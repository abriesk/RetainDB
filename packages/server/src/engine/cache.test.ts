/**
 * Unit tests for the semantic cache's scope handling (RF semantic-cache-scope).
 *
 * A cached entry may only be returned to a search whose scope matches the scope the
 * entry was computed for. SCOPE_A and SCOPE_B below stand for two different projects:
 * same org and user, different project id - the shape searchMemories builds.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { getFromSemanticCache, setInSemanticCache, searchSemanticScope } from "./cache.js";

const SCOPE_A = "org=default|project=P1|user=u1|scopes=[PROJECT]|topK=10|fastMode=1";
const SCOPE_B = "org=default|project=P2|user=u1|scopes=[PROJECT]|topK=10|fastMode=1";

const RESULTS_A = [{ memory: { id: "m1", content: "P1 fact", memoryType: "factual" }, similarity: 0.99 }];
const RESULTS_B = [{ memory: { id: "m2", content: "P2 fact", memoryType: "factual" }, similarity: 0.98 }];

/** Deterministic 16-dim vector (the cache buckets and hashes on the first dims). */
function baseEmbedding(): number[] {
  // Every value sits mid-bucket (floor(v/0.1) is 10,9,8,...,1,0 - safe from float drift).
  return [1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
}

/** Near-identical embedding: cosine similarity ~0.98 with baseEmbedding (>= the 0.85 hit threshold). */
function nearEmbedding(): number[] {
  // Quantized this differs from baseEmbedding() only on dim 0 (10 -> 9), inside the
  // bucket index neighbor range (±1 on the first 3 dims) - so without the scope check
  // this would be a hit.
  return [0.99, 0.96, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

test("semantic cache: an entry stored under scope A is not returned for scope B", async () => {
  await setInSemanticCache(baseEmbedding(), SCOPE_A, RESULTS_A);
  const hit = await getFromSemanticCache(baseEmbedding(), SCOPE_B);
  assert.equal(hit, null);
});

test("semantic cache: the same entry IS returned for its own scope A", async () => {
  await setInSemanticCache(baseEmbedding(), SCOPE_A, RESULTS_A);
  const hit = await getFromSemanticCache(baseEmbedding(), SCOPE_A);
  assert.ok(hit, "expected a hit for scope A");
  assert.equal(hit!.results, RESULTS_A);
  assert.ok(hit!.similarity >= 0.85, `similarity ${hit!.similarity} below threshold`);
});

test("semantic cache: a near-identical embedding (>= 0.85) under scope B does not hit scope A's entry", async () => {
  const similarity = cosine(nearEmbedding(), baseEmbedding());
  assert.ok(similarity >= 0.85, `test vectors must be above the hit threshold, got ${similarity}`);
  await setInSemanticCache(baseEmbedding(), SCOPE_A, RESULTS_A);
  const hit = await getFromSemanticCache(nearEmbedding(), SCOPE_B);
  assert.equal(hit, null);
  // Sanity check: without the scope difference this near-identical embedding DOES hit.
  const sameScope = await getFromSemanticCache(nearEmbedding(), SCOPE_A);
  assert.ok(sameScope, "near-identical embedding under the same scope should hit");
  assert.equal(sameScope!.results, RESULTS_A);
});

/** The inputs searchMemories holds when it checks the semantic cache, for one project,
 * one user, PROJECT scope, fast mode, no filters and a "today" temporal constraint.
 * The two questions below differ ONLY in questionDate. */
function scopeInputs(questionDate: Date) {
  return {
    orgId: "default",
    projectId: "P1",
    userId: "u1",
    sessionId: undefined,
    agentId: undefined,
    taskId: undefined,
    scopes: ["PROJECT" as const],
    includeInactive: false,
    memoryTypes: undefined,
    namespace: undefined,
    tags: undefined,
    temporal: {
      hasTemporalConstraint: true,
      relative: "today" as const,
      dateRange: {
        start: new Date("2026-11-01T00:00:00.000Z"),
        end: new Date("2026-11-01T23:59:59.999Z"),
      },
    },
    topK: 10,
    fastMode: true,
    disableLatencyDegradation: false,
    questionDate,
  };
}

test("(d) semantic-cache scope: the question's day, not its millisecond - two searches 1s apart on one UTC day share a scope, and an entry set under one hits under the other", async () => {
  const scopeA = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:00.000Z")));
  const scopeB = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:01.000Z")));
  assert.equal(scopeA, scopeB, "1s apart on the same UTC day must give the same scope");
  assert.match(scopeA, /questionDate=2026-11-01\b/, "the scope holds the question's UTC day");
  // Day precision everywhere: neither the question's time of day nor the temporal
  // filter's 23:59:59.999 range end may appear in the scope.
  assert.ok(!scopeA.includes("10:00"), "the scope must not carry the question's millisecond");
  assert.ok(scopeA.includes("start:2026-11-01"), "temporal dates serialize at UTC-day precision");
  assert.ok(!scopeA.includes("23:59:59.999"), "temporal dates serialize at UTC-day precision");

  await setInSemanticCache(baseEmbedding(), scopeA, RESULTS_A);
  const hit = await getFromSemanticCache(nearEmbedding(), scopeB);
  assert.ok(hit, "an entry set under one of the two equal scopes must hit under the other");
  assert.equal(hit!.results, RESULTS_A);
  assert.ok(hit!.similarity >= 0.85, `similarity ${hit!.similarity} below threshold`);
});

test("(e) semantic-cache scope: a different UTC day is a different scope, so the entry does not hit", async () => {
  const scopeA = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:00.000Z")));
  const scopeTomorrow = searchSemanticScope(scopeInputs(new Date("2026-11-02T10:00:00.000Z")));
  assert.notEqual(scopeA, scopeTomorrow, "a different UTC day must give a different scope");
  assert.match(scopeA, /questionDate=2026-11-01\b/);
  assert.match(scopeTomorrow, /questionDate=2026-11-02\b/);

  await setInSemanticCache(baseEmbedding(), scopeA, RESULTS_A);
  // Sanity: the entry is there and hits its own scope...
  const ownScope = await getFromSemanticCache(baseEmbedding(), scopeA);
  assert.ok(ownScope, "the entry must hit its own scope");
  // ...but the next UTC day's scope is a different scope: a miss.
  const hit = await getFromSemanticCache(baseEmbedding(), scopeTomorrow);
  assert.equal(hit, null);
});
