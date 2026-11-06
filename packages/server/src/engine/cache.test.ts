/**
 * Unit tests for the semantic cache's scope handling (RF semantic-cache-scope).
 *
 * A cached entry may only be returned to a search whose scope matches the scope the
 * entry was computed for. SCOPE_A and SCOPE_B below stand for two different projects:
 * same org and user, different project id - the shape searchMemories builds.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  getFromSemanticCache,
  setInSemanticCache,
  getFromCache,
  setInCache,
  clearAfterMemoryWrite, clearSearchCaches,
  searchSemanticScope,
  searchSimpleCacheKey,
} from "./cache.js";

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
  assert.match(scopeA, /questionDate="2026-11-01"/, "the scope holds the question's UTC day");
  // Day precision everywhere: neither the question's time of day nor the temporal
  // filter's 23:59:59.999 range end may appear in the scope.
  assert.ok(!scopeA.includes("10:00"), "the scope must not carry the question's millisecond");
  assert.ok(scopeA.includes('start:"2026-11-01"'), "temporal dates serialize at UTC-day precision");
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
  assert.match(scopeA, /questionDate="2026-11-01"/);
  assert.match(scopeTomorrow, /questionDate="2026-11-02"/);

  await setInSemanticCache(baseEmbedding(), scopeA, RESULTS_A);
  // Sanity: the entry is there and hits its own scope...
  const ownScope = await getFromSemanticCache(baseEmbedding(), scopeA);
  assert.ok(ownScope, "the entry must hit its own scope");
  // ...but the next UTC day's scope is a different scope: a miss.
  const hit = await getFromSemanticCache(baseEmbedding(), scopeTomorrow);
  assert.equal(hit, null);
});

/** (f) Values that could imitate a separator are quoted: a comma inside a tag cannot
 * look like two tags, and a | inside a namespace cannot look like a part separator. */
test("(f) semantic-cache scope: a comma in a tag cannot imitate two tags, and a | in a namespace cannot imitate a separator", () => {
  const base = scopeInputs(new Date("2026-11-01T10:00:00.000Z"));

  const commad = searchSemanticScope({ ...base, tags: ["a,b"] });
  const twoTags = searchSemanticScope({ ...base, tags: ["a", "b"] });
  assert.notEqual(commad, twoTags, "one tag holding a comma must not serialize like two tags");

  const piped = searchSemanticScope({ ...base, namespace: "ns|user=other" });
  const plain = searchSemanticScope({ ...base, namespace: "ns" });
  assert.notEqual(piped, plain, "a | inside a namespace must not imitate the separator between parts");
  assert.ok(piped.includes('namespace="ns|user=other"'), "the namespace value is JSON-quoted");
});

test("(g) semantic-cache scope: one set of inputs gives one string, and two calls on the same day stay equal", () => {
  const scopeA = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:00.000Z")));
  const scopeAgain = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:00.000Z")));
  assert.equal(scopeA, scopeAgain, "the same inputs must give the same scope string");
  const oneSecondLater = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:01.000Z")));
  assert.equal(scopeA, oneSecondLater, "two calls on the same UTC day share a scope (case (d))");
  const tomorrow = searchSemanticScope(scopeInputs(new Date("2026-11-02T10:00:00.000Z")));
  assert.notEqual(scopeA, tomorrow, "the next UTC day is a different scope (case (e))");
});

test("(h) simple-cache key: the key prefix, then exactly search:<projectId>:<userId>: - what a write now clears", () => {
  const scope = searchSemanticScope(scopeInputs(new Date("2026-11-01T10:00:00.000Z")));
  const inputs = {
    projectId: "P1",
    userId: "u1",
    sessionId: undefined,
    agentId: undefined,
    taskId: undefined,
    scopes: ["PROJECT"],
    query: "qpd",
    topK: 10,
  };
  const key = searchSimpleCacheKey(inputs, scope);
  assert.equal(
    key.slice(0, "whisper:context:search:P1:u1:".length),
    "whisper:context:search:P1:u1:",
    "the key prefix plus search:<projectId>:<userId>: is the prefix clearSearchCaches matches"
  );
  assert.ok(key.endsWith(`:${scope}`), "today's key, in today's order, with the scope appended");
  const noUser = searchSimpleCacheKey({ projectId: "P1", scopes: [], query: "q", topK: 10 }, "s");
  assert.ok(noUser.startsWith("whisper:context:search:P1::"), "a missing userId keeps today's empty part");
});

test("(i) simple-cache key: keys differing only in tags, namespace, memory types, include_inactive or the temporal filter are different keys", () => {
  const base = { projectId: "P1", userId: "u1", scopes: ["PROJECT"], query: "qpd", topK: 10 };
  const day = new Date("2026-11-01T10:00:00.000Z");
  const keyFor = (extra: Partial<Parameters<typeof searchSemanticScope>[0]>) =>
    searchSimpleCacheKey(base, searchSemanticScope({ ...scopeInputs(day), ...extra }));

  const plain = keyFor({});
  assert.notEqual(keyFor({ tags: ["a"] }), plain, "a different tag filter is a different key");
  assert.notEqual(keyFor({ namespace: "ns" }), plain, "a different namespace is a different key");
  assert.notEqual(keyFor({ memoryTypes: ["factual"] }), plain, "a different memory-type filter is a different key");
  assert.notEqual(keyFor({ includeInactive: true }), plain, "a different include_inactive is a different key");
  const yesterday = {
    hasTemporalConstraint: true as const,
    relative: "today" as const,
    dateRange: {
      start: new Date("2026-10-31T00:00:00.000Z"),
      end: new Date("2026-10-31T23:59:59.999Z"),
    },
  };
  assert.notEqual(keyFor({ temporal: yesterday }), plain, "a different temporal filter is a different key");
});

/** One simple-cache key for a search of this project and user (RF search-cache-clear). */
function simpleEntry(projectId: string, userId: string, scope: string): string {
  return searchSimpleCacheKey(
    {
      projectId,
      userId,
      sessionId: undefined,
      agentId: undefined,
      taskId: undefined,
      scopes: ["PROJECT"],
      query: "qpd",
      topK: 10,
    },
    scope
  );
}

test("(j) simple cache: clearSearchCaches removes the writing user's entry, and another user's survives [pin]", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  const mine = simpleEntry("P1", "u1", searchSemanticScope(scopeInputs(day)));
  const theirs = simpleEntry("P1", "u2", searchSemanticScope({ ...scopeInputs(day), userId: "u2" }));
  await setInCache(mine, RESULTS_A);
  await setInCache(theirs, RESULTS_B);
  assert.ok(await getFromCache(mine), "the entry is there before the write");

  await clearSearchCaches("P1", "u1");

  assert.equal(await getFromCache(mine), null, "a write by u1 must clear u1's simple search entry");
  assert.ok(await getFromCache(theirs), "another user's entry survives the write");
});

test("(k) semantic cache: clearSearchCaches removes the project+user scope's entry; another user's and another project's survive [pin]", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  const scopeP1U1 = searchSemanticScope(scopeInputs(day));
  const scopeP1U2 = searchSemanticScope({ ...scopeInputs(day), userId: "u2" });
  const scopeP2U1 = searchSemanticScope({ ...scopeInputs(day), projectId: "P2" });
  await setInSemanticCache(baseEmbedding(), scopeP1U1, RESULTS_A);
  await setInSemanticCache(baseEmbedding(), scopeP1U2, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), scopeP2U1, RESULTS_B);
  assert.ok(await getFromSemanticCache(baseEmbedding(), scopeP1U1), "the entry is there before the write");

  await clearSearchCaches("P1", "u1");

  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP1U1), null, "a write by u1 in P1 must clear P1/u1's semantic entry");
  assert.ok(await getFromSemanticCache(baseEmbedding(), scopeP1U2), "another user's semantic entry survives");
  assert.ok(await getFromSemanticCache(baseEmbedding(), scopeP2U1), "another project's semantic entry survives");
});

test("(l) a project-wide write (no userId) clears every user's entries of that project, and no other project's", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  const scopeP3U1 = searchSemanticScope({ ...scopeInputs(day), projectId: "P3" });
  const scopeP3U2 = searchSemanticScope({ ...scopeInputs(day), projectId: "P3", userId: "u2" });
  const scopeP4U1 = searchSemanticScope({ ...scopeInputs(day), projectId: "P4" });
  const p3u1Simple = simpleEntry("P3", "u1", scopeP3U1);
  const p3u2Simple = simpleEntry("P3", "u2", scopeP3U2);
  const p4u1Simple = simpleEntry("P4", "u1", scopeP4U1);
  await setInCache(p3u1Simple, RESULTS_A);
  await setInCache(p3u2Simple, RESULTS_B);
  await setInCache(p4u1Simple, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), scopeP3U1, RESULTS_A);
  await setInSemanticCache(baseEmbedding(), scopeP3U2, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), scopeP4U1, RESULTS_B);

  await clearSearchCaches("P3");

  assert.equal(await getFromCache(p3u1Simple), null, "one user's simple entry of the project goes");
  assert.equal(await getFromCache(p3u2Simple), null, "every user's simple entry of the project goes");
  assert.ok(await getFromCache(p4u1Simple), "another project's simple entry survives");
  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP3U1), null, "one user's semantic entry of the project goes");
  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP3U2), null, "every user's semantic entry of the project goes");
  assert.ok(await getFromSemanticCache(baseEmbedding(), scopeP4U1), "another project's semantic entry survives");
});

test("(m) a userId holding `.` or `+` clears only its own entries", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  // `+`: as a regex, `u+1:` also matches `u1:` - unescaped it would clear another user.
  const plusScope = searchSemanticScope({ ...scopeInputs(day), projectId: "P5", userId: "u+1" });
  const plusSiblingScope = searchSemanticScope({ ...scopeInputs(day), projectId: "P5", userId: "u1" });
  // `.`: as a regex, `u.1:` also matches `u-1:` - unescaped it would clear another user.
  const dotScope = searchSemanticScope({ ...scopeInputs(day), projectId: "P6", userId: "u.1" });
  const dotSiblingScope = searchSemanticScope({ ...scopeInputs(day), projectId: "P6", userId: "u-1" });
  const plus = simpleEntry("P5", "u+1", plusScope);
  const plusSibling = simpleEntry("P5", "u1", plusSiblingScope);
  const dot = simpleEntry("P6", "u.1", dotScope);
  const dotSibling = simpleEntry("P6", "u-1", dotSiblingScope);
  await setInCache(plus, RESULTS_A);
  await setInCache(plusSibling, RESULTS_B);
  await setInCache(dot, RESULTS_A);
  await setInCache(dotSibling, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), plusScope, RESULTS_A);
  await setInSemanticCache(baseEmbedding(), plusSiblingScope, RESULTS_B);

  await clearSearchCaches("P5", "u+1");
  await clearSearchCaches("P6", "u.1");

  assert.equal(await getFromCache(plus), null, "u+1's own simple entry is cleared");
  assert.ok(await getFromCache(plusSibling), "u+1 must not match u1: the clear's pattern is escaped");
  assert.equal(await getFromCache(dot), null, "u.1's own simple entry is cleared");
  assert.ok(await getFromCache(dotSibling), "u.1 must not match u-1: the clear's pattern is escaped");
  assert.equal(await getFromSemanticCache(baseEmbedding(), plusScope), null, "u+1's own semantic entry is cleared");
  assert.ok(await getFromSemanticCache(baseEmbedding(), plusSiblingScope), "u1's semantic entry survives");
});

test("(n) the retriever's context entries go by project, never by user", async () => {
  const p1Context = "whisper:context:query:P1:x";
  const p2Context = "whisper:context:query:P2:x";
  await setInCache(p1Context, RESULTS_A);
  await setInCache(p2Context, RESULTS_A);

  await clearSearchCaches("P1", "u1");

  assert.equal(await getFromCache(p1Context), null, "the cleared project's context entry goes");
  assert.ok(await getFromCache(p2Context), "another project's context entry survives");
});

test("(o) clearAfterMemoryWrite on a PROJECT memory clears every user's entries", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  const scopeP1U1 = searchSemanticScope({ ...scopeInputs(day), projectId: "P1", userId: "u1" });
  const scopeP1U2 = searchSemanticScope({ ...scopeInputs(day), projectId: "P1", userId: "u2" });
  const p1u1Simple = simpleEntry("P1", "u1", scopeP1U1);
  const p1u2Simple = simpleEntry("P1", "u2", scopeP1U2);
  await setInCache(p1u1Simple, RESULTS_A);
  await setInCache(p1u2Simple, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), scopeP1U1, RESULTS_A);
  await setInSemanticCache(baseEmbedding(), scopeP1U2, RESULTS_B);

  await clearAfterMemoryWrite({ projectId: "P1", scope: "PROJECT" }, "u1");

  assert.equal(await getFromCache(p1u1Simple), null, "the writing user's simple entry goes");
  assert.equal(await getFromCache(p1u2Simple), null, "another user's simple entry of the project goes");
  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP1U1), null, "the writing user's semantic entry goes");
  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP1U2), null, "another user's semantic entry of the project goes");
});

test("(p) clearAfterMemoryWrite on a USER memory clears only the writing user's entries", async () => {
  const day = new Date("2026-11-01T10:00:00.000Z");
  const scopeP1U1 = searchSemanticScope({ ...scopeInputs(day), projectId: "P1", userId: "u1" });
  const scopeP1U2 = searchSemanticScope({ ...scopeInputs(day), projectId: "P1", userId: "u2" });
  const p1u1Simple = simpleEntry("P1", "u1", scopeP1U1);
  const p1u2Simple = simpleEntry("P1", "u2", scopeP1U2);
  await setInCache(p1u1Simple, RESULTS_A);
  await setInCache(p1u2Simple, RESULTS_B);
  await setInSemanticCache(baseEmbedding(), scopeP1U1, RESULTS_A);
  await setInSemanticCache(baseEmbedding(), scopeP1U2, RESULTS_B);

  await clearAfterMemoryWrite({ projectId: "P1", scope: "USER" }, "u1");

  assert.equal(await getFromCache(p1u1Simple), null, "the writing user's simple entry goes");
  assert.ok(await getFromCache(p1u2Simple), "another user's simple entry survives");
  assert.equal(await getFromSemanticCache(baseEmbedding(), scopeP1U1), null, "the writing user's semantic entry goes");
  assert.ok(await getFromSemanticCache(baseEmbedding(), scopeP1U2), "another user's semantic entry survives");
});
