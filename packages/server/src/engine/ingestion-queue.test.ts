import { test, before } from "node:test";
import assert from "node:assert/strict";
import { extractImplicitMemories } from "./memory/inference.js";
import { extractMemories as extractMemoriesUnified } from "./memory/extractor-unified.js";

const prismaCalls: any[] = [];
const prismaMock = {
  $executeRaw: async (...args: any[]) => {
    prismaCalls.push(args);
    return 1;
  },
  $queryRaw: async (..._args: any[]) => [],
};

const ingestState: { result: any } = {
  result: {
    memoriesCreated: 0,
    relationsCreated: 0,
    memoriesInvalidated: 0,
    errors: [],
    scopeCounts: {},
    scopesTouched: [],
  },
};

let ingestionQueue: any;
let documentOutcome: any;
before(async () => {
  process.env.DATABASE_URL = process.env.DATABASE_URL || "postgresql://u:p@localhost:5432/x";
  // engine/extractor.ts builds its client at import; these keep the test offline.
  process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || "test";
  process.env.OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || "http://127.0.0.1:1";
  const qMod: any = await import("./ingestion-queue.js");
  ingestionQueue = qMod.ingestionQueue;
  documentOutcome = qMod.documentOutcome;
});

// The mocks go in as deps, so no global hooks are needed.
const DEPS = () => ({
  prisma: prismaMock,
  ingestSession: async () => ingestState.result,
});

// Point the inference model at a dead port for the duration of the callback.
async function withDeadModelEndpoint(fn: () => Promise<void>) {
  const prevUrl = process.env.OPENAI_BASE_URL;
  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_BASE_URL = "http://127.0.0.1:1";
  process.env.OPENAI_API_KEY = prevKey || "test";
  try {
    await fn();
  } finally {
    if (prevUrl === undefined) delete process.env.OPENAI_BASE_URL; else process.env.OPENAI_BASE_URL = prevUrl;
    if (prevKey === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = prevKey;
  }
}

const JOB = { id: "job-1", project_id: "proj-1", org_id: "org-1", metadata: {} };
const DOC = {
  id: "doc-1",
  metadata: {
    type: "conversation",
    messages: [{ role: "user", content: "hello world", timestamp: new Date().toISOString() }],
  },
};

function makeResult(overrides: any) {
  return {
    memoriesCreated: 0,
    relationsCreated: 0,
    memoriesInvalidated: 0,
    errors: [],
    scopeCounts: {},
    scopesTouched: [],
    ...overrides,
  };
}

// ── (d) documentOutcome ───────────────────────────────────────────────────

test("documentOutcome: all documents failed reports the first error", () => {
  const settled = [
    { status: "fulfilled", value: { success: false, error: "first failed" } },
    { status: "fulfilled", value: { success: false, error: "second failed" } },
  ];
  const o = documentOutcome(settled, 2);
  assert.equal(o.failed, 2);
  assert.equal(o.firstError, "first failed");
});

test("documentOutcome: some documents failed counts them and keeps the first error", () => {
  const settled = [
    { status: "fulfilled", value: { success: true } },
    { status: "fulfilled", value: { success: false, error: "third failed" } },
    { status: "fulfilled", value: { success: true } },
  ];
  const o = documentOutcome(settled, 3);
  assert.equal(o.failed, 1);
  assert.equal(o.firstError, "third failed");
});

test("documentOutcome: none failed reports zero and no error", () => {
  const settled = [
    { status: "fulfilled", value: { success: true } },
    { status: "fulfilled", value: { success: true } },
  ];
  const o = documentOutcome(settled, 2);
  assert.equal(o.failed, 0);
  assert.equal(o.firstError, undefined);
});

test("documentOutcome: a rejected allSettled result counts as a failed document", () => {
  const settled = [
    { status: "rejected", reason: new Error("rejected boom") },
    { status: "fulfilled", value: { success: true } },
  ];
  const o = documentOutcome(settled, 2);
  assert.equal(o.failed, 1);
  assert.equal(o.firstError, "rejected boom");
});

test("documentOutcome: a job with no documents is not a failed job", () => {
  const o = documentOutcome([], 0);
  assert.equal(o.failed, 0);
  assert.equal(o.firstError, undefined);
  assert.equal(o.allFailed, false);
});

// ── (e) extractImplicitMemories against a dead model endpoint ─────────────

test("extractImplicitMemories: a dead endpoint resolves [] and calls onError once", async () => {
  await withDeadModelEndpoint(async () => {
    let calls = 0;
    const res = await extractImplicitMemories(
      "the weather is quite pleasant today, nothing explicit here",
      "",
      { onError: () => { calls++; } },
    );
    assert.deepEqual(res, []);
    assert.equal(calls, 1);
  });
});

test("extractImplicitMemories: a dead endpoint with no onError resolves [] and does not throw", async () => {
  await withDeadModelEndpoint(async () => {
    const res = await extractImplicitMemories(
      "the weather is quite pleasant today, nothing explicit here",
      "",
    );
    assert.deepEqual(res, []);
  });
});

// ── (f) extractMemories keeps the pattern match, fires onInferenceError ───

test("extractMemories: a pattern match survives a dead endpoint and onInferenceError fires", async () => {
  await withDeadModelEndpoint(async () => {
    let inferred = 0;
    const result = await extractMemoriesUnified(
      "I live in San Francisco, by the way",
      "",
      { onInferenceError: () => { inferred++; } },
    );
    assert.ok(result.explicit.length >= 1);
    assert.ok(result.all.some(m => /lives in San Francisco/i.test(m.content)));
    assert.equal(inferred, 1);
  });
});

// ── the three existing ingest-errors-fail cases, now passing deps ─────────

test("ingest-errors-fail: extraction errors with nothing stored marks the document FAILED", async () => {
  ingestState.result = makeResult({ memoriesCreated: 0, errors: ["boom: model refused"] });
  prismaCalls.length = 0;
  const errors: any[] = [];
  const orig = console.error;
  console.error = (...args: any[]) => { errors.push(args); };
  let r: any;
  try {
    r = await ingestionQueue.processDocument(JOB, DOC, DEPS());
  } finally {
    console.error = orig;
  }
  assert.equal(r.success, false);
  assert.equal(r.error, "boom: model refused");
  const joined = prismaCalls.map((c) => JSON.stringify(c)).join("\n");
  assert.match(joined, /FAILED/);
  assert.match(joined, /boom: model refused/);
  assert.doesNotMatch(joined, /processed_chunks/);
});

test("ingest-errors-fail: no errors marks the document COMPLETED as before", async () => {
  ingestState.result = makeResult({ memoriesCreated: 2, errors: [] });
  prismaCalls.length = 0;
  const errors: any[] = [];
  const orig = console.error;
  console.error = (...args: any[]) => { errors.push(args); };
  let r: any;
  try {
    r = await ingestionQueue.processDocument(JOB, DOC, DEPS());
  } finally {
    console.error = orig;
  }
  assert.equal(r.success, true);
  const joined = prismaCalls.map((c) => JSON.stringify(c)).join("\n");
  assert.match(joined, /COMPLETED/);
  assert.equal(errors.length, 0);
});

test("ingest-errors-fail: partial success stays COMPLETED and logs once with the document id", async () => {
  ingestState.result = makeResult({ memoriesCreated: 1, errors: ["one write failed"] });
  prismaCalls.length = 0;
  const errors: any[] = [];
  const orig = console.error;
  console.error = (...args: any[]) => { errors.push(args); };
  let r: any;
  try {
    r = await ingestionQueue.processDocument(JOB, DOC, DEPS());
  } finally {
    console.error = orig;
  }
  assert.equal(r.success, true);
  const joined = prismaCalls.map((c) => JSON.stringify(c)).join("\n");
  assert.match(joined, /COMPLETED/);
  assert.match(joined, /processed_chunks/);
  assert.equal(errors.length, 1);
  assert.match(JSON.stringify(errors[0]), /doc-1/);
});
