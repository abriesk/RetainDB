import { test, before } from "node:test";
import assert from "node:assert/strict";

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
before(async () => {
  process.env.DATABASE_URL = process.env.DATABASE_URL || "postgresql://x:x@localhost:5432/x";
  (globalThis as any).__PRISMA_MOCK__ = prismaMock;
  (globalThis as any).__INGEST_SESSION_STUB__ = async () => ingestState.result;
  process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || "test";
  process.env.OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || "http://127.0.0.1:1";
  const dbMod: any = await import("../db/index.js");
  dbMod.prisma = prismaMock;

  const memMod: any = await import("./memory/index.js");
  memMod.ingestSession = async () => ingestState.result;

  const qMod: any = await import("./ingestion-queue.js");
  ingestionQueue = qMod.ingestionQueue;
});

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

test("ingest-errors-fail: extraction errors with nothing stored marks the document FAILED", async () => {
  ingestState.result = makeResult({ memoriesCreated: 0, errors: ["boom: model refused"] });
  prismaCalls.length = 0;
  const errors: any[] = [];
  const orig = console.error;
  console.error = (...args: any[]) => { errors.push(args); };
  let r: any;
  try {
    r = await ingestionQueue.processDocument(JOB, DOC);
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
    r = await ingestionQueue.processDocument(JOB, DOC);
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
    r = await ingestionQueue.processDocument(JOB, DOC);
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
