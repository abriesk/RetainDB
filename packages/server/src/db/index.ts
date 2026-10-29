import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

function buildDatabaseUrl(): string | undefined {
  const raw = process.env.DATABASE_URL;
  if (!raw) return raw;

  try {
    const url = new URL(raw);

    if (process.env.DB_CONNECTION_LIMIT && !url.searchParams.has("connection_limit")) {
      url.searchParams.set("connection_limit", process.env.DB_CONNECTION_LIMIT);
    }
    if (process.env.DB_POOL_TIMEOUT && !url.searchParams.has("pool_timeout")) {
      url.searchParams.set("pool_timeout", process.env.DB_POOL_TIMEOUT);
    }
    if (process.env.DB_MAX_CONNECTION_LIFETIME && !url.searchParams.has("max_connection_lifetime")) {
      url.searchParams.set("max_connection_lifetime", process.env.DB_MAX_CONNECTION_LIFETIME);
    }

    return url.toString();
  } catch {
    return raw;
  }
}

export const db = globalForPrisma.prisma ?? new PrismaClient({
  datasources: {
    db: {
      url: buildDatabaseUrl(),
    },
  },
});
export const prisma = db;

// BigInt results are stringified when serialized to JSON. Prisma 6.19.3 removed
// $use, so the middleware is replaced by this equivalent effect at the JSON boundary
// (arrays are covered automatically: JSON.stringify walks them).
(BigInt.prototype as any).toJSON = function () { return this.toString(); };

// Fast check before doing full recursive traversal
function hasBigInt(obj: any): boolean {
  if (obj === null || obj === undefined) return false;
  if (typeof obj === "bigint") return true;
  if (Array.isArray(obj)) return obj.some(hasBigInt);
  if (typeof obj === "object") return Object.values(obj).some(hasBigInt);
  return false;
}

function convertBigIntToString(obj: any): any {
  if (obj === null || obj === undefined) return obj;
  if (typeof obj === "bigint") return obj.toString();
  if (Array.isArray(obj)) return obj.map(convertBigIntToString);
  if (typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [k, convertBigIntToString(v)])
    );
  }
  return obj;
}

// Cache singleton in production to prevent exhausting connection pool on hot reloads
if (process.env.NODE_ENV === "production") {
  globalForPrisma.prisma = db;
}

// Re-export Prisma types for convenience
export { Prisma } from "@prisma/client";
export type {
  Project,
  Source,
  Document,
  Chunk,
  Embedding,
  Entity,
  EntityRelation,
  Memory,
  MemoryRelation,
  Session,
  Message,
  ChunkMemory,
  SyncJob,
  Webhook,
  WebhookDelivery,
  SharedFile,
  AgentTask,
  AgentRun,
  AgentRunStep,
} from "@prisma/client";
