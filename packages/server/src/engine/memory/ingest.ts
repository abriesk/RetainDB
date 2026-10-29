/**
 * SOTA Memory Ingestion Pipeline
 * Session-based ingestion with extraction, disambiguation, and relation detection
 */

import { db } from "../../db/index.js";
import { buildEntityContext, validateMemory } from "./extractor.js";
import { extractMemories as extractMemoriesUnified } from "./extractor-unified.js";
import { extractEventDate } from "./temporal.js";
import type { ExtractionContext, MemoryScopeTarget, PromotionMode, SessionWorkEvent } from "./types.js";
import {
  emitMemoryConflictDetectedEvent,
  getTenantExtractionPolicy,
} from "../extraction-observability.js";
import { writeMemoryCanonical } from "./write.js";

export interface IngestionResult {
  memoriesCreated: number;
  relationsCreated: number;
  memoriesInvalidated: number;
  errors: string[];
  scopeCounts: Partial<Record<Exclude<MemoryScopeTarget, "DROPPED">, number>>;
  scopesTouched: Exclude<MemoryScopeTarget, "DROPPED">[];
}

const MEMORY_CONFLICT_AUTOMATION_ENABLED = /^true$/i.test(
  process.env.MEMORY_CONFLICT_AUTOMATION_ENABLED || "false"
);
const MEMORY_CONFLICT_DEMOTE_DELTA = Math.max(
  Math.min(parseFloat(process.env.MEMORY_CONFLICT_DEMOTE_DELTA || "0.2"), 1),
  0
);
const MEMORY_CONFLICT_DEACTIVATE_BELOW = Math.max(
  Math.min(parseFloat(process.env.MEMORY_CONFLICT_DEACTIVATE_BELOW || "0.25"), 1),
  0
);

function buildSessionExtractionWindow(messages: Array<{ role: string; content: string }>) {
  if (messages.length === 0) {
    return {
      sourceMessageIds: [] as string[],
      extractionInput: "",
      previousMessages: [] as string[],
    };
  }
  const selectedIndices = new Set<number>();
  const userIndices = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.role === "user" && message.content?.trim())
    .slice(-4)
    .map(({ index }) => index);

  for (const index of userIndices) {
    selectedIndices.add(index);
  }

  let assistantCount = 0;
  for (const index of userIndices) {
    for (const candidate of [index - 1, index + 1]) {
      if (assistantCount >= 2) break;
      const message = messages[candidate];
      if (!message || message.role !== "assistant" || !message.content?.trim()) continue;
      selectedIndices.add(candidate);
      assistantCount += 1;
    }
  }

  const orderedWindow = messages.filter((_, index) => selectedIndices.has(index));
  return {
    sourceMessageIds: Array.from(selectedIndices)
      .sort((left, right) => left - right)
      .map((index) => `${index}`),
    extractionInput:
      orderedWindow.length > 0
        ? orderedWindow.map((message) => `${message.role}: ${message.content.trim()}`).join("\n")
        : messages[messages.length - 1].content,
    previousMessages: messages
      .slice(Math.max(0, messages.length - 6), -1)
      .map((message) => `${message.role}: ${message.content}`),
  };
}

function buildUnifiedContext(context: ExtractionContext): string {
  const parts: string[] = [];
  if (context.previousMessages?.length) {
    parts.push(context.previousMessages.join("\n"));
  }
  if (context.entityContext?.size) {
    parts.push(
      Array.from(context.entityContext.entries())
        .map(([pronoun, entity]) => `${pronoun} => ${entity}`)
        .join("\n")
    );
  }
  return parts.join("\n");
}

function recordScope(result: IngestionResult, scopeTarget: MemoryScopeTarget): void {
  if (scopeTarget === "DROPPED") return;
  result.scopeCounts[scopeTarget] = (result.scopeCounts[scopeTarget] || 0) + 1;
  if (!result.scopesTouched.includes(scopeTarget)) {
    result.scopesTouched.push(scopeTarget);
  }
}

function supportsAssistantPromotion(memoryType: string, events: SessionWorkEvent[]): boolean {
  if (events.length === 0) return false;
  if (memoryType === "decision") {
    return events.some((event) => event.kind === "decision" || event.kind === "task_update");
  }
  if (memoryType === "constraint") {
    return events.some((event) => event.kind === "constraint" || event.kind === "failure");
  }
  if (memoryType === "solution" || memoryType === "correction") {
    return events.some((event) => event.kind === "outcome" || event.kind === "tool_result" || event.success === true);
  }
  return false;
}

function eventWritePayload(
  event: SessionWorkEvent,
  context: {
    projectId: string;
    orgId?: string;
    userId?: string;
    sessionId: string;
    agentId?: string;
    taskId?: string;
    documentDate: Date;
    sessionRetentionDays: number;
    promotionMode?: PromotionMode;
  }
) {
  const timestamp = event.timestamp ? new Date(event.timestamp) : context.documentDate;
  const salience = event.salience || "medium";
  const confidenceRaw =
    salience === "high" ? 0.9 :
    salience === "medium" ? 0.8 : 0.68;

  const content = [event.summary, event.details].filter(Boolean).join(". ").trim();
  if (!content) return null;

  const base = {
    projectId: context.projectId,
    orgId: context.orgId,
    userId: context.userId,
    sessionId: context.sessionId,
    agentId: context.agentId,
    taskId: context.taskId,
    content,
    eventDate: timestamp,
    documentDate: context.documentDate,
    sessionRetentionDays: context.sessionRetentionDays,
    promotionMode: context.promotionMode,
    writeSource: "memory.ingest.session.event",
    writeMode: "session_extract" as const,
    extractionMethod: "manual",
    sourceRole: "event" as const,
    supportingEvent: true,
    metadata: {
      event_kind: event.kind,
      event_salience: salience,
      file_paths: event.filePaths || [],
      tool_name: event.toolName || null,
      success: event.success ?? null,
    },
  };

  switch (event.kind) {
    case "decision":
      return { ...base, memoryType: "decision", confidenceRaw, scopeHint: context.taskId ? "TASK" as const : "PROJECT" as const };
    case "constraint":
    case "failure":
      return { ...base, memoryType: "constraint", confidenceRaw, scopeHint: context.taskId ? "TASK" as const : "PROJECT" as const };
    case "outcome":
    case "tool_result":
      return {
        ...base,
        memoryType: event.success === false ? "constraint" : "solution",
        confidenceRaw,
        scopeHint: context.taskId ? "TASK" as const : "PROJECT" as const,
      };
    case "task_update":
      return { ...base, memoryType: "project_state", confidenceRaw, scopeHint: context.taskId ? "TASK" as const : "PROJECT" as const };
    case "file_edit":
      return {
        ...base,
        memoryType: "workflow",
        confidenceRaw: Math.max(confidenceRaw - 0.05, 0.6),
        scopeHint: context.agentId ? "AGENT" as const : (context.taskId ? "TASK" as const : "PROJECT" as const),
      };
    default:
      return { ...base, memoryType: "event", confidenceRaw, scopeHint: "SESSION" as const };
  }
}

async function ensureSessionPersistence(params: {
  sessionId: string;
  projectId: string;
  orgId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  messages: Array<{
    role: string;
    content: string;
    timestamp: Date;
  }>;
}) {
  const { sessionId, projectId, orgId, userId, agentId, taskId, messages } = params;
  const latestTimestamp = messages[messages.length - 1]?.timestamp || new Date();
  const session =
    await db.session.findFirst({
      where: {
        projectId,
        OR: [{ id: sessionId }, { sessionId }],
      },
    })
    || await db.session.create({
      data: {
        orgId,
        projectId,
        sessionId,
        userId,
        agentId,
        taskId,
        title: messages.find((message) => message.role === "user")?.content?.slice(0, 120) || `Session ${sessionId}`,
        lastMessageAt: latestTimestamp,
      },
    });

  if (messages.length > 0) {
    await db.message.createMany({
      data: messages.map((message) => ({
        sessionId: session.id,
        role: message.role,
        content: message.content,
        metadata: {
          source: "memory.ingest.session",
          external_session_id: sessionId,
          original_timestamp: message.timestamp.toISOString(),
        },
        createdAt: message.timestamp,
      })),
    });
  }

  await db.session.update({
    where: { id: session.id },
    data: {
      agentId: agentId || undefined,
      taskId: taskId || undefined,
      lastMessageAt: latestTimestamp,
      messageCount: { increment: messages.length },
      updatedAt: latestTimestamp,
    },
  });

  return session;
}

/**
 * Ingest a session (multiple messages/chunks) and extract memories
 * This is the main entry point for memory creation
 */
export async function ingestSession(params: {
  sessionId: string;
  projectId: string;
  orgId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  events?: SessionWorkEvent[];
  promotionMode?: PromotionMode;
  messages: Array<{
    role: string;
    content: string;
    timestamp: Date;
  }>;
}): Promise<IngestionResult> {
  const {
    sessionId,
    projectId,
    orgId,
    userId,
    agentId,
    taskId,
    events = [],
    promotionMode = "session_state_v1",
    messages,
  } = params;

  const result: IngestionResult = {
    memoriesCreated: 0,
    relationsCreated: 0,
    memoriesInvalidated: 0,
    errors: [],
    scopeCounts: {},
    scopesTouched: [],
  };

  if (messages.length === 0 && events.length === 0) {
    return result;
  }

  try {
    await ensureSessionPersistence({ sessionId, projectId, orgId, userId, agentId, taskId, messages });

    // Build context for extraction
    const window = buildSessionExtractionWindow(messages);
    const context: ExtractionContext = {
      sessionId,
      userId: userId || "unknown",
      projectId,
      orgId,
      agentId,
      taskId,
      promotionMode,
      documentDate: messages[messages.length - 1]?.timestamp || new Date(),
      previousMessages: window.previousMessages,
    };

    // Get recent memories for entity context building
    const recentMemories = await db.memory.findMany({
      where: {
        sessionId,
        projectId,
        isActive: true,
      },
      orderBy: {
        createdAt: "desc",
      },
      take: 20,
      select: {
        content: true,
        entityMentions: true,
        confidence: true,
      },
    });

    // Build entity context for pronoun resolution
    context.entityContext = buildEntityContext(recentMemories);

    const tenantPolicy = orgId
      ? await getTenantExtractionPolicy(orgId).catch(() => null)
      : null;
    const extraction = await extractMemoriesUnified(
      window.extractionInput,
      buildUnifiedContext(context),
      {
        enablePattern: true,
        enableInference: true,
        tieredEscalation: Boolean(
          tenantPolicy?.orchestrator_v2_enabled && tenantPolicy?.tiered_escalation_enabled
        ),
      }
    );
    const extractedMemories = extraction.all;

    // Filter valid memories
    const validMemories = extractedMemories.filter(validateMemory);

    const sessionRetentionDays = tenantPolicy?.session_only_retention_days || 14;
    for (const extracted of validMemories) {
      try {
        const eventDate = extracted.eventDate || await extractEventDate(
          extracted.content,
          context.documentDate
        );
        const writeResult = await writeMemoryCanonical({
          projectId,
          orgId,
          userId,
          sessionId,
          agentId,
          taskId,
          content: extracted.content,
          memoryType: extracted.memoryType,
          entityMentions: extracted.entityMentions,
          confidenceRaw: extracted.confidence,
          documentDate: context.documentDate,
          eventDate,
          metadata: {
            reasoning: extracted.reasoning,
            extraction_policy: tenantPolicy
              ? {
                  orchestrator_v2_enabled: tenantPolicy.orchestrator_v2_enabled,
                  threshold_enforcement_active: tenantPolicy.threshold_enforcement_active,
                  threshold_enforcement_reason: tenantPolicy.threshold_enforcement_reason,
                }
              : undefined,
          },
          writeSource: "memory.ingest.session",
          writeMode: "session_extract",
          extractionMethod:
            extracted.inferred
              ? extraction.extractionMethod === "hybrid"
                ? "inference"
                : extraction.extractionMethod
              : extraction.extractionMethod === "hybrid"
                ? "pattern"
                : extraction.extractionMethod,
          sourceMessageIds: window.sourceMessageIds.map((id) => `${sessionId}:${id}`),
          sessionRetentionDays,
          pendingOverlayTtlMs: 10000,
          sourceRole: extracted.sourceRole,
          userConfirmed: extracted.userConfirmed,
          supportingEvent: extracted.supportingEvent || (extracted.sourceRole === "assistant" && supportsAssistantPromotion(extracted.memoryType, events)),
          promotionMode,
        });

        if (writeResult.outcome === "dropped") {
          continue;
        }

        if (writeResult.outcome === "created") result.memoriesCreated++;
        result.relationsCreated += writeResult.relationCount;
        result.memoriesInvalidated += writeResult.invalidatedCount;
        recordScope(result, writeResult.scopeTarget);

        if (orgId && writeResult.memory && writeResult.relationCount > 0) {
          const relatedRows = await db.memoryRelation.findMany({
            where: { fromMemoryId: writeResult.memory.id },
            select: {
              toMemoryId: true,
              relationType: true,
              reasoning: true,
            },
          });

          for (const relation of relatedRows) {
            if (relation.relationType === "contradicts") {
              emitMemoryConflictDetectedEvent({
                tenantId: orgId,
                projectId,
                conflictTargetMemoryId: relation.toMemoryId,
                conflictEvidenceMemoryId: writeResult.memory.id,
                conflictType: "contradicts",
                recommendedAction: MEMORY_CONFLICT_AUTOMATION_ENABLED ? "demote" : "needs_review",
                metadata: {
                  relation_type: relation.relationType,
                  reasoning: relation.reasoning,
                },
              }).catch((eventError) => {
                console.warn("[Phase0] Failed to emit conflict event:", eventError);
              });

              if (MEMORY_CONFLICT_AUTOMATION_ENABLED) {
                const target = await db.memory.findUnique({
                  where: { id: relation.toMemoryId },
                  select: {
                    confidence: true,
                    metadata: true,
                  },
                });
                if (target) {
                  const updatedConfidence = Math.max(0, target.confidence - MEMORY_CONFLICT_DEMOTE_DELTA);
                  const currentMetadata =
                    target.metadata && typeof target.metadata === "object" && !Array.isArray(target.metadata)
                      ? (target.metadata as Record<string, unknown>)
                      : {};
                  await db.memory.update({
                    where: { id: relation.toMemoryId },
                    data: {
                      confidence: updatedConfidence,
                      isActive: updatedConfidence > MEMORY_CONFLICT_DEACTIVATE_BELOW,
                      validUntil:
                        updatedConfidence > MEMORY_CONFLICT_DEACTIVATE_BELOW
                          ? undefined
                          : new Date(),
                      metadata: {
                        ...currentMetadata,
                        conflict_demoted: true,
                        conflict_demoted_at: new Date().toISOString(),
                        conflict_evidence_memory_id: writeResult.memory.id,
                        conflict_demotion_reason: relation.reasoning || null,
                      },
                    },
                  });
                }
              }
            }
          }
        }
      } catch (error) {
        result.errors.push(`Failed to process memory: ${error}`);
      }
    }

    for (const event of events) {
      try {
        const payload = eventWritePayload(event, {
          projectId,
          orgId,
          userId,
          sessionId,
          agentId,
          taskId,
          documentDate: context.documentDate,
          sessionRetentionDays,
          promotionMode,
        });
        if (!payload) continue;

        const writeResult = await writeMemoryCanonical(payload);
        if (writeResult.outcome === "dropped" || !writeResult.memory) {
          continue;
        }

        if (writeResult.outcome === "created") result.memoriesCreated++;
        result.relationsCreated += writeResult.relationCount;
        result.memoriesInvalidated += writeResult.invalidatedCount;
        recordScope(result, writeResult.scopeTarget);
      } catch (error) {
        result.errors.push(`Failed to process event memory: ${error}`);
      }
    }

    return result;
  } catch (error) {
    result.errors.push(`Ingestion failed: ${error}`);
    return result;
  }
}

/**
 * Ingest a single chunk (from document indexing)
 * Different from session ingestion - no conversation context
 */
export async function ingestChunk(params: {
  chunkId: string;
  chunkContent: string;
  projectId: string;
  orgId?: string;
  documentDate: Date;
  metadata?: Record<string, any>;
}): Promise<IngestionResult> {
  const { chunkId, chunkContent, projectId, orgId, documentDate, metadata } = params;

  const result: IngestionResult = {
    memoriesCreated: 0,
    relationsCreated: 0,
    memoriesInvalidated: 0,
    errors: [],
    scopeCounts: {},
    scopesTouched: [],
  };

  try {
    const context: ExtractionContext = {
      sessionId: `chunk_${chunkId}`,
      userId: "system",
      projectId,
      orgId,
      documentDate,
    };

    const extraction = await extractMemoriesUnified(chunkContent, buildUnifiedContext(context), {
      enablePattern: true,
      enableInference: true,
    });
    const extractedMemories = extraction.all;
    const validMemories = extractedMemories.filter(validateMemory);

    for (const extracted of validMemories) {
      const eventDate = extracted.eventDate || await extractEventDate(
        extracted.content,
        documentDate
      );

      const writeResult = await writeMemoryCanonical({
        projectId,
        orgId,
        sourceChunkId: chunkId,
        content: extracted.content,
        memoryType: extracted.memoryType,
        entityMentions: extracted.entityMentions,
        confidenceRaw: extracted.confidence,
        documentDate,
        eventDate,
        metadata: {
          ...metadata,
          reasoning: extracted.reasoning,
          parser: metadata?.parser || null,
          parser_confidence: metadata?.parser_confidence || null,
          source_family: metadata?.source_type || metadata?.source_family || null,
          source_span: metadata?.source_span || {
            chunk_id: chunkId,
            page: metadata?.page || null,
            heading_path: metadata?.heading_path || null,
            section_path: metadata?.section_path || null,
          },
        },
        writeSource: "memory.ingest.chunk",
        writeMode: "source_extract",
        extractionMethod:
          metadata?.extraction_method
          || (extracted.inferred ? extraction.extractionMethod : extraction.extractionMethod === "hybrid" ? "pattern" : extraction.extractionMethod),
        sourceChunkIds: [chunkId],
        scopeHint: "DOCUMENT",
        enableRelationDetection: false,
        publishPendingOverlay: false,
        sourceRole: "document",
      });

      if (writeResult.outcome === "created") {
        result.memoriesCreated++;
      }
      recordScope(result, writeResult.scopeTarget);
    }

    return result;
  } catch (error) {
    result.errors.push(`Chunk ingestion failed: ${error}`);
    return result;
  }
}

/**
 * Batch ingest multiple chunks
 * Used during document indexing
 */
export async function ingestChunksBatch(params: {
  chunks: Array<{
    id: string;
    content: string;
    metadata?: Record<string, any>;
  }>;
  projectId: string;
  orgId?: string;
  documentDate: Date;
}): Promise<IngestionResult> {
  const { chunks, projectId, orgId, documentDate } = params;

  const aggregateResult: IngestionResult = {
    memoriesCreated: 0,
    relationsCreated: 0,
    memoriesInvalidated: 0,
    errors: [],
    scopeCounts: {},
    scopesTouched: [],
  };

  // Process in batches to avoid overwhelming the database
  const batchSize = 10;

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);

    const settled = await Promise.allSettled(
      batch.map((chunk) =>
        ingestChunk({
          chunkId: chunk.id,
          chunkContent: chunk.content,
          projectId,
          orgId,
          documentDate,
          metadata: chunk.metadata,
        })
      )
    );

    const results = settled.map((r, i) => {
      if (r.status === "rejected") {
        const chunkId = batch[i]?.id ?? "unknown";
        console.error(`[ingestChunksBatch] Chunk ${chunkId} failed:`, r.reason);
        return {
          memoriesCreated: 0,
          relationsCreated: 0,
          memoriesInvalidated: 0,
          errors: [String(r.reason)],
          scopeCounts: {},
          scopesTouched: [],
        } satisfies IngestionResult;
      }
      return r.value;
    });

    for (const result of results) {
      aggregateResult.memoriesCreated += result.memoriesCreated;
      aggregateResult.relationsCreated += result.relationsCreated;
      aggregateResult.memoriesInvalidated += result.memoriesInvalidated;
      aggregateResult.errors.push(...result.errors);
      for (const [scope, count] of Object.entries(result.scopeCounts)) {
        if (!scope || !count) continue;
        aggregateResult.scopeCounts[scope as Exclude<MemoryScopeTarget, "DROPPED">] =
          (aggregateResult.scopeCounts[scope as Exclude<MemoryScopeTarget, "DROPPED">] || 0) + count;
      }
      for (const scope of result.scopesTouched) {
        if (!aggregateResult.scopesTouched.includes(scope)) {
          aggregateResult.scopesTouched.push(scope);
        }
      }
    }
  }

  return aggregateResult;
}

/**
 * Update existing memory with new information
 * Creates a new version and invalidates the old one
 */
export async function updateMemory(params: {
  memoryId: string;
  newContent: string;
  reasoning?: string;
}): Promise<{ newMemoryId: string; oldMemoryId: string }> {
  const { memoryId, newContent, reasoning } = params;

  const oldMemory = await db.memory.findUnique({
    where: { id: memoryId },
  });

  if (!oldMemory) {
    throw new Error("Memory not found");
  }

  const writeResult = await writeMemoryCanonical({
    projectId: oldMemory.projectId || "",
    orgId: oldMemory.orgId || undefined,
    userId: oldMemory.userId || undefined,
    sessionId: oldMemory.sessionId || undefined,
    agentId: oldMemory.agentId || undefined,
    taskId: (oldMemory as any).taskId || undefined,
    content: newContent,
    memoryType: oldMemory.memoryType,
    entityMentions: oldMemory.entityMentions,
    confidenceRaw: oldMemory.confidence,
    importance: oldMemory.importance,
    documentDate: oldMemory.documentDate,
    eventDate: oldMemory.eventDate,
    metadata: {
      ...(oldMemory.metadata && typeof oldMemory.metadata === "object" && !Array.isArray(oldMemory.metadata)
        ? oldMemory.metadata as Record<string, unknown>
        : {}),
      updateReasoning: reasoning,
    },
    writeSource: "memory.update",
    writeMode: "direct_write",
    extractionMethod: "manual",
    scopeHint:
      oldMemory.scope === "DOCUMENT"
        ? "DOCUMENT"
        : oldMemory.scope === "SESSION"
          ? "SESSION"
          : oldMemory.scope === "PROJECT"
            ? "PROJECT"
            : oldMemory.scope === "AGENT"
              ? "AGENT"
              : oldMemory.scope === "TASK"
                ? "TASK"
          : "USER",
    publishPendingOverlay: false,
    enableRelationDetection: false,
    sourceRole: "user",
  });

  if (!writeResult.memory) {
    throw new Error("Failed to create updated memory version");
  }

  // Invalidate old memory
  await db.memory.update({
    where: { id: memoryId },
    data: {
      validUntil: new Date(),
      supersededBy: writeResult.memory.id,
    },
  });

  // Create update relation
  await db.memoryRelation.create({
    data: {
      fromMemoryId: writeResult.memory.id,
      toMemoryId: memoryId,
      relationType: "updates",
      confidence: 1.0,
      reasoning: reasoning || "Manual update",
    },
  });

  return {
    newMemoryId: writeResult.memory.id,
    oldMemoryId: memoryId,
  };
}
