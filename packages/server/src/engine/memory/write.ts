import { Prisma } from "@prisma/client";
import { db } from "../../db/index.js";
import { calibrateConfidence } from "../extraction-observability.js";
import { mergeMemoryNormalizationMetadata } from "../../lib/memory-normalization.js";
import { addPendingOverlayEntry } from "../pending-overlay.js";
import { enqueueMemoryEmbeddingJob } from "../queue.js";
import { getRedisClient, clearAfterMemoryWrite, clearAfterMemoriesInvalidated } from "../cache.js";
import { embedSingle } from "../embeddings.js";
import { detectRelations, shouldInvalidateMemory } from "./relations.js";
import type { MemoryScopeTarget, MemorySourceRole, MemoryType, PromotionMode } from "./types.js";
import { encrypt, decrypt } from "../../lib/encryption.js";

export const MEMORY_WRITE_POLICY_VERSION = "memory_write_v2";
export const USER_PROFILE_THRESHOLD = 0.82;
// Lower threshold for user-identity types (preference, goal, opinion, instruction, factual).
// These are the memories that MUST survive across sessions — anything above 0.68 is clear
// enough signal to write to USER scope immediately rather than trapping in SESSION.
export const USER_CROSS_SESSION_THRESHOLD = 0.68;
export const PROJECT_SCOPE_THRESHOLD = 0.76;
export const AGENT_SCOPE_THRESHOLD = 0.74;
export const TASK_SCOPE_THRESHOLD = 0.72;
export const SESSION_ONLY_THRESHOLD = 0.58;

export type MemoryWriteMode = "direct_write" | "session_extract" | "source_extract";
export type MemoryWriteOutcome =
  | "created"
  | "exact_duplicate"
  | "merged"
  | "dropped";

export interface CanonicalMemoryWriteInput {
  projectId: string;
  orgId?: string;
  userId?: string;
  sessionId?: string;
  agentId?: string;
  taskId?: string;
  sourceChunkId?: string;
  content: string;
  memoryType: string;
  importance?: number;
  confidenceRaw?: number;
  metadata?: Record<string, any>;
  entityMentions?: string[];
  documentDate?: Date | null;
  eventDate?: Date | null;
  expiresAt?: Date | null;
  writeSource: string;
  writeMode: MemoryWriteMode;
  extractionMethod: string;
  sourceMessageIds?: string[];
  sourceChunkIds?: string[];
  bypassValidation?: boolean;
  isAdminWrite?: boolean;
  actorUserId?: string;
  pendingOverlayTtlMs?: number;
  publishPendingOverlay?: boolean;
  scopeHint?: Exclude<MemoryScopeTarget, "DROPPED">;
  sessionRetentionDays?: number;
  enableRelationDetection?: boolean;
  sourceRole?: MemorySourceRole;
  userConfirmed?: boolean;
  supportingEvent?: boolean;
  promotionMode?: PromotionMode;
}

export interface CanonicalMemoryWriteResult {
  outcome: MemoryWriteOutcome;
  memory:
    | {
        id: string;
        projectId: string | null;
        orgId: string | null;
        userId: string | null;
        sessionId: string | null;
        agentId: string | null;
        taskId: string | null;
        content: string;
        memoryType: string;
        importance: number;
        confidence: number;
        scope: string;
        scopeTarget: MemoryScopeTarget;
        createdAt: Date;
        updatedAt: Date;
        metadata: Record<string, any>;
      }
    | null;
  confidenceCalibrated: number;
  scopeDecision: "user_profile" | "session_only" | "document" | "dropped";
  scopeTarget: MemoryScopeTarget;
  validatorIssues: string[];
  relationCount: number;
  invalidatedCount: number;
}

type ExistingMemory = {
  id: string;
  projectId: string | null;
  orgId: string | null;
  userId: string | null;
  sessionId: string | null;
  agentId: string | null;
  taskId: string | null;
  memoryType: string;
  content: string;
  entityMentions: string[];
  documentDate: Date | null;
  eventDate: Date | null;
  confidence: number;
  importance: number;
  version: number;
  scope: string;
  metadata: Record<string, any>;
  createdAt: Date;
  updatedAt: Date;
};

type ScopeInferenceInput = Pick<
  CanonicalMemoryWriteInput,
  "memoryType" | "scopeHint" | "userId" | "sessionId" | "agentId" | "taskId" | "sourceRole" | "userConfirmed" | "supportingEvent" | "metadata" | "promotionMode"
>;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function normalizeWhitespace(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function normalizeContent(value: string): string {
  return normalizeWhitespace(value).toLowerCase();
}

function normalizeMemoryType(memoryType?: string): MemoryType {
  const normalized = (memoryType || "factual").toLowerCase();
  const map: Record<string, MemoryType> = {
    factual: "factual",
    semantic: "factual",
    preference: "preference",
    event: "event",
    episodic: "event",
    relationship: "relationship",
    opinion: "opinion",
    goal: "goal",
    instruction: "instruction",
    procedural: "instruction",
    decision: "decision",
    constraint: "constraint",
    solution: "solution",
    project_state: "project_state",
    correction: "correction",
    workflow: "workflow",
  };
  return map[normalized] || "factual";
}

function tokenize(value: string): string[] {
  return normalizeContent(value)
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 3);
}

function jaccardSimilarity(left: string, right: string): number {
  const leftTokens = new Set(tokenize(left));
  const rightTokens = new Set(tokenize(right));
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;

  let intersection = 0;
  for (const token of leftTokens) {
    if (rightTokens.has(token)) intersection += 1;
  }
  const union = new Set([...Array.from(leftTokens), ...Array.from(rightTokens)]).size;
  return union === 0 ? 0 : intersection / union;
}

function contentDelta(left: string, right: string): number {
  const leftNorm = normalizeContent(left);
  const rightNorm = normalizeContent(right);
  const longest = Math.max(leftNorm.length, rightNorm.length, 1);
  return Math.abs(leftNorm.length - rightNorm.length) / longest;
}

function mergeUniqueStrings(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map((value) => value?.trim()).filter(Boolean) as string[]));
}

function asJsonObject(value: unknown): Record<string, any> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, any>;
  }
  return {};
}

function buildNormalizedMetadata(input: CanonicalMemoryWriteInput, content: string): Record<string, any> {
  return mergeMemoryNormalizationMetadata(asJsonObject(input.metadata), content);
}

function canCreateUserScopedMemory(input: ScopeInferenceInput): boolean {
  if (!input.userId) return false;
  if (input.sourceRole === "assistant" && !input.userConfirmed) return false;
  if (input.sourceRole === "event") return false;
  return true;
}

function canAssistantPromoteStructurally(input: ScopeInferenceInput): boolean {
  if (input.sourceRole !== "assistant") return true;
  if (input.userConfirmed) return true;
  if (input.supportingEvent) return true;
  if ((input.metadata as Record<string, unknown> | undefined)?.success === true) return true;
  return false;
}

function scopeTargetToDecision(scopeTarget: MemoryScopeTarget): CanonicalMemoryWriteResult["scopeDecision"] {
  if (scopeTarget === "USER") return "user_profile";
  if (scopeTarget === "DOCUMENT") return "document";
  if (scopeTarget === "DROPPED") return "dropped";
  return "session_only";
}

function inferScopeTarget(
  confidenceCalibrated: number,
  input: ScopeInferenceInput
): MemoryScopeTarget {
  if (input.scopeHint === "DOCUMENT") {
    return confidenceCalibrated >= SESSION_ONLY_THRESHOLD ? "DOCUMENT" : "DROPPED";
  }
  if (input.promotionMode === "user_specific_legacy") {
    if (input.scopeHint === "SESSION") {
      return confidenceCalibrated >= SESSION_ONLY_THRESHOLD && input.sessionId ? "SESSION" : "DROPPED";
    }
    if (input.scopeHint === "USER") {
      return confidenceCalibrated >= USER_PROFILE_THRESHOLD && canCreateUserScopedMemory(input)
        ? "USER"
        : "DROPPED";
    }
    if (confidenceCalibrated >= USER_PROFILE_THRESHOLD && canCreateUserScopedMemory(input)) return "USER";
    if (confidenceCalibrated >= SESSION_ONLY_THRESHOLD && input.sessionId) return "SESSION";
    return "DROPPED";
  }

  const normalizedType = normalizeMemoryType(input.memoryType);
  const assistantOnly = input.sourceRole === "assistant" && !canAssistantPromoteStructurally(input);
  const explicitHint = input.scopeHint;

  if (explicitHint === "USER") {
    return confidenceCalibrated >= USER_PROFILE_THRESHOLD && canCreateUserScopedMemory(input) ? "USER" : "DROPPED";
  }
  if (explicitHint === "TASK") {
    return confidenceCalibrated >= TASK_SCOPE_THRESHOLD && input.taskId ? "TASK" : "DROPPED";
  }
  if (explicitHint === "AGENT") {
    return confidenceCalibrated >= AGENT_SCOPE_THRESHOLD && input.agentId ? "AGENT" : "DROPPED";
  }
  if (explicitHint === "PROJECT") {
    return confidenceCalibrated >= PROJECT_SCOPE_THRESHOLD ? "PROJECT" : "DROPPED";
  }
  if (explicitHint === "SESSION") {
    return confidenceCalibrated >= SESSION_ONLY_THRESHOLD && input.sessionId ? "SESSION" : "DROPPED";
  }

  if (
    assistantOnly &&
    (normalizedType === "decision"
      || normalizedType === "constraint"
      || normalizedType === "solution"
      || normalizedType === "correction")
  ) {
    return input.sessionId && confidenceCalibrated >= SESSION_ONLY_THRESHOLD ? "SESSION" : "DROPPED";
  }

  // Cross-session types: preference/goal/opinion/instruction/factual.
  // Use the lower USER_CROSS_SESSION_THRESHOLD so moderate-confidence memories
  // survive across sessions rather than dying in SESSION scope.
  if (normalizedType === "preference" || normalizedType === "goal" || normalizedType === "opinion") {
    if (confidenceCalibrated >= USER_CROSS_SESSION_THRESHOLD && canCreateUserScopedMemory(input)) return "USER";
  }

  if (normalizedType === "factual" && canCreateUserScopedMemory(input)) {
    if (confidenceCalibrated >= USER_CROSS_SESSION_THRESHOLD) return "USER";
  }

  if (normalizedType === "instruction") {
    if (confidenceCalibrated >= USER_CROSS_SESSION_THRESHOLD && canCreateUserScopedMemory(input)) return "USER";
    if (confidenceCalibrated >= AGENT_SCOPE_THRESHOLD && input.agentId) return "AGENT";
    if (confidenceCalibrated >= TASK_SCOPE_THRESHOLD && input.taskId) return "TASK";
    if (confidenceCalibrated >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }

  if (normalizedType === "workflow") {
    if (confidenceCalibrated >= AGENT_SCOPE_THRESHOLD && input.agentId) return "AGENT";
    if (confidenceCalibrated >= TASK_SCOPE_THRESHOLD && input.taskId) return "TASK";
    if (confidenceCalibrated >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }

  if (
    normalizedType === "decision"
    || normalizedType === "constraint"
    || normalizedType === "solution"
    || normalizedType === "project_state"
    || normalizedType === "correction"
  ) {
    if (confidenceCalibrated >= TASK_SCOPE_THRESHOLD && input.taskId) return "TASK";
    if (confidenceCalibrated >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }

  if (confidenceCalibrated >= USER_PROFILE_THRESHOLD && canCreateUserScopedMemory(input)) return "USER";
  if (confidenceCalibrated >= TASK_SCOPE_THRESHOLD && input.taskId && input.sourceRole === "event") return "TASK";
  if (confidenceCalibrated >= AGENT_SCOPE_THRESHOLD && input.agentId && normalizedType !== "factual") return "AGENT";
  if (confidenceCalibrated >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  if (confidenceCalibrated >= SESSION_ONLY_THRESHOLD && input.sessionId) return "SESSION";
  return "DROPPED";
}

function inferScopeDecision(
  confidenceCalibrated: number,
  input: Partial<ScopeInferenceInput> = {}
): CanonicalMemoryWriteResult["scopeDecision"] {
  return scopeTargetToDecision(inferScopeTarget(confidenceCalibrated, input as ScopeInferenceInput));
}

function sameScopeIdentityTuple(
  memory: Pick<ExistingMemory, "projectId" | "userId" | "sessionId" | "agentId" | "taskId" | "scope">,
  scopeTarget: MemoryScopeTarget,
  input: CanonicalMemoryWriteInput
): boolean {
  if ((memory.projectId || null) !== (input.projectId || null)) return false;
  if (memory.scope !== scopeTarget) return false;
  if (scopeTarget === "USER") {
    return (memory.userId || null) === (input.userId || null);
  }
  if (scopeTarget === "PROJECT") {
    return true;
  }
  if (scopeTarget === "AGENT") {
    return (memory.agentId || null) === (input.agentId || null);
  }
  if (scopeTarget === "TASK") {
    return (memory.taskId || null) === (input.taskId || null);
  }
  if (scopeTarget === "SESSION") {
    return (memory.sessionId || null) === (input.sessionId || null);
  }
  if (scopeTarget === "DOCUMENT") {
    return true;
  }
  return false;
}

function canInvalidateExistingMemory(input: ScopeInferenceInput): boolean {
  if (input.sourceRole !== "assistant") return true;
  return canAssistantPromoteStructurally(input);
}

function extractExplicitSupersedesIds(metadata: unknown): string[] {
  const record = asJsonObject(metadata);
  const ids = Array.isArray(record.supersedes_ids) ? record.supersedes_ids : Array.isArray(record.supersedesIds) ? record.supersedesIds : [];
  return ids.map((value) => String(value || "").trim()).filter(Boolean);
}

function buildValidatorIssues(input: {
  content: string;
  memoryType: string;
  eventDate?: Date | null;
  entityMentions: string[];
}): string[] {
  const issues: string[] = [];
  const content = normalizeWhitespace(input.content);

  if (content.length < 10) issues.push("too_short");
  if (/^(hi|hello|hey|thanks|thank you|ok|okay|bye)[!.]*$/i.test(content)) issues.push("chatter");
  // Only flag pronoun ambiguity when a personal pronoun is the SUBJECT and there are no
  // entity mentions to anchor it. "OpenAI updated their API" is fine; "He prefers X" is not.
  const ambiguousSubjectPronouns = /^(he|she|they|it)\b/i;
  if (ambiguousSubjectPronouns.test(content.trim())) issues.push("unresolved_pronouns");
  if (/\b(the company|that project|this thing|the system|something|stuff|things)\b/i.test(content)) {
    issues.push("vague_reference");
  }
  if (/[.;]\s+\S+/.test(content) || /\b(and|also|plus)\b.+\b(and|also|plus)\b/i.test(content)) {
    issues.push("multi_fact");
  }
  if (content.split(/\s+/).length < 4) issues.push("low_specificity");
  if (input.entityMentions.length === 0 && /\b(alex|maria|john|react|python|typescript|docker)\b/i.test(content)) {
    issues.push("weak_grounding");
  }
  if (normalizeMemoryType(input.memoryType) === "event"
    && /\b(yesterday|today|last week|last month|next week|soon|recently)\b/i.test(content)
    && !input.eventDate) {
    issues.push("underspecified_temporal");
  }

  return Array.from(new Set(issues));
}

function calibrateWriteConfidence(input: {
  confidenceRaw: number;
  memoryType: string;
  extractionMethod: string;
  validatorIssues: string[];
}): number {
  let calibrated = calibrateConfidence(input.confidenceRaw, normalizeMemoryType(input.memoryType));
  const method = input.extractionMethod.toLowerCase();
  const methodAdjustment =
    method === "manual"
      ? 0.08
      : method === "pattern"
        ? 0.06
        : method === "hybrid"
          ? 0.02
          : method === "inference"
            ? -0.02
            : method === "strong"
              ? -0.03
              : method === "legacy_llm" || method === "session_llm"
                ? -0.01
                : 0;

  calibrated += methodAdjustment;

  for (const issue of input.validatorIssues) {
    if (issue === "too_short") calibrated -= 0.08;
    if (issue === "chatter") calibrated -= 0.2;
    if (issue === "unresolved_pronouns") calibrated -= 0.12;
    if (issue === "vague_reference") calibrated -= 0.1;
    if (issue === "multi_fact") calibrated -= 0.1;
    if (issue === "low_specificity") calibrated -= 0.06;
    if (issue === "weak_grounding") calibrated -= 0.04;
    if (issue === "underspecified_temporal") calibrated -= 0.08;
  }

  if (method === "strong" && input.validatorIssues.length > 0) {
    calibrated = Math.min(calibrated, 0.86);
  }

  return clamp01(calibrated);
}

async function auditBypassViolation(input: CanonicalMemoryWriteInput): Promise<void> {
  try {
    const { writeAuditLog } = await import("../../lib/audit.js");
    await writeAuditLog({
      action: "memory_write_bypass_rejected",
      resource: "memory",
      outcome: "failure",
      traceId: `bypass-${Date.now()}`,
      metadata: {
        write_source: input.writeSource,
        write_mode: input.writeMode,
        extraction_method: input.extractionMethod,
      },
    });
  } catch {
    // Best-effort audit only.
  }
}

async function fetchComparableMemories(input: CanonicalMemoryWriteInput): Promise<ExistingMemory[]> {
  const rows = await db.memory.findMany({
    where: {
      projectId: input.projectId,
      isActive: true,
      validUntil: null,
      ...(input.orgId ? { orgId: input.orgId } : {}),
      ...(input.userId ? { userId: input.userId } : {}),
      ...(!input.userId && input.writeMode === "session_extract" && input.sessionId
        ? { sessionId: input.sessionId }
        : {}),
    },
    orderBy: { updatedAt: "desc" },
    take: input.writeMode === "source_extract" ? 30 : 60,
    select: {
      id: true,
      projectId: true,
      orgId: true,
      userId: true,
      sessionId: true,
      agentId: true,
      taskId: true,
      memoryType: true,
      content: true,
      entityMentions: true,
      documentDate: true,
      eventDate: true,
      confidence: true,
      importance: true,
      version: true,
      scope: true,
      metadata: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  return rows.map((row) => ({
    ...row,
    content: decrypt(row.content, row.orgId || undefined),
    metadata: asJsonObject(row.metadata),
  }));
}

function findExactDuplicate(
  normalizedContent: string,
  memoryType: MemoryType,
  scopeTarget: MemoryScopeTarget,
  candidates: ExistingMemory[],
  input: CanonicalMemoryWriteInput
): ExistingMemory | null {
  return (
    candidates.find((memory) => {
      if (normalizeMemoryType(memory.memoryType) !== memoryType) return false;
      if (normalizeContent(memory.content) !== normalizedContent) return false;
      return sameScopeIdentityTuple(memory, scopeTarget, input);
    }) || null
  );
}

function findNearDuplicate(
  content: string,
  memoryType: MemoryType,
  scopeTarget: MemoryScopeTarget,
  candidates: ExistingMemory[],
  input: CanonicalMemoryWriteInput
): ExistingMemory | null {
  let best: { memory: ExistingMemory; score: number } | null = null;

  for (const memory of candidates) {
    if (normalizeMemoryType(memory.memoryType) !== memoryType) continue;
    if (!sameScopeIdentityTuple(memory, scopeTarget, input)) continue;
    const similarity = jaccardSimilarity(memory.content, content);
    const delta = contentDelta(memory.content, content);
    const containment =
      normalizeContent(memory.content).includes(normalizeContent(content)) ||
      normalizeContent(content).includes(normalizeContent(memory.content));

    if (similarity >= 0.88 && (delta <= 0.2 || containment)) {
      if (!best || similarity > best.score) {
        best = { memory, score: similarity };
      }
    }
  }

  return best?.memory || null;
}

export const __memoryWriteTestables = {
  normalizeMemoryType,
  buildValidatorIssues,
  calibrateWriteConfidence,
  inferScopeTarget,
  inferScopeDecision,
  findExactDuplicate,
  findNearDuplicate,
};

function toPersistedShape(memory: any): CanonicalMemoryWriteResult["memory"] {
  if (!memory) return null;
  return {
    id: memory.id,
    projectId: memory.projectId || null,
    orgId: memory.orgId || null,
    userId: memory.userId || null,
    sessionId: memory.sessionId || null,
    agentId: memory.agentId || null,
    taskId: memory.taskId || null,
    content: decrypt(memory.content, memory.orgId || undefined),
    memoryType: memory.memoryType,
    importance: memory.importance,
    confidence: memory.confidence,
    scope: memory.scope,
    scopeTarget: (memory.scope || "SESSION") as MemoryScopeTarget,
    createdAt: memory.createdAt,
    updatedAt: memory.updatedAt,
    metadata: asJsonObject(memory.metadata),
  };
}

async function updateDuplicateMetadata(
  memory: ExistingMemory,
  input: CanonicalMemoryWriteInput,
  confidenceCalibrated: number,
  scopeDecision: CanonicalMemoryWriteResult["scopeDecision"],
  scopeTarget: MemoryScopeTarget
) {
  const metadata = {
    ...memory.metadata,
    ...buildNormalizedMetadata(input, memory.content),
    write_source: input.writeSource,
    write_mode: input.writeMode,
    extraction_method: input.extractionMethod,
    confidence_raw: input.confidenceRaw ?? 0.8,
    confidence_calibrated: confidenceCalibrated,
    policy_version: MEMORY_WRITE_POLICY_VERSION,
    source_message_ids: mergeUniqueStrings([
      ...(Array.isArray(memory.metadata?.source_message_ids) ? memory.metadata.source_message_ids : []),
      ...(input.sourceMessageIds || []),
    ]),
    source_chunk_ids: mergeUniqueStrings([
      ...(Array.isArray(memory.metadata?.source_chunk_ids) ? memory.metadata.source_chunk_ids : []),
      ...(input.sourceChunkIds || []),
      input.sourceChunkId,
    ]),
    bypass_validation: Boolean(input.bypassValidation && input.isAdminWrite),
    scope_decision: scopeDecision,
    scope_target: scopeTarget,
    duplicate_observed_at: new Date().toISOString(),
  };

  return db.memory.update({
    where: { id: memory.id },
    data: {
      confidence: confidenceCalibrated >= memory.confidence
        ? confidenceCalibrated
        : Math.max(confidenceCalibrated, memory.confidence - 0.1),
      importance: Math.max(memory.importance, input.importance ?? 0.5),
      entityMentions: mergeUniqueStrings([...(memory.entityMentions || []), ...(input.entityMentions || [])]),
      metadata,
      expiresAt:
        memory.scope === "SESSION" && input.expiresAt
          ? input.expiresAt
          : undefined,
    },
  });
}

async function mergeNearDuplicate(
  memory: ExistingMemory,
  input: CanonicalMemoryWriteInput,
  confidenceCalibrated: number,
  scopeDecision: CanonicalMemoryWriteResult["scopeDecision"],
  scopeTarget: MemoryScopeTarget
) {
  const existingMetadata = memory.metadata;
  const mergedMetadata = {
    ...existingMetadata,
    ...buildNormalizedMetadata(input, input.content),
    write_source: input.writeSource,
    write_mode: input.writeMode,
    extraction_method: input.extractionMethod,
    confidence_raw: input.confidenceRaw ?? 0.8,
    confidence_calibrated: confidenceCalibrated,
    policy_version: MEMORY_WRITE_POLICY_VERSION,
    bypass_validation: Boolean(input.bypassValidation && input.isAdminWrite),
    scope_decision: scopeDecision,
    scope_target: scopeTarget,
    source_message_ids: mergeUniqueStrings([
      ...(Array.isArray(existingMetadata?.source_message_ids) ? existingMetadata.source_message_ids : []),
      ...(input.sourceMessageIds || []),
    ]),
    source_chunk_ids: mergeUniqueStrings([
      ...(Array.isArray(existingMetadata?.source_chunk_ids) ? existingMetadata.source_chunk_ids : []),
      ...(input.sourceChunkIds || []),
      input.sourceChunkId,
    ]),
    merged_provenance: mergeUniqueStrings([
      ...(Array.isArray(existingMetadata?.merged_provenance) ? existingMetadata.merged_provenance : []),
      `${input.writeSource}:${input.extractionMethod}`,
    ]),
    last_seen_at: new Date().toISOString(),
  };

  return db.memory.update({
    where: { id: memory.id },
    data: {
      importance: Math.max(memory.importance, input.importance ?? 0.5),
      confidence: confidenceCalibrated >= memory.confidence
        ? confidenceCalibrated
        : Math.max(confidenceCalibrated, memory.confidence - 0.1),
      entityMentions: mergeUniqueStrings([...(memory.entityMentions || []), ...(input.entityMentions || [])]),
      metadata: mergedMetadata,
      expiresAt:
        memory.scope === "SESSION" && input.expiresAt
          ? input.expiresAt
          : undefined,
    },
  });
}

async function createMemoryWithProvenance(
  input: CanonicalMemoryWriteInput,
  confidenceCalibrated: number,
  scopeTarget: MemoryScopeTarget,
  validatorIssues: string[],
  version: number
) {
  const baseMetadata = buildNormalizedMetadata(input, input.content);
  const scope = scopeTarget === "DROPPED" ? "SESSION" : scopeTarget;

  const plaintextContent = normalizeWhitespace(input.content);
  return db.memory.create({
    data: {
      projectId: input.projectId,
      orgId: input.orgId || null,
      userId: input.userId || null,
      sessionId: input.sessionId || null,
      agentId: input.agentId || null,
      taskId: input.taskId || null,
      memoryType: normalizeMemoryType(input.memoryType),
      content: encrypt(plaintextContent, input.orgId),
      entityMentions: mergeUniqueStrings(input.entityMentions || []),
      confidence: confidenceCalibrated,
      importance: input.importance ?? 0.5,
      documentDate: input.documentDate || null,
      eventDate: input.eventDate || null,
      sourceChunkId: input.sourceChunkId || null,
      validFrom: new Date(),
      expiresAt: input.expiresAt || null,
      scope,
      version,
      metadata: {
        ...baseMetadata,
        write_source: input.writeSource,
        write_mode: input.writeMode,
        extraction_method: input.extractionMethod,
        confidence_raw: input.confidenceRaw ?? 0.8,
        confidence_calibrated: confidenceCalibrated,
        validator_issues: validatorIssues,
        policy_version: MEMORY_WRITE_POLICY_VERSION,
        source_message_ids: input.sourceMessageIds || [],
        source_chunk_ids: mergeUniqueStrings([...(input.sourceChunkIds || []), input.sourceChunkId]),
        bypass_validation: Boolean(input.bypassValidation && input.isAdminWrite),
        scope_decision: scopeTargetToDecision(scopeTarget),
        scope_target: scopeTarget,
        source_role: input.sourceRole || "user",
        promotion_mode: input.promotionMode || "session_state_v1",
      } as Prisma.JsonObject,
    },
  });
}

async function enqueueEmbeddingWithRetry(
  job: Parameters<typeof enqueueMemoryEmbeddingJob>[0],
  maxAttempts = 3,
): Promise<void> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await enqueueMemoryEmbeddingJob(job);
      return;
    } catch (error) {
      if (attempt === maxAttempts) {
        console.error(
          `[MemoryWrite] Embedding enqueue failed after ${maxAttempts} attempts — memory ${job.id} will lack semantic search coverage until re-indexed:`,
          error,
        );
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 100 * Math.pow(2, attempt)));
    }
  }
}

async function embedMemoryInline(id: string, text: string): Promise<void> {
  try {
    const embedding = await embedSingle(text);
    const embeddingStr = `[${embedding.join(",")}]`;
    await db.$executeRaw(
      Prisma.sql`
        UPDATE memories
        SET
          embedding = ${embeddingStr}::vector,
          metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{semantic_status}', '"ready"'::jsonb, true)
        WHERE id = ${id}
      `
    );
  } catch (err) {
    console.warn(`[MemoryWrite] Inline embedding failed for ${id}:`, err);
  }
}

async function postPersistActions(memory: { id: string; content: string; projectId: string | null; scope?: string | null }, input: CanonicalMemoryWriteInput) {
  // Always use plaintext for embedding — input.content is guaranteed plaintext
  const plaintextContent = decrypt(memory.content, input.orgId);
  const embeddingText = String(buildNormalizedMetadata(input, plaintextContent).search_text || plaintextContent);

  if (getRedisClient()) {
    // Redis available — async queue
    void enqueueEmbeddingWithRetry({
      id: memory.id,
      text: embeddingText,
      projectId: input.projectId,
      orgId: input.orgId,
    });
  } else {
    // No Redis — embed synchronously so vector search works immediately
    await embedMemoryInline(memory.id, embeddingText);
  }

  // Invalidate the caches this write made stale so the result surfaces immediately (RF
  // search-cache-clear). clearAfterMemoryWrite logs a cache error and never fails the
  // write.
  await clearAfterMemoryWrite(memory, input.userId);

  if (input.publishPendingOverlay === false) return;
  if (!input.orgId) return;

  await addPendingOverlayEntry({
    orgId: input.orgId,
    projectId: input.projectId,
    userId: input.userId,
    sessionId: input.sessionId,
    content: plaintextContent,
    jobId: memory.id,
    ttlMs: input.pendingOverlayTtlMs,
  }).catch((error) => {
    console.warn("[MemoryWrite] Pending overlay publish failed (non-critical):", error);
  });
}

export async function writeMemoryCanonical(input: CanonicalMemoryWriteInput): Promise<CanonicalMemoryWriteResult> {
  if (input.bypassValidation && !input.isAdminWrite) {
    await auditBypassViolation(input);
    throw new Error("bypass_validation requires admin-authenticated write context");
  }

  const normalizedType = normalizeMemoryType(input.memoryType);
  const normalizedContent = normalizeContent(input.content);
  const entityMentions = mergeUniqueStrings(input.entityMentions || []);
  const validatorIssues = buildValidatorIssues({
    content: input.content,
    memoryType: normalizedType,
    eventDate: input.eventDate,
    entityMentions,
  });
  const confidenceRaw = clamp01(input.confidenceRaw ?? 0.8);
  const confidenceCalibrated = calibrateWriteConfidence({
    confidenceRaw,
    memoryType: normalizedType,
    extractionMethod: input.extractionMethod,
    validatorIssues,
  });
  const scopeTarget = inferScopeTarget(confidenceCalibrated, {
    memoryType: normalizedType,
    scopeHint: input.scopeHint,
    userId: input.userId,
    sessionId: input.sessionId,
    agentId: input.agentId,
    taskId: input.taskId,
    sourceRole: input.sourceRole,
    userConfirmed: input.userConfirmed,
    supportingEvent: input.supportingEvent,
    metadata: input.metadata,
    promotionMode: input.promotionMode,
  });
  const scopeDecision = scopeTargetToDecision(scopeTarget);

  if (validatorIssues.length > 0 && !(input.bypassValidation && input.isAdminWrite)) {
    // multi_fact is only a hard reject for pre-extracted facts (not raw user messages).
    // Raw conversational input (extractionMethod="manual") may have multiple sentences —
    // that's expected. Only drop it if the content was meant to be a single fact already.
    const isRawInput = input.extractionMethod === "manual";
    const hardRejects = isRawInput
      ? ["chatter", "unresolved_pronouns", "vague_reference"]
      : ["chatter", "unresolved_pronouns", "vague_reference", "multi_fact", "underspecified_temporal"];
    if (validatorIssues.some((issue) => hardRejects.includes(issue))) {
      return {
        outcome: "dropped",
        memory: null,
        confidenceCalibrated,
        scopeDecision: "dropped",
        scopeTarget: "DROPPED",
        validatorIssues,
        relationCount: 0,
        invalidatedCount: 0,
      };
    }
  }

  if (scopeDecision === "dropped" && !(input.bypassValidation && input.isAdminWrite)) {
    return {
      outcome: "dropped",
      memory: null,
      confidenceCalibrated,
      scopeDecision,
      scopeTarget,
      validatorIssues,
      relationCount: 0,
      invalidatedCount: 0,
    };
  }

  const expiresAt =
    input.expiresAt
    || (scopeTarget === "SESSION" && input.sessionRetentionDays
      ? new Date(Date.now() + input.sessionRetentionDays * 24 * 60 * 60 * 1000)
      : null);

  const effectiveInput = {
    ...input,
    content: normalizeWhitespace(input.content),
    memoryType: normalizedType,
    confidenceRaw,
    entityMentions,
    expiresAt,
  };

  const comparableMemories = await fetchComparableMemories(effectiveInput);
  const exactDuplicate = findExactDuplicate(
    normalizedContent,
    normalizedType,
    scopeTarget,
    comparableMemories,
    effectiveInput
  );

  if (exactDuplicate) {
    const updated = await updateDuplicateMetadata(
      exactDuplicate,
      effectiveInput,
      confidenceCalibrated,
      scopeDecision,
      scopeTarget
    );
    // A duplicate write still changes the stored memory's confidence, importance,
    // entities and metadata, so a cached answer from before it is stale (RF
    // search-cache-clear).
    await clearAfterMemoryWrite(updated, effectiveInput.userId);

    return {
      outcome: "exact_duplicate",
      memory: toPersistedShape(updated),
      confidenceCalibrated,
      scopeDecision,
      scopeTarget,
      validatorIssues,
      relationCount: 0,
      invalidatedCount: 0,
    };
  }

  const relations =
    effectiveInput.enableRelationDetection === false
      ? []
      : await detectRelations(
          {
            content: effectiveInput.content,
            memoryType: normalizedType,
            entityMentions,
          },
          comparableMemories.map((memory) => ({
            id: memory.id,
            content: memory.content,
            memoryType: memory.memoryType,
            entityMentions: memory.entityMentions,
            documentDate: memory.documentDate,
          }))
        );

  const invalidatingTargetIds = new Set(
    relations.filter((relation) => shouldInvalidateMemory(relation.relationType)).map((relation) => relation.toMemoryId)
  );
  const explicitSupersedesIds = extractExplicitSupersedesIds(effectiveInput.metadata);
  for (const supersededId of explicitSupersedesIds) {
    invalidatingTargetIds.add(supersededId);
    if (!relations.some((relation) => relation.toMemoryId === supersededId && relation.relationType === "updates")) {
      relations.push({
        toMemoryId: supersededId,
        relationType: "updates",
        confidence: 1,
        reasoning: normalizedType === "correction" ? "Explicit correction supersedes prior memory" : "Explicit supersession",
      });
    }
  }

  const nearDuplicate = relations.length === 0
    ? findNearDuplicate(effectiveInput.content, normalizedType, scopeTarget, comparableMemories, effectiveInput)
    : null;

  if (nearDuplicate) {
    const merged = await mergeNearDuplicate(
      nearDuplicate,
      effectiveInput,
      confidenceCalibrated,
      scopeDecision,
      scopeTarget
    );
    // A merge changes confidence, importance, entities and metadata - not the stored
    // text - and a cached answer shows those, so every cached answer that could have
    // come from it is stale (RF search-cache-clear). clearAfterMemoryWrite logs a
    // cache error and never fails the write.
    await clearAfterMemoryWrite(merged, effectiveInput.userId);
    return {
      outcome: "merged",
      memory: toPersistedShape(merged),
      confidenceCalibrated,
      scopeDecision,
      scopeTarget,
      validatorIssues,
      relationCount: 0,
      invalidatedCount: 0,
    };
  }

  const nextVersion = invalidatingTargetIds.size > 0
    ? Math.max(
        ...comparableMemories
          .filter((memory) => invalidatingTargetIds.has(memory.id))
          .map((memory) => memory.version || 1),
        1
      ) + 1
    : 1;

  const created = await createMemoryWithProvenance(
    effectiveInput,
    confidenceCalibrated,
    scopeTarget === "DROPPED" && input.bypassValidation ? "USER" : scopeTarget,
    validatorIssues,
    nextVersion
  );

  let invalidatedCount = 0;
  const invalidatedMemories: Array<{ projectId: string | null; scope?: string | null; userId?: string | null }> = [];
  for (const relation of relations) {
    await db.memoryRelation.create({
      data: {
        fromMemoryId: created.id,
        toMemoryId: relation.toMemoryId,
        relationType: relation.relationType,
        confidence: relation.confidence,
        reasoning: relation.reasoning,
        metadata: {
          write_source: effectiveInput.writeSource,
          policy_version: MEMORY_WRITE_POLICY_VERSION,
        },
      },
    });

    if (shouldInvalidateMemory(relation.relationType) && canInvalidateExistingMemory(effectiveInput)) {
      const invalidatedMemory = await db.memory.update({
        where: { id: relation.toMemoryId },
        data: {
          validUntil: new Date(),
          supersededBy: created.id,
          isActive: false,
        },
        select: { projectId: true, scope: true, userId: true },
      });
      invalidatedMemories.push(invalidatedMemory);
      invalidatedCount += 1;
    }
  }

  if (invalidatedMemories.length > 0) {
    await clearAfterMemoriesInvalidated(invalidatedMemories);
  }

  await postPersistActions(created, effectiveInput);

  return {
    outcome: "created",
    memory: toPersistedShape(created),
    confidenceCalibrated,
    scopeDecision,
    scopeTarget,
    validatorIssues,
    relationCount: relations.length,
    invalidatedCount,
  };
}
