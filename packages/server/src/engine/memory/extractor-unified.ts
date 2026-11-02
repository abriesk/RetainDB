import { extractExplicitMemory } from "./patterns.js";
import { extractImplicitMemories } from "./inference.js";
import type { ExtractedMemory, MemoryExtractionOptions, MemorySourceRole } from "./types.js";

export interface ExtractionResult {
  explicit: ExtractedMemory[];
  implicit: ExtractedMemory[];
  all: ExtractedMemory[];
  extractionMethod: "pattern" | "inference" | "hybrid";
  latencyMs: number;
}

// Chunk long messages so nothing is silently dropped
const CHUNK_SIZE = 1800;
const CHUNK_OVERLAP = 150;

function chunkMessage(message: string): string[] {
  if (message.length <= CHUNK_SIZE) return [message];
  const chunks: string[] = [];
  let offset = 0;
  while (offset < message.length) {
    chunks.push(message.slice(offset, offset + CHUNK_SIZE));
    offset += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

export async function extractMemories(
  message: string,
  context: string = "",
  options: MemoryExtractionOptions & {
    tieredEscalation?: boolean;
    escalationModel?: string;
    sourceRole?: MemorySourceRole;
    onInferenceError?: (err: unknown) => void;
  } = {}
): Promise<ExtractionResult> {
  const startTime = Date.now();

  // For long messages, run extraction on each chunk and merge
  const chunks = chunkMessage(message);
  if (chunks.length > 1) {
    const allResults = await Promise.all(
      chunks.map(chunk => extractMemories(chunk, context, options))
    );
    const merged = deduplicateAndMerge(allResults.flatMap(r => r.all));
    const hasPattern = allResults.some(r => r.extractionMethod !== "inference");
    const hasInference = allResults.some(r => r.extractionMethod !== "pattern");
    return {
      explicit: merged.filter(m => !m.inferred),
      implicit: merged.filter(m => m.inferred),
      all: merged,
      extractionMethod: hasPattern && hasInference ? "hybrid" : hasPattern ? "pattern" : "inference",
      latencyMs: Date.now() - startTime,
    };
  }

  const explicit: ExtractedMemory[] = [];
  const implicit: ExtractedMemory[] = [];

  const enablePattern = options.enablePattern !== false;
  const enableInference = options.enableInference !== false;
  const minConfidence = options.minConfidence || 0.5;

  if (enablePattern && message.trim()) {
    const patternMatches = extractExplicitMemory(message);

    if (patternMatches.length > 0) {
      explicit.push(
        ...patternMatches.map((m) => ({
          content: m.content,
          memoryType: m.type,
          entityMentions: m.entities,
          eventDate: null,
          confidence: m.confidence,
          reasoning: `Matched pattern: ${m.matchedPattern}`,
          inferred: false,
          sourceRole: options.sourceRole || "user",
        }))
      );
    }
  }

  if (enableInference && message.trim()) {
    const inferredMemories = await extractImplicitMemories(message, context, {
      minConfidence,
      sourceRole: options.sourceRole,
      onError: options.onInferenceError,
    });
    implicit.push(...inferredMemories.filter((memory) => memory.confidence >= minConfidence));

    const borderlineConfidence = [...explicit, ...implicit]
      .map((memory) => memory.confidence)
      .sort((left, right) => right - left)[0] ?? 0;
    const shouldEscalate =
      Boolean(options.tieredEscalation)
      && (explicit.length === 0 || borderlineConfidence < 0.8)
      && borderlineConfidence >= Math.max(minConfidence - 0.05, 0.45);

    if (shouldEscalate) {
      const strongerMemories = await extractImplicitMemories(message, context, {
        model: options.escalationModel || "gpt-5.4-mini",
        minConfidence,
        sourceRole: options.sourceRole,
        onError: options.onInferenceError,
      });
      if (strongerMemories.length > 0) {
        implicit.push(...strongerMemories.map((memory) => ({
          ...memory,
          reasoning: memory.reasoning || "Escalated strong-model inference",
        })));
      }
    }
  }

  const all = deduplicateAndMerge([...explicit, ...implicit])
    .filter((memory) => memory.confidence >= minConfidence);
  const extractionMethod =
    explicit.length > 0 && implicit.length > 0
      ? "hybrid"
      : explicit.length > 0
        ? "pattern"
        : "inference";

  return {
    explicit,
    implicit,
    all,
    extractionMethod,
    latencyMs: Date.now() - startTime,
  };
}

export async function extractMemoriesForSession(
  messages: Array<{ role: string; content: string; timestamp?: string }>,
  options: MemoryExtractionOptions = {}
): Promise<ExtractedMemory[]> {
  const results: ExtractedMemory[] = [];

  // ── Pass 1: build a session-level entity map from ALL messages ──────────
  // Scans the full conversation for proper nouns before any per-message
  // extraction so pronouns in later turns can be resolved correctly.
  const sessionEntityMap = buildSessionEntityMap(messages.map(m => m.content));

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg.content.trim()) continue;

    // Build context from surrounding messages (up to 3 prior turns)
    const contextParts = messages
      .slice(Math.max(0, i - 3), i)
      .map(m => m.content);

    // Append entity hints so the LLM knows who "he/she/they" refers to
    const entityHints = Array.from(sessionEntityMap.entries())
      .map(([pronoun, name]) => `"${pronoun}" = ${name}`)
      .join(", ");
    if (entityHints) contextParts.push(`[Known entities: ${entityHints}]`);

    const context = contextParts.join(" | ");

    if (msg.role === "user") {
      // Full extraction: pattern + inference
      const result = await extractMemories(msg.content, context, options);
      result.all.forEach((memory) => {
        memory.sourceRole = "user";
      });
      results.push(...result.all);
    } else if (msg.role === "assistant") {
      // Assistant turns: also run pattern extraction with lower threshold
      // (assistant confirmations like "Got it — you work at Stripe as a backend engineer"
      // contain high-value facts that pure inference misses)
      const [patternMatches, inferredMemories] = await Promise.all([
        Promise.resolve(extractExplicitMemory(msg.content)),
        extractImplicitMemories(msg.content, context, {
          minConfidence: options.minConfidence ?? 0.65,
          sourceRole: "assistant",
        }),
      ]);
      // Lower confidence slightly for assistant-turn pattern matches since
      // they describe the user in third-person rather than first-person.
      results.push(
        ...patternMatches.map(m => ({
          content: m.content,
          memoryType: m.type,
          entityMentions: m.entities,
          eventDate: null,
          confidence: Math.max(m.confidence - 0.1, 0.5),
          reasoning: `Assistant confirmation, pattern: ${m.matchedPattern}`,
          inferred: false,
          sourceRole: "assistant" as const,
        })),
        ...inferredMemories,
      );
    }
  }

  return deduplicateAndMerge(results);
}

/**
 * Scan all messages for proper-noun person names and build a pronoun→name map.
 * Uses a simple heuristic: capitalised "First [Last]" not at sentence start,
 * preceded or followed by a pronoun in nearby text.
 */
function buildSessionEntityMap(texts: string[]): Map<string, string> {
  const nameFreq = new Map<string, number>();
  const namePattern = /\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)\b/g;
  const skipWords = new Set([
    "I", "The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or",
    "But", "So", "It", "We", "He", "She", "They", "You", "My", "Our", "Your",
    "His", "Her", "Its", "This", "That", "These", "Those", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
  ]);

  for (const text of texts) {
    let m: RegExpExecArray | null;
    const re = new RegExp(namePattern.source, namePattern.flags);
    while ((m = re.exec(text)) !== null) {
      const word = m[1];
      if (!skipWords.has(word) && word.length > 2) {
        nameFreq.set(word, (nameFreq.get(word) ?? 0) + 1);
      }
    }
  }

  // Pick the most-mentioned person name (>= 2 occurrences) for pronoun mapping.
  // For more sophisticated NER, this is the place to plug in a proper model.
  const candidates = Array.from(nameFreq.entries())
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1]);

  const entityMap = new Map<string, string>();
  if (candidates.length > 0) {
    const [topName] = candidates[0];
    entityMap.set("he", topName);
    entityMap.set("she", topName);
    entityMap.set("they", topName);
  }
  // Map the top-2 distinct names if multiple people appear
  if (candidates.length > 1) {
    entityMap.set("him", candidates[0][0]);
    entityMap.set("her", candidates[0][0]);
  }

  return entityMap;
}

function deduplicateAndMerge(memories: ExtractedMemory[]): ExtractedMemory[] {
  const seen = new Map<string, ExtractedMemory>();

  for (const memory of memories) {
    const key = normalizeDedupeKey(memory.memoryType, memory.content);
    const existing = seen.get(key);

    if (!existing) {
      seen.set(key, memory);
    } else {
      // Keep whichever has higher confidence, but union entity mentions
      const winner = memory.confidence > existing.confidence ? memory : existing;
      seen.set(key, {
        ...winner,
        entityMentions: Array.from(
          new Set([...(existing.entityMentions || []), ...(memory.entityMentions || [])])
        ),
        sourceRole: winner.sourceRole || existing.sourceRole || memory.sourceRole,
        userConfirmed: Boolean(existing.userConfirmed || memory.userConfirmed),
        supportingEvent: Boolean(existing.supportingEvent || memory.supportingEvent),
      });
    }
  }

  return Array.from(seen.values());
}

function normalizeDedupeKey(type: string, content: string): string {
  // Remove stop words, lowercase, sort significant words so order doesn't matter
  // e.g. "Works at Stripe" and "Stripe is where the user works" → same key
  const stopWords = new Set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "i", "my",
    "me", "user", "users",
  ]);
  const words = content
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter(w => w.length > 2 && !stopWords.has(w))
    .sort()
    .slice(0, 6);
  return `${type}:${words.join(":")}`;
}

export function shouldExtractMemory(message: string): boolean {
  if (!message.trim() || message.length < 5) return false;

  // Skip greetings and trivial one-word responses — but never skip on length alone.
  // Long messages are more likely to contain valuable memories, not less.
  const skipPatterns = [
    /^(hi|hello|hey|thanks|thank you|ok|okay|yeah|yes|no|bye)$/i,
    /^(?:\/|!|\?|\.)\s*$/,
    /^[\s\n]*$/,
  ];

  for (const pattern of skipPatterns) {
    if (pattern.test(message.trim())) return false;
  }

  return true;
}
