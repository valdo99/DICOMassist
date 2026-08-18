import { generateText, stepCountIs, type ModelMessage, type LanguageModel } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage } from '../llm/types';
import { buildAgentSystemPrompt } from './systemPrompt';
import { createDicomTools, createRunContext, sliceKey, circleAnnotationUid, type AgentRunContext } from './tools';
import { chooseModel, type AgentProvider } from './modelRouter';
import type { AgentBridge, AgentStepEvent, AgentFindingRef } from './types';
import { logger } from '../utils/logger';

export const MAX_STEPS = 32;
// A few steps before the cap, tell the agent to stop exploring, annotate what
// it found, and answer — so the budget never runs out mid-review.
const WIND_DOWN_STEPS_LEFT = 4;
// Anthropic caps a request at 100 images. Batches are 30, so this leaves room
// for the current batch plus a healthy set of annotated slices.
const MAX_IMAGES_IN_CONTEXT = 70;
// Anthropic also caps a request at 32MB. Base64 image data dominates the
// payload, so budget it well below the cap (text, metadata and JSON overhead
// ride on top). Large-matrix series (DX/MG at 1568px) can hit ~1MB per slice,
// where 70 images would sail past 32MB even though the count budget is met.
const MAX_IMAGE_BYTES_IN_CONTEXT = 20_000_000;

export interface RunAgentParams {
  /** Which API-backed provider to run the tool loop on. */
  provider: AgentProvider;
  apiKey: string;
  metadata: StudyMetadata;
  /** Full conversation so far (the newest user message included). */
  history: ChatMessage[];
  bridge: AgentBridge;
  /** First analysis of a study vs a follow-up — feeds model selection. */
  isNewAnalysis: boolean;
  surveyMode?: boolean;
  /** Pin a specific model id instead of letting the harness choose (''/'auto' = auto). */
  modelOverride?: string;
  /** Called as the agent works, for the live trace. */
  onStep?: (event: AgentStepEvent) => void;
  signal?: AbortSignal;
}

export interface RunAgentResult {
  text: string;
  steps: number;
  modelId: string;
}

type ContentPart = Record<string, unknown>;

function isImagePart(part: unknown): boolean {
  const t = (part as { type?: string })?.type;
  return t === 'file' || t === 'image' || t === 'media';
}

/** Approximate request bytes for an image part — the base64 string dominates. */
function imagePartSize(part: unknown): number {
  const data = (part as { data?: unknown })?.data;
  if (typeof data === 'string') return data.length;
  const nested = (data as { data?: unknown } | undefined)?.data;
  if (typeof nested === 'string') return nested.length;
  // Unknown shape — assume a typical exported slice so the budget stays safe.
  return 400_000;
}

/**
 * Keep the model's image context bounded and relevant.
 *
 * Every batch of slices stays in the conversation, so a long review would blow
 * past Anthropic's 100-image limit (and pay to re-send every image on every
 * step). The agent tells us what matters by annotating: so we keep the batch it
 * is currently working on, plus every slice it has circled, and drop the rest —
 * replacing them with a short note so the agent knows they can be re-viewed.
 *
 * Uses the run context (tool call id → slices returned) rather than parsing
 * captions, so the mapping from image to slice is exact.
 */
function pruneImages(messages: ModelMessage[], ctx: AgentRunContext): ModelMessage[] {
  // Locate every image-bearing tool result, oldest → newest.
  const batches: Array<{ mi: number; pi: number; toolCallId: string; imageCount: number; imageSizes: number[] }> = [];
  messages.forEach((m, mi) => {
    if (m.role !== 'tool' || !Array.isArray(m.content)) return;
    (m.content as ContentPart[]).forEach((part, pi) => {
      if (part?.type !== 'tool-result') return;
      const toolCallId = part.toolCallId as string | undefined;
      if (!toolCallId || !ctx.viewedByCall.has(toolCallId)) return;
      const output = part.output as { type?: string; value?: unknown[] } | undefined;
      if (output?.type !== 'content' || !Array.isArray(output.value)) return;
      const imageSizes = output.value.filter(isImagePart).map(imagePartSize);
      if (imageSizes.length > 0) batches.push({ mi, pi, toolCallId, imageCount: imageSizes.length, imageSizes });
    });
  });
  if (batches.length === 0) return messages;

  // Decide, per batch, which image indexes survive. The newest batch has first
  // claim on the budget; older batches keep only annotated slices. Both an
  // image-count and a byte budget apply — either one running out drops the
  // rest, keeping every request under Anthropic's 100-image / 32MB caps.
  const keepByBatch = new Map<string, Set<number>>();
  let budget = MAX_IMAGES_IN_CONTEXT;
  let byteBudget = MAX_IMAGE_BYTES_IN_CONTEXT;
  let dropped = 0;

  for (let b = batches.length - 1; b >= 0; b--) {
    const batch = batches[b];
    const refs = ctx.viewedByCall.get(batch.toolCallId) ?? [];
    const keep = new Set<number>();
    const isNewest = b === batches.length - 1;

    for (let i = 0; i < batch.imageCount; i++) {
      const ref = refs[i];
      const size = batch.imageSizes[i] ?? 0;
      const annotated = ref ? ctx.annotated.has(sliceKey(ref.seriesNumber, ref.instanceNumber)) : false;
      if ((isNewest || annotated) && budget > 0 && size <= byteBudget) {
        keep.add(i);
        budget--;
        byteBudget -= size;
      } else {
        dropped++;
      }
    }
    keepByBatch.set(batch.toolCallId, keep);
  }
  if (dropped === 0) return messages;

  // Rebuild only the messages that actually change.
  const out = messages.slice();
  const edited = new Map<number, ContentPart[]>();

  for (const batch of batches) {
    const keep = keepByBatch.get(batch.toolCallId);
    if (!keep || keep.size === batch.imageCount) continue;

    if (!edited.has(batch.mi)) edited.set(batch.mi, (out[batch.mi].content as ContentPart[]).slice());
    const content = edited.get(batch.mi)!;
    const part = { ...content[batch.pi] } as ContentPart;
    const value = ((part.output as { value?: ContentPart[] }).value ?? []) as ContentPart[];

    const next: ContentPart[] = [];
    let imageIdx = 0;
    let removed = 0;
    for (const p of value) {
      if (!isImagePart(p)) {
        next.push(p);
        continue;
      }
      if (keep.has(imageIdx)) {
        next.push(p);
      } else {
        // Drop the image and the caption line that introduced it.
        const last = next[next.length - 1] as { type?: string } | undefined;
        if (last?.type === 'text' && next.length > 1) next.pop();
        removed++;
      }
      imageIdx++;
    }
    if (removed > 0) {
      next.push({
        type: 'text',
        text:
          `[${removed} slice image${removed === 1 ? '' : 's'} from this batch removed from context ` +
          `(not annotated, or trimmed to keep the request within API limits). ` +
          `Call view_slices or compare_slices again to look at them once more.]`,
      });
    }
    part.output = { type: 'content', value: next };
    content[batch.pi] = part;

    // Keep the recorded slices in step with the images that actually remain.
    // prepareStep hands us the previously-pruned messages, so without this the
    // index → slice mapping would drift and annotated slices would be dropped
    // on a later pass.
    const refs = ctx.viewedByCall.get(batch.toolCallId) ?? [];
    ctx.viewedByCall.set(
      batch.toolCallId,
      refs.filter((_, i) => keep.has(i)),
    );
  }

  for (const [mi, content] of edited) {
    out[mi] = { ...(out[mi] as Record<string, unknown>), content } as ModelMessage;
  }
  logger.log(`[Agent] pruned ${dropped} un-annotated image(s); ${MAX_IMAGES_IN_CONTEXT - budget} kept in context`);
  return out;
}

/**
 * Remove every image from tool results, keeping the text (headers, slice
 * captions). Last-resort shape for the closing summary when even the pruned
 * context is rejected by the API: the agent's own narration and the slice
 * captions are enough to write up what it saw.
 */
function stripImages(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((m) => {
    if (m.role !== 'tool' || !Array.isArray(m.content)) return m;
    let changed = false;
    const content = (m.content as ContentPart[]).map((part) => {
      if (part?.type !== 'tool-result') return part;
      const output = part.output as { type?: string; value?: ContentPart[] } | undefined;
      if (output?.type !== 'content' || !Array.isArray(output.value)) return part;
      const kept = output.value.filter((p) => !isImagePart(p));
      if (kept.length === output.value.length) return part;
      changed = true;
      return {
        ...part,
        output: {
          type: 'content',
          value: [...kept, { type: 'text', text: '[Slice images omitted here to fit the request.]' }],
        },
      };
    });
    return changed ? ({ ...m, content } as ModelMessage) : m;
  });
}

/**
 * Build the AI SDK language model for the chosen provider. Both providers speak
 * the same `generateText` + tools interface, so the rest of the agent (tool
 * loop, image pruning, closing summary) is provider-agnostic. Gemini can see
 * images returned from tool results as of @ai-sdk/google 2.0.13+ (PR #8357).
 */
function buildModel(provider: AgentProvider, apiKey: string, modelId: string): LanguageModel {
  if (provider === 'gemini') {
    const google = createGoogleGenerativeAI({ apiKey });
    return google(modelId);
  }
  const anthropic = createAnthropic({
    apiKey,
    // Required for direct browser calls — keeps the app client-only.
    headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
  });
  return anthropic(modelId);
}

/**
 * Runs the DICOM agent: one tool-using loop (review a batch → mark findings →
 * compare → answer). The model sees the images it selects and marks them up,
 * and the harness picks the model tier from how hard the task looks.
 */
export async function runDicomAgent(params: RunAgentParams): Promise<RunAgentResult> {
  const { provider, apiKey, metadata, history, bridge, isNewAnalysis, surveyMode, modelOverride, onStep, signal } = params;

  const question = [...history].reverse().find((m) => m.role === 'user')?.content ?? '';
  const choice = chooseModel({ provider, question, metadata, isNewAnalysis, surveyMode, override: modelOverride });
  onStep?.({ type: 'model', detail: `${choice.modelId} · ${choice.reason}` });

  const ctx = createRunContext();
  const tools = createDicomTools(metadata, bridge, ctx);
  const model: LanguageModel = buildModel(provider, apiKey, choice.modelId);
  const system = buildAgentSystemPrompt(metadata, MAX_STEPS);

  const messages: ModelMessage[] = history.map((m) => ({ role: m.role, content: m.content }));

  logger.group('[DICOMassist] Agent run');
  logger.log(`Model: ${choice.modelId} (${choice.tier}, score ${choice.score}) — ${choice.reason}`);

  // The pruned messages actually sent on the latest step. The SDK's
  // `result.response.messages` holds the RAW tool results (every image of
  // every batch), so a closing call built from those blows Anthropic's
  // 100-image / 32MB request limits and kills the whole run — exactly the
  // "agent goes silent after many batches" failure. Track the pruned state
  // instead and build the closing call from it.
  let latestStepMessages: ModelMessage[] = messages;
  let windDownInjected = false;
  let stepsCompleted = 0;

  const closingPrompt: ModelMessage = {
    role: 'user',
    content:
      'Summarise what you found for the user now, based on the slices you reviewed. ' +
      'Do not call any more tools. Cite the series and slice for each finding, tier them ' +
      '(definite / probable / possible), and include the limitations of this review.',
  };

  /** Ask for the final write-up from a conversation state, tools disabled. */
  const requestClosingSummary = async (base: ModelMessage[], withImages: boolean): Promise<string> => {
    const closing = await generateText({
      model,
      system,
      messages: [...(withImages ? pruneImages(base, ctx) : stripImages(base)), closingPrompt],
      tools,
      toolChoice: 'none',
      abortSignal: signal,
    });
    return closing.text?.trim() ?? '';
  };

  try {
    let result;
    try {
      result = await generateText({
        model,
        system,
        messages,
        tools,
        stopWhen: stepCountIs(MAX_STEPS),
        prepareStep: ({ messages: stepMessages, stepNumber }) => {
          let prepared = pruneImages(stepMessages, ctx);
          if (!windDownInjected && stepNumber >= MAX_STEPS - WIND_DOWN_STEPS_LEFT) {
            windDownInjected = true;
            prepared = [
              ...prepared,
              {
                role: 'user',
                content:
                  `[Harness notice — not the user speaking] You have at most ${MAX_STEPS - stepNumber - 1} tool ` +
                  'steps left in this turn, then one final message for your answer. Stop exploring new regions ' +
                  'now. First mark every finding you have identified with draw_circle (you can issue several ' +
                  'draw_circle calls in a single step), then write your final answer for the user. Do not review ' +
                  'any further batches.',
              },
            ];
            onStep?.({ type: 'text', text: 'Step budget nearly used — annotating findings and preparing the answer…' });
          }
          latestStepMessages = prepared;
          return { messages: prepared };
        },
        abortSignal: signal,
        onStepFinish: (step) => {
          stepsCompleted++;
          // Surface the agent's own narration, not just which tool it called.
          const text = step.text?.trim();
          if (text) onStep?.({ type: 'text', text });
          for (const call of step.toolCalls ?? []) {
            onStep?.({
              type: 'tool-call',
              toolName: call.toolName,
              detail: summarizeToolCall(call.toolName, call.input),
              finding: call.toolName === 'draw_circle' ? resolveFinding(metadata, call.input) : undefined,
            });
          }
        },
      });
    } catch (err) {
      // A failed request mid-loop used to kill the whole run and lose every
      // batch already reviewed. If the agent got anywhere, salvage a written
      // summary from the last good conversation state (text-only: if the
      // failure was an oversized request, the images are the culprit).
      if (signal?.aborted || ctx.viewedByCall.size === 0) throw err;
      logger.warn('[Agent] tool loop failed mid-run — salvaging a summary from what was reviewed', err);
      onStep?.({ type: 'text', text: 'The review hit an error — writing up what was examined so far…' });
      const text = await requestClosingSummary(latestStepMessages, false).catch((salvageErr) => {
        logger.warn('[Agent] salvage summary failed too', salvageErr);
        return '';
      });
      if (!text) throw err;
      logger.log(`Agent salvaged a summary after ${stepsCompleted} steps`);
      logger.groupEnd();
      return { text, steps: stepsCompleted, modelId: choice.modelId };
    }

    const lastStep = result.steps[result.steps.length - 1];
    // A turn that ends on tool calls hit the step cap mid-work: any trailing
    // narration ("Now I'll check the coronals…") is not an answer.
    const endedOnToolCalls = (lastStep?.toolCalls?.length ?? 0) > 0;
    let text = endedOnToolCalls ? '' : (result.text?.trim() ?? '');

    // However the loop ended, never leave the user with nothing.
    if (!text) {
      logger.log('Loop ended without a final answer — requesting a closing summary');
      onStep?.({ type: 'text', text: 'Wrapping up the findings…' });
      // latestStepMessages + the final step's own response messages is exactly
      // the conversation state the loop would have continued from.
      const closingBase: ModelMessage[] = [
        ...latestStepMessages,
        ...((lastStep?.response.messages ?? []) as ModelMessage[]),
      ];
      try {
        text = await requestClosingSummary(closingBase, true);
      } catch (err) {
        if (signal?.aborted) throw err;
        // Even the pruned context was rejected (or the call failed) — retry
        // once with every image stripped. The agent's narration and slice
        // captions are enough to write the summary from.
        logger.warn('[Agent] closing summary failed, retrying without images', err);
        try {
          text = await requestClosingSummary(closingBase, false);
        } catch (retryErr) {
          if (signal?.aborted) throw retryErr;
          // Don't throw away a run's worth of review and annotations over a
          // failed summary call — fall through to the canned message.
          logger.warn('[Agent] text-only closing summary also failed', retryErr);
        }
      }
    }

    if (!text) {
      text =
        'I reviewed the study but was not able to produce a written summary this time. ' +
        'Any regions I marked are shown on the images — please ask me to try again.';
    }

    logger.log(`Agent finished in ${stepsCompleted} steps`);
    logger.groupEnd();
    return { text, steps: stepsCompleted, modelId: choice.modelId };
  } catch (err) {
    logger.groupEnd();
    throw err;
  }
}

/**
 * Resolve a `draw_circle` tool call to the concrete finding it produced, so the
 * trace can link straight to that circle. Reuses the same series lookup and uid
 * formula as the tool itself, so the uid matches the annotation that was added.
 * Returns undefined if the slice can't be resolved (nothing to open).
 */
function resolveFinding(metadata: StudyMetadata, input: unknown): AgentFindingRef | undefined {
  const i = (input ?? {}) as Record<string, unknown>;
  const series = metadata.series.find((s) => String(s.seriesNumber) === String(i.seriesNumber));
  const instanceNumber = Number(i.instanceNumber);
  const slice = series?.slices.find((s) => s.instanceNumber === instanceNumber);
  if (!series || !slice) return undefined;
  return {
    uid: circleAnnotationUid(series.seriesNumber, instanceNumber, Number(i.cx), Number(i.cy)),
    seriesNumber: String(series.seriesNumber),
    instanceNumber,
    imageId: slice.imageId,
    label: (typeof i.label === 'string' && i.label ? i.label : 'Finding').slice(0, 40),
  };
}

function summarizeToolCall(toolName: string, input: unknown): string {
  const i = (input ?? {}) as Record<string, unknown>;
  switch (toolName) {
    case 'view_slices':
      return `Series #${i.seriesNumber}, slices ${i.fromInstance}–${i.toInstance}`;
    case 'compare_slices': {
      const slices = Array.isArray(i.slices) ? i.slices.length : 0;
      return `${slices} slices${i.reason ? ` · ${i.reason}` : ''}`;
    }
    case 'draw_circle':
      return `${i.label} on Series #${i.seriesNumber} Slice ${i.instanceNumber}`;
    case 'navigate_to_slice':
      return `Series #${i.seriesNumber} Slice ${i.instanceNumber}`;
    case 'set_window_level':
      return `W:${i.windowWidth} C:${i.windowCenter}`;
    default:
      return '';
  }
}
