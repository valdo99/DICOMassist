import { generateText, stepCountIs, type ModelMessage } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage } from '../llm/types';
import { buildAgentSystemPrompt } from './systemPrompt';
import { createDicomTools, createRunContext, sliceKey, type AgentRunContext } from './tools';
import { chooseModel, type ModelTier } from './modelRouter';
import type { AgentBridge, AgentStepEvent } from './types';
import { logger } from '../utils/logger';

const MAX_STEPS = 14;
// Anthropic caps a request at 100 images. Batches are 30, so this leaves room
// for the current batch plus a healthy set of annotated slices.
const MAX_IMAGES_IN_CONTEXT = 70;

export interface RunAgentParams {
  apiKey: string;
  metadata: StudyMetadata;
  /** Full conversation so far (the newest user message included). */
  history: ChatMessage[];
  bridge: AgentBridge;
  /** First analysis of a study vs a follow-up — feeds model selection. */
  isNewAnalysis: boolean;
  surveyMode?: boolean;
  /** Pin a tier instead of letting the harness choose. */
  modelOverride?: ModelTier | 'auto';
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
  const batches: Array<{ mi: number; pi: number; toolCallId: string; imageCount: number }> = [];
  messages.forEach((m, mi) => {
    if (m.role !== 'tool' || !Array.isArray(m.content)) return;
    (m.content as ContentPart[]).forEach((part, pi) => {
      if (part?.type !== 'tool-result') return;
      const toolCallId = part.toolCallId as string | undefined;
      if (!toolCallId || !ctx.viewedByCall.has(toolCallId)) return;
      const output = part.output as { type?: string; value?: unknown[] } | undefined;
      if (output?.type !== 'content' || !Array.isArray(output.value)) return;
      const imageCount = output.value.filter(isImagePart).length;
      if (imageCount > 0) batches.push({ mi, pi, toolCallId, imageCount });
    });
  });
  if (batches.length === 0) return messages;

  // Decide, per batch, which image indexes survive. Newest batch stays whole;
  // older batches keep only annotated slices, within the global budget.
  const keepByBatch = new Map<string, Set<number>>();
  let budget = MAX_IMAGES_IN_CONTEXT;
  let dropped = 0;

  for (let b = batches.length - 1; b >= 0; b--) {
    const batch = batches[b];
    const refs = ctx.viewedByCall.get(batch.toolCallId) ?? [];
    const keep = new Set<number>();
    const isNewest = b === batches.length - 1;

    for (let i = 0; i < batch.imageCount; i++) {
      const ref = refs[i];
      const annotated = ref ? ctx.annotated.has(sliceKey(ref.seriesNumber, ref.instanceNumber)) : false;
      if ((isNewest || annotated) && budget > 0) {
        keep.add(i);
        budget--;
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
          `(no finding was marked on them). Call view_slices or compare_slices again to look at them once more.]`,
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
 * Runs the DICOM agent: one tool-using loop (review a batch → mark findings →
 * compare → answer). The model sees the images it selects and marks them up,
 * and the harness picks the model tier from how hard the task looks.
 */
export async function runDicomAgent(params: RunAgentParams): Promise<RunAgentResult> {
  const { apiKey, metadata, history, bridge, isNewAnalysis, surveyMode, modelOverride, onStep, signal } = params;

  const anthropic = createAnthropic({
    apiKey,
    // Required for direct browser calls — keeps the app client-only.
    headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
  });

  const question = [...history].reverse().find((m) => m.role === 'user')?.content ?? '';
  const choice = chooseModel({ question, metadata, isNewAnalysis, surveyMode, override: modelOverride });
  onStep?.({ type: 'model', detail: `${choice.modelId} · ${choice.reason}` });

  const ctx = createRunContext();
  const tools = createDicomTools(metadata, bridge, ctx);
  const model = anthropic(choice.modelId);
  const system = buildAgentSystemPrompt(metadata);

  const messages: ModelMessage[] = history.map((m) => ({ role: m.role, content: m.content }));

  logger.group('[DICOMassist] Agent run');
  logger.log(`Model: ${choice.modelId} (${choice.tier}, score ${choice.score}) — ${choice.reason}`);

  try {
    const result = await generateText({
      model,
      system,
      messages,
      tools,
      stopWhen: stepCountIs(MAX_STEPS),
      prepareStep: ({ messages: stepMessages }) => ({ messages: pruneImages(stepMessages, ctx) }),
      abortSignal: signal,
      onStepFinish: (step) => {
        // Surface the agent's own narration, not just which tool it called.
        const text = step.text?.trim();
        if (text) onStep?.({ type: 'text', text });
        for (const call of step.toolCalls ?? []) {
          onStep?.({
            type: 'tool-call',
            toolName: call.toolName,
            detail: summarizeToolCall(call.toolName, call.input),
          });
        }
      },
    });

    let text = result.text?.trim() ?? '';

    // The loop can end on a tool call (e.g. it hit the step cap), which would
    // leave the user with nothing. Always come back with an answer.
    if (!text) {
      logger.log('No final text — requesting a closing summary');
      onStep?.({ type: 'text', text: 'Wrapping up the findings…' });
      const closing = await generateText({
        model,
        system,
        messages: [
          ...messages,
          ...result.response.messages,
          {
            role: 'user',
            content:
              'Summarise what you found for the user now, based on the slices you reviewed. ' +
              'Do not call any more tools. Cite the series and slice for each finding, tier them ' +
              '(definite / probable / possible), and include the limitations of this review.',
          },
        ],
        tools,
        toolChoice: 'none',
        abortSignal: signal,
      });
      text = closing.text?.trim() ?? '';
    }

    if (!text) {
      text =
        'I reviewed the study but was not able to produce a written summary this time. ' +
        'Any regions I marked are shown on the images — please ask me to try again.';
    }

    logger.log(`Agent finished in ${result.steps.length} steps`);
    logger.groupEnd();
    return { text, steps: result.steps.length, modelId: choice.modelId };
  } catch (err) {
    logger.groupEnd();
    throw err;
  }
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
