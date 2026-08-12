import { generateText, stepCountIs, type ModelMessage } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage } from '../llm/types';
import { buildAgentSystemPrompt } from './systemPrompt';
import { createDicomTools } from './tools';
import type { AgentBridge, AgentStepEvent } from './types';
import { logger } from '../utils/logger';

const DEFAULT_MODEL = 'claude-sonnet-4-5';
const MAX_STEPS = 12;
// Anthropic caps a request at 100 images. The agent calls view_slices repeatedly
// and every batch persists in history, so we keep only the most recent images
// (well under the cap) and replace older batches with a text placeholder — the
// agentic equivalent of context editing.
const MAX_IMAGES_IN_CONTEXT = 60;

/**
 * Strip images from all but the most recent view_slices tool results (keeping
 * the total under MAX_IMAGES_IN_CONTEXT). Older batches become a short text note
 * so the model still knows what it saw and can re-view if needed. Idempotent.
 */
function pruneOldImages(messages: ModelMessage[]): ModelMessage[] {
  const isImagePart = (v: unknown) => {
    const t = (v as { type?: string })?.type;
    return t === 'file' || t === 'image' || t === 'media';
  };

  // Collect image-bearing view_slices results, oldest → newest.
  const positions: Array<{ mi: number; pi: number; count: number }> = [];
  messages.forEach((m, mi) => {
    if (m.role !== 'tool' || !Array.isArray(m.content)) return;
    (m.content as Array<Record<string, unknown>>).forEach((part, pi) => {
      if (part?.type !== 'tool-result' || part.toolName !== 'view_slices') return;
      const output = part.output as { type?: string; value?: unknown[] } | undefined;
      if (output?.type === 'content' && Array.isArray(output.value)) {
        const count = output.value.filter(isImagePart).length;
        if (count > 0) positions.push({ mi, pi, count });
      }
    });
  });

  // Keep newest batches within the budget; always keep at least the latest.
  const keep = new Set<string>();
  let total = 0;
  for (let i = positions.length - 1; i >= 0; i--) {
    const p = positions[i];
    if (keep.size === 0 || total + p.count <= MAX_IMAGES_IN_CONTEXT) {
      keep.add(`${p.mi}:${p.pi}`);
      total += p.count;
    }
  }
  const strip = positions.filter((p) => !keep.has(`${p.mi}:${p.pi}`));
  if (strip.length === 0) return messages;

  const out = messages.slice();
  const touched = new Map<number, Array<Record<string, unknown>>>();
  for (const p of strip) {
    if (!touched.has(p.mi)) touched.set(p.mi, (out[p.mi].content as Array<Record<string, unknown>>).slice());
    const content = touched.get(p.mi)!;
    const part = { ...content[p.pi] } as Record<string, unknown>;
    const output = part.output as { value?: Array<Record<string, unknown>> };
    const label = (output.value ?? [])
      .filter((v) => v?.type === 'text')
      .map((v) => v.text as string)
      .join(' ');
    part.output = {
      type: 'text',
      value: `${label} [${p.count} slice image${p.count === 1 ? '' : 's'} omitted to save context — call view_slices again to see them]`.trim(),
    };
    content[p.pi] = part;
  }
  for (const [mi, content] of touched) {
    out[mi] = { ...(out[mi] as Record<string, unknown>), content } as ModelMessage;
  }
  logger.log(`[Agent] pruned images from ${strip.length} old view(s); keeping ~${total} in context`);
  return out;
}

export interface RunAgentParams {
  apiKey: string;
  model?: string;
  metadata: StudyMetadata;
  /** Full conversation so far (the newest user message included). */
  history: ChatMessage[];
  bridge: AgentBridge;
  /** Called after each step for the trace/pipeline UI. */
  onStep?: (event: AgentStepEvent) => void;
  /** Abort signal to cancel a run. */
  signal?: AbortSignal;
}

export interface RunAgentResult {
  text: string;
  steps: number;
}

/**
 * Runs the DICOM agent: a single tool-using loop (view slices → reason → draw →
 * answer) that replaces the old two-call pipeline. The model sees the images it
 * selects (via the view_slices tool's image output) and can mark them up.
 */
export async function runDicomAgent(params: RunAgentParams): Promise<RunAgentResult> {
  const { apiKey, model = DEFAULT_MODEL, metadata, history, bridge, onStep, signal } = params;

  const anthropic = createAnthropic({
    apiKey,
    // Required for direct browser calls — keeps the app client-only.
    headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
  });

  const tools = createDicomTools(metadata, bridge);

  const messages: ModelMessage[] = history.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  logger.group('[DICOMassist] Agent run');
  logger.log(`Model: ${model}, messages: ${messages.length}`);

  try {
    const result = await generateText({
      model: anthropic(model),
      system: buildAgentSystemPrompt(metadata),
      messages,
      tools,
      stopWhen: stepCountIs(MAX_STEPS),
      // Keep the number of images sent per step under Anthropic's 100-image cap.
      prepareStep: ({ messages: stepMessages }) => ({ messages: pruneOldImages(stepMessages) }),
      abortSignal: signal,
      onStepFinish: (step) => {
        if (step.text && step.text.trim()) {
          onStep?.({ type: 'text', detail: step.text.slice(0, 160) });
        }
        for (const call of step.toolCalls ?? []) {
          onStep?.({ type: 'tool-call', toolName: call.toolName, detail: summarizeToolCall(call.toolName, call.input) });
        }
      },
    });

    logger.log(`Agent finished in ${result.steps.length} steps`);
    logger.groupEnd();
    return { text: result.text, steps: result.steps.length };
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
