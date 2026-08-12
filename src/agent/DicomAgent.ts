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
