import type { StudyMetadata } from '../dicom/types';
import type { SelectionPlan, SeriesSelection, ChatMessage, ProviderConfig, LLMService, ViewportContext, AnalysisResult, SliceCircle } from './types';
import {
  buildSelectionSystemPrompt,
  buildSelectionUserPrompt,
  buildAnalysisSystemPrompt,
  buildAnalysisUserPrompt,
  buildFollowUpSystemPrompt,
} from './PromptBuilder';

// --- Shared Helpers ---

function extractJson(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  const braceStart = text.indexOf('{');
  const braceEnd = text.lastIndexOf('}');
  if (braceStart !== -1 && braceEnd > braceStart) {
    return text.slice(braceStart, braceEnd + 1);
  }
  return text.trim();
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

function normalizeCircle(raw: Record<string, unknown>): SliceCircle | null {
  const image = Number(raw.image);
  const cx = Number(raw.cx);
  const cy = Number(raw.cy);
  const radius = Number(raw.radius);
  if (![image, cx, cy, radius].every(Number.isFinite)) return null;
  return {
    image: Math.round(image),
    cx: clamp01(cx),
    cy: clamp01(cy),
    // keep the circle sane: never smaller than a couple px, never over half the image
    radius: Math.min(0.5, Math.max(0.01, radius)),
    label: String(raw.label ?? 'Finding').trim().slice(0, 40) || 'Finding',
  };
}

/**
 * Split a Call-2 response into its prose and any annotation circles. The model
 * is asked to append a fenced ```annotations``` (or ```json) block containing
 * { "circles": [...] }. We parse it out and strip it from the displayed text.
 * Missing/invalid blocks degrade to no annotations — never throws.
 */
function extractAnnotations(text: string): AnalysisResult {
  const fenceRe = /```(?:annotations|json)?\s*(\{[\s\S]*?\})\s*```/gi;
  let match: RegExpExecArray | null;
  while ((match = fenceRe.exec(text)) !== null) {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(match[1]);
    } catch {
      continue; // not JSON — keep scanning
    }
    const list = Array.isArray(parsed.circles)
      ? parsed.circles
      : Array.isArray(parsed.annotations)
        ? parsed.annotations
        : null;
    if (!list) continue; // some other JSON block — leave it, keep scanning
    const annotations = (list as Record<string, unknown>[])
      .map(normalizeCircle)
      .filter((c): c is SliceCircle => c !== null);
    const cleaned = (text.slice(0, match.index) + text.slice(match.index + match[0].length)).trim();
    return { text: cleaned, annotations };
  }
  return { text: text.trim(), annotations: [] };
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function parseSeriesSelection(raw: Record<string, unknown>): SeriesSelection {
  return {
    seriesNumber: String(raw.seriesNumber),
    role: (raw.role as string) === 'supplementary' ? 'supplementary' : 'primary',
    rationale: String(raw.rationale ?? ''),
    sliceRange: [Number((raw.sliceRange as number[])[0]), Number((raw.sliceRange as number[])[1])],
    samplingStrategy: (raw.samplingStrategy as string) ?? 'uniform',
    samplingParam: raw.samplingParam != null ? Number(raw.samplingParam) : undefined,
    windowWidth: Number(raw.windowWidth),
    windowCenter: Number(raw.windowCenter),
  };
}

function populateLegacyFields(selections: SeriesSelection[], reasoning: string, totalImages: number): SelectionPlan {
  const primary = selections[0];
  return {
    reasoning,
    selections,
    totalImages,
    targetSeries: primary.seriesNumber,
    sliceRange: primary.sliceRange,
    windowCenter: primary.windowCenter,
    windowWidth: primary.windowWidth,
    samplingStrategy: primary.samplingStrategy,
    samplingParam: primary.samplingParam,
  };
}

function parseSelectionPlan(raw: string): SelectionPlan {
  let json: Record<string, unknown>;
  try {
    json = JSON.parse(extractJson(raw));
  } catch {
    throw new Error(
      'The LLM did not return valid JSON. This can happen with smaller models. ' +
      'Try a more specific clinical prompt (e.g., "Evaluate for lung nodules") or switch to Claude.',
    );
  }

  // New multi-series format: { reasoning, selections: [...], totalImages }
  if (Array.isArray(json.selections) && json.selections.length > 0) {
    const selections = (json.selections as Record<string, unknown>[]).map(parseSeriesSelection);
    const reasoning = String(json.reasoning ?? '');
    const totalImages = json.totalImages != null ? Number(json.totalImages) : 0;
    return populateLegacyFields(selections, reasoning, totalImages);
  }

  // Legacy single-series format: { targetSeries, sliceRange, ... }
  if (!json.targetSeries || !json.sliceRange) {
    throw new Error(
      'The LLM response is missing required fields (targetSeries, sliceRange). ' +
      'Try a more specific clinical prompt or a larger model.',
    );
  }

  const selection: SeriesSelection = {
    seriesNumber: String(json.targetSeries),
    role: 'primary',
    rationale: String(json.reasoning ?? ''),
    sliceRange: [Number((json.sliceRange as number[])[0]), Number((json.sliceRange as number[])[1])],
    samplingStrategy: (json.samplingStrategy as string) ?? 'uniform',
    samplingParam: json.samplingParam != null ? Number(json.samplingParam) : undefined,
    windowCenter: Number(json.windowCenter),
    windowWidth: Number(json.windowWidth),
  };

  return populateLegacyFields([selection], selection.rationale, 0);
}

// --- Claude Service ---

class ClaudeService implements LLMService {
  private apiKey: string;
  private model = 'claude-sonnet-4-5-20250929';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async getSelectionPlan(metadata: StudyMetadata, clinicalHint: string, viewportContext?: ViewportContext): Promise<SelectionPlan> {
    const response = await this.callClaude({
      system: buildSelectionSystemPrompt(),
      messages: [{ role: 'user', content: buildSelectionUserPrompt(metadata, clinicalHint, viewportContext) }],
      temperature: 0,
      maxTokens: 1024,
    });
    return parseSelectionPlan(response);
  }

  async analyzeSlices(
    images: Blob[],
    metadata: StudyMetadata,
    clinicalHint: string,
    plan: SelectionPlan,
    sliceLabels: string[],
    surveyMode?: boolean,
  ): Promise<AnalysisResult> {
    const imageContents = await Promise.all(
      images.map(async (blob, i) => [
        {
          type: 'image' as const,
          source: {
            type: 'base64' as const,
            media_type: 'image/jpeg' as const,
            data: await blobToBase64(blob),
          },
        },
        {
          type: 'text' as const,
          text: sliceLabels[i] ?? `Image ${i + 1}`,
        },
      ]),
    );

    const content = [
      ...imageContents.flat(),
      {
        type: 'text' as const,
        text: buildAnalysisUserPrompt(metadata, clinicalHint, plan, sliceLabels),
      },
    ];

    const raw = await this.callClaude({
      system: buildAnalysisSystemPrompt(surveyMode),
      messages: [{ role: 'user', content }],
      temperature: 0,
      maxTokens: 4096,
    });
    return extractAnnotations(raw);
  }

  async sendFollowUp(
    conversationHistory: ChatMessage[],
    metadata: StudyMetadata,
    images?: Blob[],
    sliceLabels?: string[],
  ): Promise<AnalysisResult> {
    const hasImages = !!images && images.length > 0;
    const messages: Array<{ role: 'user' | 'assistant'; content: unknown }> =
      conversationHistory.map((msg) => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content as unknown,
      }));

    // Re-attach the analyzed images to the latest user turn so the model can
    // actually see them again — for both visual reasoning and annotations.
    if (hasImages) {
      const imageBlocks = await Promise.all(
        images!.map(async (blob) => ({
          type: 'image' as const,
          source: {
            type: 'base64' as const,
            media_type: 'image/jpeg' as const,
            data: await blobToBase64(blob),
          },
        })),
      );
      const manifest = (sliceLabels ?? [])
        .map((l, i) => `  Image ${i + 1}: ${l}`)
        .join('\n');
      const lastIdx = messages.length - 1;
      const lastText = typeof messages[lastIdx]?.content === 'string'
        ? (messages[lastIdx].content as string)
        : '';
      messages[lastIdx] = {
        role: 'user',
        content: [
          ...imageBlocks,
          {
            type: 'text' as const,
            text: `The ${imageBlocks.length} analyzed images (in order):\n${manifest}\n\n${lastText}`,
          },
        ],
      };
    }

    const raw = await this.callClaude({
      system: buildFollowUpSystemPrompt(hasImages) + '\n\nStudy context: ' + metadata.studyDescription,
      messages,
      temperature: 0,
      maxTokens: 4096,
    });
    return extractAnnotations(raw);
  }

  private async callClaude(params: {
    system: string;
    messages: Array<{ role: string; content: unknown }>;
    temperature: number;
    maxTokens: number;
  }): Promise<string> {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: params.maxTokens,
        temperature: params.temperature,
        system: params.system,
        messages: params.messages,
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      if (res.status === 401) throw new Error('Invalid API key. Check your Claude API key.');
      throw new Error(`Claude API error (${res.status}): ${body}`);
    }

    const data = await res.json();
    const textBlock = data.content?.find((b: { type: string }) => b.type === 'text');
    return textBlock?.text ?? '';
  }
}

// --- Ollama Service ---

class OllamaService implements LLMService {
  private baseUrl: string;
  private textModel: string;
  private visionModel: string;

  constructor(textModel: string, visionModel: string, baseUrl: string) {
    this.textModel = textModel;
    this.visionModel = visionModel;
    this.baseUrl = baseUrl;
  }

  async getSelectionPlan(metadata: StudyMetadata, clinicalHint: string, viewportContext?: ViewportContext): Promise<SelectionPlan> {
    const response = await this.callOllama({
      model: this.textModel,
      system: buildSelectionSystemPrompt(),
      userContent: buildSelectionUserPrompt(metadata, clinicalHint, viewportContext),
    });
    return parseSelectionPlan(response);
  }

  async analyzeSlices(
    images: Blob[],
    metadata: StudyMetadata,
    clinicalHint: string,
    plan: SelectionPlan,
    sliceLabels: string[],
    surveyMode?: boolean,
  ): Promise<AnalysisResult> {
    const base64Images = await Promise.all(images.map(blobToBase64));
    const manifest = sliceLabels.map((l, i) => `  ${i + 1}. ${l}`).join('\n');
    const userContent =
      `IMAGE MANIFEST (${sliceLabels.length} images, in sequential order):\n${manifest}\n\nThe images are provided in the exact order listed above.\n\n` +
      buildAnalysisUserPrompt(metadata, clinicalHint, plan, sliceLabels);

    const raw = await this.callOllama({
      model: this.visionModel,
      system: buildAnalysisSystemPrompt(surveyMode),
      userContent,
      images: base64Images,
    });
    return extractAnnotations(raw);
  }

  async sendFollowUp(
    conversationHistory: ChatMessage[],
    metadata: StudyMetadata,
    images?: Blob[],
    sliceLabels?: string[],
  ): Promise<AnalysisResult> {
    const hasImages = !!images && images.length > 0;
    const base64Images = hasImages ? await Promise.all(images!.map(blobToBase64)) : undefined;

    const messages: Array<{ role: string; content: string; images?: string[] }> = [
      { role: 'system', content: buildFollowUpSystemPrompt(hasImages) + '\n\nStudy context: ' + metadata.studyDescription },
      ...conversationHistory.map((msg) => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
      })),
    ];

    // Re-attach images to the latest user turn (vision model required for these).
    if (hasImages && base64Images) {
      const manifest = (sliceLabels ?? []).map((l, i) => `  ${i + 1}. ${l}`).join('\n');
      const lastIdx = messages.length - 1;
      messages[lastIdx] = {
        ...messages[lastIdx],
        content: `Analyzed images (in order):\n${manifest}\n\n${messages[lastIdx].content}`,
        images: base64Images,
      };
    }

    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: hasImages ? this.visionModel : this.textModel,
        messages,
        stream: false,
        options: { temperature: 0 },
      }),
      signal: AbortSignal.timeout(300_000),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Ollama error (${res.status}): ${body}`);
    }

    const data = await res.json();
    return extractAnnotations(data.message?.content ?? '');
  }

  private async callOllama(params: {
    model: string;
    system: string;
    userContent: string;
    images?: string[];
  }): Promise<string> {
    const messages = [
      { role: 'system', content: params.system },
      {
        role: 'user',
        content: params.userContent,
        ...(params.images?.length ? { images: params.images } : {}),
      },
    ];

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: params.model,
          messages,
          stream: false,
          options: { temperature: 0 },
        }),
        signal: AbortSignal.timeout(300_000),
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'TimeoutError') {
        throw new Error(`Ollama request timed out (5min). Model: ${params.model}. Try fewer slices or a smaller model.`);
      }
      throw new Error('Cannot connect to Ollama. Is it running? (ollama serve)');
    }

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Ollama error (${res.status}): ${body}`);
    }

    const data = await res.json();
    return data.message?.content ?? '';
  }
}

// --- Factory ---

const DEFAULT_TEXT_MODEL = 'alibayram/medgemma:4b';
const DEFAULT_VISION_MODEL = 'gemma3:4b';

export function createLLMService(config: ProviderConfig): LLMService {
  if (config.provider === 'claude') {
    const key = config.apiKey || import.meta.env.VITE_ANTHROPIC_API_KEY;
    if (!key) throw new Error('Claude API key is required. Enter it in Settings.');
    return new ClaudeService(key);
  }
  if (config.provider === 'gemini') {
    // Gemini runs through the in-app tool-using agent (needs the viewer bridge),
    // not the legacy two-call pipeline. Reaching here means the bridge was
    // missing — surface it clearly instead of silently using Ollama.
    throw new Error('Gemini requires the in-app agent and is not available in this context.');
  }
  const baseUrl = config.ollamaUrl || 'http://localhost:11434';
  const textModel = config.ollamaTextModel || DEFAULT_TEXT_MODEL;
  const visionModel = config.ollamaVisionModel || DEFAULT_VISION_MODEL;
  return new OllamaService(textModel, visionModel, baseUrl);
}

// --- Ollama Management API ---

export interface OllamaModelInfo {
  name: string;
  size: number;
  modified_at: string;
}

export async function fetchOllamaModels(baseUrl = 'http://localhost:11434'): Promise<OllamaModelInfo[]> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models ?? []).map((m: { name: string; size: number; modified_at: string }) => ({
      name: m.name,
      size: m.size,
      modified_at: m.modified_at,
    }));
  } catch {
    return [];
  }
}

export async function pingOllama(baseUrl = 'http://localhost:11434'): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function pullOllamaModel(
  modelName: string,
  onProgress: (status: string, percent: number | null) => void,
  baseUrl = 'http://localhost:11434',
): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName, stream: true }),
    });

    if (!res.ok || !res.body) {
      onProgress('Failed to start download', null);
      return false;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.error) {
            onProgress(`Error: ${data.error}`, null);
            return false;
          }
          const percent = data.total ? Math.round((data.completed / data.total) * 100) : null;
          onProgress(data.status ?? 'Downloading...', percent);
        } catch { /* skip malformed lines */ }
      }
    }

    onProgress('Complete', 100);
    return true;
  } catch {
    onProgress('Connection failed', null);
    return false;
  }
}
