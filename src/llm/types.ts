import type { StudyMetadata } from '../dicom/types';

export interface SeriesSelection {
  seriesNumber: string;
  role: 'primary' | 'supplementary';
  rationale: string;
  sliceRange: [number, number];
  samplingStrategy: 'every_nth' | 'uniform' | 'all';
  samplingParam?: number;
  windowWidth: number;
  windowCenter: number;
}

export interface SelectionPlan {
  reasoning: string;
  selections: SeriesSelection[];
  totalImages: number;
  // Legacy shortcuts from selections[0] — used by App.tsx viewport logic
  targetSeries: string;
  sliceRange: [number, number];
  windowCenter: number;
  windowWidth: number;
  samplingStrategy: 'every_nth' | 'uniform' | 'all';
  samplingParam?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

/**
 * A circle the LLM wants drawn on an image to mark a finding — coordinates
 * are normalized to the image (0..1) so they're resolution-independent.
 * Returned by Call 2 (multimodal analysis).
 */
export interface SliceCircle {
  image: number;   // 1-based image number in the manifest sent to the model
  cx: number;      // center x, fraction of width  (0 = left, 1 = right)
  cy: number;      // center y, fraction of height (0 = top,  1 = bottom)
  radius: number;  // radius, fraction of image width
  label: string;   // short finding name, e.g. "ACL tear"
}

/** Result of Call 2: the prose analysis plus any circle annotations. */
export interface AnalysisResult {
  text: string;
  annotations: SliceCircle[];
}

/**
 * A SliceCircle resolved to a concrete slice (imageId), ready to draw on the
 * viewport. The image number the model returned has been mapped to the actual
 * DICOM slice it was rendered from.
 */
export interface ResolvedCircleAnnotation {
  uid: string;            // stable Cornerstone annotationUID
  imageId: string;        // Cornerstone imageId of the slice to draw on
  seriesNumber: string;
  instanceNumber: number;
  label: string;
  cx: number;             // normalized center x (0..1)
  cy: number;             // normalized center y (0..1)
  radius: number;         // normalized radius, fraction of image width
}

export type ProviderType = 'claude' | 'gemini' | 'ollama';

export interface ProviderConfig {
  provider: ProviderType;
  apiKey?: string;           // Claude API key
  claudeModel?: string;      // '' | 'auto' = auto-route by tier; else a pinned Claude model id
  geminiApiKey?: string;     // Gemini (Google Generative AI) API key
  geminiModel?: string;      // '' | 'auto' = auto-route by tier; else a pinned Gemini model id
  ollamaTextModel?: string;  // Ollama model for Call 1 (text-only planning)
  ollamaVisionModel?: string; // Ollama model for Call 2 (multimodal analysis)
  ollamaUrl?: string;        // Ollama base URL override
}

export interface ViewportContext {
  currentInstanceNumber: number;
  currentZPosition: number;
  seriesNumber: string;
  totalSlicesInSeries: number;
}

export interface LLMService {
  getSelectionPlan(metadata: StudyMetadata, clinicalHint: string, viewportContext?: ViewportContext): Promise<SelectionPlan>;
  analyzeSlices(
    images: Blob[],
    metadata: StudyMetadata,
    clinicalHint: string,
    plan: SelectionPlan,
    sliceLabels: string[],
    surveyMode?: boolean,
  ): Promise<AnalysisResult>;
  sendFollowUp(
    conversationHistory: ChatMessage[],
    metadata: StudyMetadata,
    images?: Blob[],
    sliceLabels?: string[],
  ): Promise<AnalysisResult>;
}
