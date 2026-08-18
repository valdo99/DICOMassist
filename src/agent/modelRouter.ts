import type { StudyMetadata } from '../dicom/types';

/** The tool-using agent works with either of these API-backed providers. */
export type AgentProvider = 'claude' | 'gemini';

/**
 * Model tiers the harness can pick from, cheapest → most capable.
 * The harness (not the model) decides, from signals it can measure: how hard
 * the clinical question is, how complex the study is, and what kind of turn
 * this is. Keeps routine work cheap and reserves the top model for hard cases.
 */
export type ModelTier = 'light' | 'standard' | 'deep';

export const TIER_MODELS: Record<AgentProvider, Record<ModelTier, string>> = {
  claude: {
    light: 'claude-sonnet-4-6',
    standard: 'claude-opus-4-8',
    deep: 'claude-opus-5',
  },
  gemini: {
    light: 'gemini-3.5-flash',
    standard: 'gemini-3.7-flash',
    deep: 'gemini-3.1-pro-preview',
  },
};

/** A model the user can pin in Settings (per provider), plus the "Auto" option. */
export interface SelectableModel {
  id: string;
  label: string;
  desc: string;
}

export const SELECTABLE_MODELS: Record<AgentProvider, SelectableModel[]> = {
  claude: [
    { id: 'claude-sonnet-4-6', label: 'Sonnet 4.6', desc: 'Fast, lighter reasoning' },
    { id: 'claude-opus-4-8', label: 'Opus 4.8', desc: 'Strong all-rounder' },
    { id: 'claude-opus-5', label: 'Opus 5', desc: 'Deepest reasoning' },
  ],
  // Current Gemini lineup (Aug 2026). All are multimodal + support tool calling.
  // 2.5-pro / 2.5-flash and 3-pro-preview are retired/deprecated, so they're not
  // offered here — see https://ai.google.dev/gemini-api/docs/models.
  gemini: [
    { id: 'gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro', desc: 'Deepest reasoning (preview)' },
    { id: 'gemini-3.7-flash', label: 'Gemini 3.7 Flash', desc: 'Most capable Flash · agentic' },
    { id: 'gemini-3.6-flash', label: 'Gemini 3.6 Flash', desc: 'Stable, token-efficient' },
    { id: 'gemini-3.5-flash', label: 'Gemini 3.5 Flash', desc: 'Fast, capable' },
    { id: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash-Lite', desc: 'Cheapest, low latency' },
  ],
};

export interface ModelChoice {
  modelId: string;
  tier: ModelTier;
  score: number;
  reason: string;
}

/** Resolve a pinned model id back to its tier (for the trace); default 'standard'. */
function tierForModel(provider: AgentProvider, modelId: string): ModelTier {
  const entry = (Object.entries(TIER_MODELS[provider]) as Array<[ModelTier, string]>)
    .find(([, id]) => id === modelId);
  return entry?.[0] ?? 'standard';
}

/** Questions that need real diagnostic reasoning. */
const HARD_PATTERNS: Array<[RegExp, number, string]> = [
  [/\b(differential|ddx|diagnosi differenziale|diagnosis)\b/i, 25, 'differential diagnosis'],
  [/\b(stage|staging|grade|grading|classif)/i, 20, 'staging/grading'],
  [/\b(compare|comparison|confront|versus|vs\.?|contralateral|bilateral)\b/i, 20, 'comparison'],
  [/\b(subtle|equivocal|uncertain|rule out|exclude|escludere|occult)\b/i, 20, 'subtle/exclusion'],
  [/\b(why|explain|reason|cause|etiolog|eziolog|pathophys)/i, 15, 'causal reasoning'],
  [/\b(compression|impingement|invasion|infiltrat|metasta|malignan|tumor|tumour|mass|lesion)\b/i, 15, 'possible malignancy/mass'],
  [/\b(neuralgia|neurovascular|nerve|plexus|cranial)\b/i, 12, 'neuro-anatomical detail'],
  [/\b(protocol|multi-?series|all series|whole study|full review|comprehensive)\b/i, 12, 'broad review'],
];

/** Questions that are navigational or trivially scoped. */
const LIGHT_PATTERNS: Array<[RegExp, number, string]> = [
  [/\b(show|go to|navigate|scroll|jump|vai a|mostra)\b/i, -20, 'navigation request'],
  [/\b(window|brighter|darker|contrast|zoom|luminos)/i, -20, 'display adjustment'],
  [/\b(which slice|what slice|how many|count|list)\b/i, -12, 'lookup'],
  [/\b(thanks|thank you|grazie|ok|yes|no)\b/i, -10, 'conversational'],
];

/**
 * Pick a model for this turn. Deterministic and explainable — the reason is
 * surfaced in the agent trace so the choice is visible to the user.
 */
export function chooseModel(params: {
  /** Which provider's model catalog to route within. */
  provider: AgentProvider;
  question: string;
  metadata: StudyMetadata;
  /** True for the first analysis of a study; follow-ups are usually lighter. */
  isNewAnalysis: boolean;
  /** Survey mode sweeps many structures — treat as harder. */
  surveyMode?: boolean;
  /** '', undefined, or 'auto' → auto-route by tier; any other value → a pinned model id. */
  override?: string;
}): ModelChoice {
  const { provider, question, metadata, isNewAnalysis, surveyMode, override } = params;
  const tiers = TIER_MODELS[provider];

  if (override && override !== 'auto') {
    return {
      modelId: override,
      tier: tierForModel(provider, override),
      score: -1,
      reason: 'model pinned in settings',
    };
  }

  const reasons: string[] = [];
  // Baseline: a fresh image analysis is inherently heavier than a text follow-up.
  let score = isNewAnalysis ? 45 : 20;
  reasons.push(isNewAnalysis ? 'new image analysis' : 'follow-up');

  for (const [re, weight, label] of HARD_PATTERNS) {
    if (re.test(question)) {
      score += weight;
      reasons.push(label);
    }
  }
  for (const [re, weight, label] of LIGHT_PATTERNS) {
    if (re.test(question)) {
      score += weight;
      reasons.push(label);
    }
  }

  if (surveyMode) {
    score += 20;
    reasons.push('survey mode');
  }

  // Study complexity: more series / slices / planes = more to reason over.
  const seriesCount = metadata.series.length;
  const totalSlices = metadata.series.reduce((sum, s) => sum + s.slices.length, 0);
  const planes = new Set(metadata.series.map((s) => s.anatomicalPlane));
  if (seriesCount >= 5) { score += 10; reasons.push(`${seriesCount} series`); }
  if (totalSlices >= 300) { score += 10; reasons.push(`${totalSlices} slices`); }
  if (planes.size >= 3) { score += 5; reasons.push('multi-plane'); }

  // Long, detailed clinical context usually signals a real case, not a poke.
  if (question.length > 180) { score += 10; reasons.push('detailed clinical context'); }

  const tier: ModelTier = score >= 75 ? 'deep' : score >= 40 ? 'standard' : 'light';

  return {
    modelId: tiers[tier],
    tier,
    score,
    reason: reasons.slice(0, 4).join(', '),
  };
}
