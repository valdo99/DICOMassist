import type { StudyMetadata } from '../dicom/types';

/**
 * Model tiers the harness can pick from, cheapest → most capable.
 * The harness (not the model) decides, from signals it can measure: how hard
 * the clinical question is, how complex the study is, and what kind of turn
 * this is. Keeps routine work cheap and reserves the top model for hard cases.
 */
export type ModelTier = 'light' | 'standard' | 'deep';

export const TIER_MODELS: Record<ModelTier, string> = {
  light: 'claude-sonnet-4-6',
  standard: 'claude-opus-4-8',
  deep: 'claude-opus-5',
};

export interface ModelChoice {
  modelId: string;
  tier: ModelTier;
  score: number;
  reason: string;
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
  question: string;
  metadata: StudyMetadata;
  /** True for the first analysis of a study; follow-ups are usually lighter. */
  isNewAnalysis: boolean;
  /** Survey mode sweeps many structures — treat as harder. */
  surveyMode?: boolean;
  /** Explicit user override; bypasses scoring. */
  override?: ModelTier | 'auto';
}): ModelChoice {
  const { question, metadata, isNewAnalysis, surveyMode, override } = params;

  if (override && override !== 'auto') {
    return {
      modelId: TIER_MODELS[override],
      tier: override,
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
    modelId: TIER_MODELS[tier],
    tier,
    score,
    reason: reasons.slice(0, 4).join(', '),
  };
}
