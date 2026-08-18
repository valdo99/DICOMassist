/**
 * Types for the PDF export feature: a structured clinical report the export
 * agent authors from the conversation + marked findings, plus the options and
 * progress states that drive report generation.
 */

/** How confident the AI is in a finding — mirrors the agent's own tiering. */
export type FindingTier = 'definite' | 'probable' | 'possible';

/** One structured finding for the report, tied (where possible) to a slice. */
export interface ExportFinding {
  /** Short finding name, e.g. "SCA vascular loop". */
  title: string;
  /** Series Number the finding sits on, e.g. "3". Optional if not localised. */
  seriesNumber?: string;
  /** Instance number of the slice, if known. */
  instanceNumber?: number;
  /** Confidence tier. Omitted when the AI did not tier it. */
  tier?: FindingTier;
  /** 1–3 sentence explanation of what was seen and why it matters. */
  description: string;
}

/**
 * The report the export agent produces from the analysis conversation. Rendered
 * into a formatted PDF. Every field is prose the model authored (or a
 * deterministic fallback), so the PDF layer never has to call an LLM itself.
 */
export interface ExportReport {
  /** Document title, e.g. "MRI Brain — Trigeminal Neuralgia Evaluation". */
  title: string;
  /** Clinical question / indication (usually derived from the user's prompt). */
  indication: string;
  /** Study & technique summary (modality, sequences/series reviewed). */
  technique: string;
  /** The headline read — the single most important paragraph. */
  impression: string;
  /** Structured findings, most significant first. May be empty. */
  findings: ExportFinding[];
  /** Suggested next steps / correlation. */
  recommendations: string;
  /** What this AI review did and did not cover. */
  limitations: string;
}

/** User-selectable toggles for what the exported PDF contains. */
export interface ExportOptions {
  /** Embed an annotated image for each marked region. */
  includeImages: boolean;
  /** Append the full user/assistant conversation transcript. */
  includeTranscript: boolean;
  /** Include the AI's slice-selection rationale (when available). */
  includeSelectionRationale: boolean;
}

export const DEFAULT_EXPORT_OPTIONS: ExportOptions = {
  includeImages: true,
  includeTranscript: true,
  includeSelectionRationale: true,
};

/** Progress phases surfaced in the export dialog while the PDF is built. */
export type ExportPhase =
  | 'idle'
  | 'summarizing'
  | 'rendering-images'
  | 'building-pdf'
  | 'done'
  | 'error';
