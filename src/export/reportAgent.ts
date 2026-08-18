import { generateObject } from 'ai';
import { z } from 'zod';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage, ProviderConfig, ResolvedCircleAnnotation, SelectionPlan } from '../llm/types';
import { formatMetadataSummary } from '../llm/PromptBuilder';
import { chooseModel } from '../agent/modelRouter';
import { buildProviderModel } from '../agent/buildModel';
import { logger } from '../utils/logger';
import type { ExportFinding, ExportReport, FindingTier } from './types';

export interface ExportReportResult {
  report: ExportReport;
  /** The model that authored the report, or null if the deterministic fallback ran. */
  modelId: string | null;
  /** True when the AI summary could not be produced and a fallback was assembled. */
  usedFallback: boolean;
  /** Human-readable reason the fallback ran (shown subtly in the dialog). */
  note?: string;
}

const TIERS: FindingTier[] = ['definite', 'probable', 'possible'];

const findingSchema = z.object({
  title: z.string().describe('Short finding name, 1–5 words, e.g. "SCA vascular loop".'),
  seriesNumber: z.string().optional().describe('Series Number the finding is on, e.g. "3".'),
  instanceNumber: z.number().optional().describe('Slice (instance) number, if localised.'),
  tier: z.enum(['definite', 'probable', 'possible']).optional().describe('Confidence tier.'),
  description: z.string().describe('1–3 sentences: what was seen and why it matters.'),
});

const reportSchema = z.object({
  title: z.string().describe('Report title, e.g. "MRI Brain — Trigeminal Neuralgia Evaluation".'),
  indication: z.string().describe('Clinical question / reason for the study.'),
  technique: z.string().describe('Modality and which series/sequences were reviewed.'),
  impression: z.string().describe('The headline read — the single most important paragraph.'),
  findings: z.array(findingSchema).describe('Structured findings, most significant first. Empty if none.'),
  recommendations: z.string().describe('Suggested next steps / clinical correlation.'),
  limitations: z.string().describe('What this AI review did and did not cover.'),
});

const SYSTEM_PROMPT = [
  'You are a medical imaging assistant writing a structured report that summarises an',
  'AI-assisted image review for a portfolio/research tool — NOT for clinical diagnosis.',
  '',
  'Write the report ONLY from the analysis conversation and the marked findings provided.',
  'Do NOT invent findings, measurements, or normal statements for anything not discussed.',
  'If the analysis is thin, keep the report short and say so in the limitations.',
  '',
  'Guidelines:',
  '- Be precise and clinical in tone; prefer the analysis\'s own wording where it is specific.',
  '- Cite the series and slice for each finding where the source gives them.',
  '- Tier findings as definite / probable / possible when the source supports it.',
  '- The impression is the key takeaway; make it a clear standalone paragraph.',
  '- Always include a limitations note stating this is an AI review of selected slices,',
  '  not a full diagnostic read, and not for clinical use.',
].join('\n');

function transcript(messages: ChatMessage[]): string {
  if (messages.length === 0) return '(no conversation)';
  return messages
    .map((m) => `${m.role === 'user' ? 'CLINICIAN' : 'AI ANALYSIS'}:\n${m.content}`)
    .join('\n\n');
}

function findingsList(findings: ResolvedCircleAnnotation[]): string {
  if (findings.length === 0) return '(no regions were marked on the images)';
  return findings
    .map((f) => `- "${f.label}" — Series #${f.seriesNumber}, Slice ${f.instanceNumber}`)
    .join('\n');
}

function selectionRationale(plan: SelectionPlan | null | undefined): string {
  if (!plan) return '';
  const perSeries = plan.selections
    .map((s) => `  - Series #${s.seriesNumber} (${s.role}): ${s.rationale}`)
    .join('\n');
  return ['\n\n## AI SLICE-SELECTION RATIONALE', plan.reasoning, perSeries]
    .filter(Boolean)
    .join('\n');
}

function buildUserPrompt(
  metadata: StudyMetadata,
  messages: ChatMessage[],
  findings: ResolvedCircleAnnotation[],
  plan: SelectionPlan | null | undefined,
): string {
  const clinicalQuestion = messages.find((m) => m.role === 'user')?.content ?? '(not specified)';
  return [
    '## STUDY METADATA',
    formatMetadataSummary(metadata),
    '',
    '## CLINICAL QUESTION',
    clinicalQuestion,
    '',
    '## ANALYSIS CONVERSATION',
    transcript(messages),
    '',
    '## MARKED FINDINGS (circles the AI placed)',
    findingsList(findings),
    selectionRationale(plan),
    '',
    'Now produce the structured report.',
  ].join('\n');
}

/** Resolve the API key / pinned model for an AI-SDK provider from config + env. */
function resolveAuth(providerConfig: ProviderConfig): {
  provider: 'claude' | 'gemini';
  apiKey: string | undefined;
  override: string | undefined;
} {
  if (providerConfig.provider === 'gemini') {
    return {
      provider: 'gemini',
      apiKey: providerConfig.geminiApiKey || import.meta.env.VITE_GOOGLE_GENERATIVE_AI_API_KEY,
      override: providerConfig.geminiModel,
    };
  }
  return {
    provider: 'claude',
    apiKey: providerConfig.apiKey || import.meta.env.VITE_ANTHROPIC_API_KEY,
    override: providerConfig.claudeModel,
  };
}

/**
 * Author a structured export report from the analysis conversation. Uses the
 * configured provider (Claude/Gemini via the AI SDK, Ollama via its HTTP API).
 * Never throws: on any failure it returns a deterministic fallback report so the
 * PDF export always succeeds — the caller can surface `note`/`usedFallback`.
 */
export async function generateExportReport(params: {
  providerConfig: ProviderConfig;
  metadata: StudyMetadata;
  messages: ChatMessage[];
  findings: ResolvedCircleAnnotation[];
  plan?: SelectionPlan | null;
  signal?: AbortSignal;
}): Promise<ExportReportResult> {
  const { providerConfig, metadata, messages, findings, plan, signal } = params;
  const userPrompt = buildUserPrompt(metadata, messages, findings, plan);

  try {
    if (providerConfig.provider === 'ollama') {
      const { report, modelId } = await generateWithOllama(providerConfig, userPrompt, signal);
      return { report: coerceReport(report, metadata, messages, findings), modelId, usedFallback: false };
    }

    const { provider, apiKey, override } = resolveAuth(providerConfig);
    if (!apiKey) {
      return fallback(
        metadata, messages, findings,
        `Add a ${provider === 'gemini' ? 'Gemini' : 'Claude'} API key in Settings for an AI-authored summary.`,
      );
    }

    const question = messages.find((m) => m.role === 'user')?.content ?? '';
    const choice = chooseModel({ provider, question, metadata, isNewAnalysis: false, override });
    const model = buildProviderModel(provider, apiKey, choice.modelId);

    logger.log(`[Export] Authoring report with ${choice.modelId}`);
    const { object } = await generateObject({
      model,
      schema: reportSchema,
      system: SYSTEM_PROMPT,
      prompt: userPrompt,
      abortSignal: signal,
    });

    return {
      report: coerceReport(object, metadata, messages, findings),
      modelId: choice.modelId,
      usedFallback: false,
    };
  } catch (err) {
    if (signal?.aborted) throw err;
    logger.warn('[Export] Report generation failed — using fallback', err);
    const reason = err instanceof Error ? err.message : 'The report model was unavailable.';
    return fallback(metadata, messages, findings, reason);
  }
}

/** Ollama path: ask the text model for a JSON report and parse it leniently. */
async function generateWithOllama(
  config: ProviderConfig,
  userPrompt: string,
  signal?: AbortSignal,
): Promise<{ report: Partial<ExportReport>; modelId: string }> {
  const baseUrl = config.ollamaUrl || 'http://localhost:11434';
  const model = config.ollamaTextModel || 'alibayram/medgemma:4b';
  const schemaHint =
    'Respond with ONLY a JSON object with these keys: title, indication, technique, ' +
    'impression, recommendations, limitations (all strings), and findings (an array of ' +
    '{ title, description, seriesNumber?, instanceNumber?, tier? } where tier is one of ' +
    '"definite" | "probable" | "possible").';

  const res = await fetch(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      format: 'json',
      stream: false,
      options: { temperature: 0 },
      messages: [
        { role: 'system', content: `${SYSTEM_PROMPT}\n\n${schemaHint}` },
        { role: 'user', content: userPrompt },
      ],
    }),
    // Always bound the request with a 120s timeout even when a caller signal is
    // present, so a stalled Ollama model still aborts → the caller's catch falls
    // back to the deterministic report instead of hanging in 'summarizing'.
    signal: signal
      ? AbortSignal.any([signal, AbortSignal.timeout(120_000)])
      : AbortSignal.timeout(120_000),
  });
  if (!res.ok) throw new Error(`Ollama error (${res.status})`);
  const data = await res.json();
  const raw = data.message?.content ?? '';
  const parsed = parseReportJson(raw);
  if (!parsed) throw new Error('Ollama returned no parseable report JSON.');
  return { report: parsed, modelId: model };
}

function parseReportJson(raw: string): Partial<ExportReport> | null {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end <= start) return null;
  try {
    return JSON.parse(raw.slice(start, end + 1)) as Partial<ExportReport>;
  } catch {
    return null;
  }
}

// --- Coercion + fallback -----------------------------------------------------

function coerceFinding(raw: unknown): ExportFinding | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  const title = typeof r.title === 'string' ? r.title.trim() : '';
  const description = typeof r.description === 'string' ? r.description.trim() : '';
  if (!title && !description) return null;
  const tier = typeof r.tier === 'string' && TIERS.includes(r.tier as FindingTier)
    ? (r.tier as FindingTier)
    : undefined;
  const instanceNumber = Number.isFinite(Number(r.instanceNumber)) && r.instanceNumber != null
    ? Number(r.instanceNumber)
    : undefined;
  return {
    title: title || 'Finding',
    description: description || title,
    seriesNumber: r.seriesNumber != null ? String(r.seriesNumber) : undefined,
    instanceNumber,
    tier,
  };
}

/**
 * Fill any missing/invalid fields of a (possibly partial) LLM report from the
 * source data, so the PDF layer always receives a complete ExportReport.
 */
function coerceReport(
  raw: Partial<ExportReport> | Record<string, unknown>,
  metadata: StudyMetadata,
  messages: ChatMessage[],
  findings: ResolvedCircleAnnotation[],
): ExportReport {
  const base = baseReport(metadata, messages, findings);
  const r = (raw ?? {}) as Record<string, unknown>;
  const str = (v: unknown, dflt: string) => (typeof v === 'string' && v.trim() ? v.trim() : dflt);

  const rawFindings = Array.isArray(r.findings) ? r.findings : [];
  const coercedFindings = rawFindings
    .map(coerceFinding)
    .filter((f): f is ExportFinding => f !== null);

  return {
    title: str(r.title, base.title),
    indication: str(r.indication, base.indication),
    technique: str(r.technique, base.technique),
    impression: str(r.impression, base.impression),
    findings: coercedFindings.length > 0 ? coercedFindings : base.findings,
    recommendations: str(r.recommendations, base.recommendations),
    limitations: str(r.limitations, base.limitations),
  };
}

const LIMITATIONS_NOTE =
  'This report was generated by an AI assistant reviewing a selected subset of slices, ' +
  'not a full diagnostic read of every image. It is a research/educational tool and is ' +
  'NOT for clinical use. All findings must be confirmed by a qualified radiologist.';

/** Deterministic report assembled purely from the source data (no LLM). */
function baseReport(
  metadata: StudyMetadata,
  messages: ChatMessage[],
  findings: ResolvedCircleAnnotation[],
): ExportReport {
  const indication = messages.find((m) => m.role === 'user')?.content?.trim() || 'Not specified.';
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')?.content?.trim();

  const study = metadata.studyDescription?.trim();
  const modality = metadata.modality && metadata.modality !== 'unknown' ? metadata.modality : '';
  const title = [modality, study].filter(Boolean).join(' — ') || 'AI-Assisted Imaging Report';

  const planes = Array.from(new Set(metadata.series.map((s) => s.anatomicalPlane))).join(', ');
  const technique =
    `${modality || 'Imaging'} study with ${metadata.series.length} series` +
    (planes ? ` (${planes}).` : '.') +
    ` A subset of slices was reviewed by the AI assistant.`;

  const exportFindings: ExportFinding[] = findings.map((f) => ({
    title: f.label,
    seriesNumber: f.seriesNumber,
    instanceNumber: f.instanceNumber,
    description: `Region marked by the AI on Series #${f.seriesNumber}, Slice ${f.instanceNumber}.`,
  }));

  return {
    title,
    indication,
    technique,
    impression:
      lastAssistant ||
      (findings.length > 0
        ? `The AI marked ${findings.length} region${findings.length === 1 ? '' : 's'} of interest; see findings below.`
        : 'No AI analysis text is available for this study.'),
    findings: exportFindings,
    recommendations:
      'Correlate with the clinical presentation and any prior imaging. Review by a qualified radiologist is recommended.',
    limitations: LIMITATIONS_NOTE,
  };
}

function fallback(
  metadata: StudyMetadata,
  messages: ChatMessage[],
  findings: ResolvedCircleAnnotation[],
  note: string,
): ExportReportResult {
  return {
    report: baseReport(metadata, messages, findings),
    modelId: null,
    usedFallback: true,
    note,
  };
}
