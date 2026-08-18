import { generateObject, generateText } from 'ai';
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
  /** True when the AI report could not be produced and a fallback was assembled. */
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

/**
 * Phase 1 — the report author. Reads the whole clinician ↔ analyzer conversation
 * and drafts a single coherent report. This is where the actual clinical
 * reasoning happens; the transcript itself never reaches the reader.
 */
const DRAFT_SYSTEM_PROMPT = [
  'You are an experienced radiologist writing the FINAL report for an AI-assisted image review.',
  '',
  'You are given: the study metadata, the FULL dialogue between the referring clinician and the',
  'AI image-analysis assistant that reviewed the study (the initial analysis AND every follow-up),',
  'and the list of regions the assistant marked on the images.',
  '',
  'Write a single, coherent clinical report that SYNTHESISES the entire conversation — not just the',
  'last message. Fold follow-up clarifications and corrections into the relevant sections so the',
  'report reads as one considered opinion. Do NOT reproduce or quote the conversation, and do NOT',
  'include a transcript: integrate everything into clean, standalone prose.',
  '',
  'Rules:',
  '- Base the report ONLY on what the conversation and marked findings support. Never invent',
  '  findings, measurements, or normal statements for anything not discussed.',
  '- Cite the series and slice for each finding where the source gives them (e.g. "Series #3, Slice 45").',
  '- Tier findings as definite / probable / possible when the source supports it.',
  '- Write in a professional clinical register; be specific, not verbose.',
  '- The Impression is the key takeaway — a clear, standalone paragraph.',
  '- Always include a Limitations note: this is an AI review of a selected subset of slices, not a',
  '  full diagnostic read, and NOT for clinical use.',
  '',
  'Format the report in markdown with a leading "# <title>" line, then these "## " sections in order:',
  'Indication, Technique, Findings, Impression, Recommendations, Limitations.',
  'Under Findings, use one bullet per finding, leading with the finding name and its series/slice.',
].join('\n');

/**
 * Phase 2 — the structurer. Turns the drafted markdown report into the schema
 * the PDF layout consumes, without changing its meaning.
 */
const STRUCTURE_SYSTEM_PROMPT = [
  'You convert a drafted radiology report into a structured schema for a document generator.',
  'Transfer the content faithfully and completely — do NOT add, drop, or reinterpret findings.',
  'Map each "## " section of the report to the matching field. For findings, extract each discrete',
  'finding with its title, description, series/slice (if stated), and tier (if stated).',
].join('\n');

function formatDialogue(messages: ChatMessage[]): string {
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

/** The material the report author reads: metadata + full dialogue + marked findings. */
function buildDraftPrompt(
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
    '## FULL CONVERSATION (clinician ↔ AI analyzer)',
    formatDialogue(messages),
    '',
    '## MARKED FINDINGS (circles the AI placed)',
    findingsList(findings),
    selectionRationale(plan),
    '',
    'Now write the final report.',
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
 * The report agent. Reads the full clinician ↔ analyzer conversation and drafts a
 * clinical report, then structures it for the PDF.
 *
 * Claude/Gemini run a two-phase agent: draft (generateText) → structure
 * (generateObject). Ollama uses a single JSON call. Never throws (except on
 * abort): on any failure it returns a deterministic fallback so the export
 * always succeeds — the caller can surface `note`/`usedFallback`.
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
  const draftPrompt = buildDraftPrompt(metadata, messages, findings, plan);

  try {
    if (providerConfig.provider === 'ollama') {
      const { report, modelId } = await generateWithOllama(providerConfig, draftPrompt, signal);
      return { report: coerceReport(report, metadata, messages, findings), modelId, usedFallback: false };
    }

    const { provider, apiKey, override } = resolveAuth(providerConfig);
    if (!apiKey) {
      return fallback(
        metadata, messages, findings,
        `Add a ${provider === 'gemini' ? 'Gemini' : 'Claude'} API key in Settings for an AI-authored report.`,
      );
    }

    const question = messages.find((m) => m.role === 'user')?.content ?? '';
    const choice = chooseModel({ provider, question, metadata, isNewAnalysis: false, override });
    const model = buildProviderModel(provider, apiKey, choice.modelId);

    // Phase 1: draft the report from the conversation.
    logger.log(`[Export] Drafting report with ${choice.modelId}`);
    const draft = await generateText({
      model,
      system: DRAFT_SYSTEM_PROMPT,
      prompt: draftPrompt,
      abortSignal: signal,
    });
    const draftText = draft.text?.trim() ?? '';
    if (!draftText) throw new Error('The report model returned an empty draft.');

    // Phase 2: structure the draft for the PDF. If structuring fails, fall back
    // to a deterministic section parse of the draft so the prose is not lost.
    let structured: Partial<ExportReport>;
    try {
      const { object } = await generateObject({
        model,
        schema: reportSchema,
        system: STRUCTURE_SYSTEM_PROMPT,
        prompt: draftText,
        abortSignal: signal,
      });
      structured = object;
    } catch (structErr) {
      if (signal?.aborted) throw structErr;
      logger.warn('[Export] Structuring failed — parsing the draft directly', structErr);
      structured = parseDraftSections(draftText);
    }

    return {
      report: coerceReport(structured, metadata, messages, findings),
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
  draftPrompt: string,
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
        { role: 'system', content: `${DRAFT_SYSTEM_PROMPT}\n\n${schemaHint}` },
        { role: 'user', content: draftPrompt },
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

/**
 * Deterministic fallback structurer: split a drafted markdown report into its
 * sections by "## " headings. Used only when phase-2 structuring fails, so the
 * drafted prose still reaches the PDF instead of being discarded.
 */
function parseDraftSections(draft: string): Partial<ExportReport> {
  const out: Partial<ExportReport> = {};

  const titleMatch = draft.match(/^#\s+(.+)$/m);
  if (titleMatch) out.title = titleMatch[1].trim();

  const marks: Array<{ name: string; start: number; contentStart: number }> = [];
  const sectionRe = /^##\s+(.+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = sectionRe.exec(draft)) !== null) {
    marks.push({ name: m[1].trim().toLowerCase(), start: m.index, contentStart: m.index + m[0].length });
  }
  const bodyOf = (keyword: string): string | undefined => {
    const i = marks.findIndex((k) => k.name.includes(keyword));
    if (i === -1) return undefined;
    const end = i + 1 < marks.length ? marks[i + 1].start : draft.length;
    return draft.slice(marks[i].contentStart, end).trim() || undefined;
  };

  out.indication = bodyOf('indication');
  out.technique = bodyOf('technique') ?? bodyOf('protocol');
  out.impression = bodyOf('impression');
  out.recommendations = bodyOf('recommend');
  out.limitations = bodyOf('limitation');

  const findingsBody = bodyOf('finding');
  if (findingsBody) {
    // Only real list items ("- x" / "1. x") become findings — intro sentences and
    // negative/normal prose (no marker) are left out rather than fabricated into
    // findings. The marker requires trailing whitespace so a decimal measurement
    // ("3.2 cm …") is never mistaken for an ordered-list number.
    const markerRe = /^\s*(?:[-•*]\s+|\d+\.\s+)/;
    out.findings = findingsBody
      .split('\n')
      .filter((l) => markerRe.test(l))
      .map((l) => l.replace(markerRe, '').trim())
      .filter(Boolean)
      .map((line) => ({
        // Title = leading phrase; split only on ':' or a period FOLLOWED by
        // whitespace, so "1.5 cm spiculated nodule" keeps its measurement.
        title: (line.split(/:|\.\s/)[0] || 'Finding').slice(0, 60).trim() || 'Finding',
        description: line,
      }));
  }
  return out;
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
 * Fill any missing/invalid fields of a (possibly partial) report from the source
 * data, so the PDF layer always receives a complete ExportReport.
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

  // Distinguish "the report gave a findings array" (even empty = a normal study)
  // from "the field is absent" (parse/JSON gap). Only the latter falls back to
  // circle-derived stubs; an explicit empty array is honored. Either way, marked
  // regions still appear in the PDF's "Other marked regions" gallery.
  const hasFindingsKey = Array.isArray(r.findings);
  const coercedFindings = (hasFindingsKey ? (r.findings as unknown[]) : [])
    .map(coerceFinding)
    .filter((f): f is ExportFinding => f !== null);

  return {
    title: str(r.title, base.title),
    indication: str(r.indication, base.indication),
    technique: str(r.technique, base.technique),
    impression: str(r.impression, base.impression),
    findings: hasFindingsKey ? coercedFindings : base.findings,
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
