import { useCallback, useEffect, useRef, useState } from 'react';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage, ProviderConfig, ResolvedCircleAnnotation, SelectionPlan } from '../llm/types';
import { generateExportReport } from './reportAgent';
import { renderAnnotatedSliceImages, type AnnotatedSliceImage } from './sliceImage';
import { buildReportPdf } from './pdfBuilder';
import type { ExportOptions, ExportPhase } from './types';
import { logger } from '../utils/logger';

interface UseExportReportParams {
  metadata: StudyMetadata | null;
  messages: ChatMessage[];
  findings: ResolvedCircleAnnotation[];
  providerConfig: ProviderConfig;
  plan?: SelectionPlan | null;
}

export interface UseExportReport {
  phase: ExportPhase;
  /** Image-render progress, present only during 'rendering-images'. */
  progress: { done: number; total: number } | null;
  error: string | null;
  /** Set when an AI summary couldn't be produced and a fallback was used. */
  note: string | null;
  run: (options: ExportOptions) => Promise<void>;
  reset: () => void;
}

const PROVIDER_LABELS: Record<ProviderConfig['provider'], string> = {
  claude: 'Claude',
  gemini: 'Gemini',
  ollama: 'Ollama',
};

function fileStamp(d: Date): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the browser a tick to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/**
 * Orchestrates the PDF export: author a report (LLM or fallback) → render the
 * annotated slice images → assemble the PDF → trigger the download. Surfaces a
 * phase + progress so the dialog can show what's happening, and never leaves the
 * app in a broken state on error.
 */
export function useExportReport(params: UseExportReportParams): UseExportReport {
  const { metadata, messages, findings, providerConfig, plan } = params;
  const [phase, setPhase] = useState<ExportPhase>('idle');
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPhase('idle');
    setProgress(null);
    setError(null);
    setNote(null);
  }, []);

  // Abort any in-flight generation if the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(
    async (options: ExportOptions) => {
      if (!metadata) {
        setError('No study is loaded.');
        setPhase('error');
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      const generatedAt = new Date();
      setError(null);
      setNote(null);

      try {
        // 1) Author the report (or fall back deterministically).
        setPhase('summarizing');
        const result = await generateExportReport({
          providerConfig,
          metadata,
          messages,
          findings,
          plan,
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (result.usedFallback && result.note) setNote(result.note);

        // 2) Render annotated slice images.
        let images = new Map<string, AnnotatedSliceImage>();
        if (options.includeImages && findings.length > 0) {
          setPhase('rendering-images');
          setProgress({ done: 0, total: findings.length });
          images = await renderAnnotatedSliceImages(findings, metadata, (done, total) => {
            if (!controller.signal.aborted) setProgress({ done, total });
          });
          if (controller.signal.aborted) return;
        }

        // 3) Build the PDF.
        setPhase('building-pdf');
        const providerLabel = result.usedFallback
          ? 'Report assembled without an AI model'
          : `${PROVIDER_LABELS[providerConfig.provider]}${result.modelId ? ` · ${result.modelId}` : ''}`;

        const blob = await buildReportPdf({
          report: result.report,
          metadata,
          findings,
          images,
          plan,
          options,
          providerLabel,
          generatedAt,
        });
        if (controller.signal.aborted) return;

        // 4) Download.
        download(blob, `DICOMassist-report-${fileStamp(generatedAt)}.pdf`);
        setPhase('done');
        setProgress(null);
      } catch (err) {
        if (controller.signal.aborted) return;
        logger.warn('[Export] Failed to build PDF', err);
        setError(err instanceof Error ? err.message : 'Failed to generate the PDF.');
        setPhase('error');
      } finally {
        abortRef.current = null;
      }
    },
    [metadata, messages, findings, providerConfig, plan],
  );

  return { phase, progress, error, note, run, reset };
}
