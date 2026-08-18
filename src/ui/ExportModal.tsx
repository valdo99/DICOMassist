import { useState } from 'react';
import {
  X,
  Download,
  FileText,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Image as ImageIcon,
  ListChecks,
} from 'lucide-react';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage, ProviderConfig, ResolvedCircleAnnotation, SelectionPlan } from '../llm/types';
import { useExportReport } from '../export/useExportReport';
import { DEFAULT_EXPORT_OPTIONS, type ExportOptions } from '../export/types';

interface ExportModalProps {
  onClose: () => void;
  metadata: StudyMetadata | null;
  messages: ChatMessage[];
  findings: ResolvedCircleAnnotation[];
  providerConfig: ProviderConfig;
  plan?: SelectionPlan | null;
}

const PROVIDER_NAMES: Record<ProviderConfig['provider'], string> = {
  claude: 'Claude',
  gemini: 'Gemini',
  ollama: 'Ollama',
};

/**
 * PDF export dialog. Mounted only while open (the parent renders it
 * conditionally), so its generation state resets naturally each time it opens
 * and any in-flight generation is aborted on close via the hook's unmount cleanup.
 */
export default function ExportModal({
  onClose,
  metadata,
  messages,
  findings,
  providerConfig,
  plan,
}: ExportModalProps) {
  const { phase, progress, error, note, run } = useExportReport({
    metadata,
    messages,
    findings,
    providerConfig,
    plan,
  });
  const hasImages = findings.length > 0;
  const [options, setOptions] = useState<ExportOptions>(() => ({
    ...DEFAULT_EXPORT_OPTIONS,
    includeImages: hasImages,
    includeSelectionRationale: !!plan,
  }));

  const running = phase === 'summarizing' || phase === 'rendering-images' || phase === 'building-pdf';
  const toggle = (key: keyof ExportOptions) => setOptions((o) => ({ ...o, [key]: !o[key] }));

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
      onMouseDown={onClose}
    >
      <div
        className="bg-neutral-900 border border-neutral-700 rounded-xl max-w-md w-full shadow-2xl overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-neutral-800">
          <span className="flex items-center gap-2 text-sm font-semibold text-neutral-100">
            <FileText className="w-4 h-4 text-blue-400" />
            Export Report (PDF)
          </span>
          <button
            onClick={onClose}
            className="p-0.5 rounded hover:bg-neutral-800 text-neutral-400 hover:text-neutral-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {/* Summary */}
          <p className="text-xs text-neutral-400 leading-relaxed">
            A report agent reads the full conversation with the analyzer and drafts a clinical
            report — no raw transcript — with the marked findings and annotated slice images.
          </p>
          <div className="text-[11px] text-neutral-500 bg-neutral-950/60 border border-neutral-800 rounded-lg px-3 py-2 space-y-0.5">
            <div className="flex justify-between">
              <span>Study</span>
              <span className="text-neutral-300 truncate ml-3 max-w-[60%] text-right">
                {metadata?.studyDescription || metadata?.modality || 'DICOM study'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Marked regions</span>
              <span className="text-neutral-300">{findings.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Summary by</span>
              <span className="text-neutral-300">{PROVIDER_NAMES[providerConfig.provider]}</span>
            </div>
          </div>

          {/* Options */}
          <div className="space-y-1">
            <OptionRow
              icon={<ImageIcon className="w-4 h-4" />}
              label="Annotated slice images"
              hint={hasImages ? `${findings.length} region${findings.length === 1 ? '' : 's'}` : 'No regions marked'}
              checked={options.includeImages && hasImages}
              disabled={!hasImages || running}
              onChange={() => toggle('includeImages')}
            />
            <OptionRow
              icon={<ListChecks className="w-4 h-4" />}
              label="AI slice-selection rationale"
              hint={plan ? 'Available' : 'Not available'}
              checked={options.includeSelectionRationale && !!plan}
              disabled={!plan || running}
              onChange={() => toggle('includeSelectionRationale')}
            />
          </div>

          {/* Status */}
          {running && (
            <div className="flex items-center gap-2.5 text-xs text-blue-300 bg-blue-950/40 border border-blue-900/50 rounded-lg px-3 py-2.5">
              <Loader2 className="w-4 h-4 animate-spin shrink-0" />
              <span className="flex-1">
                {phase === 'summarizing' && 'Writing the report…'}
                {phase === 'rendering-images' && (
                  <>
                    Rendering annotated images
                    {progress ? ` (${progress.done}/${progress.total})` : '…'}
                  </>
                )}
                {phase === 'building-pdf' && 'Assembling the PDF…'}
              </span>
            </div>
          )}

          {note && phase !== 'error' && (
            <div className="flex items-start gap-2 text-[11px] text-amber-300 bg-amber-950/30 border border-amber-900/40 rounded-lg px-3 py-2">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{note}</span>
            </div>
          )}

          {phase === 'done' && (
            <div className="flex items-center gap-2 text-xs text-green-300 bg-green-950/30 border border-green-900/40 rounded-lg px-3 py-2.5">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>Report downloaded. Check your downloads folder.</span>
            </div>
          )}

          {phase === 'error' && error && (
            <div className="flex items-start gap-2 text-xs text-red-300 bg-red-950/30 border border-red-900/40 rounded-lg px-3 py-2.5">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-3.5 border-t border-neutral-800">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-sm text-neutral-300 hover:bg-neutral-800 transition-colors"
          >
            {phase === 'done' ? 'Close' : 'Cancel'}
          </button>
          <button
            onClick={() => run(options)}
            disabled={running}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-40"
          >
            {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            {phase === 'done' ? 'Export again' : 'Generate PDF'}
          </button>
        </div>
      </div>
    </div>
  );
}

function OptionRow({
  icon,
  label,
  hint,
  checked,
  disabled,
  onChange,
}: {
  icon: React.ReactNode;
  label: string;
  hint: string;
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}) {
  return (
    <button
      onClick={onChange}
      disabled={disabled}
      className={`flex items-center gap-3 w-full px-3 py-2 rounded-lg text-left transition-colors ${
        disabled ? 'opacity-40 cursor-not-allowed' : 'hover:bg-neutral-800'
      }`}
    >
      <span
        className={`flex items-center justify-center w-4 h-4 rounded border transition-colors shrink-0 ${
          checked ? 'bg-blue-600 border-blue-600' : 'border-neutral-600'
        }`}
      >
        {checked && <CheckCircle2 className="w-3.5 h-3.5 text-white" />}
      </span>
      <span className="text-neutral-400 shrink-0">{icon}</span>
      <span className="flex-1 min-w-0">
        <span className="block text-sm text-neutral-200">{label}</span>
      </span>
      <span className="text-[10px] text-neutral-500 shrink-0">{hint}</span>
    </button>
  );
}
