import { useMemo } from 'react';
import { X, Eye, EyeOff, Circle, Crosshair } from 'lucide-react';
import type { ResolvedCircleAnnotation } from '../llm/types';
import type { StudyMetadata } from '../dicom/types';

export interface FindingRef {
  uid: string;
  imageId: string;
  seriesNumber: string;
  instanceNumber: number;
  label: string;
}

interface FindingsPanelProps {
  annotations: ResolvedCircleAnnotation[];
  studyMetadata: StudyMetadata | null;
  focusedUid: string | null;
  showCircles: boolean;
  onToggleShowCircles: () => void;
  onSelect: (finding: FindingRef) => void;
  onClose: () => void;
}

interface SeriesGroup {
  seriesNumber: string;
  description: string;
  items: ResolvedCircleAnnotation[];
}

/**
 * A running list of every AI-marked finding, grouped by series. Each row jumps
 * the viewer to that finding's slice and focuses its circle. Covers findings
 * from both providers — App merges the agent and legacy annotation sets before
 * passing them here.
 */
export default function FindingsPanel({
  annotations,
  studyMetadata,
  focusedUid,
  showCircles,
  onToggleShowCircles,
  onSelect,
  onClose,
}: FindingsPanelProps) {
  const groups = useMemo<SeriesGroup[]>(() => {
    const bySeries = new Map<string, ResolvedCircleAnnotation[]>();
    for (const ann of annotations) {
      const list = bySeries.get(ann.seriesNumber);
      if (list) list.push(ann);
      else bySeries.set(ann.seriesNumber, [ann]);
    }
    return Array.from(bySeries.entries())
      .map(([seriesNumber, items]) => {
        const series = studyMetadata?.series.find((s) => String(s.seriesNumber) === seriesNumber);
        return {
          seriesNumber,
          description: series?.seriesDescription || '(no description)',
          items: [...items].sort((a, b) => a.instanceNumber - b.instanceNumber),
        };
      })
      .sort((a, b) => Number(a.seriesNumber) - Number(b.seriesNumber));
  }, [annotations, studyMetadata]);

  return (
    <div className="w-72 h-full bg-neutral-900 border-l border-neutral-700 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-neutral-700 shrink-0">
        <span className="flex items-center gap-1.5 text-sm font-medium text-neutral-200">
          Findings
          {annotations.length > 0 && (
            <span className="text-[10px] font-semibold text-blue-300 bg-blue-900/60 px-1.5 py-0.5 rounded-full">
              {annotations.length}
            </span>
          )}
        </span>
        <div className="flex items-center gap-1">
          {annotations.length > 0 && (
            <button
              onClick={onToggleShowCircles}
              title={showCircles ? 'Hide all circles on the viewer' : 'Show all circles on the viewer'}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200"
            >
              {showCircles ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              {showCircles ? 'Hide' : 'Show'}
            </button>
          )}
          <button
            onClick={onClose}
            className="p-0.5 rounded hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {annotations.length === 0 ? (
          <div className="text-center text-neutral-500 text-xs px-4 mt-8">
            <p>No findings marked yet.</p>
            <p className="mt-1">Run an analysis (Cmd+K) — the AI marks findings here as it reviews.</p>
          </div>
        ) : (
          <div className="py-1">
            {groups.map((group) => (
              <div key={group.seriesNumber}>
                <div className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wide text-neutral-500 truncate">
                  Series #{group.seriesNumber} · {group.description}
                </div>
                {group.items.map((ann) => {
                  const active = ann.uid === focusedUid;
                  return (
                    <button
                      key={ann.uid}
                      onClick={() =>
                        onSelect({
                          uid: ann.uid,
                          imageId: ann.imageId,
                          seriesNumber: ann.seriesNumber,
                          instanceNumber: ann.instanceNumber,
                          label: ann.label,
                        })
                      }
                      title={`Jump to "${ann.label}" — Series #${ann.seriesNumber}, Slice ${ann.instanceNumber}`}
                      className={`group flex items-center gap-2.5 w-full px-3 py-2 text-left transition-colors ${
                        active ? 'bg-blue-950/60' : 'hover:bg-neutral-800'
                      }`}
                    >
                      <Circle
                        className={`w-3.5 h-3.5 shrink-0 ${active ? 'text-blue-300' : 'text-blue-500/80'}`}
                        strokeWidth={2.5}
                      />
                      <span className="flex-1 min-w-0">
                        <span className={`block text-xs font-medium truncate ${active ? 'text-blue-200' : 'text-neutral-200'}`}>
                          {ann.label}
                        </span>
                        <span className="block text-[10px] text-neutral-500">Slice {ann.instanceNumber}</span>
                      </span>
                      <Crosshair
                        className={`w-3.5 h-3.5 shrink-0 transition-opacity ${
                          active ? 'opacity-100 text-blue-300' : 'opacity-0 group-hover:opacity-60 text-neutral-400'
                        }`}
                      />
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer disclaimer */}
      <div className="px-3 py-2 border-t border-neutral-800 text-[10px] text-neutral-600 text-center shrink-0">
        AI-suggested regions — approximate guides, not measurements.
      </div>
    </div>
  );
}
