import { useCallback, useEffect, useState } from 'react';
import { Clock, Trash2, HardDrive, Layers, FileStack, Loader2 } from 'lucide-react';
import {
  listStudies,
  deleteStudy,
  clearAllStudies,
  estimateStorage,
  type StoredStudyMeta,
} from '../persistence/studyStore';

interface RecentStudiesProps {
  /** Called when the user picks a study to reopen. */
  onRestore: (meta: StoredStudyMeta) => void;
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, i);
  return `${value >= 100 || i === 0 ? Math.round(value) : value.toFixed(1)} ${units[i]}`;
}

function formatWhen(ts: number): string {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60_000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day}d ago`;
  return new Date(ts).toLocaleDateString();
}

export default function RecentStudies({ onRestore }: RecentStudiesProps) {
  const [studies, setStudies] = useState<StoredStudyMeta[] | null>(null);
  const [usage, setUsage] = useState<{ usage: number; quota: number } | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [list, est] = await Promise.all([
      listStudies().catch(() => [] as StoredStudyMeta[]),
      estimateStorage().catch(() => null),
    ]);
    setStudies(list);
    setUsage(est);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleDelete = useCallback(
    async (id: string) => {
      setBusyId(id);
      try {
        await deleteStudy(id);
      } finally {
        setBusyId(null);
        void refresh();
      }
    },
    [refresh],
  );

  const handleClearAll = useCallback(async () => {
    await clearAllStudies().catch(() => {});
    void refresh();
  }, [refresh]);

  // Nothing stored yet — keep the landing screen uncluttered.
  if (!studies || studies.length === 0) return null;

  return (
    <div className="mb-10">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wide">
          Recent studies
        </h3>
        <button
          onClick={handleClearAll}
          className="text-xs text-zinc-600 hover:text-red-400 transition-colors"
        >
          Clear all
        </button>
      </div>

      <div className="space-y-2">
        {studies.map((s) => (
          <div
            key={s.id}
            className="group flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 hover:border-zinc-600 hover:bg-zinc-900 transition-colors"
          >
            <button
              onClick={() => onRestore(s)}
              disabled={busyId === s.id}
              className="flex-1 min-w-0 flex items-center gap-3 px-4 py-3 text-left disabled:opacity-50"
            >
              <FileStack className="w-5 h-5 text-blue-400 shrink-0" />
              <div className="min-w-0">
                <div className="text-sm text-zinc-200 truncate">
                  {s.label || 'DICOM study'}
                </div>
                <div className="flex items-center gap-3 text-xs text-zinc-500 mt-0.5">
                  {s.modality && s.modality !== 'unknown' && (
                    <span className="uppercase">{s.modality}</span>
                  )}
                  <span className="flex items-center gap-1">
                    <Layers className="w-3 h-3" />
                    {s.seriesCount} series
                  </span>
                  <span>{s.fileCount} slices</span>
                  <span>{formatBytes(s.totalBytes)}</span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatWhen(s.savedAt)}
                  </span>
                </div>
              </div>
            </button>
            <button
              onClick={() => handleDelete(s.id)}
              disabled={busyId === s.id}
              title="Remove from local storage"
              className="px-3 py-3 text-zinc-600 hover:text-red-400 transition-colors shrink-0"
            >
              {busyId === s.id ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Trash2 className="w-4 h-4" />
              )}
            </button>
          </div>
        ))}
      </div>

      {usage && usage.usage > 0 && (
        <div className="flex items-center gap-1.5 text-xs text-zinc-600 mt-3">
          <HardDrive className="w-3 h-3" />
          <span>
            {formatBytes(usage.usage)} used locally
            {usage.quota > 0 && ` of ~${formatBytes(usage.quota)} available`}
          </span>
        </div>
      )}
    </div>
  );
}
