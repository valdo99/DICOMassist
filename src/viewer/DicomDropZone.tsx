import { useCallback, useRef, useState } from 'react';
import { Upload, FolderOpen, FlaskConical, Loader2 } from 'lucide-react';
import { processDicomFiles, isDicomFile, type LoadResult } from '../dicom/loadDicomFiles';
import { loadSampleData, type SampleDataProgress } from '../utils/sampleDataLoader';

export type { LoadResult };

interface DicomDropZoneProps {
  onFilesLoaded: (result: LoadResult, sourceFiles: File[]) => void;
}

async function getAllFiles(dataTransfer: DataTransfer): Promise<File[]> {
  const files: File[] = [];
  const entries: FileSystemEntry[] = [];

  for (let i = 0; i < dataTransfer.items.length; i++) {
    const entry = dataTransfer.items[i].webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }

  async function readEntry(entry: FileSystemEntry): Promise<void> {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve) =>
        (entry as FileSystemFileEntry).file(resolve)
      );
      if (isDicomFile(file)) files.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const subEntries: FileSystemEntry[] = [];
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise<FileSystemEntry[]>((resolve) =>
          reader.readEntries(resolve)
        );
        subEntries.push(...batch);
      } while (batch.length > 0);
      for (const sub of subEntries) {
        await readEntry(sub);
      }
    }
  }

  if (entries.length > 0) {
    for (const entry of entries) {
      await readEntry(entry);
    }
  } else {
    for (let i = 0; i < dataTransfer.files.length; i++) {
      const file = dataTransfer.files[i];
      if (isDicomFile(file)) files.push(file);
    }
  }

  return files;
}

export default function DicomDropZone({ onFilesLoaded }: DicomDropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<'reading' | 'sorting'>('reading');
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [sampleProgress, setSampleProgress] = useState<SampleDataProgress | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) {
        setLoading(false);
        return;
      }

      setLoadingPhase('reading');
      setProgress({ loaded: 0, total: files.length });

      const result = await processDicomFiles(files, (p) => {
        setLoadingPhase(p.phase);
        setProgress({ loaded: p.loaded, total: p.total });
      });

      setLoading(false);
      if (result) onFilesLoaded(result, files);
    },
    [onFilesLoaded]
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      setLoading(true);

      const files = await getAllFiles(e.dataTransfer);
      await processFiles(files);
    },
    [processFiles]
  );

  const handleInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = e.target.files;
      if (!fileList || fileList.length === 0) return;

      setLoading(true);

      const files: File[] = [];
      for (let i = 0; i < fileList.length; i++) {
        if (isDicomFile(fileList[i])) files.push(fileList[i]);
      }
      await processFiles(files);

      // Reset input so the same folder can be re-selected
      if (inputRef.current) inputRef.current.value = '';
    },
    [processFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleLoadSample = useCallback(async () => {
    setSampleError(null);
    setSampleProgress({ phase: 'downloading', percent: 0 });
    try {
      const files = await loadSampleData((p) => setSampleProgress(p));
      setLoading(true);
      setSampleProgress(null);
      await processFiles(files);
    } catch (err) {
      setSampleProgress(null);
      setSampleError(err instanceof Error ? err.message : 'Failed to load sample data');
    }
  }, [processFiles]);

  if (loading) {
    const pct = progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : 0;
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <p className="text-neutral-400 text-sm">
          {loadingPhase === 'reading'
            ? `Reading DICOM headers... ${progress.loaded} / ${progress.total}`
            : `Sorting slices...`}
        </p>
        <div className="w-64 h-2 bg-neutral-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-100"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  const sampleBusy = sampleProgress != null;
  const sampleLabel = sampleProgress
    ? sampleProgress.phase === 'downloading'
      ? `Downloading... ${sampleProgress.percent}%`
      : sampleProgress.phase === 'extracting'
        ? `Extracting... ${sampleProgress.percent}%`
        : 'Loading into viewer...'
    : null;

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`flex flex-col items-center justify-center h-full border-2 border-dashed rounded-lg transition-colors ${
        dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-neutral-700 hover:border-neutral-500'
      }`}
    >
      <Upload className="w-12 h-12 text-neutral-500 mb-3" />
      <p className="text-neutral-400 text-lg">Drop DICOM files or folder here</p>
      <p className="text-neutral-600 text-sm mt-1">Supports .dcm files and DICOM directories</p>
      <input
        ref={inputRef}
        type="file"
        // @ts-expect-error webkitdirectory is a non-standard attribute
        webkitdirectory=""
        multiple
        hidden
        onChange={handleInputChange}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={sampleBusy}
        className="mt-3 flex items-center gap-2 px-4 py-2 rounded-md bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 transition-colors text-sm disabled:opacity-50"
      >
        <FolderOpen className="w-4 h-4" />
        Browse Folder
      </button>

      {/* Divider */}
      <div className="flex items-center gap-3 w-48 mt-4 mb-2">
        <div className="flex-1 h-px bg-neutral-700" />
        <span className="text-xs text-neutral-600">or</span>
        <div className="flex-1 h-px bg-neutral-700" />
      </div>

      {/* Sample data button */}
      <button
        type="button"
        onClick={handleLoadSample}
        disabled={sampleBusy}
        className="flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600/20 border border-blue-500/30 text-blue-300 hover:bg-blue-600/30 hover:text-blue-200 transition-colors text-sm disabled:opacity-70"
      >
        {sampleBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <FlaskConical className="w-4 h-4" />}
        {sampleBusy ? sampleLabel : 'Try with Sample Knee MRI'}
      </button>
      <p className="text-neutral-600 text-xs mt-1">Public anonymized dataset &middot; ~32 MB</p>

      {sampleError && (
        <p className="text-red-400 text-xs mt-2">{sampleError}</p>
      )}
    </div>
  );
}
