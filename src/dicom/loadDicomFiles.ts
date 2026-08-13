import cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';
import dicomParser from 'dicom-parser';
import type { AnatomicalPlane } from './orientationUtils';
import { extractFileMetadata, buildStudyMetadata, type RawFileRecord } from './MetadataExtractor';
import type { StudyMetadata } from './types';

export interface LoadResult {
  imageIds: string[];
  primaryAxis: AnatomicalPlane;
  studyMetadata: StudyMetadata;
}

export interface LoadProgress {
  phase: 'reading' | 'sorting';
  loaded: number;
  total: number;
}

const PARSE_BATCH_SIZE = 20;

export function isDicomFile(file: File): boolean {
  const name = file.name.toLowerCase();
  return name.endsWith('.dcm') || !name.includes('.');
}

function hasDicomPreamble(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 132) return false;
  const view = new Uint8Array(buffer, 128, 4);
  return view[0] === 0x44 && view[1] === 0x49 && view[2] === 0x43 && view[3] === 0x4d; // "DICM"
}

const FALLBACK_META: Omit<RawFileRecord, 'imageId'> = {
  instanceNumber: 0,
  zPosition: 0,
  imagePositionPatient: [0, 0, 0],
  imageOrientationPatient: [1, 0, 0, 0, 1, 0],
  seriesInstanceUID: 'unknown',
  seriesNumber: 0,
  seriesDescription: '',
  modality: 'unknown',
  studyDescription: '',
};

/**
 * Parse DICOM headers, register each file with Cornerstone's fileManager, and
 * build the structured study metadata. Shared by the drop zone (fresh loads)
 * and the local-persistence restore path so both produce identical results.
 *
 * Returns null when no file in the set parses as DICOM.
 */
export async function processDicomFiles(
  files: File[],
  onProgress?: (p: LoadProgress) => void,
): Promise<LoadResult | null> {
  if (files.length === 0) return null;

  onProgress?.({ phase: 'reading', loaded: 0, total: files.length });

  const parsed: { file: File; meta: Omit<RawFileRecord, 'imageId'> }[] = [];

  for (let start = 0; start < files.length; start += PARSE_BATCH_SIZE) {
    const batch = files.slice(start, start + PARSE_BATCH_SIZE);
    const results = await Promise.all(
      batch.map(async (file) => {
        try {
          let dataSet: dicomParser.DataSet;
          try {
            const partial = await file.slice(0, 131072).arrayBuffer();
            if (!file.name.toLowerCase().endsWith('.dcm') && !hasDicomPreamble(partial)) {
              return null; // Skip non-DICOM files without .dcm extension
            }
            dataSet = dicomParser.parseDicom(new Uint8Array(partial), { untilTag: 'x7fe00010' });
          } catch {
            // Partial read failed (e.g., large private tags) — retry with full file
            const full = await file.arrayBuffer();
            if (!file.name.toLowerCase().endsWith('.dcm') && !hasDicomPreamble(full)) {
              return null;
            }
            dataSet = dicomParser.parseDicom(new Uint8Array(full), { untilTag: 'x7fe00010' });
          }

          const meta = extractFileMetadata(dataSet);
          return { file, meta };
        } catch {
          return { file, meta: { ...FALLBACK_META } };
        }
      }),
    );
    for (const r of results) {
      if (r) parsed.push(r);
    }
    onProgress?.({
      phase: 'reading',
      loaded: Math.min(start + PARSE_BATCH_SIZE, files.length),
      total: files.length,
    });
  }

  onProgress?.({ phase: 'sorting', loaded: files.length, total: files.length });

  if (parsed.length === 0) return null;

  // Register files with fileManager and assign imageIds
  const records: RawFileRecord[] = parsed.map((p) => {
    const imageId = cornerstoneDICOMImageLoader.wadouri.fileManager.add(p.file);
    return { ...p.meta, imageId };
  });

  const studyMetadata = buildStudyMetadata(records);

  const primarySeries = studyMetadata.series.find(
    (s) => s.seriesInstanceUID === studyMetadata.primarySeriesUID,
  );
  const imageIds = primarySeries
    ? primarySeries.slices.map((s) => s.imageId)
    : records.map((r) => r.imageId);
  const plane = primarySeries?.anatomicalPlane;
  const primaryAxis: AnatomicalPlane =
    plane === 'axial' || plane === 'coronal' || plane === 'sagittal' ? plane : 'axial';

  return { imageIds, primaryAxis, studyMetadata };
}
