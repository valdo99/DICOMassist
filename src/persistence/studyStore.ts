/**
 * Local persistence for loaded DICOM studies.
 *
 * DICOM datasets are far too large for localStorage (300–500 MB is common), so
 * the raw file bytes are stored in IndexedDB. On refresh we re-run the normal
 * load pipeline (`processDicomFiles`) over the restored bytes, which produces
 * fresh Cornerstone imageIds and rebuilds metadata deterministically — nothing
 * session-specific needs to be persisted.
 *
 * Schema (DB "dicomassist"):
 *   - studies: { id }              — one small record per stored study (the
 *                                    "commit marker", written last on save)
 *   - files:   { key: studyId::i } — one record per DICOM file, holding its Blob,
 *                                    indexed by `studyId`
 *
 * The "last opened" pointer lives in localStorage (tiny, synchronous) so the app
 * can decide whether to auto-restore before touching IndexedDB.
 */

const DB_NAME = 'dicomassist';
const DB_VERSION = 1;
const STUDIES_STORE = 'studies';
const FILES_STORE = 'files';
const BY_STUDY_INDEX = 'by-study';
const LAST_OPENED_KEY = 'dicomassist-last-study';

/** Number of file blobs written per IndexedDB transaction (progress + memory). */
const WRITE_BATCH_SIZE = 25;

/** Summary of a persisted study — small enough to list cheaply. */
export interface StoredStudyMeta {
  id: string;
  label: string;
  /** Content fingerprint used to avoid storing the same folder twice. */
  signature: string;
  fileCount: number;
  totalBytes: number;
  modality: string;
  studyDescription: string;
  seriesCount: number;
  /** ms epoch of the last save/open; drives recency ordering. */
  savedAt: number;
}

/** Info the caller supplies alongside the raw files when saving. */
export interface SaveStudyInfo {
  label: string;
  modality: string;
  studyDescription: string;
  seriesCount: number;
  /**
   * Series Instance UIDs of the study. Used to build a collision-free content
   * signature — DICOM UIDs are globally unique, so two genuinely different
   * studies can never share one even when description/file-count/byte-size match.
   */
  seriesUIDs: string[];
}

interface StoredFile {
  key: string; // `${studyId}::${index}`
  studyId: string;
  index: number;
  name: string;
  blob: Blob;
}

export type ProgressFn = (done: number, total: number) => void;

export const isPersistenceSupported = (): boolean =>
  typeof indexedDB !== 'undefined';

const now = () => Date.now();

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STUDIES_STORE)) {
        db.createObjectStore(STUDIES_STORE, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(FILES_STORE)) {
        const files = db.createObjectStore(FILES_STORE, { keyPath: 'key' });
        files.createIndex(BY_STUDY_INDEX, 'studyId', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function runTx(
  db: IDBDatabase,
  storeName: string,
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, mode);
    fn(tx.objectStore(storeName));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

// --- Last-opened pointer (localStorage) --------------------------------------

export function getLastOpenedId(): string | null {
  try {
    return localStorage.getItem(LAST_OPENED_KEY);
  } catch {
    return null;
  }
}

export function setLastOpenedId(id: string | null): void {
  try {
    if (id) localStorage.setItem(LAST_OPENED_KEY, id);
    else localStorage.removeItem(LAST_OPENED_KEY);
  } catch {
    /* ignore */
  }
}

// --- Reads -------------------------------------------------------------------

export async function listStudies(): Promise<StoredStudyMeta[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STUDIES_STORE, 'readonly');
    const req = tx.objectStore(STUDIES_STORE).getAll();
    req.onsuccess = () =>
      resolve((req.result as StoredStudyMeta[]).sort((a, b) => b.savedAt - a.savedAt));
    req.onerror = () => reject(req.error);
  });
}

export async function getStudyMeta(id: string): Promise<StoredStudyMeta | undefined> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STUDIES_STORE, 'readonly');
    const req = tx.objectStore(STUDIES_STORE).get(id);
    req.onsuccess = () => resolve(req.result as StoredStudyMeta | undefined);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Reconstruct the File[] for a stored study. File order is irrelevant to the
 * caller (metadata is re-derived and re-sorted per slice), but we sort by the
 * original index anyway for determinism.
 */
export async function loadStudyFiles(id: string, onProgress?: ProgressFn): Promise<File[]> {
  const db = await openDB();
  const meta = await getStudyMeta(id);
  const total = meta?.fileCount ?? 0;
  return new Promise((resolve, reject) => {
    const records: StoredFile[] = [];
    const tx = db.transaction(FILES_STORE, 'readonly');
    const index = tx.objectStore(FILES_STORE).index(BY_STUDY_INDEX);
    const req = index.openCursor(IDBKeyRange.only(id));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        records.push(cursor.value as StoredFile);
        onProgress?.(records.length, total || records.length);
        cursor.continue();
      }
    };
    tx.oncomplete = () => {
      records.sort((a, b) => a.index - b.index);
      resolve(records.map((r) => new File([r.blob], r.name)));
    };
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

// --- Writes ------------------------------------------------------------------

/**
 * Persist a study's raw files. Idempotent by content signature: re-saving the
 * same folder just bumps its recency instead of storing a second copy.
 *
 * File records are written first (in batches) and the small `studies` record is
 * written last, so it acts as a commit marker — an interrupted save leaves only
 * orphan file records (cleaned by `pruneOrphans`), never a half-listed study.
 */
export async function saveStudy(
  files: File[],
  info: SaveStudyInfo,
  onProgress?: ProgressFn,
): Promise<StoredStudyMeta> {
  if (files.length === 0) throw new Error('No files to save');
  const db = await openDB();

  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
  // Identity, not a size heuristic: sorted Series UIDs uniquely pin the study.
  // (UIDs are dot-separated numeric OIDs, so the '|'/',' delimiters are safe.)
  const uidKey = [...info.seriesUIDs].sort().join(',');
  const signature = `${uidKey}|${files.length}|${totalBytes}`;

  const existing = (await listStudies()).find((s) => s.signature === signature);
  if (existing) {
    const touched = { ...existing, savedAt: now() };
    await runTx(db, STUDIES_STORE, 'readwrite', (store) => store.put(touched));
    onProgress?.(files.length, files.length);
    return touched;
  }

  const id = crypto.randomUUID();

  for (let i = 0; i < files.length; i += WRITE_BATCH_SIZE) {
    const batch = files.slice(i, i + WRITE_BATCH_SIZE);
    await runTx(db, FILES_STORE, 'readwrite', (store) => {
      batch.forEach((file, j) => {
        const index = i + j;
        const record: StoredFile = {
          key: `${id}::${index}`,
          studyId: id,
          index,
          name: file.name,
          blob: file,
        };
        store.put(record);
      });
    });
    onProgress?.(Math.min(i + WRITE_BATCH_SIZE, files.length), files.length);
  }

  const meta: StoredStudyMeta = {
    id,
    label: info.label,
    signature,
    fileCount: files.length,
    totalBytes,
    modality: info.modality,
    studyDescription: info.studyDescription,
    seriesCount: info.seriesCount,
    savedAt: now(),
  };
  await runTx(db, STUDIES_STORE, 'readwrite', (store) => store.put(meta));
  return meta;
}

/** Bump a study's recency without rewriting its bytes. */
export async function touchStudy(id: string): Promise<void> {
  const meta = await getStudyMeta(id);
  if (!meta) return;
  meta.savedAt = now();
  const db = await openDB();
  await runTx(db, STUDIES_STORE, 'readwrite', (store) => store.put(meta));
}

export async function deleteStudy(id: string): Promise<void> {
  const db = await openDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([STUDIES_STORE, FILES_STORE], 'readwrite');
    tx.objectStore(STUDIES_STORE).delete(id);
    const filesStore = tx.objectStore(FILES_STORE);
    const req = filesStore.index(BY_STUDY_INDEX).openKeyCursor(IDBKeyRange.only(id));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        filesStore.delete(cursor.primaryKey);
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  if (getLastOpenedId() === id) setLastOpenedId(null);
}

export async function clearAllStudies(): Promise<void> {
  const db = await openDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction([STUDIES_STORE, FILES_STORE], 'readwrite');
    tx.objectStore(STUDIES_STORE).clear();
    tx.objectStore(FILES_STORE).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  setLastOpenedId(null);
}

/** Remove file records whose parent study record no longer exists. */
export async function pruneOrphans(): Promise<void> {
  const db = await openDB();
  const validIds = new Set((await listStudies()).map((s) => s.id));
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(FILES_STORE, 'readwrite');
    const store = tx.objectStore(FILES_STORE);
    const req = store.openKeyCursor();
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        const studyId = String(cursor.primaryKey).split('::')[0];
        if (!validIds.has(studyId)) store.delete(cursor.primaryKey);
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

// --- Storage usage -----------------------------------------------------------

export async function estimateStorage(): Promise<{ usage: number; quota: number } | null> {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) return null;
  const est = await navigator.storage.estimate();
  return { usage: est.usage ?? 0, quota: est.quota ?? 0 };
}
