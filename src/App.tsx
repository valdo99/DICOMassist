import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Circle, Eye, EyeOff, Loader2 } from 'lucide-react';
import { imageLoader, getRenderingEngine, cache } from '@cornerstonejs/core';
import type { IStackViewport } from '@cornerstonejs/core';
import cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';
import { initCornerstone } from './viewer/CornerstoneInit';
import DicomDropZone, { type LoadResult } from './viewer/DicomDropZone';
import ViewportGrid, { type ActiveToolName, type LayoutType, type OrientationMarkerType } from './viewer/ViewportGrid';
import Toolbar from './viewer/Toolbar';
import LoadingOverlay from './viewer/LoadingOverlay';
import MetadataPanel from './ui/MetadataPanel';
import SeriesBrowser from './ui/SeriesBrowser';
import ChatSidebar, { type ChatSidebarHandle } from './ui/ChatSidebar';
import SettingsPanel from './ui/SettingsPanel';
import DisclaimerModal from './ui/DisclaimerModal';
import LandingScreen from './ui/LandingScreen';
import RecentStudies from './ui/RecentStudies';
import type { AnatomicalPlane } from './dicom/orientationUtils';
import type { StudyMetadata } from './dicom/types';
import type { ProviderConfig, ViewportContext, ResolvedCircleAnnotation } from './llm/types';
import { useLLMChat, type SliceMapping } from './llm/useLLMChat';
import { processDicomFiles } from './dicom/loadDicomFiles';
import * as studyStore from './persistence/studyStore';
import { drawCircleAnnotations } from './viewer/AnnotationDrawer';
import type { AgentBridge } from './agent/types';
import { logger } from './utils/logger';

const STORAGE_KEY = 'dicomassist-llm-config';

function loadConfig(): ProviderConfig {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch { /* ignore */ }
  return { provider: 'ollama' };
}

function saveConfig(config: ProviderConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export default function App() {
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const [ready, setReady] = useState(false);
  const [imageIds, setImageIds] = useState<string[]>([]);
  const [primaryAxis, setPrimaryAxis] = useState<AnatomicalPlane>('axial');
  const [orientation, setOrientation] = useState<AnatomicalPlane>('axial');
  const [activeTool, setActiveTool] = useState<ActiveToolName>('WindowLevel');
  const [layout, setLayout] = useState<LayoutType>('1x1');
  const [orientationMarkerType, setOrientationMarkerType] = useState<OrientationMarkerType>('cube');
  const [prefetchProgress, setPrefetchProgress] = useState({ loaded: 0, total: 0 });
  const [studyMetadata, setStudyMetadata] = useState<StudyMetadata | null>(null);
  const [showMetadata, setShowMetadata] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [providerConfig, setProviderConfig] = useState<ProviderConfig>(loadConfig);
  const [showSeriesBrowser, setShowSeriesBrowser] = useState(false);
  const [activeSeriesUID, setActiveSeriesUID] = useState<string>('');
  const [invert, setInvert] = useState(false);
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [cineEnabled, setCineEnabled] = useState(false);
  const [showAiCircles, setShowAiCircles] = useState(true);
  const [agentAnnotations, setAgentAnnotations] = useState<ResolvedCircleAnnotation[]>([]);
  const [restoring, setRestoring] = useState<{ label: string; loaded: number; total: number } | null>(null);
  const resetRef = useRef<(() => void) | null>(null);
  const chatSidebarRef = useRef<ChatSidebarHandle>(null);
  const autoRestoredRef = useRef(false);
  // Monotonic token identifying the current "load intent". Bumped on every new
  // load, restore, cancel, and close, so a slow async load/restore/save can
  // detect that it has been superseded and skip its state writes. Fixes the
  // cancel-then-pick-another and save-after-close races.
  const loadEpochRef = useRef(0);

  // Bridge the agent's tools use to drive the viewer. Rebuilt only when the
  // study changes; the state setters it closes over are stable.
  const agentBridge = useMemo<AgentBridge>(() => {
    type Series = StudyMetadata['series'][number];
    const showSlice = (series: Series, instanceNumber: number, wc?: number, ww?: number) => {
      const engine = getRenderingEngine('dicomRenderingEngine');
      const targetImageIds = series.slices.map((s) => s.imageId);
      const vp0 = engine?.getViewport('CT_STACK') as IStackViewport | undefined;
      const currentIds = vp0?.getImageIds?.() ?? [];
      const alreadyActive = currentIds.length > 0 && currentIds[0] === targetImageIds[0];
      if (!alreadyActive) {
        setImageIds(targetImageIds);
        setActiveSeriesUID(series.seriesInstanceUID);
        const plane = series.anatomicalPlane === 'oblique' ? 'axial' : series.anatomicalPlane;
        setPrimaryAxis(plane);
        setOrientation(plane);
        setLayout('1x1');
      }
      let attempts = 0;
      const apply = () => {
        const eng = getRenderingEngine('dicomRenderingEngine');
        const vp = (eng?.getViewport('CT_STACK') ?? eng?.getViewport('CT_SINGLE_VOL')) as IStackViewport | undefined;
        const ids = vp?.getImageIds?.() ?? [];
        if (!vp || ids.length === 0) {
          if (attempts++ < 8) setTimeout(apply, 200);
          return;
        }
        if (wc != null && ww != null) {
          vp.setProperties({ voiRange: { lower: wc - ww / 2, upper: wc + ww / 2 } });
        }
        const idx = series.slices.findIndex((s) => s.instanceNumber === instanceNumber);
        if (idx >= 0 && idx < ids.length) vp.setImageIdIndex(idx);
        vp.render();
      };
      setTimeout(apply, alreadyActive ? 0 : 350);
    };
    return {
      viewSeries(seriesNumber, instanceNumber, wc, ww) {
        const series = studyMetadata?.series.find((s) => String(s.seriesNumber) === seriesNumber);
        if (series) showSlice(series, instanceNumber, wc, ww);
      },
      navigateToSlice(seriesNumber, instanceNumber) {
        const series = studyMetadata?.series.find((s) => String(s.seriesNumber) === seriesNumber);
        if (series) showSlice(series, instanceNumber);
      },
      setWindowLevel(wc, ww) {
        const eng = getRenderingEngine('dicomRenderingEngine');
        const vp = (eng?.getViewport('CT_STACK') ?? eng?.getViewport('CT_SINGLE_VOL')) as IStackViewport | undefined;
        if (!vp) return;
        vp.setProperties({ voiRange: { lower: wc - ww / 2, upper: wc + ww / 2 } });
        vp.render();
      },
      drawCircle(ann) {
        setAgentAnnotations((prev) => (prev.some((a) => a.uid === ann.uid) ? prev : [...prev, ann]));
      },
      clearCircles() {
        setAgentAnnotations([]);
      },
    };
  }, [studyMetadata]);

  const {
    messages,
    status,
    statusText,
    error,
    currentPlan,
    pipeline,
    startAnalysis,
    confirmPlan,
    cancelPlan,
    sendFollowUp,
    clearChat,
    agentSteps,
  } = useLLMChat(studyMetadata, providerConfig, agentBridge);

  useEffect(() => {
    initCornerstone().then(() => setReady(true));
  }, []);

  // Push a loaded/restored study into the viewer state. Shared by fresh loads
  // (drop zone) and local-storage restores so both behave identically. Resets
  // per-view transform/layout/tool state so the new study never inherits the
  // previous one's inverted/flipped/cine/MPR view after an in-session switch.
  const applyLoadResult = useCallback((result: LoadResult) => {
    setImageIds(result.imageIds);
    setPrimaryAxis(result.primaryAxis);
    setOrientation(result.primaryAxis);
    setStudyMetadata(result.studyMetadata);
    setActiveSeriesUID(result.studyMetadata.primarySeriesUID);
    setShowSeriesBrowser(result.studyMetadata.series.length > 1);
    setInvert(false);
    setFlipH(false);
    setFlipV(false);
    setCineEnabled(false);
    setLayout('1x1');
    setActiveTool('WindowLevel');
  }, []);

  const handleFilesLoaded = useCallback(
    (result: LoadResult, sourceFiles: File[]) => {
      const epoch = ++loadEpochRef.current;
      applyLoadResult(result);

      // Persist the raw bytes locally so a refresh restores this study without
      // a re-drop. Runs in the background; failure (e.g. quota) is non-fatal.
      const m = result.studyMetadata;
      const label =
        m.studyDescription?.trim() ||
        m.series[0]?.seriesDescription?.trim() ||
        (m.modality && m.modality !== 'unknown' ? `${m.modality} study` : 'DICOM study');
      studyStore
        .saveStudy(sourceFiles, {
          label,
          modality: m.modality,
          studyDescription: m.studyDescription,
          seriesCount: m.series.length,
          seriesUIDs: m.series.map((s) => s.seriesInstanceUID),
        })
        .then((saved) => {
          // Skip if the user has since closed this study or loaded another —
          // otherwise a late save would re-point auto-restore at a closed study.
          if (loadEpochRef.current === epoch) studyStore.setLastOpenedId(saved.id);
        })
        .catch((err) => logger.warn('[DICOMassist] Could not save study locally:', err));
    },
    [applyLoadResult],
  );

  // Reload a persisted study's bytes from IndexedDB and run them back through
  // the normal pipeline. Each call claims a fresh epoch; if a newer load/restore/
  // cancel/close supersedes it, its async continuation bails without touching
  // state — so a cancelled or stale restore can never clobber the active study.
  const restoreStudy = useCallback(
    async (meta: studyStore.StoredStudyMeta) => {
      const epoch = ++loadEpochRef.current;
      const superseded = () => loadEpochRef.current !== epoch;
      setRestoring({ label: meta.label, loaded: 0, total: meta.fileCount });
      try {
        const files = await studyStore.loadStudyFiles(meta.id, (loaded, total) => {
          if (!superseded()) setRestoring({ label: meta.label, loaded, total });
        });
        if (superseded()) return;
        const result = await processDicomFiles(files);
        if (superseded()) return;
        if (!result) throw new Error('restored study contained no readable DICOM files');
        applyLoadResult(result);
        studyStore.setLastOpenedId(meta.id);
        studyStore.touchStudy(meta.id).catch(() => {});
      } catch (err) {
        if (!superseded()) {
          logger.warn('[DICOMassist] Could not restore study:', err);
          studyStore.setLastOpenedId(null);
        }
      } finally {
        if (!superseded()) setRestoring(null);
      }
    },
    [applyLoadResult],
  );

  const handleCancelRestore = useCallback(() => {
    loadEpochRef.current++; // supersede the in-flight restore
    setRestoring(null);
    studyStore.setLastOpenedId(null);
  }, []);

  // Close the current study and return to the landing/library. The study stays
  // in local storage (still listed under "Recent studies"); only the auto-open
  // pointer is cleared so a refresh won't reopen it. Purges Cornerstone's caches
  // so repeated close/reopen cycles don't accumulate retained file bytes.
  const handleCloseStudy = useCallback(() => {
    loadEpochRef.current++; // supersede any in-flight save/restore
    setImageIds([]);
    setStudyMetadata(null);
    setActiveSeriesUID('');
    setAgentAnnotations([]);
    setShowChat(false);
    setShowMetadata(false);
    setShowSeriesBrowser(false);
    clearChat();
    studyStore.setLastOpenedId(null);
    try {
      cache.purgeCache();
      cornerstoneDICOMImageLoader.wadouri.fileManager.purge();
    } catch { /* best-effort cleanup */ }
  }, [clearChat]);

  // On first ready render, auto-restore the last-opened study (if any) and
  // sweep any orphaned file records left by an interrupted save.
  useEffect(() => {
    if (!ready || autoRestoredRef.current) return;
    autoRestoredRef.current = true;
    studyStore.pruneOrphans().catch(() => {});
    const id = studyStore.getLastOpenedId();
    if (!id) return;
    studyStore
      .getStudyMeta(id)
      .then((meta) => {
        if (meta) restoreStudy(meta);
        else studyStore.setLastOpenedId(null);
      })
      .catch(() => {});
  }, [ready, restoreStudy]);

  // Prefetch all images after they're set
  useEffect(() => {
    if (imageIds.length === 0) return;

    let cancelled = false;
    const total = imageIds.length;
    let loaded = 0;

    setPrefetchProgress({ loaded: 0, total });

    const BATCH_SIZE = 6;
    async function prefetch() {
      for (let i = 0; i < total; i += BATCH_SIZE) {
        if (cancelled) return;
        const batch = imageIds.slice(i, i + BATCH_SIZE);
        const promises = batch.map((id) =>
          imageLoader.loadAndCacheImage(id).catch(() => {})
        );
        await Promise.all(promises);
        loaded += batch.length;
        if (!cancelled) {
          setPrefetchProgress({ loaded: Math.min(loaded, total), total });
        }
      }
    }

    prefetch();

    return () => {
      cancelled = true;
    };
  }, [imageIds]);

  // Apply SelectionPlan to viewport (W/L + scroll + switch series if needed)
  useEffect(() => {
    if (!currentPlan || !studyMetadata) return;

    const targetSeries = studyMetadata.series.find(
      (s) => String(s.seriesNumber) === currentPlan.targetSeries,
    );

    // If the plan targets a different series, switch the viewport to it
    if (targetSeries) {
      const targetImageIds = targetSeries.slices.map((s) => s.imageId);
      if (targetImageIds.length > 0 && targetImageIds[0] !== imageIds[0]) {
        setImageIds(targetImageIds);
        // W/L and scroll will be applied after the viewport reloads with new imageIds
      }
      setActiveSeriesUID(targetSeries.seriesInstanceUID);
    }

    // Apply W/L and scroll (may run before or after series switch)
    let attempts = 0;
    const applyPlan = () => {
      try {
        const engine = getRenderingEngine('dicomRenderingEngine');
        if (!engine) {
          // Viewport not ready yet — retry
          if (attempts++ < 5) setTimeout(applyPlan, 200);
          return;
        }
        const viewport = engine.getViewport('CT_STACK') as IStackViewport | undefined;
        if (!viewport) {
          if (attempts++ < 5) setTimeout(applyPlan, 200);
          return;
        }

        const viewportIds = viewport.getImageIds();
        if (viewportIds.length === 0) {
          if (attempts++ < 5) setTimeout(applyPlan, 200);
          return;
        }

        const { windowCenter, windowWidth } = currentPlan;
        viewport.setProperties({ voiRange: { lower: windowCenter - windowWidth / 2, upper: windowCenter + windowWidth / 2 } });

        if (targetSeries) {
          const [rangeStart, rangeEnd] = currentPlan.sliceRange;
          const midInstance = Math.round((rangeStart + rangeEnd) / 2);
          // Find the slice closest to midInstance in the target series
          const sliceIdx = targetSeries.slices.findIndex((s) => s.instanceNumber >= midInstance);
          if (sliceIdx >= 0 && sliceIdx < viewportIds.length) {
            viewport.setImageIdIndex(sliceIdx);
          }
        }

        viewport.render();
      } catch {
        // viewport may not be ready yet — retry
        if (attempts++ < 5) setTimeout(applyPlan, 200);
      }
    };

    // Delay to let series switch + viewport setup take effect
    const timer = setTimeout(applyPlan, 300);
    return () => clearTimeout(timer);
  }, [currentPlan, studyMetadata]); // intentionally omitting imageIds to avoid loop

  // When plan arrives, ensure sidebar is open
  useEffect(() => {
    if (status === 'awaiting-confirmation') {
      setShowChat(true);
      setShowMetadata(false);
    }
  }, [status]);

  // Auto-open chat when analysis completes
  useEffect(() => {
    if (messages.length > 0 && status === 'idle') {
      setShowChat(true);
    }
  }, [messages.length, status]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;

      if (mod && e.key === 'k') {
        e.preventDefault();
        if (imageIds.length > 0 && studyMetadata) {
          setShowChat(true);
          setShowMetadata(false);
          // Focus the input after the sidebar renders
          requestAnimationFrame(() => chatSidebarRef.current?.focusInput());
        }
      }

      if (e.key === 'Escape') {
        if (status === 'awaiting-confirmation') {
          cancelPlan();
        } else if (settingsOpen) {
          setSettingsOpen(false);
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [imageIds.length, studyMetadata, settingsOpen, status, cancelPlan]);

  // Fall back to W/L if leaving MPR while Crosshairs is active
  useEffect(() => {
    if (layout !== 'mpr' && activeTool === 'Crosshairs') {
      setActiveTool('WindowLevel');
    }
  }, [layout, activeTool]);

  const handleReset = useCallback(() => {
    resetRef.current?.();
    setInvert(false);
    setFlipH(false);
    setFlipV(false);
    setCineEnabled(false);
  }, []);

  const handleAcceptDisclaimer = useCallback(() => {
    setDisclaimerAccepted(true);
  }, []);

  const handleConfigChange = useCallback((config: ProviderConfig) => {
    setProviderConfig(config);
    saveConfig(config);
  }, []);

  const handleStartAnalysis = useCallback((hint: string, options?: { surveyMode?: boolean }) => {
    // Capture current viewport position as context for slice selection
    let viewportContext: ViewportContext | undefined;
    try {
      const engine = getRenderingEngine('dicomRenderingEngine');
      const viewport = engine?.getViewport('CT_STACK') as IStackViewport | undefined;
      if (viewport && studyMetadata) {
        const sliceIndex = viewport.getCurrentImageIdIndex();
        // Find which series is currently displayed
        const currentIds = viewport.getImageIds();
        const currentSeries = studyMetadata.series.find((s) =>
          s.slices.length === currentIds.length && s.slices[0]?.imageId === currentIds[0],
        ) ?? studyMetadata.series.find((s) =>
          s.slices.some((sl) => sl.imageId === currentIds[0]),
        );
        if (currentSeries && sliceIndex >= 0 && sliceIndex < currentSeries.slices.length) {
          const slice = currentSeries.slices[sliceIndex];
          viewportContext = {
            currentInstanceNumber: slice.instanceNumber,
            currentZPosition: slice.imagePositionPatient[2],
            seriesNumber: String(currentSeries.seriesNumber),
            totalSlicesInSeries: currentSeries.slices.length,
          };
          logger.log('[DICOMassist] Viewport context:', viewportContext);
        }
      }
    } catch { /* viewport may not be ready */ }

    startAnalysis(hint, viewportContext, options);
  }, [startAnalysis, studyMetadata]);

  const navigateTargetRef = useRef<{ instanceNumber: number; imageId: string; seriesNumber: string } | null>(null);

  const handleNavigateToSlice = useCallback((mapping: SliceMapping) => {
    if (!studyMetadata || !currentPlan) return;

    // Use the mapping's seriesNumber (multi-series aware) to find the correct series
    const seriesNum = mapping.seriesNumber || currentPlan.targetSeries;
    const targetSeries = studyMetadata.series.find(
      (s) => String(s.seriesNumber) === seriesNum,
    );
    if (!targetSeries) return;

    // Check if we need to switch series first
    const needsSeriesSwitch = targetSeries.seriesInstanceUID !== activeSeriesUID;

    if (needsSeriesSwitch) {
      logger.log(`[Navigate] Switching from series ${activeSeriesUID} → ${targetSeries.seriesInstanceUID} (${targetSeries.seriesDescription})`);
      // Store the target so we can scroll after series loads
      navigateTargetRef.current = { instanceNumber: mapping.instanceNumber, imageId: mapping.imageId, seriesNumber: seriesNum };
      const targetImageIds = targetSeries.slices.map((s) => s.imageId);
      setImageIds(targetImageIds);
      setActiveSeriesUID(targetSeries.seriesInstanceUID);
      const plane = targetSeries.anatomicalPlane === 'oblique' ? 'axial' : targetSeries.anatomicalPlane;
      setPrimaryAxis(plane);
      setOrientation(plane);
      setLayout('1x1');
      return; // scrollToSlice will be called by the effect below once images load
    }

    // Already on the correct series — scroll directly
    scrollToSlice(mapping.instanceNumber, mapping.imageId, targetSeries);
  }, [studyMetadata, currentPlan, activeSeriesUID]);

  // After a series switch triggered by slice navigation, scroll to the target slice
  useEffect(() => {
    const target = navigateTargetRef.current;
    if (!target || !studyMetadata) return;

    const targetSeries = studyMetadata.series.find(
      (s) => String(s.seriesNumber) === target.seriesNumber,
    );
    if (!targetSeries || targetSeries.seriesInstanceUID !== activeSeriesUID) return;

    // Series is now active — try to scroll (with retries for viewport readiness)
    let attempts = 0;
    const tryScroll = () => {
      const success = scrollToSlice(target.instanceNumber, target.imageId, targetSeries);
      if (!success && attempts++ < 5) {
        setTimeout(tryScroll, 200);
      } else {
        navigateTargetRef.current = null;
      }
    };
    const timer = setTimeout(tryScroll, 100);
    return () => clearTimeout(timer);
  }, [activeSeriesUID, studyMetadata]);

  // Draw (or clear) the LLM's circle annotations on the viewer. Re-runs when the
  // annotation set changes, when the loaded series changes (imageIds), or when
  // the user toggles visibility. drawCircleAnnotations clears prior AI circles
  // first, so passing [] is a clean "remove all". The delay lets a series switch
  // settle so the target slice's viewport is ready.
  // Circles come from either the legacy pipeline (Ollama) or the agent path
  // (Claude); whichever is populated is the active set.
  const annotations = useMemo(
    () => (pipeline?.annotations && pipeline.annotations.length > 0 ? pipeline.annotations : agentAnnotations),
    [pipeline?.annotations, agentAnnotations],
  );

  useEffect(() => {
    const anns = showAiCircles ? annotations : [];
    // drawCircleAnnotations clears prior AI circles then redraws, so it's
    // idempotent. Two passes cover the case where a series switch is still
    // setting up the stack viewport on the first pass.
    const timers = [
      setTimeout(() => drawCircleAnnotations(anns), 300),
      setTimeout(() => drawCircleAnnotations(anns), 900),
    ];
    return () => timers.forEach(clearTimeout);
  }, [annotations, imageIds, showAiCircles]);

  // Legacy pipeline (Ollama): jump the viewer to the first circle when analysis
  // completes. The agent path drives navigation itself via its tools.
  useEffect(() => {
    const anns = pipeline?.annotations;
    if (!anns || anns.length === 0) return;
    setShowAiCircles(true);
    const first = anns[0];
    handleNavigateToSlice({
      imageIndex: 0,
      instanceNumber: first.instanceNumber,
      imageId: first.imageId,
      zPosition: 0,
      label: '',
      seriesNumber: first.seriesNumber,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipeline?.annotations]);

  // Agent path: reveal circles as the agent draws them.
  useEffect(() => {
    if (agentAnnotations.length > 0) setShowAiCircles(true);
  }, [agentAnnotations]);

  function scrollToSlice(
    instanceNumber: number,
    imageId: string,
    targetSeries: StudyMetadata['series'][number],
  ): boolean {
    try {
      const engine = getRenderingEngine('dicomRenderingEngine');
      if (!engine) return false;

      let viewport = engine.getViewport('CT_STACK') as IStackViewport | undefined;
      if (!viewport) {
        viewport = engine.getViewport('CT_SINGLE_VOL') as IStackViewport | undefined;
      }
      if (!viewport) return false;

      const viewportIds = viewport.getImageIds();
      if (viewportIds.length === 0) return false;

      // Strategy 1: Find by instance number in the target series metadata
      const sliceIdx = targetSeries.slices.findIndex(
        (s) => s.instanceNumber === instanceNumber,
      );
      if (sliceIdx >= 0 && sliceIdx < viewportIds.length) {
        logger.log(`[Navigate] Instance #${instanceNumber} → series index ${sliceIdx}`);
        viewport.setImageIdIndex(sliceIdx);
        viewport.render();
        return true;
      }

      // Strategy 2: Direct imageId match
      const exactIdx = viewportIds.indexOf(imageId);
      if (exactIdx >= 0) {
        logger.log(`[Navigate] Exact imageId match at index ${exactIdx}`);
        viewport.setImageIdIndex(exactIdx);
        viewport.render();
        return true;
      }

      // Strategy 3: Partial imageId match
      const partialIdx = viewportIds.findIndex(
        (id) => id.includes(imageId) || imageId.includes(id),
      );
      if (partialIdx >= 0) {
        logger.log(`[Navigate] Partial imageId match at index ${partialIdx}`);
        viewport.setImageIdIndex(partialIdx);
        viewport.render();
        return true;
      }

      logger.warn(`[Navigate] Failed to find slice for instance #${instanceNumber}`, {
        imageId,
        viewportIdCount: viewportIds.length,
      });
      return false;
    } catch {
      return false;
    }
  }

  const handleToggleMetadata = useCallback(() => {
    if (!studyMetadata) return;
    setShowMetadata((v) => {
      if (!v) setShowChat(false);
      return !v;
    });
  }, [studyMetadata]);

  const handleSelectSeries = useCallback((seriesUID: string) => {
    if (!studyMetadata || seriesUID === activeSeriesUID) return;
    const series = studyMetadata.series.find((s) => s.seriesInstanceUID === seriesUID);
    if (!series) return;
    setImageIds(series.slices.map((s) => s.imageId));
    setActiveSeriesUID(seriesUID);
    const plane = series.anatomicalPlane === 'oblique' ? 'axial' : series.anatomicalPlane;
    setPrimaryAxis(plane);
    setOrientation(plane);
    setLayout('1x1');
  }, [studyMetadata, activeSeriesUID]);

  if (!ready) {
    return (
      <>
        {!disclaimerAccepted && <DisclaimerModal onAccept={handleAcceptDisclaimer} />}
        <div className="flex items-center justify-center h-full text-neutral-500">
          Initializing viewer...
        </div>
      </>
    );
  }

  if (imageIds.length === 0) {
    return (
      <div className="h-full overflow-y-auto">
        {!disclaimerAccepted && <DisclaimerModal onAccept={handleAcceptDisclaimer} />}
        {restoring ? (
          <div className="min-h-screen flex flex-col items-center justify-center bg-black text-zinc-300 gap-4 p-8">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            <p className="text-sm text-zinc-400">
              Restoring <span className="text-zinc-200">{restoring.label}</span>…
            </p>
            <div className="w-64 h-2 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-100"
                style={{
                  width: `${restoring.total > 0 ? Math.round((restoring.loaded / restoring.total) * 100) : 0}%`,
                }}
              />
            </div>
            <p className="text-xs text-zinc-600">
              {restoring.loaded} / {restoring.total} files from local storage
            </p>
            <button
              onClick={handleCancelRestore}
              className="mt-1 text-xs text-zinc-600 hover:text-zinc-300 transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <LandingScreen recent={<RecentStudies onRestore={restoreStudy} />}>
            <DicomDropZone onFilesLoaded={handleFilesLoaded} />
          </LandingScreen>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Toolbar
        activeTool={activeTool}
        onToolChange={setActiveTool}
        layout={layout}
        onLayoutChange={setLayout}
        onReset={handleReset}
        onCloseStudy={handleCloseStudy}
        showSeriesBrowser={showSeriesBrowser}
        onToggleSeriesBrowser={studyMetadata && studyMetadata.series.length > 1 ? () => setShowSeriesBrowser((v) => !v) : undefined}
        showMetadata={showMetadata}
        onToggleMetadata={studyMetadata ? handleToggleMetadata : undefined}
        onOpenSpotlight={() => {
          setShowChat(true);
          setShowMetadata(false);
          requestAnimationFrame(() => chatSidebarRef.current?.focusInput());
        }}
        onOpenSettings={() => setSettingsOpen((v) => !v)}
        orientationMarkerType={orientationMarkerType}
        onOrientationMarkerTypeChange={setOrientationMarkerType}
        invert={invert}
        onInvertToggle={() => setInvert((v) => !v)}
        flipH={flipH}
        onFlipHToggle={() => setFlipH((v) => !v)}
        flipV={flipV}
        onFlipVToggle={() => setFlipV((v) => !v)}
        cineEnabled={cineEnabled}
        onCineToggle={() => setCineEnabled((v) => !v)}
      />
      <div className="flex-1 min-h-0 flex overflow-hidden">
        {showSeriesBrowser && studyMetadata && studyMetadata.series.length > 1 && (
          <SeriesBrowser
            metadata={studyMetadata}
            activeSeriesUID={activeSeriesUID}
            onSelectSeries={handleSelectSeries}
            onClose={() => setShowSeriesBrowser(false)}
          />
        )}
        <div className="flex-1 min-w-0 relative overflow-hidden">
          <div className="absolute inset-0">
            <ViewportGrid
              imageIds={imageIds}
              activeTool={activeTool}
              layout={layout}
              orientation={orientation}
              primaryAxis={primaryAxis}
              orientationMarkerType={orientationMarkerType}
              onResetRef={resetRef}
              invert={invert}
              flipH={flipH}
              flipV={flipV}
              cineEnabled={cineEnabled}
              studyMetadata={studyMetadata}
            />
          </div>
          {annotations.length > 0 && (
            <div
              className="absolute top-2 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-neutral-900/85 border border-blue-700/50 shadow-lg backdrop-blur-sm"
              title="AI-suggested regions — approximate visual guides, not measurements"
            >
              <span className="flex items-center gap-1.5 text-xs font-medium text-blue-300">
                <Circle className="w-3.5 h-3.5" />
                {annotations.length} AI region{annotations.length === 1 ? '' : 's'} marked
              </span>
              <span className="w-px h-3.5 bg-neutral-700" />
              <button
                onClick={() => setShowAiCircles((v) => !v)}
                className="flex items-center gap-1 text-xs text-neutral-400 hover:text-neutral-100 transition-colors"
              >
                {showAiCircles ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                {showAiCircles ? 'Hide' : 'Show'}
              </button>
            </div>
          )}
          <LoadingOverlay
            loaded={prefetchProgress.loaded}
            total={prefetchProgress.total}
          />
        </div>
        {showMetadata && studyMetadata && (
          <MetadataPanel
            metadata={studyMetadata}
            activeSeriesUID={activeSeriesUID}
            onClose={() => setShowMetadata(false)}
          />
        )}
        {showChat && (
          <ChatSidebar
            ref={chatSidebarRef}
            messages={messages}
            status={status}
            statusText={statusText}
            error={error}
            pipeline={pipeline}
            agentSteps={agentSteps}
            currentPlan={currentPlan}
            studyMetadata={studyMetadata}
            onConfirmPlan={confirmPlan}
            onCancelPlan={cancelPlan}
            onStartAnalysis={handleStartAnalysis}
            onSendFollowUp={sendFollowUp}
            onClear={clearChat}
            onClose={() => setShowChat(false)}
            onNavigateToSlice={handleNavigateToSlice}
          />
        )}
      </div>

      {/* Overlays */}
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        config={providerConfig}
        onConfigChange={handleConfigChange}
      />
    </div>
  );
}
