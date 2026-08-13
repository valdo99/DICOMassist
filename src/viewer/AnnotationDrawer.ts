import { getRenderingEngine, utilities as csUtils, metaData } from '@cornerstonejs/core';
import type { Types } from '@cornerstonejs/core';
import {
  EllipticalROITool,
  annotation as csAnnotation,
  utilities as csToolsUtils,
} from '@cornerstonejs/tools';
import type { ResolvedCircleAnnotation } from '../llm/types';
import { logger } from '../utils/logger';

/**
 * Programmatically draws EllipticalROI circles on the viewer — the same tool a
 * human uses to circle a region — from LLM-provided normalized coordinates.
 *
 * Coordinate flow: normalized (cx, cy, radius) in [0..1] over the exported JPEG
 * → DICOM pixel (col, row) [the JPEG is the full, uncropped slice] →
 * patient world coordinates via imageToWorldCoords → EllipticalROI world handles.
 * Because we go to world coordinates, circles land on the correct anatomy
 * regardless of the current window/level, zoom, or flip state.
 *
 * Scope: circles are tied to a slice's `referencedImageId`, so each shows only
 * when its slice is displayed. Only slices currently loaded in a stack viewport
 * are drawn; navigating to another series re-runs drawing for that series.
 */

// UIDs of the AI-generated annotations we've added, so we can clear exactly
// those without touching a human's manual measurements.
const aiAnnotationUids = new Set<string>();

type Point3 = Types.Point3;

/** Find a viewport in the engine whose loaded stack contains this imageId. */
function findViewportForImage(
  engine: ReturnType<typeof getRenderingEngine>,
  imageId: string,
) {
  if (!engine) return undefined;
  for (const vp of engine.getViewports()) {
    const getImageIds = (vp as { getImageIds?: () => string[] }).getImageIds;
    if (typeof getImageIds !== 'function') continue;
    let ids: string[] = [];
    try {
      ids = getImageIds.call(vp);
    } catch {
      continue;
    }
    if (ids.includes(imageId)) return vp;
  }
  return undefined;
}

/**
 * Compute the four EllipticalROI world handle points for a normalized circle.
 * EllipticalROI expects handles in [bottom, top, left, right] order.
 * Returns null if the slice has no image-plane metadata.
 */
function computeEllipseWorldPoints(
  imageId: string,
  cx: number,
  cy: number,
  radius: number,
): [Point3, Point3, Point3, Point3] | null {
  const plane = metaData.get('imagePlaneModule', imageId) as
    | { rows?: number; columns?: number }
    | undefined;
  const columns = plane?.columns;
  const rows = plane?.rows;
  if (!columns || !rows) return null;

  const col = cx * columns;
  const row = cy * rows;
  const rPx = radius * columns; // radius in pixels (fraction of width)

  try {
    const toWorld = (c: number, r: number) =>
      csUtils.imageToWorldCoords(imageId, [c, r]) as Point3 | undefined;
    const bottom = toWorld(col, row + rPx);
    const top = toWorld(col, row - rPx);
    const left = toWorld(col - rPx, row);
    const right = toWorld(col + rPx, row);
    if (!bottom || !top || !left || !right) return null;
    return [bottom, top, left, right];
  } catch {
    return null;
  }
}

/**
 * Clear every AI-drawn circle and re-render. Leaves the user's own manual
 * annotations untouched.
 */
export function clearAiAnnotations(renderingEngineId = 'dicomRenderingEngine'): void {
  if (aiAnnotationUids.size === 0) return;
  for (const uid of aiAnnotationUids) {
    try {
      csAnnotation.state.removeAnnotation(uid);
    } catch {
      /* annotation may already be gone (e.g. viewport torn down) */
    }
  }
  aiAnnotationUids.clear();

  const engine = getRenderingEngine(renderingEngineId);
  if (engine) {
    const ids = engine.getViewports().map((vp) => vp.id);
    if (ids.length > 0) {
      try {
        csToolsUtils.triggerAnnotationRenderForViewportIds(ids);
      } catch {
        /* ignore */
      }
    }
  }
}

// The circle currently carrying Cornerstone's "selected" styling. Tracked so we
// can explicitly deselect it when focus moves or clears: removeAnnotation (via
// clearAiAnnotations) does NOT purge Cornerstone's global selection set, so a
// re-added circle with the same uid would otherwise stay selected forever.
let focusedAnnotationUid: string | null = null;

/**
 * Emphasise a single circle so a "jump to this finding" action is visible: it
 * gets Cornerstone's selected styling plus the `highlighted` flag (a bolder
 * outline). Because focus is applied here — inside the draw pass — it survives
 * the redraw passes that fire after a series switch, unlike a one-shot select
 * that a later `clearAiAnnotations` → re-add would wipe. Passing `null` (or a
 * uid whose slice isn't loaded) clears the previous focus back to plain.
 */
function focusAnnotation(focusedUid: string | null): void {
  const next = focusedUid && aiAnnotationUids.has(focusedUid) ? focusedUid : null;
  // Return the previously-focused circle to plain styling when focus changes.
  if (focusedAnnotationUid && focusedAnnotationUid !== next) {
    try {
      csAnnotation.selection.deselectAnnotation(focusedAnnotationUid);
    } catch {
      /* already gone — nothing to clear */
    }
  }
  focusedAnnotationUid = next;
  if (!next) return;
  try {
    // preserveSelected = false → this circle becomes the sole selection.
    csAnnotation.selection.setAnnotationSelected(next, true, false);
  } catch {
    /* selection API unavailable / annotation gone — highlight flag still applies */
  }
}

/**
 * Draw the given circles on whichever stack viewport currently holds each
 * slice. Existing AI circles are cleared first, so calling this with `[]` is a
 * clean "remove all AI circles". Circles whose series is not currently loaded
 * are skipped (they render once the user navigates to that series and this runs
 * again). `focusedUid` emphasises one circle (see focusAnnotation). Returns the
 * number of circles actually drawn.
 */
export function drawCircleAnnotations(
  annotations: ResolvedCircleAnnotation[],
  opts: { focusedUid?: string | null; renderingEngineId?: string } = {},
): number {
  const { focusedUid = null, renderingEngineId = 'dicomRenderingEngine' } = opts;
  clearAiAnnotations(renderingEngineId);
  if (annotations.length === 0) {
    focusAnnotation(null); // release any tracked selection so it can't linger
    return 0;
  }

  const engine = getRenderingEngine(renderingEngineId);
  if (!engine) return 0;

  const touchedViewports = new Set<string>();
  let drawn = 0;

  for (const ann of annotations) {
    const viewport = findViewportForImage(engine, ann.imageId);
    if (!viewport) continue; // series not currently loaded — skip

    const points = computeEllipseWorldPoints(ann.imageId, ann.cx, ann.cy, ann.radius);
    if (!points) {
      logger.warn('[AnnotationDrawer] No image-plane metadata for', ann.imageId);
      continue;
    }

    const camera = viewport.getCamera();
    const newAnnotation = {
      annotationUID: ann.uid,
      highlighted: ann.uid === focusedUid,
      invalidated: true,
      isLocked: false,
      isVisible: true,
      metadata: {
        toolName: EllipticalROITool.toolName,
        viewPlaneNormal: [...(camera.viewPlaneNormal ?? [0, 0, 1])] as Point3,
        viewUp: [...(camera.viewUp ?? [0, -1, 0])] as Point3,
        FrameOfReferenceUID: viewport.getFrameOfReferenceUID(),
        referencedImageId: ann.imageId,
      },
      data: {
        label: ann.label,
        handles: {
          points,
          activeHandleIndex: null,
          textBox: { hasMoved: false },
        },
        cachedStats: {},
      },
    };

    try {
      csAnnotation.state.addAnnotation(
        newAnnotation as Parameters<typeof csAnnotation.state.addAnnotation>[0],
        viewport.element,
      );
      aiAnnotationUids.add(ann.uid);
      touchedViewports.add(viewport.id);
      drawn++;
    } catch (err) {
      logger.warn('[AnnotationDrawer] Failed to add annotation', err);
    }
  }

  // Apply focus before the render so the selected styling paints in this pass.
  focusAnnotation(focusedUid);

  if (touchedViewports.size > 0) {
    try {
      csToolsUtils.triggerAnnotationRenderForViewportIds([...touchedViewports]);
    } catch {
      /* ignore */
    }
  }

  logger.log(`[AnnotationDrawer] Drew ${drawn}/${annotations.length} AI circles`);
  return drawn;
}
