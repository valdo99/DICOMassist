import type { StudyMetadata } from '../dicom/types';
import type { ResolvedCircleAnnotation } from '../llm/types';
import { renderWindowedSliceToCanvas } from '../filtering/SliceExporter';
import { logger } from '../utils/logger';

/**
 * Renders slice images with their AI circle annotations baked in, for the PDF
 * export. Unlike the live viewer (where circles live on a separate SVG overlay),
 * this re-renders the slice offscreen and draws the circle onto the pixels — so
 * the result is a self-contained image that needs no viewport and works for any
 * slice, whether or not it is currently displayed.
 */

/** Window/level defaults by modality — mirrors the agent's slice tools. */
function defaultWindow(modality: string): { wc: number; ww: number } {
  return modality === 'CT' ? { wc: 40, ww: 400 } : { wc: 127, ww: 256 };
}

/** Pick the window/level to render an annotation's slice with. */
function windowForAnnotation(
  ann: ResolvedCircleAnnotation,
  metadata: StudyMetadata,
): { wc: number; ww: number } {
  const series = metadata.series.find((s) => String(s.seriesNumber) === ann.seriesNumber);
  const def = defaultWindow(series?.modality ?? metadata.modality ?? 'MR');
  return {
    wc: series?.windowCenter ?? def.wc,
    ww: series?.windowWidth ?? def.ww,
  };
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** A rendered annotated slice, ready to place in the PDF. */
export interface AnnotatedSliceImage {
  uid: string;
  dataUrl: string;
  /** Final pixel dimensions — used to preserve aspect ratio in the PDF. */
  width: number;
  height: number;
}

/**
 * Render a single annotation onto its slice and return a JPEG data URL plus the
 * image dimensions. Returns null if the slice can't be rendered.
 */
export async function renderAnnotatedSliceImage(
  ann: ResolvedCircleAnnotation,
  metadata: StudyMetadata,
): Promise<AnnotatedSliceImage | null> {
  try {
    const { wc, ww } = windowForAnnotation(ann, metadata);
    const rendered = await renderWindowedSliceToCanvas(ann.imageId, wc, ww);
    if (!rendered) return null;

    const { canvas, width, height } = rendered;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    drawCircleOnCanvas(ctx, ann, width, height);

    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
    const dataUrl = await blobToDataUrl(blob);
    return { uid: ann.uid, dataUrl, width, height };
  } catch (err) {
    logger.warn('[Export] Failed to render annotated slice', ann.imageId, err);
    return null;
  }
}

/**
 * Render every annotation to an annotated image. Runs sequentially (each render
 * decodes a slice); a failed render is skipped. Returns a uid → image map.
 */
export async function renderAnnotatedSliceImages(
  annotations: ResolvedCircleAnnotation[],
  metadata: StudyMetadata,
  onProgress?: (done: number, total: number) => void,
): Promise<Map<string, AnnotatedSliceImage>> {
  const out = new Map<string, AnnotatedSliceImage>();
  for (let i = 0; i < annotations.length; i++) {
    const img = await renderAnnotatedSliceImage(annotations[i], metadata);
    if (img) out.set(img.uid, img);
    onProgress?.(i + 1, annotations.length);
  }
  return out;
}

/** Draw the amber circle + label chip for one annotation onto the slice. */
function drawCircleOnCanvas(
  ctx: OffscreenCanvasRenderingContext2D,
  ann: ResolvedCircleAnnotation,
  width: number,
  height: number,
): void {
  const cxPx = ann.cx * width;
  const cyPx = ann.cy * height;
  const rPx = Math.max(6, ann.radius * width);
  const lw = Math.max(2, Math.round(width / 280));

  // Dark halo behind the ring so it reads on both bright and dark anatomy.
  ctx.beginPath();
  ctx.arc(cxPx, cyPx, rPx, 0, Math.PI * 2);
  ctx.lineWidth = lw + 2;
  ctx.strokeStyle = 'rgba(0,0,0,0.65)';
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cxPx, cyPx, rPx, 0, Math.PI * 2);
  ctx.lineWidth = lw;
  ctx.strokeStyle = '#f5b301';
  ctx.stroke();

  // Label chip
  const label = (ann.label || 'Finding').slice(0, 40);
  const fontSize = Math.max(15, Math.round(width / 46));
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.textBaseline = 'top';
  const textW = ctx.measureText(label).width;
  const padX = fontSize * 0.5;
  const padY = fontSize * 0.35;
  const chipW = textW + padX * 2;
  const chipH = fontSize + padY * 2;

  let chipX = cxPx - chipW / 2;
  chipX = Math.min(Math.max(4, chipX), width - chipW - 4);
  // Prefer above the circle; drop below if it would clip the top edge.
  let chipY = cyPx - rPx - chipH - lw * 2;
  if (chipY < 4) chipY = cyPx + rPx + lw * 2;
  chipY = Math.min(chipY, height - chipH - 4);

  ctx.fillStyle = 'rgba(0,0,0,0.72)';
  ctx.fillRect(chipX, chipY, chipW, chipH);
  ctx.fillStyle = '#fbbf24';
  ctx.fillText(label, chipX + padX, chipY + padY);
}
