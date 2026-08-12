import { tool } from 'ai';
import { z } from 'zod';
import type { StudyMetadata, SliceMetadata, SeriesMetadata } from '../dicom/types';
import type { ResolvedCircleAnnotation } from '../llm/types';
import type { SelectedSlice } from '../filtering/types';
import { exportSlicesToJpeg } from '../filtering/SliceExporter';
import type { AgentBridge } from './types';
import { logger } from '../utils/logger';

/** A batch is 30 slices — enough to read a region, small enough to stay in budget. */
export const BATCH_SIZE = 30;

export interface SliceRef {
  seriesNumber: string;
  instanceNumber: number;
}

export const sliceKey = (seriesNumber: string, instanceNumber: number) =>
  `${seriesNumber}:${instanceNumber}`;

/**
 * Per-run bookkeeping shared between the tools and the context pruner: which
 * slices each tool call returned images for, and which slices the agent marked
 * as significant. Lets the pruner drop un-annotated batches per-slice without
 * parsing any text.
 */
export interface AgentRunContext {
  viewedByCall: Map<string, SliceRef[]>;
  annotated: Set<string>;
}

export function createRunContext(): AgentRunContext {
  return { viewedByCall: new Map(), annotated: new Set() };
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

/** Which spatial axis varies as you scroll a series of the given plane. */
function varyingAxisIndex(plane: string): 0 | 1 | 2 {
  if (plane === 'sagittal') return 0;
  if (plane === 'coronal') return 1;
  return 2; // axial / oblique
}

function defaultWindow(modality: string): { wc: number; ww: number } {
  return modality === 'CT' ? { wc: 40, ww: 400 } : { wc: 127, ww: 256 };
}

/** Evenly pick up to `count` items across the array (keeps first and last). */
function uniformSample<T>(items: T[], count: number): T[] {
  if (items.length <= count) return items;
  const out: T[] = [];
  for (let i = 0; i < count; i++) {
    out.push(items[Math.round((i * (items.length - 1)) / (count - 1))]);
  }
  return out;
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

interface ViewedImage {
  seriesNumber: string;
  seriesDescription: string;
  instanceNumber: number;
  totalSlices: number;
  zPosition: number;
  base64: string;
}

interface ImageBatchOutput {
  error?: string;
  header: string;
  windowCenter: number;
  windowWidth: number;
  images: ViewedImage[];
}

/** Render a set of slices to JPEG and record them against this tool call. */
async function loadImages(
  entries: Array<{ series: SeriesMetadata; slice: SliceMetadata }>,
  windowCenter: number | undefined,
  windowWidth: number | undefined,
  toolCallId: string,
  ctx: AgentRunContext,
): Promise<{ images: ViewedImage[]; wc: number; ww: number }> {
  const first = entries[0]?.series;
  const def = defaultWindow(first?.modality ?? 'MR');
  const wc = windowCenter ?? first?.windowCenter ?? def.wc;
  const ww = windowWidth ?? first?.windowWidth ?? def.ww;

  const images: ViewedImage[] = [];
  for (const { series, slice } of entries) {
    const axis = varyingAxisIndex(series.anatomicalPlane);
    const selected: SelectedSlice[] = [
      {
        imageId: slice.imageId,
        instanceNumber: slice.instanceNumber,
        sliceLocation: slice.sliceLocation,
        zPosition: slice.imagePositionPatient[axis],
      },
    ];
    const [exported] = await exportSlicesToJpeg(selected, wc, ww);
    if (!exported) continue;
    images.push({
      seriesNumber: String(series.seriesNumber),
      seriesDescription: series.seriesDescription,
      instanceNumber: exported.instanceNumber,
      totalSlices: series.slices.length,
      zPosition: exported.zPosition,
      base64: await blobToBase64(exported.blob),
    });
  }

  ctx.viewedByCall.set(
    toolCallId,
    images.map((i) => ({ seriesNumber: i.seriesNumber, instanceNumber: i.instanceNumber })),
  );
  return { images, wc, ww };
}

/** Turn a batch result into image content the model can actually look at. */
function batchToModelOutput(output: ImageBatchOutput) {
  if (output.error) return { type: 'error-text' as const, value: output.error };

  const parts: Array<
    | { type: 'text'; text: string }
    | { type: 'file'; mediaType: string; data: { type: 'data'; data: string } }
  > = [{ type: 'text', text: output.header }];

  for (const img of output.images) {
    parts.push({
      type: 'text',
      text: `Series #${img.seriesNumber} Slice ${img.instanceNumber}/${img.totalSlices} (z=${Math.round(img.zPosition)}mm):`,
    });
    parts.push({ type: 'file', mediaType: 'image/jpeg', data: { type: 'data', data: img.base64 } });
  }
  return { type: 'content' as const, value: parts };
}

/**
 * Build the agent's tool set, bound to a study, a viewer bridge, and the run
 * context. The tools wrap existing domain logic (slice export, annotation
 * drawing) and drive the viewer through the bridge — no React in here.
 */
export function createDicomTools(metadata: StudyMetadata, bridge: AgentBridge, ctx: AgentRunContext) {
  const findSeries = (seriesNumber: string) =>
    metadata.series.find((s) => String(s.seriesNumber) === String(seriesNumber));

  return {
    view_slices: tool({
      description:
        `Review a batch of up to ${BATCH_SIZE} slices from one series: renders them at the given window, ` +
        'RETURNS the images so you can see them, and moves the viewer there. ' +
        'Work one region at a time — review the batch, then mark anything significant with draw_circle. ' +
        'Slices you do not annotate are dropped from your context after the next batch, so annotate before moving on.',
      inputSchema: z.object({
        seriesNumber: z.string().describe('Series Number as shown in the metadata, e.g. "3".'),
        fromInstance: z.number().describe('First slice (instance number) of the range to review.'),
        toInstance: z.number().describe('Last slice (instance number) of the range to review.'),
        maxImages: z
          .number()
          .optional()
          .describe(`How many slices to sample across the range (default ${BATCH_SIZE}, max ${BATCH_SIZE}).`),
        windowCenter: z.number().optional().describe('Window center (level). Omit for the series/modality default.'),
        windowWidth: z.number().optional().describe('Window width. Omit for the series/modality default.'),
      }),
      execute: async (
        { seriesNumber, fromInstance, toInstance, maxImages, windowCenter, windowWidth },
        { toolCallId },
      ): Promise<ImageBatchOutput> => {
        const series = findSeries(seriesNumber);
        if (!series) {
          return { error: `No series #${seriesNumber} in this study.`, header: '', windowCenter: 0, windowWidth: 0, images: [] };
        }
        const [lo, hi] = fromInstance <= toInstance ? [fromInstance, toInstance] : [toInstance, fromInstance];

        let inRange = series.slices
          .filter((s) => s.instanceNumber >= lo && s.instanceNumber <= hi)
          .sort((a, b) => a.instanceNumber - b.instanceNumber);
        if (inRange.length === 0) inRange = [...series.slices].sort((a, b) => a.instanceNumber - b.instanceNumber);

        const cap = Math.min(BATCH_SIZE, Math.max(1, Math.round(maxImages ?? BATCH_SIZE)));
        const sampled = uniformSample(inRange, cap);

        const { images, wc, ww } = await loadImages(
          sampled.map((slice) => ({ series, slice })),
          windowCenter,
          windowWidth,
          toolCallId,
          ctx,
        );

        const mid = sampled[Math.floor(sampled.length / 2)]?.instanceNumber ?? sampled[0]?.instanceNumber;
        if (mid != null) bridge.viewSeries(String(series.seriesNumber), mid, wc, ww);

        logger.log(`[Agent] view_slices: Series #${series.seriesNumber}, ${images.length} images (W:${ww} C:${wc})`);
        return {
          header:
            `Reviewing ${images.length} slices from Series #${series.seriesNumber} "${series.seriesDescription}" ` +
            `(instances ${lo}–${hi} of ${series.slices.length}, window W:${ww} C:${wc}). The images, in order:`,
          windowCenter: wc,
          windowWidth: ww,
          images,
        };
      },
      toModelOutput: ({ output }) => batchToModelOutput(output as ImageBatchOutput),
    }),

    compare_slices: tool({
      description:
        `Put up to ${BATCH_SIZE} specific slices side by side to compare them — across series (e.g. confirm a ` +
        'sagittal finding on coronal), across sides (left vs right), or across levels. ' +
        'List the exact slices you want; use this instead of re-reviewing a whole range.',
      inputSchema: z.object({
        slices: z
          .array(
            z.object({
              seriesNumber: z.string().describe('Series Number of this slice.'),
              instanceNumber: z.number().describe('Instance number of this slice.'),
            }),
          )
          .describe(`The exact slices to compare, in the order you want to see them (max ${BATCH_SIZE}).`),
        reason: z.string().optional().describe('What you are comparing, e.g. "right vs left trigeminal nerve".'),
        windowCenter: z.number().optional(),
        windowWidth: z.number().optional(),
      }),
      execute: async ({ slices, reason, windowCenter, windowWidth }, { toolCallId }): Promise<ImageBatchOutput> => {
        const entries: Array<{ series: SeriesMetadata; slice: SliceMetadata }> = [];
        const missing: string[] = [];
        for (const ref of slices.slice(0, BATCH_SIZE)) {
          const series = findSeries(ref.seriesNumber);
          const slice = series?.slices.find((s) => s.instanceNumber === ref.instanceNumber);
          if (series && slice) entries.push({ series, slice });
          else missing.push(`#${ref.seriesNumber}/${ref.instanceNumber}`);
        }
        if (entries.length === 0) {
          return {
            error: `None of the requested slices exist (${missing.join(', ')}).`,
            header: '', windowCenter: 0, windowWidth: 0, images: [],
          };
        }

        const { images, wc, ww } = await loadImages(entries, windowCenter, windowWidth, toolCallId, ctx);

        const first = entries[0];
        bridge.viewSeries(String(first.series.seriesNumber), first.slice.instanceNumber, wc, ww);

        logger.log(`[Agent] compare_slices: ${images.length} images${reason ? ` (${reason})` : ''}`);
        return {
          header:
            `Comparison set: ${images.length} slices${reason ? ` — ${reason}` : ''} (window W:${ww} C:${wc}).` +
            (missing.length ? ` Not found: ${missing.join(', ')}.` : '') +
            ' The images, in order:',
          windowCenter: wc,
          windowWidth: ww,
          images,
        };
      },
      toModelOutput: ({ output }) => batchToModelOutput(output as ImageBatchOutput),
    }),

    draw_circle: tool({
      description:
        'Mark a finding on a slice with a labelled circle, like a radiologist annotating an image. ' +
        'The label is what the reader sees, so name the finding ("ACL tear", "vascular loop"). ' +
        'Annotating also keeps that slice in your context — anything you do not annotate is dropped later.',
      inputSchema: z.object({
        seriesNumber: z.string().describe('Series Number the slice belongs to.'),
        instanceNumber: z.number().describe('Slice (instance number) to draw on.'),
        cx: z.number().describe('Center x as a fraction of image width (0 = left, 1 = right).'),
        cy: z.number().describe('Center y as a fraction of image height (0 = top, 1 = bottom).'),
        radius: z.number().describe('Circle radius as a fraction of image width (typically 0.03–0.15).'),
        label: z.string().describe('What this marks — 1–4 words, shown on the image, e.g. "SCA vascular loop".'),
      }),
      execute: async ({ seriesNumber, instanceNumber, cx, cy, radius, label }) => {
        const series = findSeries(seriesNumber);
        const slice = series?.slices.find((s) => s.instanceNumber === instanceNumber);
        if (!series || !slice) {
          return { ok: false, message: `Could not find Series #${seriesNumber} Slice ${instanceNumber} to draw on.` };
        }
        const ann: ResolvedCircleAnnotation = {
          uid: `ai-circle-${series.seriesNumber}-${instanceNumber}-${Math.round(clamp01(cx) * 1000)}-${Math.round(clamp01(cy) * 1000)}`,
          imageId: slice.imageId,
          seriesNumber: String(series.seriesNumber),
          instanceNumber,
          label: (label || 'Finding').slice(0, 40),
          cx: clamp01(cx),
          cy: clamp01(cy),
          radius: Math.min(0.5, Math.max(0.01, radius)),
        };
        bridge.drawCircle(ann);
        ctx.annotated.add(sliceKey(String(series.seriesNumber), instanceNumber));
        logger.log(`[Agent] draw_circle: "${ann.label}" on Series #${seriesNumber} Slice ${instanceNumber}`);
        return {
          ok: true,
          message: `Marked "${ann.label}" on Series #${seriesNumber} Slice ${instanceNumber}; this slice stays in context.`,
        };
      },
    }),

    navigate_to_slice: tool({
      description: 'Scroll the viewer to a specific slice so the user sees it.',
      inputSchema: z.object({
        seriesNumber: z.string(),
        instanceNumber: z.number(),
      }),
      execute: async ({ seriesNumber, instanceNumber }) => {
        const series = findSeries(seriesNumber);
        if (!series) return { ok: false, message: `No series #${seriesNumber}.` };
        bridge.navigateToSlice(String(series.seriesNumber), instanceNumber);
        return { ok: true, message: `Navigated to Series #${seriesNumber} Slice ${instanceNumber}.` };
      },
    }),

    set_window_level: tool({
      description: 'Adjust the window/level (contrast) of the current viewport.',
      inputSchema: z.object({
        windowCenter: z.number(),
        windowWidth: z.number(),
      }),
      execute: async ({ windowCenter, windowWidth }) => {
        bridge.setWindowLevel(windowCenter, windowWidth);
        return { ok: true, message: `Window set to W:${windowWidth} C:${windowCenter}.` };
      },
    }),
  };
}

export type DicomTools = ReturnType<typeof createDicomTools>;
