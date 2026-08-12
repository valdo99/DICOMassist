import { tool } from 'ai';
import { z } from 'zod';
import type { StudyMetadata } from '../dicom/types';
import type { ResolvedCircleAnnotation } from '../llm/types';
import type { SelectedSlice } from '../filtering/types';
import { exportSlicesToJpeg } from '../filtering/SliceExporter';
import type { AgentBridge } from './types';
import { logger } from '../utils/logger';

const MAX_IMAGES_PER_CALL = 16;

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
  instanceNumber: number;
  zPosition: number;
  base64: string;
}

/**
 * Build the agent's tool set, bound to a study and a viewer bridge. The tools
 * wrap the existing domain logic (slice export, annotation drawing) and drive
 * the viewer through the bridge — no React in here.
 */
export function createDicomTools(metadata: StudyMetadata, bridge: AgentBridge) {
  const findSeries = (seriesNumber: string) =>
    metadata.series.find((s) => String(s.seriesNumber) === String(seriesNumber));

  return {
    view_slices: tool({
      description:
        'Load slices from a series, apply a window/level, and RETURN the resulting images so you can see them. ' +
        'Also updates the viewer to show these slices. Call this before making any claim about the images. ' +
        'Pick the series and slice range from the study metadata and the clinical question.',
      inputSchema: z.object({
        seriesNumber: z.string().describe('Series Number as shown in the metadata, e.g. "3".'),
        fromInstance: z.number().describe('First slice (instance number) of the range to view.'),
        toInstance: z.number().describe('Last slice (instance number) of the range to view.'),
        maxImages: z
          .number()
          .optional()
          .describe(`How many slices to sample across the range (default 12, max ${MAX_IMAGES_PER_CALL}).`),
        windowCenter: z.number().optional().describe('Window center (level). Omit to use the series/modality default.'),
        windowWidth: z.number().optional().describe('Window width. Omit to use the series/modality default.'),
      }),
      execute: async ({ seriesNumber, fromInstance, toInstance, maxImages, windowCenter, windowWidth }) => {
        const series = findSeries(seriesNumber);
        if (!series) {
          return { error: `No series #${seriesNumber} in this study.`, images: [] as ViewedImage[] };
        }
        const [lo, hi] = fromInstance <= toInstance ? [fromInstance, toInstance] : [toInstance, fromInstance];
        const axis = varyingAxisIndex(series.anatomicalPlane);

        let inRange = series.slices
          .filter((s) => s.instanceNumber >= lo && s.instanceNumber <= hi)
          .sort((a, b) => a.instanceNumber - b.instanceNumber);
        if (inRange.length === 0) inRange = [...series.slices].sort((a, b) => a.instanceNumber - b.instanceNumber);

        const cap = Math.min(MAX_IMAGES_PER_CALL, Math.max(1, Math.round(maxImages ?? 12)));
        const sampled = uniformSample(inRange, cap);

        const def = defaultWindow(series.modality);
        const wc = windowCenter ?? series.windowCenter ?? def.wc;
        const ww = windowWidth ?? series.windowWidth ?? def.ww;

        const selected: SelectedSlice[] = sampled.map((s) => ({
          imageId: s.imageId,
          instanceNumber: s.instanceNumber,
          sliceLocation: s.sliceLocation,
          zPosition: s.imagePositionPatient[axis],
        }));

        const exported = await exportSlicesToJpeg(selected, wc, ww);
        const images: ViewedImage[] = await Promise.all(
          exported.map(async (e) => ({
            seriesNumber: String(series.seriesNumber),
            instanceNumber: e.instanceNumber,
            zPosition: e.zPosition,
            base64: await blobToBase64(e.blob),
          })),
        );

        // Sync the viewer to what the agent is looking at.
        const mid = sampled[Math.floor(sampled.length / 2)]?.instanceNumber ?? sampled[0]?.instanceNumber;
        if (mid != null) bridge.viewSeries(String(series.seriesNumber), mid, wc, ww);

        logger.log(`[Agent] view_slices: Series #${series.seriesNumber}, ${images.length} images (W:${ww} C:${wc})`);
        return {
          seriesNumber: String(series.seriesNumber),
          seriesDescription: series.seriesDescription,
          windowCenter: wc,
          windowWidth: ww,
          totalSlices: series.slices.length,
          images,
        };
      },
      // Feed the exported JPEGs back to the model as real image content.
      toModelOutput: ({ output }) => {
        const o = output as {
          error?: string;
          seriesNumber?: string;
          seriesDescription?: string;
          windowCenter?: number;
          windowWidth?: number;
          totalSlices?: number;
          images: ViewedImage[];
        };
        if (o.error) return { type: 'error-text', value: o.error };

        const parts: Array<
          | { type: 'text'; text: string }
          | { type: 'file'; mediaType: string; data: { type: 'data'; data: string } }
        > = [
          {
            type: 'text',
            text:
              `Loaded ${o.images.length} slices from Series #${o.seriesNumber} "${o.seriesDescription}" ` +
              `(window W:${o.windowWidth} C:${o.windowCenter}). The images, in order:`,
          },
        ];
        for (const img of o.images) {
          parts.push({
            type: 'text',
            text: `Series #${img.seriesNumber} Slice ${img.instanceNumber}/${o.totalSlices} (z=${Math.round(img.zPosition)}mm):`,
          });
          parts.push({ type: 'file', mediaType: 'image/jpeg', data: { type: 'data', data: img.base64 } });
        }
        return { type: 'content', value: parts };
      },
    }),

    draw_circle: tool({
      description:
        'Circle a finding on a specific slice, exactly like a radiologist marking a region. ' +
        'Only use this for findings you can point to a specific spot for, on a slice you have viewed.',
      inputSchema: z.object({
        seriesNumber: z.string().describe('Series Number the slice belongs to.'),
        instanceNumber: z.number().describe('Slice (instance number) to draw on.'),
        cx: z.number().describe('Center x as a fraction of image width (0 = left, 1 = right).'),
        cy: z.number().describe('Center y as a fraction of image height (0 = top, 1 = bottom).'),
        radius: z.number().describe('Circle radius as a fraction of image width (typically 0.03–0.15).'),
        label: z.string().describe('Short 1–3 word name of the finding.'),
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
        logger.log(`[Agent] draw_circle: "${ann.label}" on Series #${seriesNumber} Slice ${instanceNumber}`);
        return { ok: true, message: `Circled "${ann.label}" on Series #${seriesNumber} Slice ${instanceNumber}.` };
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
