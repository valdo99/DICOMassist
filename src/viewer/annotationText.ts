import { utilities as csUtils } from '@cornerstonejs/core';

interface RoiStats {
  area?: number;
  mean?: number;
  stdDev?: number;
  max?: number;
  min?: number;
  isEmptyArea?: boolean;
  areaUnit?: string;
  modalityUnit?: string;
}

interface RoiAnnotationData {
  label?: string;
  cachedStats?: Record<string, RoiStats | undefined>;
}

/**
 * Text shown next to an ROI annotation.
 *
 * A label says what the region *is* ("ACL tear"), which is what a reader needs;
 * raw statistics (area in mm², mean HU, std dev) are only meaningful when the
 * ROI was drawn as a measurement. So: show the label when there is one, and
 * fall back to the measurement stats only for unlabeled (hand-drawn) ROIs.
 *
 * This also guards `cachedStats[targetId]`, which Cornerstone's default
 * implementation destructures without a check — it throws for annotations whose
 * stats have not been computed yet (e.g. ones added programmatically).
 */
export function labelFirstGetTextLines(data: RoiAnnotationData, targetId: string): string[] {
  const label = data?.label?.trim();
  if (label) return [label];

  const stats = data?.cachedStats?.[targetId];
  if (!stats) return [];

  const lines: string[] = [];
  const round = (n: number) => csUtils.roundNumber(n);
  if (typeof stats.area === 'number') {
    lines.push(
      stats.isEmptyArea ? 'Area: Oblique not supported' : `Area: ${round(stats.area)} ${stats.areaUnit ?? ''}`.trim(),
    );
  }
  if (typeof stats.mean === 'number') lines.push(`Mean: ${round(stats.mean)} ${stats.modalityUnit ?? ''}`.trim());
  if (typeof stats.max === 'number') lines.push(`Max: ${round(stats.max)} ${stats.modalityUnit ?? ''}`.trim());
  if (typeof stats.min === 'number') lines.push(`Min: ${round(stats.min)} ${stats.modalityUnit ?? ''}`.trim());
  if (typeof stats.stdDev === 'number') lines.push(`Std Dev: ${round(stats.stdDev)} ${stats.modalityUnit ?? ''}`.trim());
  return lines;
}
