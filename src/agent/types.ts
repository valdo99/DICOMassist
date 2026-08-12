import type { ResolvedCircleAnnotation } from '../llm/types';

/**
 * The bridge the agent's tools use to drive the viewer. Implemented by the
 * React layer (which owns the rendering engine + component state); the tools
 * themselves stay free of React so they can run anywhere.
 */
export interface AgentBridge {
  /**
   * Show a series in the primary viewport, scroll to `instanceNumber`, and
   * apply the given window/level. Used by `view_slices` so the user sees what
   * the agent is looking at.
   */
  viewSeries(
    seriesNumber: string,
    instanceNumber: number,
    windowCenter?: number,
    windowWidth?: number,
  ): void;
  /** Scroll the viewer to a specific slice. */
  navigateToSlice(seriesNumber: string, instanceNumber: number): void;
  /** Apply window/level to the current viewport. */
  setWindowLevel(windowCenter: number, windowWidth: number): void;
  /** Draw one circle annotation (accumulates across calls within a turn). */
  drawCircle(annotation: ResolvedCircleAnnotation): void;
  /** Remove all agent-drawn circles. */
  clearCircles(): void;
}

/** A step the agent took, surfaced to the UI for the live trace. */
export interface AgentStepEvent {
  type: 'text' | 'tool-call' | 'tool-result' | 'model';
  /** Tool name for tool events. */
  toolName?: string;
  /** Short human-readable summary for the trace UI. */
  detail?: string;
  /** The agent's own narration/reasoning for this step. */
  text?: string;
}
