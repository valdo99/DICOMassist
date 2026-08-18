import type { LanguageModel } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import type { AgentProvider } from './modelRouter';

/**
 * Build the AI SDK language model for the chosen provider. Both providers speak
 * the same `generateText` / `generateObject` + tools interface, so callers stay
 * provider-agnostic. Shared by the tool-using agent (DicomAgent) and the
 * text-only export report writer (reportAgent).
 */
export function buildProviderModel(
  provider: AgentProvider,
  apiKey: string,
  modelId: string,
): LanguageModel {
  if (provider === 'gemini') {
    const google = createGoogleGenerativeAI({ apiKey });
    return google(modelId);
  }
  const anthropic = createAnthropic({
    apiKey,
    // Required for direct browser calls — keeps the app client-only.
    headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
  });
  return anthropic(modelId);
}
