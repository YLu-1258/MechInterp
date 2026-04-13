import { useState, useCallback } from 'react';
import type { PatchResponse, SupportedModel, AnswerTokens } from '../types';

interface UsePatchReturn {
  result: PatchResponse | null;
  isLoading: boolean;
  error: string | null;
  runPatching: (
    sourcePrompt: string,
    targetPrompt: string,
    model: SupportedModel,
    answerTokens?: AnswerTokens
  ) => Promise<void>;
  reset: () => void;
}

export function usePatch(): UsePatchReturn {
  const [result, setResult] = useState<PatchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  const runPatching = useCallback(
    async (
      sourcePrompt: string,
      targetPrompt: string,
      model: SupportedModel,
      answerTokens?: AnswerTokens
    ) => {
      setIsLoading(true);
      setError(null);
      setResult(null); // Clear previous results while loading

      try {
        const body: Record<string, any> = {
          source_prompt: sourcePrompt,
          target_prompt: targetPrompt,
          model,
        };

        if (answerTokens?.correct && answerTokens?.incorrect) {
          body.answer_tokens = answerTokens;
        }

        const response = await fetch('/api/patch', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          let errMsg = 'Failed to run patching.';
          try {
            const errData = await response.json();
            errMsg = errData.detail || errMsg;
          } catch {
            // Fallback if parsing fails
          }
          throw new Error(errMsg);
        }

        const data: PatchResponse = await response.json();
        setResult(data);
      } catch (err: any) {
        setError(err.message || 'An unknown error occurred.');
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return { result, isLoading, error, runPatching, reset };
}
