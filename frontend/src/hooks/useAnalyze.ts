import { useState, useCallback } from 'react';
import type { AnalyzeResponse, SupportedModel } from '../types';

interface UseAnalyzeReturn {
  result: AnalyzeResponse | null;
  isLoading: boolean;
  error: string | null;
  analyzePrompt: (prompt: string, model: SupportedModel) => Promise<void>;
  reset: () => void;
}

export function useAnalyze(): UseAnalyzeReturn {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  const analyzePrompt = useCallback(async (prompt: string, model: SupportedModel) => {
    setIsLoading(true);
    setError(null);
    setResult(null); // Clear previous results while loading

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, model }),
      });

      if (!response.ok) {
        let errMsg = 'Failed to analyze prompt.';
        try {
          const errData = await response.json();
          errMsg = errData.detail || errMsg;
        } catch {
          // Fallback if parsing fails
        }
        throw new Error(errMsg);
      }

      const data: AnalyzeResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { result, isLoading, error, analyzePrompt, reset };
}
