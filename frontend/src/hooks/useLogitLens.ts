import { useState, useCallback, useRef, useEffect } from 'react';
import type { LogitLensFrame, SupportedModel } from '../types';

interface UseLogitLensReturn {
  frames: LogitLensFrame[];
  isStreaming: boolean;
  convergenceLayer: number | null;
  error: string | null;
  startStreaming: (prompt: string, model: SupportedModel, topK?: number, position?: number) => void;
  reset: () => void;
}

export function useLogitLens(): UseLogitLensReturn {
  const [frames, setFrames] = useState<LogitLensFrame[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [convergenceLayer, setConvergenceLayer] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const socketRef = useRef<WebSocket | null>(null);

  const reset = useCallback(() => {
    setFrames([]);
    setConvergenceLayer(null);
    setError(null);
    setIsStreaming(false);
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }, []);

  const startStreaming = useCallback((
    prompt: string, 
    model: SupportedModel, 
    topK: number = 10, 
    position: number = -1
  ) => {
    reset();
    setIsStreaming(true);

    // Determine the WS URL (using the same host as the Vite proxy)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/logit-lens`;
    
    // Connect through Vite proxy
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      // Send the initial configuration message
      ws.send(JSON.stringify({
        prompt,
        model,
        top_k: topK,
        position
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'layer') {
          setFrames(prev => {
            // Ensure we don't duplicate layers (in case of weird network issues)
            if (prev.some(f => f.layer === data.layer)) return prev;
            return [...prev, data as LogitLensFrame];
          });
        } else if (data.type === 'done') {
          setConvergenceLayer(data.convergence_layer);
          setIsStreaming(false);
          // Socket will be closed by the server, but we can close it here too
          ws.close();
        } else if (data.type === 'error') {
          setError(data.message || 'Unknown backend error');
          setIsStreaming(false);
          ws.close();
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error.');
      setIsStreaming(false);
    };

    ws.onclose = () => {
      setIsStreaming(false);
    };

  }, [reset]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  return {
    frames,
    isStreaming,
    convergenceLayer,
    error,
    startStreaming,
    reset
  };
}
