import React, { useState, useCallback } from 'react';
import Tooltip from './Tooltip';
import { deltaLogitToColor } from '../utils/colorScale';
import { usePatch } from '../hooks/usePatch';
import type { SupportedModel } from '../types';

const DEFAULT_SOURCE_PROMPT = 'When Mary and John went to the store, John gave a drink to';
const DEFAULT_TARGET_PROMPT = 'When John and Mary went to the store, Mary gave a drink to';
const DEFAULT_CORRECT_TOKEN = ' Mary';
const DEFAULT_INCORRECT_TOKEN = ' John';

interface PatchDiffViewProps {
  nLayers: number;
  nHeads: number;
  model: SupportedModel;
}

const PatchDiffView: React.FC<PatchDiffViewProps> = ({
  nLayers,
  nHeads,
  model,
}) => {
  const [sourcePrompt, setSourcePrompt] = useState(DEFAULT_SOURCE_PROMPT);
  const [targetPrompt, setTargetPrompt] = useState(DEFAULT_TARGET_PROMPT);
  const [correctToken, setCorrectToken] = useState(DEFAULT_CORRECT_TOKEN);
  const [incorrectToken, setIncorrectToken] = useState(DEFAULT_INCORRECT_TOKEN);
  const [threshold, setThreshold] = useState(0);

  const patchLogic = usePatch();
  const isLoading = patchLogic.isLoading;
  const deltaLogits = patchLogic.result?.delta_logits.data as unknown as number[][] || [];

  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: React.ReactNode;
  }>({ visible: false, x: 0, y: 0, content: null });

  // Compute max absolute ΔLogit for color normalization
  const maxAbs = React.useMemo(() => {
    if (deltaLogits.length === 0) return 0;
    let max = 0;
    for (const row of deltaLogits) {
      for (const v of row) {
        const abs = Math.abs(v);
        if (abs > max) max = abs;
      }
    }
    return max || 1;
  }, [deltaLogits]);

  // Count heads above threshold
  const headsAboveThreshold = React.useMemo(() => {
    let count = 0;
    for (const row of deltaLogits) {
      for (const v of row) {
        if (Math.abs(v) >= threshold) count++;
      }
    }
    return count;
  }, [deltaLogits, threshold]);

  const handleRunPatch = useCallback(() => {
    patchLogic.runPatching(sourcePrompt, targetPrompt, model, {
      correct: correctToken,
      incorrect: incorrectToken,
    });
  }, [patchLogic, sourcePrompt, targetPrompt, model, correctToken, incorrectToken]);

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, layer: number, head: number, value: number) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setTooltip({
        visible: true,
        x: rect.left + rect.width / 2,
        y: rect.top,
        content: (
          <div className="flex flex-col gap-0.5">
            <span style={{ color: 'var(--mech-accent)', fontWeight: 600 }}>
              L{layer}H{head}
            </span>
            <span
              style={{
                color: value > 0 ? 'var(--mech-blue-positive)' : value < 0 ? 'var(--mech-red)' : 'var(--mech-text-dim)',
                fontWeight: 600,
              }}
            >
              ΔLogit: {value > 0 ? '+' : ''}{value.toFixed(4)}
            </span>
          </div>
        ),
      });
    },
    []
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  }, []);

  const CELL_SIZE = 40;

  return (
    <div className="flex flex-col gap-5" style={{ padding: '24px' }}>
      {/* ── Header ────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <h2 className="text-label" style={{ fontSize: 11 }}>
          Activation Patching
        </h2>
        <span
          className="text-readout"
          style={{ color: 'var(--mech-text-dim)', marginLeft: 'auto' }}
        >
          {nLayers}×{nHeads} ΔLogit matrix
        </span>
      </div>

      {/* ── Dual Prompt Input ──────────────────────────── */}
      <div
        className="flex gap-0"
        style={{
          background: 'var(--mech-surface)',
          borderRadius: 8,
          border: '1px solid var(--mech-border)',
          overflow: 'hidden',
        }}
      >
        {/* Source Prompt */}
        <div className="flex flex-col gap-2 flex-1" style={{ padding: 16 }}>
          <label className="text-label">Source Prompt</label>
          <textarea
            value={sourcePrompt}
            onChange={(e) => setSourcePrompt(e.target.value)}
            disabled={isLoading}
            rows={3}
            style={{ fontSize: 11, lineHeight: 1.6, background: 'var(--mech-surface-raised)' }}
          />
        </div>

        {/* Divider */}
        <div
          style={{
            width: 1,
            background: 'var(--mech-border)',
            alignSelf: 'stretch',
          }}
        />

        {/* Target Prompt */}
        <div className="flex flex-col gap-2 flex-1" style={{ padding: 16 }}>
          <label className="text-label">Target Prompt</label>
          <textarea
            value={targetPrompt}
            onChange={(e) => setTargetPrompt(e.target.value)}
            disabled={isLoading}
            rows={3}
            style={{ fontSize: 11, lineHeight: 1.6, background: 'var(--mech-surface-raised)' }}
          />
        </div>
      </div>

      {/* ── Token Inputs + Run Button ──────────────────── */}
      <div className="flex items-end gap-3">
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-label">Correct Token</label>
          <input
            value={correctToken}
            onChange={(e) => setCorrectToken(e.target.value)}
            disabled={isLoading}
            style={{ fontSize: 11, padding: '6px 10px' }}
          />
        </div>
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-label">Incorrect Token</label>
          <input
            value={incorrectToken}
            onChange={(e) => setIncorrectToken(e.target.value)}
            disabled={isLoading}
            style={{ fontSize: 11, padding: '6px 10px' }}
          />
        </div>
        <button
          className={`btn-primary ${isLoading ? 'scanline-loader' : ''}`}
          onClick={handleRunPatch}
          disabled={isLoading}
          style={{ minWidth: 160, flex: 0 }}
        >
          {isLoading ? 'Patching...' : 'Run Patching'}
        </button>
      </div>

      {/* ── Threshold Slider ───────────────────────────── */}
      {deltaLogits.length > 0 && (
        <div
          className="flex items-center gap-4"
          style={{
            padding: '12px 16px',
            background: 'var(--mech-surface)',
            borderRadius: 8,
            border: '1px solid var(--mech-border)',
          }}
        >
          <label className="text-label" style={{ flexShrink: 0 }}>
            ΔLogit Threshold
          </label>
          <input
            type="range"
            min={0}
            max={maxAbs}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            style={{
              flex: 1,
              accentColor: 'var(--mech-accent)',
              cursor: 'pointer',
            }}
          />
          <span
            className="text-readout"
            style={{ color: 'var(--mech-accent)', minWidth: 42 }}
          >
            {threshold.toFixed(2)}
          </span>
          <span
            className="text-readout"
            style={{ color: 'var(--mech-text-dim)' }}
          >
            {headsAboveThreshold}/{nLayers * nHeads} heads
          </span>
        </div>
      )}

      {/* ── ΔLogit Heatmap ─────────────────────────────── */}
      {deltaLogits.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <div className="flex gap-1" style={{ display: 'inline-flex' }}>
            {/* Layer labels (Y axis) */}
            <div
              className="flex flex-col items-end"
              style={{ paddingTop: CELL_SIZE + 8, gap: 2, minWidth: 32 }}
            >
              {Array.from({ length: nLayers }, (_, l) => (
                <div
                  key={l}
                  className="text-readout flex items-center justify-end"
                  style={{
                    height: CELL_SIZE,
                    color: 'var(--mech-text-dim)',
                    paddingRight: 6,
                  }}
                >
                  L{l}
                </div>
              ))}
            </div>

            <div className="flex flex-col">
              {/* Head labels (X axis) */}
              <div className="flex" style={{ gap: 2, marginBottom: 6 }}>
                {Array.from({ length: nHeads }, (_, h) => (
                  <div
                    key={h}
                    className="text-readout flex items-center justify-center"
                    style={{
                      width: CELL_SIZE,
                      height: CELL_SIZE,
                      color: 'var(--mech-text-dim)',
                    }}
                  >
                    H{h}
                  </div>
                ))}
              </div>

              {/* Cells */}
              {deltaLogits.map((row, l) => (
                <div key={l} className="flex" style={{ gap: 2, marginBottom: 2 }}>
                  {row.map((value, h) => {
                    const aboveThreshold = Math.abs(value) >= threshold;
                    const { color, glowIntensity, glowType, textColor } = deltaLogitToColor(
                      value,
                      maxAbs
                    );

                    let boxShadow = 'none';
                    if (aboveThreshold && glowIntensity > 0.2) {
                      if (glowType === 'blue') {
                        boxShadow = `0 0 ${6 + glowIntensity * 12}px rgba(66, 165, 245, ${glowIntensity * 0.4}),
                                     0 0 ${12 + glowIntensity * 20}px rgba(66, 165, 245, ${glowIntensity * 0.15})`;
                      } else if (glowType === 'red') {
                        boxShadow = `0 0 ${6 + glowIntensity * 12}px rgba(239, 83, 80, ${glowIntensity * 0.4}),
                                     0 0 ${12 + glowIntensity * 20}px rgba(239, 83, 80, ${glowIntensity * 0.15})`;
                      }
                    }

                    return (
                      <div
                        key={h}
                        onMouseEnter={(e) => handleMouseEnter(e, l, h, value)}
                        onMouseLeave={handleMouseLeave}
                        className="flex items-center justify-center font-mono"
                        style={{
                          width: CELL_SIZE,
                          height: CELL_SIZE,
                          borderRadius: 4,
                          background: aboveThreshold ? color : 'var(--mech-surface)',
                          border: `1px solid ${aboveThreshold ? 'var(--mech-border-subtle)' : 'transparent'}`,
                          opacity: aboveThreshold ? 1 : 0.15,
                          fontSize: 8,
                          fontWeight: 600,
                          color: aboveThreshold ? textColor : 'var(--mech-text-dim)',
                          boxShadow,
                          transition: 'all 200ms ease',
                          cursor: 'default',
                        }}
                      >
                        {Math.abs(value) >= 0.01
                          ? `${value > 0 ? '+' : ''}${value.toFixed(1)}`
                          : '·'}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Context Info ───────────────────────────────── */}
      {deltaLogits.length > 0 && (
        <div
          className="flex items-center gap-6"
          style={{
            padding: '12px 16px',
            background: 'var(--mech-surface)',
            borderRadius: 8,
            border: '1px solid var(--mech-border)',
          }}
        >
          <div className="flex items-center gap-2">
            <span className="text-label">Clean ΔLogit:</span>
            <span className="text-readout" style={{ color: 'var(--mech-accent)', fontWeight: 700 }}>
              {(patchLogic.result?.answer_token_info?.clean_logit_diff as number)?.toFixed(2) ?? '0.00'}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <div style={{ width: 10, height: 10, borderRadius: 2, background: '#42a5f5' }} />
              <span className="text-readout" style={{ color: 'var(--mech-text-dim)' }}>
                +ΔLogit (pushes correct)
              </span>
            </div>
            <div className="flex items-center gap-1">
              <div style={{ width: 10, height: 10, borderRadius: 2, background: '#ef5350' }} />
              <span className="text-readout" style={{ color: 'var(--mech-text-dim)' }}>
                −ΔLogit (pushes incorrect)
              </span>
            </div>
          </div>
        </div>
      )}

      <Tooltip
        content={tooltip.content}
        x={tooltip.x}
        y={tooltip.y}
        visible={tooltip.visible}
      />
    </div>
  );
};

export default React.memo(PatchDiffView);
