import React, { useState, useCallback } from 'react';
import Tooltip from './Tooltip';
import { entropyToColor } from '../utils/colorScale';
import { meanHeadEntropy } from '../utils/entropy';
import type { SelectedHead } from '../types';

interface AttentionHeadGridProps {
  attentionPatterns: number[][][][]; // [n_layers, n_heads, seq_len, seq_len]
  nLayers: number;
  nHeads: number;
  selectedHead: SelectedHead | null;
  onHeadClick: (head: SelectedHead) => void;
}

const AttentionHeadGrid: React.FC<AttentionHeadGridProps> = ({
  attentionPatterns,
  nLayers,
  nHeads,
  selectedHead,
  onHeadClick,
}) => {
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: React.ReactNode;
  }>({ visible: false, x: 0, y: 0, content: null });

  // Compute entropy for each head
  const entropies: number[][] = React.useMemo(() => {
    return attentionPatterns.map((layerHeads) =>
      layerHeads.map((pattern) => meanHeadEntropy(pattern))
    );
  }, [attentionPatterns]);

  // Find max entropy for color normalization
  const maxEntropy = React.useMemo(() => {
    let max = 0;
    for (const row of entropies) {
      for (const e of row) {
        if (e > max) max = e;
      }
    }
    return max || 1;
  }, [entropies]);

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, layer: number, head: number) => {
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
            <span style={{ color: 'var(--mech-amber)' }}>
              entropy: {entropies[layer][head].toFixed(3)}
            </span>
          </div>
        ),
      });
    },
    [entropies]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  }, []);

  const CELL_SIZE = 36;
  const GAP = 2;

  return (
    <div className="flex flex-col gap-4" style={{ padding: '24px' }}>
      {/* ── Header ────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <h2 className="text-label" style={{ fontSize: 11 }}>
          Attention Head Grid
        </h2>
        <span
          className="text-readout"
          style={{ color: 'var(--mech-text-dim)', marginLeft: 'auto' }}
        >
          {nLayers}×{nHeads} · colored by entropy
        </span>
      </div>

      {/* ── Grid ──────────────────────────────────────── */}
      <div className="flex gap-1" style={{ overflowX: 'auto' }}>
        {/* Layer labels (Y axis) */}
        <div
          className="flex flex-col items-end"
          style={{
            paddingTop: CELL_SIZE + GAP + 4,
            gap: GAP,
            minWidth: 36,
          }}
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
          <div className="flex" style={{ gap: GAP, marginBottom: 4 }}>
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

          {/* Grid cells */}
          {Array.from({ length: nLayers }, (_, l) => (
            <div key={l} className="flex" style={{ gap: GAP, marginBottom: GAP }}>
              {Array.from({ length: nHeads }, (_, h) => {
                const entropy = entropies[l]?.[h] ?? 0;
                const { color, glowIntensity } = entropyToColor(entropy, maxEntropy);
                const isSelected =
                  selectedHead?.layer === l && selectedHead?.head === h;

                return (
                  <button
                    key={h}
                    onClick={() => onHeadClick({ layer: l, head: h })}
                    onMouseEnter={(e) => handleMouseEnter(e, l, h)}
                    onMouseLeave={handleMouseLeave}
                    className={`no-select ${isSelected ? 'pulse-selected' : ''}`}
                    style={{
                      width: CELL_SIZE,
                      height: CELL_SIZE,
                      borderRadius: 4,
                      background: color,
                      border: isSelected
                        ? '2px solid var(--mech-accent)'
                        : '1px solid var(--mech-border-subtle)',
                      cursor: 'pointer',
                      transition: 'all 200ms ease',
                      boxShadow:
                        glowIntensity > 0.4
                          ? `0 0 ${6 + glowIntensity * 10}px rgba(255, 183, 77, ${glowIntensity * 0.3}),
                             0 0 ${12 + glowIntensity * 16}px rgba(255, 183, 77, ${glowIntensity * 0.12})`
                          : 'none',
                    }}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* ── Legend ─────────────────────────────────────── */}
      <div className="flex items-center gap-3" style={{ marginTop: 4 }}>
        <span className="text-readout" style={{ color: 'var(--mech-text-dim)' }}>
          Focused
        </span>
        <div
          style={{
            width: 80,
            height: 6,
            borderRadius: 3,
            background: 'linear-gradient(90deg, #1a1a1f, #ffb74d)',
          }}
        />
        <span className="text-readout" style={{ color: 'var(--mech-amber)' }}>
          Diffuse
        </span>
        <span
          className="text-readout"
          style={{ color: 'var(--mech-text-dim)', marginLeft: 'auto' }}
        >
          Click a cell to expand
        </span>
      </div>

      <Tooltip
        content={tooltip.content}
        x={tooltip.x}
        y={tooltip.y}
        visible={tooltip.visible}
      />
    </div>
  );
};

export default React.memo(AttentionHeadGrid);
