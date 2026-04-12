import React, { useState, useCallback } from 'react';
import Tooltip from './Tooltip';
import { attributionToAmber } from '../utils/colorScale';

interface TokenHeatmapProps {
  tokens: string[];
  attributionScores: number[];
  selectedPosition: number | null;
  onTokenClick: (position: number) => void;
  isLoading: boolean;
}

const TokenHeatmap: React.FC<TokenHeatmapProps> = ({
  tokens,
  attributionScores,
  selectedPosition,
  onTokenClick,
  isLoading,
}) => {
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: React.ReactNode;
  }>({ visible: false, x: 0, y: 0, content: null });

  const maxAbs = Math.max(...attributionScores.map(Math.abs), 0.01);

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, idx: number) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setTooltip({
        visible: true,
        x: rect.left + rect.width / 2,
        y: rect.top,
        content: (
          <div className="flex flex-col gap-1">
            <span style={{ color: 'var(--mech-text-secondary)' }}>
              Token [{idx}]
            </span>
            <span style={{ color: 'var(--mech-text)', fontWeight: 600 }}>
              "{tokens[idx]}"
            </span>
            <span style={{ color: 'var(--mech-amber)' }}>
              Score: {attributionScores[idx].toFixed(3)}
            </span>
          </div>
        ),
      });
    },
    [tokens, attributionScores]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  }, []);

  return (
    <div className="flex flex-col gap-4" style={{ padding: '24px' }}>
      {/* ── Section Header ────────────────────────── */}
      <div className="flex items-center gap-3">
        <h2
          className="text-label"
          style={{ fontSize: 11, color: 'var(--mech-text-secondary)' }}
        >
          Token Attribution
        </h2>
        <div
          className="flex items-center gap-2"
          style={{ marginLeft: 'auto', fontSize: 10 }}
        >
          <span className="text-readout" style={{ color: 'var(--mech-text-dim)' }}>
            {tokens.length} tokens
          </span>
          <div
            style={{
              width: 60,
              height: 4,
              borderRadius: 2,
              background: 'linear-gradient(90deg, #1a1a1f, #ffb74d)',
            }}
          />
          <span className="text-readout" style={{ color: 'var(--mech-amber)' }}>
            max
          </span>
        </div>
      </div>

      {/* ── Token Chips ────────────────────────────── */}
      <div
        className={`flex flex-wrap gap-2 ${isLoading ? 'scanline-loader' : ''}`}
        style={{
          position: 'relative',
          minHeight: 40,
          opacity: isLoading ? 0.6 : 1,
          transition: 'opacity 200ms ease',
        }}
      >
        {tokens.map((token, idx) => {
          const { color, glowIntensity, textColor } = attributionToAmber(
            attributionScores[idx],
            maxAbs
          );
          const isSelected = selectedPosition === idx;

          return (
            <button
              key={idx}
              role="button"
              tabIndex={0}
              aria-label={`Token "${token}", attribution score ${attributionScores[idx].toFixed(3)}`}
              className={`font-mono no-select ${isSelected ? 'pulse-selected' : ''}`}
              onClick={() => !isLoading && onTokenClick(idx)}
              onMouseEnter={(e) => handleMouseEnter(e, idx)}
              onMouseLeave={handleMouseLeave}
              onKeyDown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && !isLoading) {
                  e.preventDefault();
                  onTokenClick(idx);
                }
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                padding: '6px 12px',
                borderRadius: 20,
                fontSize: 12,
                fontWeight: 500,
                cursor: isLoading ? 'not-allowed' : 'pointer',
                border: isSelected
                  ? '2px solid var(--mech-accent)'
                  : '1px solid transparent',
                background: color,
                color: textColor,
                '--glow-intensity': glowIntensity,
                boxShadow: glowIntensity > 0.3
                  ? `0 0 ${4 + glowIntensity * 12}px rgba(255, 183, 77, ${glowIntensity * 0.35}),
                     0 0 ${8 + glowIntensity * 20}px rgba(255, 183, 77, ${glowIntensity * 0.15})`
                  : 'none',
                transition: 'all 200ms ease',
                position: 'relative',
              } as React.CSSProperties}
            >
              {token === ' ' ? '·' : token.replace(/^ /, '␣')}
            </button>
          );
        })}
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

export default React.memo(TokenHeatmap);
