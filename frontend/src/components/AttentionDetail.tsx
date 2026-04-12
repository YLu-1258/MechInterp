import React, { useState, useCallback, useEffect, useRef } from 'react';
import Tooltip from './Tooltip';
import { attentionToAmber } from '../utils/colorScale';

interface AttentionDetailProps {
  pattern: number[][]; // [seq_len, seq_len]
  tokens: string[];
  layer: number;
  head: number;
  onClose: () => void;
}

const AttentionDetail: React.FC<AttentionDetailProps> = ({
  pattern,
  tokens,
  layer,
  head,
  onClose,
}) => {
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: React.ReactNode;
  }>({ visible: false, x: 0, y: 0, content: null });

  const panelRef = useRef<HTMLDivElement>(null);
  const [isClosing, setIsClosing] = useState(false);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const handleClose = useCallback(() => {
    setIsClosing(true);
    setTimeout(onClose, 200);
  }, [onClose]);

  // Click outside to close
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) handleClose();
    },
    [handleClose]
  );

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, q: number, k: number, weight: number) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setTooltip({
        visible: true,
        x: rect.left + rect.width / 2,
        y: rect.top,
        content: (
          <div className="flex flex-col gap-0.5">
            <span style={{ color: 'var(--mech-accent)' }}>
              Q: "{tokens[q]}" → K: "{tokens[k]}"
            </span>
            <span style={{ color: 'var(--mech-amber)', fontWeight: 600 }}>
              {weight.toFixed(4)}
            </span>
          </div>
        ),
      });
    },
    [tokens]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }));
  }, []);

  const seqLen = tokens.length;
  const CELL_SIZE = seqLen > 20 ? 18 : seqLen > 12 ? 24 : 32;
  const LABEL_WIDTH = 56;

  return (
    <div
      onClick={handleBackdropClick}
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        width: '100%',
        zIndex: 100,
        display: 'flex',
        justifyContent: 'flex-end',
      }}
    >
      <div
        ref={panelRef}
        className={isClosing ? 'panel-exit' : 'panel-enter'}
        style={{
          width: Math.min(500, LABEL_WIDTH + CELL_SIZE * seqLen + 60),
          maxWidth: '75vw',
          height: '100%',
          background: 'var(--mech-surface)',
          borderLeft: '1px solid var(--mech-border)',
          boxShadow: '-8px 0 24px rgba(0, 0, 0, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* ── Header ────────────────────────────────── */}
        <div
          className="flex items-center justify-between no-select"
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--mech-border)',
            flexShrink: 0,
          }}
        >
          <div>
            <h3
              className="font-mono"
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: 'var(--mech-accent)',
              }}
            >
              Layer {layer} · Head {head}
            </h3>
            <p
              className="text-readout"
              style={{ color: 'var(--mech-text-dim)', marginTop: 2 }}
            >
              {seqLen}×{seqLen} attention pattern
            </p>
          </div>
          <button
            onClick={handleClose}
            className="font-mono"
            style={{
              background: 'var(--mech-surface-raised)',
              border: '1px solid var(--mech-border)',
              borderRadius: 6,
              padding: '6px 12px',
              cursor: 'pointer',
              fontSize: 11,
              color: 'var(--mech-text-secondary)',
            }}
          >
            ESC
          </button>
        </div>

        {/* ── Matrix ────────────────────────────────── */}
        <div
          style={{
            flex: 1,
            overflow: 'auto',
            padding: 20,
          }}
        >
          <div style={{ display: 'inline-block' }}>
            {/* Column token labels (key tokens) */}
            <div className="flex" style={{ marginLeft: LABEL_WIDTH, marginBottom: 4 }}>
              {tokens.map((tok, k) => (
                <div
                  key={k}
                  className="text-readout"
                  style={{
                    width: CELL_SIZE,
                    textAlign: 'center',
                    transform: 'rotate(-45deg)',
                    transformOrigin: 'bottom center',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    color: 'var(--mech-text-dim)',
                    fontSize: 9,
                    height: 40,
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'center',
                  }}
                >
                  {tok.trim() || '·'}
                </div>
              ))}
            </div>

            {/* Rows (query tokens) */}
            {pattern.map((row, q) => (
              <div key={q} className="flex items-center">
                {/* Row label */}
                <div
                  className="text-readout"
                  style={{
                    width: LABEL_WIDTH,
                    textAlign: 'right',
                    paddingRight: 8,
                    color: 'var(--mech-text-dim)',
                    fontSize: 9,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {tokens[q].trim() || '·'}
                </div>

                {/* Attention cells */}
                {row.map((weight, k) => {
                  const { color, glowIntensity } = attentionToAmber(weight);
                  const isZero = k > q; // Autoregressive mask

                  return (
                    <div
                      key={k}
                      onMouseEnter={(e) =>
                        !isZero && handleMouseEnter(e, q, k, weight)
                      }
                      onMouseLeave={handleMouseLeave}
                      style={{
                        width: CELL_SIZE,
                        height: CELL_SIZE,
                        background: isZero ? 'transparent' : color,
                        border: `0.5px solid ${isZero ? 'transparent' : 'var(--mech-border-subtle)'}`,
                        borderRadius: 2,
                        boxShadow:
                          !isZero && glowIntensity > 0.5
                            ? `0 0 ${4 + glowIntensity * 8}px rgba(255, 183, 77, ${glowIntensity * 0.3})`
                            : 'none',
                        transition: 'box-shadow 200ms ease',
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* ── Legend ─────────────────────────────────── */}
        <div
          className="flex items-center gap-3 no-select"
          style={{
            padding: '12px 20px',
            borderTop: '1px solid var(--mech-border)',
            flexShrink: 0,
          }}
        >
          <span className="text-readout" style={{ color: 'var(--mech-text-dim)' }}>
            0.0
          </span>
          <div
            style={{
              flex: 1,
              height: 6,
              borderRadius: 3,
              background: 'linear-gradient(90deg, #1a1a1f, #ffb74d)',
              maxWidth: 100,
            }}
          />
          <span className="text-readout" style={{ color: 'var(--mech-amber)' }}>
            1.0
          </span>
        </div>
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

export default React.memo(AttentionDetail);
