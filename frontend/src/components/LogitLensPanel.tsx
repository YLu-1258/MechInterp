import React, { useState } from 'react';
import type { LogitLensFrame } from '../types';

interface LogitLensPanelProps {
  frames: LogitLensFrame[];
  convergenceLayer: number | null;
  isStreaming: boolean;
  correctToken?: string;
}

const LogitLensPanel: React.FC<LogitLensPanelProps> = ({
  frames,
  convergenceLayer,
  isStreaming,
  correctToken = ' Paris',
}) => {
  const [expandedLayer, setExpandedLayer] = useState<number | null>(null);

  const handleLayerClick = (layer: number) => {
    setExpandedLayer(expandedLayer === layer ? null : layer);
  };

  // Find the first layer where correctToken appears as top-1
  const firstCorrectLayer = frames.findIndex(
    (f) => f.top_tokens[0]?.trim() === correctToken.trim()
  );

  return (
    <div className="flex flex-col gap-4" style={{ padding: '24px' }}>
      {/* ── Header ────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <h2 className="text-label" style={{ fontSize: 11 }}>
          Logit Lens
        </h2>
        {isStreaming && (
          <div
            className="text-readout"
            style={{ color: 'var(--mech-accent)', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'var(--mech-accent)',
                animation: 'pulse-halo 1.5s ease infinite',
              }}
            />
            Streaming...
          </div>
        )}
        <span
          className="text-readout"
          style={{ color: 'var(--mech-text-dim)', marginLeft: 'auto' }}
        >
          {frames.length} layers · predicting next token
        </span>
      </div>

      {/* ── Timeline (vertical) ────────────────────────── */}
      <div
        className="flex flex-col gap-1"
        style={{ maxHeight: 'calc(100vh - 240px)', overflowY: 'auto' }}
      >
        {frames.map((frame, idx) => {
          const isConvergence = convergenceLayer === frame.layer;
          const isFirstCorrect = firstCorrectLayer === idx;
          const isExpanded = expandedLayer === frame.layer;

          return (
            <div
              key={frame.layer}
              className={isConvergence ? 'convergence-glow' : 'fade-in'}
              style={{
                flexShrink: 0,
                animationDelay: isConvergence ? undefined : `${idx * 80}ms`,
                display: 'flex',
                flexDirection: 'column',
                background: isConvergence ? 'var(--mech-accent-dim)' : 'var(--mech-surface)',
                borderRadius: 8,
                border: `1px solid ${isConvergence ? 'var(--mech-accent)' : 'var(--mech-border-subtle)'}`,
                borderLeft: isFirstCorrect
                  ? '3px solid var(--mech-accent)'
                  : isConvergence
                    ? '3px solid var(--mech-accent)'
                    : '1px solid var(--mech-border-subtle)',
                overflow: 'hidden',
                cursor: 'pointer',
                transition: 'all 200ms ease',
              }}
              onClick={() => handleLayerClick(frame.layer)}
            >
              {/* Main row */}
              <div
                className="flex items-center gap-3"
                style={{
                  padding: '10px 16px',
                }}
              >
                {/* Layer label */}
                <span
                  className="text-readout"
                  style={{
                    color: isConvergence ? 'var(--mech-accent)' : 'var(--mech-text-dim)',
                    minWidth: 28,
                    fontWeight: isConvergence ? 700 : 500,
                  }}
                >
                  L{frame.layer}
                </span>

                {/* Top-1 token */}
                <span
                  className="font-mono"
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: isConvergence ? 'var(--mech-accent)' : 'var(--mech-text)',
                    padding: '2px 8px',
                    background: isConvergence
                      ? 'var(--mech-accent-dim)'
                      : 'var(--mech-surface-raised)',
                    borderRadius: 4,
                    minWidth: 80,
                  }}
                >
                  {frame.top_tokens[0]}
                </span>

                {/* Probability bar */}
                <div
                  className="flex items-center gap-2"
                  style={{ flex: 1, minWidth: 0 }}
                >
                  <div
                    style={{
                      flex: 1,
                      height: 6,
                      background: 'var(--mech-surface-raised)',
                      borderRadius: 3,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${frame.top_probs[0] * 100}%`,
                        height: '100%',
                        background: isConvergence
                          ? 'var(--mech-accent)'
                          : 'var(--mech-amber)',
                        borderRadius: 3,
                        transition: 'width 300ms ease',
                        boxShadow:
                          frame.top_probs[0] > 0.5
                            ? `0 0 8px ${isConvergence ? 'rgba(79, 195, 247, 0.4)' : 'rgba(255, 183, 77, 0.4)'}`
                            : 'none',
                      }}
                    />
                  </div>
                  <span
                    className="text-readout"
                    style={{
                      color: frame.top_probs[0] > 0.5 ? 'var(--mech-amber)' : 'var(--mech-text-dim)',
                      minWidth: 42,
                      textAlign: 'right',
                    }}
                  >
                    {(frame.top_probs[0] * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Convergence badge */}
                {isConvergence && (
                  <span
                    className="font-mono"
                    style={{
                      fontSize: 9,
                      fontWeight: 600,
                      color: 'var(--mech-accent)',
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      flexShrink: 0,
                    }}
                  >
                    ◆ CONVERGE
                  </span>
                )}

                {/* Expand indicator */}
                <span
                  className="text-readout"
                  style={{
                    color: 'var(--mech-text-dim)',
                    transform: isExpanded ? 'rotate(180deg)' : 'rotate(0)',
                    transition: 'transform 200ms ease',
                    fontSize: 10,
                  }}
                >
                  ▼
                </span>
              </div>

              {/* Expanded top-K view */}
              {isExpanded && (
                <div
                  style={{
                    borderTop: '1px solid var(--mech-border-subtle)',
                    padding: '12px 16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                    animation: 'fade-slide-in 200ms ease',
                  }}
                >
                  {frame.top_tokens.map((token, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3"
                      style={{ fontSize: 12 }}
                    >
                      <span
                        className="text-readout"
                        style={{
                          color: 'var(--mech-text-dim)',
                          minWidth: 16,
                          textAlign: 'right',
                        }}
                      >
                        {i + 1}.
                      </span>
                      <span
                        className="font-mono"
                        style={{
                          color: i === 0 ? 'var(--mech-text)' : 'var(--mech-text-secondary)',
                          fontWeight: i === 0 ? 600 : 400,
                          minWidth: 80,
                        }}
                      >
                        {token}
                      </span>
                      <div
                        style={{
                          flex: 1,
                          height: 4,
                          background: 'var(--mech-surface-raised)',
                          borderRadius: 2,
                          overflow: 'hidden',
                          maxWidth: 200,
                        }}
                      >
                        <div
                          style={{
                            width: `${frame.top_probs[i] * 100}%`,
                            height: '100%',
                            background:
                              i === 0 ? 'var(--mech-amber)' : 'var(--mech-text-dim)',
                            borderRadius: 2,
                          }}
                        />
                      </div>
                      <span
                        className="text-readout"
                        style={{
                          color: 'var(--mech-text-dim)',
                          minWidth: 42,
                          textAlign: 'right',
                        }}
                      >
                        {(frame.top_probs[i] * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Loading indicator ────────────────────────── */}
      {isStreaming && frames.length > 0 && (
        <div className="scanline-loader" style={{ height: 2, borderRadius: 1 }} />
      )}
    </div>
  );
};

export default React.memo(LogitLensPanel);
