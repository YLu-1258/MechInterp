import React from 'react';
import ThemeToggle from './ThemeToggle';
import type { SupportedModel } from '../types';

interface SidebarProps {
  model: SupportedModel;
  onModelChange: (model: SupportedModel) => void;
  prompt: string;
  onPromptChange: (prompt: string) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  isDark: boolean;
  onToggleTheme: () => void;
}

const MODELS: { value: SupportedModel; label: string }[] = [
  { value: 'gpt2-small', label: 'GPT-2 Small' },
  { value: 'gpt2-medium', label: 'GPT-2 Medium' },
  { value: 'qwen3-0.5b', label: 'Qwen3 0.5B' },
  { value: 'qwen3-1.5b', label: 'Qwen3 1.5B' },
];

const Sidebar: React.FC<SidebarProps> = ({
  model,
  onModelChange,
  prompt,
  onPromptChange,
  onAnalyze,
  isLoading,
  isDark,
  onToggleTheme,
}) => {
  return (
    <aside
      className="flex flex-col h-full no-select"
      style={{
        width: 240,
        minWidth: 240,
        background: 'var(--mech-surface)',
        borderRight: '1px solid var(--mech-border)',
        padding: '20px 16px',
        gap: 20,
      }}
    >
      {/* ── Logo / Title ────────────────────────── */}
      <div>
        <h1
          className="font-mono"
          style={{
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: '0.08em',
            color: 'var(--mech-accent)',
            marginBottom: 2,
          }}
        >
          MECHINTERP
        </h1>
        <p
          className="font-mono"
          style={{
            fontSize: 9,
            letterSpacing: '0.12em',
            color: 'var(--mech-text-dim)',
            textTransform: 'uppercase',
          }}
        >
          Interpretability Toolkit
        </p>
      </div>

      {/* ── Divider ─────────────────────────────── */}
      <div style={{ height: 1, background: 'var(--mech-border)' }} />

      {/* ── Model Selector ──────────────────────── */}
      <div className="flex flex-col gap-2">
        <label className="text-label">Model</label>
        <select
          value={model}
          onChange={(e) => onModelChange(e.target.value as SupportedModel)}
          disabled={isLoading}
          style={{ fontSize: 12, padding: '8px 10px' }}
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* ── Prompt Input ────────────────────────── */}
      <div className="flex flex-col gap-2 flex-1">
        <label className="text-label">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          placeholder="Enter prompt to analyze..."
          disabled={isLoading}
          rows={6}
          style={{
            fontSize: 12,
            lineHeight: 1.6,
            flex: 1,
            minHeight: 120,
          }}
        />
      </div>

      {/* ── Run Button ──────────────────────────── */}
      <button
        className={`btn-primary ${isLoading ? 'scanline-loader' : ''}`}
        onClick={onAnalyze}
        disabled={isLoading || !prompt.trim()}
      >
        {isLoading ? 'Analyzing...' : 'Analyze'}
      </button>

      {/* ── Spacer ──────────────────────────────── */}
      <div className="flex-1" />

      {/* ── Theme Toggle ────────────────────────── */}
      <div style={{ borderTop: '1px solid var(--mech-border)', paddingTop: 16 }}>
        <ThemeToggle isDark={isDark} onToggle={onToggleTheme} />
      </div>
    </aside>
  );
};

export default React.memo(Sidebar);
