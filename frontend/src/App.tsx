import React, { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TokenHeatmap from './components/TokenHeatmap';
import AttentionHeadGrid from './components/AttentionHeadGrid';
import AttentionDetail from './components/AttentionDetail';
import LogitLensPanel from './components/LogitLensPanel';
import PatchDiffView from './components/PatchDiffView';
import type { TabId, SupportedModel, SelectedHead } from './types';
import {
  MOCK_TOKENS,
  MOCK_ATTRIBUTION_SCORES,
  MOCK_ATTENTION_PATTERNS,
  MOCK_LOGIT_LENS_FRAMES,
  MOCK_CONVERGENCE_LAYER,
  MOCK_MODEL_INFO,
} from './mockData';

const TABS: { id: TabId; label: string }[] = [
  { id: 'tokens', label: 'Tokens' },
  { id: 'heads', label: 'Heads' },
  { id: 'lens', label: 'Logit Lens' },
  { id: 'patch', label: 'Patch Diff' },
];

const App: React.FC = () => {
  // ── Theme ──────────────────────────────────────
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggleTheme = useCallback(() => setIsDark((d) => !d), []);

  // ── Sidebar State ──────────────────────────────
  const [model, setModel] = useState<SupportedModel>('gpt2-small');
  const [prompt, setPrompt] = useState('The capital of France is');
  const [isLoading, setIsLoading] = useState(false);

  // ── Tab State ──────────────────────────────────
  const [activeTab, setActiveTab] = useState<TabId>('tokens');

  // ── Analysis Data (mock) ───────────────────────
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [selectedHead, setSelectedHead] = useState<SelectedHead | null>(null);

  // ── Handlers ───────────────────────────────────
  const handleAnalyze = useCallback(() => {
    setIsLoading(true);
    // Simulate a loading delay for the scan-line effect
    setTimeout(() => setIsLoading(false), 1500);
  }, []);

  const handleTokenClick = useCallback((pos: number) => {
    setSelectedToken((prev) => (prev === pos ? null : pos));
  }, []);

  const handleHeadClick = useCallback((head: SelectedHead) => {
    setSelectedHead((prev) =>
      prev?.layer === head.layer && prev?.head === head.head ? null : head
    );
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedHead(null);
  }, []);

  return (
    <div className="flex h-full w-full" style={{ background: 'var(--mech-bg)' }}>
      {/* ── Left Sidebar ────────────────────────────── */}
      <Sidebar
        model={model}
        onModelChange={setModel}
        prompt={prompt}
        onPromptChange={setPrompt}
        onAnalyze={handleAnalyze}
        isLoading={isLoading}
        isDark={isDark}
        onToggleTheme={toggleTheme}
      />

      {/* ── Main Content Area ───────────────────────── */}
      <main className="flex flex-col flex-1 h-full" style={{ position: 'relative', overflow: 'hidden' }}>
        {/* Tab Bar */}
        <nav className="tab-bar no-select">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}

          {/* Model info badge */}
          <div
            className="flex items-center gap-2"
            style={{ marginLeft: 'auto', padding: '0 8px' }}
          >
            <span
              className="text-readout"
              style={{ color: 'var(--mech-text-dim)' }}
            >
              {MOCK_MODEL_INFO.name}
            </span>
            <span
              className="text-readout"
              style={{
                color: 'var(--mech-text-dim)',
                padding: '2px 6px',
                background: 'var(--mech-surface-raised)',
                borderRadius: 4,
                fontSize: 9,
              }}
            >
              {MOCK_MODEL_INFO.n_layers}L · {MOCK_MODEL_INFO.n_heads}H · {MOCK_MODEL_INFO.d_model}d
            </span>
          </div>
        </nav>

        {/* Active View */}
        <div
          className="flex-1"
          style={{ overflow: 'auto', position: 'relative' }}
        >
          {activeTab === 'tokens' && (
            <TokenHeatmap
              tokens={MOCK_TOKENS}
              attributionScores={MOCK_ATTRIBUTION_SCORES}
              selectedPosition={selectedToken}
              onTokenClick={handleTokenClick}
              isLoading={isLoading}
            />
          )}

          {activeTab === 'heads' && (
            <AttentionHeadGrid
              attentionPatterns={MOCK_ATTENTION_PATTERNS}
              nLayers={MOCK_MODEL_INFO.n_layers}
              nHeads={MOCK_MODEL_INFO.n_heads}
              selectedHead={selectedHead}
              onHeadClick={handleHeadClick}
            />
          )}

          {activeTab === 'lens' && (
            <LogitLensPanel
              frames={MOCK_LOGIT_LENS_FRAMES}
              convergenceLayer={MOCK_CONVERGENCE_LAYER}
              isStreaming={false}
            />
          )}

          {activeTab === 'patch' && (
            <PatchDiffView
              nLayers={MOCK_MODEL_INFO.n_layers}
              nHeads={MOCK_MODEL_INFO.n_heads}
            />
          )}
        </div>

        {/* ── Right Panel (Attention Detail) ──────────── */}
        {selectedHead && activeTab === 'heads' && (
          <AttentionDetail
            pattern={MOCK_ATTENTION_PATTERNS[selectedHead.layer][selectedHead.head]}
            tokens={MOCK_TOKENS}
            layer={selectedHead.layer}
            head={selectedHead.head}
            onClose={handleCloseDetail}
          />
        )}
      </main>
    </div>
  );
};

export default App;
