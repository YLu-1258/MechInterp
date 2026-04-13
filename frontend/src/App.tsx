import React, { useState, useCallback, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TokenHeatmap from './components/TokenHeatmap';
import AttentionHeadGrid from './components/AttentionHeadGrid';
import AttentionDetail from './components/AttentionDetail';
import LogitLensPanel from './components/LogitLensPanel';
import PatchDiffView from './components/PatchDiffView';
import { useLogitLens } from './hooks/useLogitLens';
import { useAnalyze } from './hooks/useAnalyze';
import type { TabId, SupportedModel, SelectedHead } from './types';
const DEFAULT_MODEL_INFO = {
  name: 'Ready for Analysis',
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
};

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

  // ── Tab State ──────────────────────────────────
  const [activeTab, setActiveTab] = useState<TabId>('tokens');

  // ── Analysis Data (mock/real) ────────────────
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [selectedHead, setSelectedHead] = useState<SelectedHead | null>(null);

  const logitLens = useLogitLens();
  const analyze = useAnalyze();

  // The backend already returns attention_patterns.data as a nested 4D array when serialized
  const attentionPatterns = React.useMemo(() => {
    if (!analyze.result?.attention_patterns) return [];
    return analyze.result.attention_patterns.data as unknown as number[][][][];
  }, [analyze.result?.attention_patterns]);

  const handleAnalyze = useCallback(() => {
    // Start WebSocket stream for Logit Lens
    logitLens.startStreaming(prompt, model);
    
    // Start REST analyze
    analyze.analyzePrompt(prompt, model);
  }, [prompt, model, logitLens, analyze]);

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

  // Use real data if available, fallback to empty arrays
  const tokens = analyze.result?.tokens || [];
  const attributionScores = analyze.result?.logit_attribution.by_token || [];
  const modelInfo = analyze.result?.model_info || DEFAULT_MODEL_INFO;
  const isAnyLoading = logitLens.isStreaming || analyze.isLoading;

  return (
    <div className="flex h-full w-full" style={{ background: 'var(--mech-bg)' }}>
      {/* ── Left Sidebar ────────────────────────────── */}
      <Sidebar
        model={model}
        onModelChange={setModel}
        prompt={prompt}
        onPromptChange={setPrompt}
        onAnalyze={handleAnalyze}
        isLoading={isAnyLoading}
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
              {modelInfo.name}
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
              {modelInfo.n_layers}L · {modelInfo.n_heads}H · {modelInfo.d_model}d
            </span>
          </div>
        </nav>

        {/* Active View */}
        <div
          className="flex-1"
          style={{ overflow: 'auto', position: 'relative' }}
        >
          {/* Analyze Error Message */}
          {analyze.error && (
            <div style={{ padding: 24, paddingBottom: 0 }}>
              <div style={{ padding: 16, background: 'rgba(239, 83, 80, 0.1)', border: '1px solid var(--mech-red)', borderRadius: 8 }}>
                <span className="font-mono text-readout" style={{ color: 'var(--mech-red)' }}>Error: {analyze.error}</span>
              </div>
            </div>
          )}

          {activeTab === 'tokens' && (
            <TokenHeatmap
              tokens={tokens}
              attributionScores={attributionScores}
              selectedPosition={selectedToken}
              onTokenClick={handleTokenClick}
              isLoading={isAnyLoading}
            />
          )}

          {activeTab === 'heads' && (
            <AttentionHeadGrid
              attentionPatterns={attentionPatterns}
              nLayers={modelInfo.n_layers}
              nHeads={modelInfo.n_heads}
              selectedHead={selectedHead}
              onHeadClick={handleHeadClick}
            />
          )}

          {activeTab === 'lens' && (
            <LogitLensPanel
              frames={logitLens.frames}
              convergenceLayer={logitLens.convergenceLayer}
              isStreaming={logitLens.isStreaming}
            />
          )}

          {activeTab === 'patch' && (
            <PatchDiffView
              nLayers={modelInfo.n_layers}
              nHeads={modelInfo.n_heads}
              model={model}
            />
          )}
        </div>

        {/* ── Right Panel (Attention Detail) ──────────── */}
        {selectedHead && activeTab === 'heads' && (
          <AttentionDetail
            pattern={attentionPatterns[selectedHead.layer]?.[selectedHead.head] || []}
            tokens={tokens}
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
