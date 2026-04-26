import React, { useState, useEffect, useRef } from 'react';
import { Layers, ChevronDown, ChevronRight, X, Calculator, Upload } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';
const withNoCache = (path: string) => `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;

interface Box {
  id: string;
  symbolName: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  verificationScore?: number;
  color: string;
  source?: string;
  rotation?: number;
  scale?: number;
  mirrored?: boolean;
  coverage?: number;
  purity?: number;
  contextPurity?: number;
  colorSimilarity?: number;
  analysisId?: string;
  analysisGeneratedUtc?: string;
  analysisSession?: string;
  sourcePdf?: string;
  hiddenLayersUsed?: string[];
}

interface ResultGroup {
  name: string;
  count: number;
  color: string;
}

interface ResultsPanelProps {
  results: ResultGroup[];
  boxes: Box[];
  focusedBoxId?: string | null;
  onFocusBox?: (id: string) => void;
  onRejectBox: (id: string) => void;
  onTemplateUploaded?: () => void;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  results,
  boxes,
  focusedBoxId,
  onFocusBox,
  onRejectBox,
  onTemplateUploaded,
}) => {
  const [activeTab, setActiveTab] = useState<'correction' | 'cost'>('correction');
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [minConfidence, setMinConfidence] = useState(0);
  const [prices, setPrices] = useState<Record<string, number>>({});
  const uploadRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  // Init prices for new results
  useEffect(() => {
    setPrices(prev => {
      const next = { ...prev };
      results.forEach(r => {
        if (next[r.name] === undefined) next[r.name] = 0;
      });
      return next;
    });
  }, [results]);

  const toggleGroup = (name: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const getDisplayConfidence = (box: Box) => box.verificationScore ?? box.confidence;

  // Group boxes by symbolName, applying confidence filter
  const filteredBoxes = boxes.filter(b => getDisplayConfidence(b) * 100 >= minConfidence);
  const boxesBySymbol: Record<string, Box[]> = {};
  filteredBoxes.forEach(b => {
    if (!boxesBySymbol[b.symbolName]) boxesBySymbol[b.symbolName] = [];
    boxesBySymbol[b.symbolName].push(b);
  });

  const totalSum = results.reduce((acc, r) => {
    const activeCount = (boxesBySymbol[r.name] || []).length;
    return acc + activeCount * (prices[r.name] || 0);
  }, 0);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(withNoCache('/api/templates/upload'), {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });
      if (res.ok) {
        onTemplateUploaded?.();
      } else {
        alert('Błąd uploadu wzorca.');
      }
    } catch {
      alert('Błąd połączenia z backendem.');
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = '';
    }
  };

  const empty = results.length === 0;

  return (
    <div className="results-panel">
      {/* Header + Tabs */}
      <div className="sidebar-header" style={{ paddingBottom: 0 }}>
        <div className="flex-row gap-2" style={{ marginBottom: 16 }}>
          <Layers size={18} color="var(--accent-gold)" />
          <h2 className="text-sm" style={{ fontWeight: 700 }}>Wyniki Analizy</h2>
          {boxes.length > 0 && (
            <span className="badge badge-gold" style={{ marginLeft: 'auto' }}>
              {filteredBoxes.length} detekcji
            </span>
          )}
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid var(--border-light)', gap: 2 }}>
          {(['correction', 'cost'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: '8px 0',
                background: 'none',
                border: 'none',
                borderBottom: activeTab === tab ? '2px solid var(--accent-gold)' : '2px solid transparent',
                color: activeTab === tab ? 'var(--accent-gold)' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: 12,
                cursor: 'pointer',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                transition: 'all 0.2s',
              }}
            >
              {tab === 'correction' ? '🔍 Korekta' : '💰 Kosztorys'}
            </button>
          ))}
        </div>
      </div>

      {empty ? (
        <div className="sidebar-content" style={{ alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <Layers size={36} style={{ opacity: 0.2, marginBottom: 12 }} />
          <p className="text-sm text-muted">Brak wyników.<br />Wgraj plan i uruchom analizę.</p>
        </div>
      ) : (
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>

          {/* ── KOREKTA TAB ── */}
          {activeTab === 'correction' && (
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>

              {/* Confidence Slider */}
              <div className="card" style={{ padding: '12px 16px' }}>
                <div className="flex-row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
                  <span className="text-xs text-muted" style={{ textTransform: 'uppercase', fontWeight: 600 }}>
                    Min. Pewność
                  </span>
                  <span className="text-xs" style={{ color: 'var(--accent-gold)', fontWeight: 700 }}>
                    {minConfidence}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={90}
                  step={5}
                  value={minConfidence}
                  onChange={e => setMinConfidence(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-gold)' }}
                />
                <div className="flex-row" style={{ justifyContent: 'space-between', marginTop: 4 }}>
                  <span className="text-xs text-muted">0%</span>
                  <span className="text-xs text-muted">90%</span>
                </div>
              </div>

              {/* Upload ręcznego wzorca */}
              <div>
                <button
                  className="btn-secondary flex-row gap-2"
                  style={{ width: '100%', justifyContent: 'center' }}
                  onClick={() => uploadRef.current?.click()}
                  disabled={uploading}
                >
                  <Upload size={14} />
                  {uploading ? 'Wgrywanie...' : 'Dodaj wzorzec PNG'}
                </button>
                <input
                  ref={uploadRef}
                  type="file"
                  accept=".png"
                  style={{ display: 'none' }}
                  onChange={handleUpload}
                />
              </div>

              {/* Accordion z grupami */}
              {results.map(group => {
                const groupBoxes = boxesBySymbol[group.name] || [];
                const isOpen = expandedGroups.has(group.name);
                return (
                  <div key={group.name} className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    {/* Nagłówek grupy */}
                    <button
                      onClick={() => toggleGroup(group.name)}
                      style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                        padding: '10px 14px',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        borderLeft: `4px solid ${group.color}`,
                        textAlign: 'left',
                      }}
                    >
                      {isOpen ? <ChevronDown size={14} color="var(--text-muted)" /> : <ChevronRight size={14} color="var(--text-muted)" />}
                      <span
                        className="text-xs"
                        style={{
                          flex: 1,
                          fontWeight: 700,
                          color: 'var(--text-primary)',
                          wordBreak: 'break-all',
                          lineHeight: 1.3,
                        }}
                      >
                        {group.name}
                      </span>
                      <span
                        style={{
                          background: group.color + '33',
                          color: group.color,
                          borderRadius: 4,
                          padding: '2px 8px',
                          fontWeight: 700,
                          fontSize: 12,
                          minWidth: 28,
                          textAlign: 'center',
                        }}
                      >
                        {groupBoxes.length}
                      </span>
                    </button>

                    {/* Rozwinięta lista detekcji */}
                    {isOpen && (
                      <div style={{ borderTop: '1px solid var(--border-light)' }}>
                        {groupBoxes.length === 0 ? (
                          <p className="text-xs text-muted" style={{ padding: '10px 14px' }}>
                            Wszystkie odfiltrowane przez próg pewności.
                          </p>
                        ) : (
                          groupBoxes.map(box => {
                            const isFocused = focusedBoxId === box.id;
                            const displayConfidence = getDisplayConfidence(box);
                            return (
                            <div
                              key={box.id}
                              onClick={() => onFocusBox?.(box.id)}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8,
                                padding: '7px 14px',
                                borderBottom: '1px solid var(--border-light)',
                                background: isFocused ? 'rgba(249,115,22,0.1)' : 'transparent',
                                borderLeft: isFocused ? '3px solid var(--accent-orange)' : '3px solid transparent',
                                cursor: 'pointer',
                                transition: 'all 0.15s',
                              }}
                            >
                              {/* Confidence chip */}
                              <span
                                style={{
                                  fontSize: 11,
                                  fontWeight: 700,
                                  color: displayConfidence < 0.55 ? '#f59e0b' : '#10b981',
                                  minWidth: 36,
                                }}
                              >
                                {(displayConfidence * 100).toFixed(0)}%
                              </span>
                              {/* Position */}
                              <span className="text-xs text-muted" style={{ flex: 1 }}>
                                x:{box.x} y:{box.y}
                              </span>
                              {/* Odrzuć */}
                              <button
                                onClick={e => { e.stopPropagation(); onRejectBox(box.id); }}
                                title="Odrzuć (fałszywe trafienie)"
                                style={{
                                  background: 'none',
                                  border: 'none',
                                  cursor: 'pointer',
                                  color: '#ef4444',
                                  padding: 2,
                                  display: 'flex',
                                  opacity: 0.5,
                                  transition: 'opacity 0.15s',
                                }}
                                onMouseEnter={e => (e.currentTarget.style.opacity = '1')}
                                onMouseLeave={e => (e.currentTarget.style.opacity = '0.5')}
                              >
                                <X size={14} />
                              </button>
                            </div>
                          );
                          })
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* ── KOSZTORYS TAB ── */}
          {activeTab === 'cost' && (
            <>
              <div style={{ padding: 16, flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {results.map((group) => {
                  const activeCount = (boxesBySymbol[group.name] || []).length;
                  return (
                    <div key={group.name} className="estimate-card">
                      <div className="estimate-card-stripe" style={{ backgroundColor: group.color }} />
                      <div className="estimate-card-content">
                        <div className="estimate-card-title">{group.name}</div>
                        <div className="estimate-inputs">
                          <div className="estimate-input-group">
                            <label className="estimate-input-label">ILOŚĆ</label>
                            <input
                              type="number"
                              className="estimate-input"
                              value={activeCount}
                              readOnly
                            />
                          </div>
                          <div className="estimate-input-group">
                            <label className="estimate-input-label">CENA NETTO (PLN)</label>
                            <input
                              type="number"
                              className="estimate-input"
                              value={prices[group.name] || 0}
                              onChange={e => {
                                const val = parseFloat(e.target.value);
                                setPrices(prev => ({ ...prev, [group.name]: isNaN(val) ? 0 : val }));
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="estimate-total-footer">
                <div className="flex-row gap-2">
                  <Calculator size={16} color="var(--text-muted)" />
                  <span className="text-xs text-muted" style={{ textTransform: 'uppercase' }}>Suma Całkowita</span>
                </div>
                <span style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-gold)' }}>
                  {totalSum.toFixed(2)} PLN
                </span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
