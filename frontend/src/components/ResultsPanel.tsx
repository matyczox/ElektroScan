import React, { useState, useRef } from 'react';
import { Layers, ChevronDown, ChevronRight, X, Upload, Edit3, Save, Download } from 'lucide-react';
import { apiFetch, projectApiPath, readApiError } from '../api';
import { formatSymbolLabel } from '../symbolLabels';

interface Box {
  id: string;
  symbolName: string;
  x: number;
  y: number;
  width: number;
  height: number;
  visualBBox?: [number, number, number, number] | null;
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
  reason?: string;
  note?: string;
  reviewStatus?: 'unchecked' | 'accepted' | 'wrong' | 'manual_check';
}

interface ResultGroup {
  name: string;
  count: number;
  color: string;
}

interface AnalysisContext {
  analysisId?: string;
  generatedAtUtc?: string;
  sessionId?: string;
  sourcePdf?: string;
}

interface ResultsPanelProps {
  results: ResultGroup[];
  boxes: Box[];
  analysisContext?: AnalysisContext | null;
  focusedBoxId?: string | null;
  onFocusBox?: (id: string) => void;
  onRejectBox: (id: string) => void;
  onChangeBoxSymbol?: (id: string, symbolName: string) => void;
  onUpdateBox?: (id: string, patch: Partial<Box>) => void;
  onRenameSymbol?: (currentName: string, nextName: string) => Promise<string | void> | string | void;
  symbolNames?: string[];
  symbolLabels?: Record<string, string>;
  projectId?: string;
  onTemplateUploaded?: () => void;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  results,
  boxes,
  analysisContext,
  focusedBoxId,
  onFocusBox,
  onRejectBox,
  onChangeBoxSymbol,
  onUpdateBox,
  onRenameSymbol,
  symbolNames = [],
  symbolLabels = {},
  projectId,
  onTemplateUploaded,
}) => {
  const [activeTab, setActiveTab] = useState<'correction' | 'export'>('correction');
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const uploadRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [renamingGroup, setRenamingGroup] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [renameBusyGroup, setRenameBusyGroup] = useState<string | null>(null);
  const [editingBoxId, setEditingBoxId] = useState<string | null>(null);
  const getSymbolDisplayName = (name: string) => symbolLabels[name] || formatSymbolLabel(name);
  const reviewLabels: Record<NonNullable<Box['reviewStatus']>, string> = {
    unchecked: 'Nieopisane',
    accepted: 'OK',
    wrong: 'Błąd',
    manual_check: 'Sprawdzić',
  };

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
  const filteredBoxes = boxes;
  const boxesBySymbol: Record<string, Box[]> = {};
  filteredBoxes.forEach(b => {
    if (!boxesBySymbol[b.symbolName]) boxesBySymbol[b.symbolName] = [];
    boxesBySymbol[b.symbolName].push(b);
  });

  const exportRows = Array.from(
    results.reduce((map, group) => {
      const activeCount = filteredBoxes.length > 0
        ? (boxesBySymbol[group.name] || []).length
        : group.count;
      if (activeCount <= 0) return map;
      const displayName = getSymbolDisplayName(group.name);
      const key = displayName.trim().toLocaleLowerCase('pl-PL');
      const existing = map.get(key);
      if (existing) {
        existing.count += activeCount;
        existing.symbolNames.push(group.name);
      } else {
        map.set(key, {
          displayName,
          count: activeCount,
          color: group.color,
          symbolNames: [group.name],
        });
      }
      return map;
    }, new Map<string, { displayName: string; count: number; color: string; symbolNames: string[] }>())
      .values()
  );
  const exportTotal = exportRows.reduce((sum, row) => sum + row.count, 0);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    if (!projectId) return;
    const file = e.target.files[0];
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await apiFetch(projectApiPath(projectId, '/templates/upload'), {
        method: 'POST',
        body: formData,
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

  const beginRename = (name: string) => {
    setRenamingGroup(name);
    setRenameDraft(getSymbolDisplayName(name));
  };

  const saveRename = async (currentName: string) => {
    const nextName = renameDraft.trim();
    if (!nextName || !onRenameSymbol) return;
    setRenameBusyGroup(currentName);
    try {
      await onRenameSymbol(currentName, nextName);
      setRenamingGroup(null);
      setRenameDraft('');
    } catch (error) {
      alert((error as Error).message || 'Nie udało się zmienić nazwy symbolu.');
    } finally {
      setRenameBusyGroup(null);
    }
  };

  const exportFileName = () => {
    const sourcePdf = analysisContext?.sourcePdf?.replace(/\.[^.]+$/, '') || 'wyniki';
    const safeName = sourcePdf.replace(/[\\/:*?"<>|]+/g, '_').trim() || 'wyniki';
    return `elektroscan_${safeName.slice(0, 80)}_wyniki.xlsx`;
  };

  const filenameFromHeader = (header: string | null) => {
    if (!header) return exportFileName();
    const utfMatch = header.match(/filename\*=UTF-8''([^;]+)/i);
    if (utfMatch?.[1]) return decodeURIComponent(utfMatch[1]);
    const asciiMatch = header.match(/filename="?([^";]+)"?/i);
    return asciiMatch?.[1] || exportFileName();
  };

  const handleExportExcel = async () => {
    if (!projectId || exportTotal <= 0) return;
    setExporting(true);
    try {
      const response = await apiFetch(projectApiPath(projectId, '/analysis-export'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          results,
          boxes: filteredBoxes,
          analysisContext,
          symbolLabels,
        }),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response, 'Nie udało się wyeksportować wyników.'));
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filenameFromHeader(response.headers.get('Content-Disposition'));
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      alert((error as Error).message || 'Nie udało się wyeksportować wyników.');
    } finally {
      setExporting(false);
    }
  };

  const empty = results.length === 0;
  const allSymbolNames = Array.from(new Set([...symbolNames, ...results.map(result => result.name)])).sort();

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
          {(['correction', 'export'] as const).map(tab => (
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
              {tab === 'correction' ? 'Korekta' : 'Eksport'}
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

          {/* KOREKTA TAB */}
          {activeTab === 'correction' && (
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>

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
                const isRenaming = renamingGroup === group.name;
                const isRenameBusy = renameBusyGroup === group.name;
                return (
                  <div key={group.name} className="card" style={{ padding: 0, overflow: 'hidden' }}>
                    {/* Nagłówek grupy */}
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        if (!isRenaming) toggleGroup(group.name);
                      }}
                      onKeyDown={event => {
                        if (!isRenaming && (event.key === 'Enter' || event.key === ' ')) {
                          event.preventDefault();
                          toggleGroup(group.name);
                        }
                      }}
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
                      {isRenaming ? (
                        <input
                          value={renameDraft}
                          onClick={event => event.stopPropagation()}
                          onChange={event => setRenameDraft(event.target.value)}
                          onKeyDown={event => {
                            if (event.key === 'Enter') void saveRename(group.name);
                            if (event.key === 'Escape') setRenamingGroup(null);
                          }}
                          autoFocus
                          disabled={isRenameBusy}
                          style={{
                            flex: 1,
                            minWidth: 0,
                            background: 'var(--bg-main)',
                            color: 'var(--text-primary)',
                            border: '1px solid var(--border-light)',
                            borderRadius: 4,
                            padding: '5px 7px',
                            fontSize: 12,
                            fontWeight: 700,
                          }}
                        />
                      ) : (
                        <span
                          className="text-xs"
                          title={getSymbolDisplayName(group.name)}
                          style={{
                            flex: 1,
                            fontWeight: 700,
                            color: 'var(--text-primary)',
                            wordBreak: 'break-word',
                            lineHeight: 1.3,
                          }}
                        >
                          {getSymbolDisplayName(group.name)}
                        </span>
                      )}
                      {onRenameSymbol && (
                        isRenaming ? (
                          <>
                            <button
                              type="button"
                              className="btn-icon"
                              title="Zapisz nazwę"
                              disabled={isRenameBusy}
                              onClick={event => {
                                event.stopPropagation();
                                void saveRename(group.name);
                              }}
                            >
                              <Save size={14} />
                            </button>
                            <button
                              type="button"
                              className="btn-icon"
                              title="Anuluj zmianę nazwy"
                              disabled={isRenameBusy}
                              onClick={event => {
                                event.stopPropagation();
                                setRenamingGroup(null);
                              }}
                            >
                              <X size={14} />
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            className="btn-icon"
                            title="Zmień nazwę symbolu"
                            onClick={event => {
                              event.stopPropagation();
                              beginRename(group.name);
                            }}
                          >
                            <Edit3 size={14} />
                          </button>
                        )
                      )}
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
                    </div>

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
                            const isEditing = editingBoxId === box.id;
                            return (
                              <React.Fragment key={box.id}>
                                <div
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
                                  <span className="text-xs text-muted" style={{ flex: 1 }}>
                                    x:{box.x} y:{box.y}
                                    {box.note?.trim() ? ' · opis' : ''}
                                  </span>
                                  {onChangeBoxSymbol && allSymbolNames.length > 0 && (
                                    <select
                                      value={box.symbolName}
                                      onClick={e => e.stopPropagation()}
                                      onChange={e => onChangeBoxSymbol(box.id, e.target.value)}
                                      title="Zmien klase symbolu"
                                      style={{
                                        maxWidth: 92,
                                        background: 'var(--bg-main)',
                                        color: 'var(--text-primary)',
                                        border: '1px solid var(--border-light)',
                                        borderRadius: 4,
                                        fontSize: 10,
                                        padding: '2px 4px',
                                      }}
                                    >
                                      {allSymbolNames.map(name => (
                                        <option key={name} value={name}>{getSymbolDisplayName(name)}</option>
                                      ))}
                                    </select>
                                  )}
                                  <button
                                    className="btn-icon"
                                    onClick={e => {
                                      e.stopPropagation();
                                      setEditingBoxId(current => current === box.id ? null : box.id);
                                      onFocusBox?.(box.id);
                                    }}
                                    title="Edytuj box i opis"
                                  >
                                    <Edit3 size={14} />
                                  </button>
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
                                {isEditing && (
                                  <div
                                    className="box-edit-panel"
                                    onClick={event => event.stopPropagation()}
                                    onMouseDown={event => event.stopPropagation()}
                                  >
                                    <div className="box-edit-grid">
                                      {(['x', 'y', 'width', 'height'] as const).map(key => (
                                        <label key={key} className="box-edit-field">
                                          <span>{key === 'width' ? 'W' : key === 'height' ? 'H' : key.toUpperCase()}</span>
                                          <input
                                            type="number"
                                            value={box[key]}
                                            min={key === 'width' || key === 'height' ? 1 : undefined}
                                            disabled={!onUpdateBox}
                                            onChange={event => {
                                              const value = Number(event.target.value);
                                              if (Number.isFinite(value)) onUpdateBox?.(box.id, { [key]: value });
                                            }}
                                          />
                                        </label>
                                      ))}
                                    </div>
                                    <div className="box-edit-grid box-edit-grid-wide">
                                      <label className="box-edit-field">
                                        <span>Status</span>
                                        <select
                                          value={box.reviewStatus ?? 'unchecked'}
                                          disabled={!onUpdateBox}
                                          onChange={event => onUpdateBox?.(box.id, { reviewStatus: event.target.value as Box['reviewStatus'] })}
                                        >
                                          {Object.entries(reviewLabels).map(([value, label]) => (
                                            <option key={value} value={value}>{label}</option>
                                          ))}
                                        </select>
                                      </label>
                                      <label className="box-edit-field">
                                        <span>Źródło</span>
                                        <input value={box.source ?? 'template'} readOnly />
                                      </label>
                                    </div>
                                    <label className="box-edit-field">
                                      <span>Opis do goldena / decyzja manualna</span>
                                      <textarea
                                        rows={3}
                                        value={box.note ?? ''}
                                        disabled={!onUpdateBox}
                                        placeholder="np. expected 28_TB11, pusty hit, manual_check"
                                        onChange={event => onUpdateBox?.(box.id, { note: event.target.value })}
                                      />
                                    </label>
                                  </div>
                                )}
                              </React.Fragment>
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

          {/* EKSPORT TAB */}
          {activeTab === 'export' && (
            <>
              <div style={{ padding: 16, flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
                <button
                  className="btn-secondary flex-row gap-2"
                  style={{ width: '100%', justifyContent: 'center' }}
                  onClick={() => void handleExportExcel()}
                  disabled={!projectId || exporting || exportTotal <= 0}
                >
                  <Download size={15} />
                  {exporting ? 'Eksportowanie...' : 'Eksportuj XLSX'}
                </button>

                {exportRows.map(row => (
                  <div key={row.displayName} className="estimate-card">
                    <div className="estimate-card-stripe" style={{ backgroundColor: row.color }} />
                    <div className="estimate-card-content">
                      <div className="estimate-card-title" title={row.displayName}>
                        {row.displayName}
                      </div>
                      <div className="estimate-inputs">
                        <div className="estimate-input-group">
                          <label className="estimate-input-label">ILOŚĆ</label>
                          <input
                            type="number"
                            className="estimate-input"
                            value={row.count}
                            readOnly
                          />
                        </div>
                        <div className="estimate-input-group">
                          <label className="estimate-input-label">WZORCE</label>
                          <input
                            className="estimate-input"
                            value={row.symbolNames.join(', ')}
                            readOnly
                            title={row.symbolNames.join(', ')}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="estimate-total-footer">
                <div className="flex-row gap-2">
                  <Layers size={16} color="var(--text-muted)" />
                  <span className="text-xs text-muted" style={{ textTransform: 'uppercase' }}>Razem</span>
                </div>
                <span style={{ fontSize: 18, fontWeight: 700, color: 'var(--accent-gold)' }}>
                  {exportTotal} szt.
                </span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
