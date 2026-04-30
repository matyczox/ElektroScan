import React, { useRef, useState } from 'react';
import { Upload, FileDown, Layers, Trash2, Cpu, Edit3 } from 'lucide-react';
import { PatternModal } from './PatternModal';

type DetectorProfile = 'auto' | 'color' | 'gray';

interface PdfDiagnostics {
  pages?: number;
  layers?: number;
  textCharsPage1?: number;
  textBlocksPage1?: number;
  drawingsPage1?: number;
  imagesPage1?: number;
  inkPct?: number;
  colorfulInkPct?: number;
  grayInkPct?: number;
  recommendedProfile?: 'color' | 'gray';
}

interface AnalysisProgress {
  stage?: string;
  percent?: number;
  detail?: string;
  done?: boolean;
  error?: string | null;
}

interface SidebarProps {
  fileName: string | null;
  onFileSelect: (file: File) => void;
  onExtractLegend: () => void;
  onDetect: () => void;
  onClear: () => void;
  onClearTemplates: () => void;
  isProcessing: boolean;
  progressText: string;
  analysisProgress?: AnalysisProgress | null;
  patterns: any[];
  onUpdatePattern: (index: number, newName: string) => void;
  onDeletePattern: (index: number) => void;
  layers?: {name: string, visible: boolean}[];
  onToggleLayer?: (name: string) => void;
  detectorProfile: DetectorProfile;
  onDetectorProfileChange: (profile: DetectorProfile) => void;
  pdfDiagnostics?: PdfDiagnostics | null;
  hasLegendZone?: boolean;
  onClearLegendZone?: () => void;
  hasPlanZone?: boolean;
  onClearPlanZone?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  fileName,
  onFileSelect,
  onExtractLegend,
  onDetect,
  onClear,
  onClearTemplates,
  isProcessing,
  progressText,
  analysisProgress,
  patterns,
  onUpdatePattern,
  onDeletePattern,
  layers = [],
  onToggleLayer,
  detectorProfile,
  onDetectorProfileChange,
  pdfDiagnostics,
  hasLegendZone = false,
  onClearLegendZone,
  hasPlanZone = false,
  onClearPlanZone,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [editingPatternIndex, setEditingPatternIndex] = useState<number | null>(null);
  const progressPercent =
    typeof analysisProgress?.percent === 'number'
      ? Math.max(0, Math.min(100, analysisProgress.percent))
      : null;
  const progressLabel =
    analysisProgress?.error ||
    analysisProgress?.detail ||
    progressText ||
    'Przetwarzanie...';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1 className="title-gradient flex-row gap-2">
          <Layers size={24} color="var(--accent-gold)" />
          ElektroScan AI
        </h1>
        <p className="text-sm text-muted mt-2">Baza Wiedzy & Korekta</p>
      </div>

      <div className="sidebar-content">
        {/* Upload Section */}
        <div className="card">
          <div className="card-header">Plik Projektu</div>
          {!fileName ? (
            <div 
              className="upload-area"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <Upload size={32} className="upload-icon" />
              <div>
                <p className="text-sm" style={{ fontWeight: 600 }}>Przeciągnij plik PDF</p>
                <p className="text-xs text-muted">lub kliknij aby wybrać</p>
              </div>
            </div>
          ) : (
            <div className="flex-row gap-4" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="flex-row gap-2">
                <FileDown size={20} color="var(--accent-gold)" />
                <span className="text-sm" style={{ fontWeight: 500, wordBreak: 'break-all' }}>{fileName}</span>
              </div>
              <button className="btn-secondary" style={{ width: 'auto', padding: '6px 10px' }} onClick={() => fileInputRef.current?.click()}>
                Zmień
              </button>
            </div>
          )}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf" 
            style={{ display: 'none' }} 
          />
        </div>

        {/* Warstwy PDF */}
        {layers && layers.length > 0 && (
          <div className="card">
            <div className="card-header">
              <Layers size={14} /> Warstwy PDF (Zoptymalizuj tło)
            </div>
            <div style={{ maxHeight: 150, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {layers.map((layer, idx) => (
                <label key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, cursor: 'pointer', color: layer.visible ? 'var(--text-main)' : 'var(--text-muted)' }}>
                  <input 
                    type="checkbox" 
                    checked={layer.visible}
                    onChange={() => onToggleLayer?.(layer.name)}
                    disabled={isProcessing}
                  />
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{layer.name}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Detection profile */}
        <div className="card">
          <div className="card-header">
            <Cpu size={14} /> Profil detekcji
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <label className="text-xs text-muted" style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              Tryb obrazu
              <select
                value={detectorProfile}
                disabled={isProcessing}
                onChange={e => onDetectorProfileChange(e.target.value as DetectorProfile)}
                style={{
                  width: '100%',
                  padding: '7px 8px',
                  borderRadius: 6,
                  border: '1px solid var(--border-light)',
                  background: 'var(--bg-primary)',
                  color: 'var(--text-main)',
                }}
              >
                <option value="auto">Auto</option>
                <option value="color">Kolor</option>
                <option value="gray">Szary / tusz</option>
              </select>
            </label>

            {pdfDiagnostics && (
              <div className="text-xs text-muted" style={{ lineHeight: 1.45 }}>
                <div>Rekomendacja: <b>{pdfDiagnostics.recommendedProfile === 'gray' ? 'Szary' : 'Kolor'}</b></div>
                <div>Warstwy: {pdfDiagnostics.layers ?? 0}, tekst: {pdfDiagnostics.textCharsPage1 ?? 0}</div>
                <div>Wektory: {pdfDiagnostics.drawingsPage1 ?? 0}, obrazy: {pdfDiagnostics.imagesPage1 ?? 0}</div>
                <div>Tusz kolorowy: {(pdfDiagnostics.colorfulInkPct ?? 0).toFixed(2)}%</div>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="card">
          <div className="card-header">Operacje</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <button 
              className="btn-primary" 
              onClick={onExtractLegend}
              disabled={!fileName || isProcessing}
            >
              <Layers size={18} />
              {hasLegendZone ? '1. Legenda z zaznaczenia' : '1. Auto-Legenda (Extract)'}
            </button>
            {hasLegendZone && (
              <button
                className="btn-secondary"
                onClick={onClearLegendZone}
                disabled={isProcessing}
              >
                Wyczysc strefe legendy
              </button>
            )}
            {hasPlanZone && (
              <button
                className="btn-secondary"
                onClick={onClearPlanZone}
                disabled={isProcessing}
              >
                Wyczysc strefe planu
              </button>
            )}
            <button 
              className="btn-primary" 
              onClick={onDetect}
              disabled={!fileName || patterns.length === 0 || isProcessing}
            >
              <Cpu size={18} />
              2. Analizuj Plan (Hybrid)
            </button>
            
            <button 
              className="btn-secondary btn-danger" 
              onClick={onClear}
              disabled={isProcessing}
            >
              <Trash2 size={18} />
              Wyczyść Wszystko
            </button>
          </div>

          {/* Progress Indicator */}
          {isProcessing && (
            <div className="mt-4">
              <div className="flex-row" style={{ justifyContent: 'space-between' }}>
                <span className="text-xs text-muted">{progressLabel}</span>
                {progressPercent !== null && (
                  <span className="text-xs text-muted">{Math.round(progressPercent)}%</span>
                )}
              </div>
              <div className="progress-container">
                <div
                  className="progress-fill"
                  style={{
                    width: `${progressPercent ?? 100}%`,
                    animation: progressPercent === null ? 'pulse 1.5s infinite' : 'none',
                    transition: 'width 260ms ease',
                  }}
                />
              </div>
              {analysisProgress?.stage && (
                <div className="text-xs text-muted mt-1">Etap: {analysisProgress.stage}</div>
              )}
            </div>
          )}
        </div>

        {/* Knowledge Base */}
        {patterns.length > 0 && (
          <div className="card" style={{ flex: 1, overflowY: 'auto' }}>
            <div className="card-header flex-row" style={{ justifyContent: 'space-between' }}>
              <span>Baza Wzorców</span>
              <div className="flex-row gap-2">
                <span className="badge badge-gold">{patterns.length}</span>
                <button className="btn-icon" onClick={onClearTemplates} title="Wyczyść całą bazę wiedzy" style={{ color: '#ef4444' }}>
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
            {patterns.map((pat, i) => (
              <div key={i} className="list-item">
                <div className="flex-row gap-2 text-sm" style={{ flex: 1, overflow: 'hidden' }}>
                  <img src={pat.imgBase64} alt="wzorzec" />
                  <span style={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }} title={pat.name}>
                    {pat.name}
                  </span>
                </div>
                <button className="btn-icon" onClick={() => setEditingPatternIndex(i)}>
                  <Edit3 size={16} />
                </button>
              </div>
            ))}
          </div>
        )}
        
      </div>

      {editingPatternIndex !== null && (
        <PatternModal 
          pattern={patterns[editingPatternIndex]} 
          onClose={() => setEditingPatternIndex(null)}
          onSave={(newName) => {
            onUpdatePattern(editingPatternIndex, newName);
            setEditingPatternIndex(null);
          }}
          onDelete={() => {
            onDeletePattern(editingPatternIndex);
            setEditingPatternIndex(null);
          }}
        />
      )}
    </div>
  );
};
