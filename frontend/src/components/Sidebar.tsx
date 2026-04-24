import React, { useRef, useState } from 'react';
import { Upload, FileDown, Layers, Trash2, Cpu, Edit3 } from 'lucide-react';
import { PatternModal } from './PatternModal';

interface SidebarProps {
  fileName: string | null;
  onFileSelect: (file: File) => void;
  onExtractLegend: () => void;
  onDetect: () => void;
  onClear: () => void;
  onClearTemplates: () => void;
  isProcessing: boolean;
  progressText: string;
  patterns: any[];
  onUpdatePattern: (index: number, newName: string) => void;
  onDeletePattern: (index: number) => void;
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
  patterns,
  onUpdatePattern,
  onDeletePattern
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [editingPatternIndex, setEditingPatternIndex] = useState<number | null>(null);

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
              1. Auto-Legenda (Extract)
            </button>
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
                <span className="text-xs text-muted">{progressText || 'Przetwarzanie...'}</span>
              </div>
              <div className="progress-container">
                <div className="progress-fill" style={{ width: '100%', animation: 'pulse 1.5s infinite' }} />
              </div>
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
