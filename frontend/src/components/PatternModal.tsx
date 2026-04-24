import React, { useState } from 'react';
import { X, Save, Trash2 } from 'lucide-react';

interface PatternModalProps {
  pattern: any;
  onClose: () => void;
  onSave: (newName: string) => void;
  onDelete: () => void;
}

export const PatternModal: React.FC<PatternModalProps> = ({ pattern, onClose, onSave, onDelete }) => {
  const [name, setName] = useState(pattern.name);

  if (!pattern) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="flex-row" style={{ justifyContent: 'space-between', marginBottom: 20 }}>
          <h2 className="text-sm" style={{ fontWeight: 600 }}>Szczegóły Wzorca</h2>
          <button className="btn-icon" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
          <img 
            src={pattern.imgBase64} 
            alt={pattern.name} 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '200px', 
              background: '#000', 
              padding: 10,
              borderRadius: 8,
              border: '1px solid var(--border-light)'
            }} 
          />
        </div>

        <div style={{ marginBottom: 20 }}>
          <label className="text-xs text-muted" style={{ display: 'block', marginBottom: 8 }}>
            Nazwa symbolu
          </label>
          <input 
            type="text" 
            className="input-text" 
            value={name}
            onChange={e => setName(e.target.value)}
          />
        </div>

        <div className="flex-row gap-4" style={{ justifyContent: 'space-between' }}>
          <button className="btn-secondary btn-danger" onClick={onDelete} style={{ flex: 1 }}>
            <Trash2 size={16} /> Usuń
          </button>
          <button className="btn-primary" onClick={() => onSave(name)} style={{ flex: 2 }}>
            <Save size={16} /> Zapisz zmiany
          </button>
        </div>
      </div>
    </div>
  );
};
