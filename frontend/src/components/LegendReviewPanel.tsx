import React, { useMemo, useState } from 'react';
import { Check, Crop, Plus, Save, Trash2, X } from 'lucide-react';

export type LegendReviewStatus = 'pending' | 'accepted' | 'fixed' | 'rejected';

export interface LegendReviewItem {
  id: string;
  name: string;
  imgBase64: string;
  status: LegendReviewStatus;
  correctedBBoxPx?: [number, number, number, number];
}

interface LegendReviewPanelProps {
  items: LegendReviewItem[];
  activeCorrectionId?: string | null;
  isProcessing?: boolean;
  onAccept: (id: string) => void;
  onAcceptAll: () => void;
  onReject: (id: string) => void;
  onStartCrop: (item: LegendReviewItem) => void;
  onCancelCrop: () => void;
  onRename: (id: string, name: string) => void;
  onAddMissing: () => void;
  onClose: () => void;
}

const statusLabel: Record<LegendReviewStatus, string> = {
  pending: 'Do sprawdzenia',
  accepted: 'OK',
  fixed: 'Poprawiony',
  rejected: 'Odrzucony',
};

const statusColor: Record<LegendReviewStatus, string> = {
  pending: '#f59e0b',
  accepted: '#22c55e',
  fixed: '#38bdf8',
  rejected: '#ef4444',
};

interface LegendReviewNameInputProps {
  item: LegendReviewItem;
  isProcessing: boolean;
  onRename: (id: string, name: string) => void;
}

const LegendReviewNameInput: React.FC<LegendReviewNameInputProps> = ({
  item,
  isProcessing,
  onRename,
}) => {
  const [draftName, setDraftName] = useState(item.name);
  const isRenamed = draftName.trim() !== item.name;

  return (
    <div className="legend-review-name-row">
      <input
        className="input-text"
        value={draftName}
        disabled={isProcessing || item.status === 'rejected'}
        onChange={event => setDraftName(event.target.value)}
      />
      <button
        className="btn-icon"
        title="Zapisz nazwę"
        disabled={!isRenamed || isProcessing || item.status === 'rejected'}
        onClick={() => onRename(item.id, draftName.trim())}
      >
        <Save size={15} />
      </button>
    </div>
  );
};

export const LegendReviewPanel: React.FC<LegendReviewPanelProps> = ({
  items,
  activeCorrectionId = null,
  isProcessing = false,
  onAccept,
  onAcceptAll,
  onReject,
  onStartCrop,
  onCancelCrop,
  onRename,
  onAddMissing,
  onClose,
}) => {
  const progress = useMemo(() => {
    const completed = items.filter(item => item.status !== 'pending').length;
    const acceptable = items.filter(
      item => item.status !== 'rejected' && Boolean(item.imgBase64)
    ).length;
    const accepted = items.filter(
      item => item.status === 'accepted' || item.status === 'fixed'
    ).length;
    return { acceptable, accepted, completed, total: items.length };
  }, [items]);

  const activeItem = activeCorrectionId
    ? items.find(item => item.id === activeCorrectionId)
    : null;

  if (items.length === 0) return null;

  return (
    <div className="legend-review-panel">
      <div className="legend-review-header">
        <div>
          <h2>Sprawdź wzorce legendy</h2>
          <div className="text-xs text-muted">
            {progress.completed}/{progress.total} gotowe
          </div>
        </div>
        <button className="btn-icon" onClick={onClose} title="Zamknij panel">
          <X size={18} />
        </button>
      </div>

      {activeItem && (
        <div className="legend-review-active">
          <div>
            <strong>{activeItem.name}</strong>
            <span className="text-xs text-muted">Zaznacz poprawny obszar na legendzie</span>
          </div>
          <button className="btn-secondary" onClick={onCancelCrop} disabled={isProcessing}>
            Anuluj
          </button>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 8 }}>
        <button
          className="btn-secondary"
          onClick={onAcceptAll}
          disabled={isProcessing || progress.accepted >= progress.acceptable}
        >
          <Check size={16} />
          Akceptuj wszystkie
        </button>
        <button className="btn-secondary" onClick={onAddMissing} disabled={isProcessing}>
        <Plus size={16} />
        Dodaj brakujący wzorzec
        </button>
      </div>

      <div className="legend-review-list">
        {items.map(item => {
          const isActive = activeCorrectionId === item.id;

          return (
            <div
              key={item.id}
              className="legend-review-item"
              style={{ borderColor: isActive ? 'rgba(198,168,124,0.7)' : undefined }}
            >
              {item.imgBase64 ? (
                <img src={item.imgBase64} alt={item.name} />
              ) : (
                <div className="legend-review-empty-preview">Nowy</div>
              )}
              <div className="legend-review-main">
                <LegendReviewNameInput
                  key={`${item.id}:${item.name}`}
                  item={item}
                  isProcessing={isProcessing}
                  onRename={onRename}
                />

                <div className="legend-review-actions">
                  <span
                    className="legend-review-status"
                    style={{ color: statusColor[item.status], borderColor: statusColor[item.status] }}
                  >
                    {statusLabel[item.status]}
                  </span>
                  <button
                    className="btn-icon"
                    title="Akceptuj"
                    disabled={isProcessing || item.status === 'rejected' || !item.imgBase64}
                    onClick={() => onAccept(item.id)}
                  >
                    <Check size={16} />
                  </button>
                  <button
                    className="btn-icon"
                    title="Popraw zaznaczenie"
                    disabled={isProcessing || item.status === 'rejected'}
                    onClick={() => onStartCrop(item)}
                  >
                    <Crop size={16} />
                  </button>
                  <button
                    className="btn-icon"
                    title="Odrzuć"
                    disabled={isProcessing}
                    onClick={() => onReject(item.id)}
                    style={{ color: '#ef4444' }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
