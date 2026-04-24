import React, { useState, useEffect } from 'react';
import { Layers } from 'lucide-react';

interface CostPanelProps {
  results: any[];
}

const COLORS = ['#ef4444', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6'];

export const CostPanel: React.FC<CostPanelProps> = ({ results }) => {
  const [prices, setPrices] = useState<Record<string, number>>({});

  useEffect(() => {
    // Inicjalizacja pustych cen dla nowych wyników
    const newPrices = { ...prices };
    results.forEach(r => {
      if (newPrices[r.name] === undefined) {
        newPrices[r.name] = 0;
      }
    });
    setPrices(newPrices);
  }, [results]);

  const handlePriceChange = (name: string, value: string) => {
    const num = parseFloat(value);
    setPrices(prev => ({
      ...prev,
      [name]: isNaN(num) ? 0 : num
    }));
  };

  const totalSum = results.reduce((acc, r) => acc + (r.count * (prices[r.name] || 0)), 0);

  if (results.length === 0) {
    return (
      <div className="results-panel">
        <div className="sidebar-header flex-row gap-2">
          <Layers size={18} />
          <h2 className="text-sm" style={{ fontWeight: 600 }}>Kosztorys Wykonawczy</h2>
        </div>
        <div className="sidebar-content" style={{ alignItems: 'center', justifyContent: 'center' }}>
          <p className="text-sm text-muted">Brak danych do kosztorysu.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-panel">
      <div className="sidebar-header flex-row gap-2">
        <Layers size={18} />
        <h2 className="text-sm" style={{ fontWeight: 600 }}>Kosztorys Wykonawczy</h2>
      </div>

      <div className="sidebar-content" style={{ padding: '16px', flex: 1, overflowY: 'auto' }}>
        {results.map((group, index) => (
          <div key={index} className="estimate-card">
            <div className="estimate-card-stripe" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
            <div className="estimate-card-content">
              <div className="estimate-card-title">{group.name}</div>
              <div className="estimate-inputs">
                <div className="estimate-input-group">
                  <label className="estimate-input-label">ILOŚĆ</label>
                  <input 
                    type="number" 
                    className="estimate-input" 
                    value={group.count}
                    readOnly
                  />
                </div>
                <div className="estimate-input-group">
                  <label className="estimate-input-label">CENA NETTO (PLN)</label>
                  <input 
                    type="number" 
                    className="estimate-input" 
                    value={prices[group.name] || 0}
                    onChange={(e) => handlePriceChange(group.name, e.target.value)}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="estimate-total-footer">
        <span className="text-xs text-muted" style={{ textTransform: 'uppercase' }}>Suma Całkowita</span>
        <span className="text-orange" style={{ fontSize: '18px', fontWeight: 700 }}>
          {totalSum.toFixed(2)} PLN
        </span>
      </div>
    </div>
  );
};
