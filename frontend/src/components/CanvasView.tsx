import React, { useRef, useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize, Move, Slash, X } from 'lucide-react';

interface Box {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  confidence: number;
}

interface ExcludedZone {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface CanvasViewProps {
  imageSrc: string | null;
  boxes?: Box[];
  onBoxClick?: (id: string) => void;
  focusedBoxId?: string | null;
  excludedZones?: ExcludedZone[];
  onAddExcludedZone?: (x: number, y: number, w: number, h: number) => void;
  onRemoveExcludedZone?: (index: number) => void;
}

export const CanvasView: React.FC<CanvasViewProps> = ({
  imageSrc,
  boxes = [],
  onBoxClick,
  focusedBoxId,
  excludedZones = [],
  onAddExcludedZone,
  onRemoveExcludedZone,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Tryb rysowania strefy wykluczonej
  const [isZoneMode, setIsZoneMode] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });
  const [drawCurrent, setDrawCurrent] = useState({ x: 0, y: 0 });

  // Pulsowanie wybranej ramki
  const [pulsingId, setPulsingId] = useState<string | null>(null);

  useEffect(() => {
    if (imageSrc) {
      setScale(0.8);
      setPosition({ x: 50, y: 50 });
    }
  }, [imageSrc]);

  // Kiedy focusedBoxId się zmienia → animuj pan do tej ramki
  useEffect(() => {
    if (!focusedBoxId || !containerRef.current) return;
    const box = boxes.find(b => b.id === focusedBoxId);
    if (!box) return;

    const container = containerRef.current;
    const centerX = container.clientWidth / 2 - (box.x + box.width / 2) * scale;
    const centerY = container.clientHeight / 2 - (box.y + box.height / 2) * scale;

    setPosition({ x: centerX, y: centerY });

    // Pulsowanie przez 2s
    setPulsingId(focusedBoxId);
    const timer = setTimeout(() => setPulsingId(null), 2000);
    return () => clearTimeout(timer);
  }, [focusedBoxId]); // eslint-disable-line

  const handleWheel = (e: React.WheelEvent) => {
    if (!imageSrc) return;
    e.preventDefault();
    const delta = -e.deltaY * 0.001;
    setScale(s => Math.min(Math.max(0.1, s * (1 + delta)), 5));
  };

  const getCanvasCoordinates = (clientX: number, clientY: number) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: (clientX - rect.left - position.x) / scale,
      y: (clientY - rect.top - position.y) / scale,
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!imageSrc) return;
    if (isZoneMode) {
      setIsDrawing(true);
      const coords = getCanvasCoordinates(e.clientX, e.clientY);
      setDrawStart(coords);
      setDrawCurrent(coords);
    } else {
      setIsDragging(true);
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isZoneMode && isDrawing) {
      setDrawCurrent(getCanvasCoordinates(e.clientX, e.clientY));
    } else if (isDragging) {
      setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    if (isZoneMode && isDrawing) {
      setIsDrawing(false);
      setIsZoneMode(false);
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      if (w > 5 && h > 5) onAddExcludedZone?.(x, y, w, h);
    }
    setIsDragging(false);
  };

  if (!imageSrc) {
    return (
      <div className="workspace" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <Move size={48} style={{ opacity: 0.2, marginBottom: 16 }} />
          <h2>Brak podglądu</h2>
          <p className="text-sm text-muted" style={{ marginTop: 8 }}>
            Wgraj plan PDF i uruchom ekstrakcję legendy.
          </p>
        </div>
      </div>
    );
  }

  const BOX_COLOR = '#c6a87c';         // złoty — normalny
  const BOX_FOCUS_COLOR = '#f97316';   // pomarańcz — zaznaczony / pulsujący
  const BOX_LOW_COLOR = '#f59e0b';     // żółty — niski confidence

  return (
    <div className="workspace" ref={containerRef} onWheel={handleWheel}>
      {/* Kontrolki */}
      <div style={{ position: 'absolute', top: 16, right: 16, zIndex: 20, display: 'flex', gap: 6 }}>
        <button
          className="btn-secondary"
          onClick={() => setIsZoneMode(z => !z)}
          title="Dodaj strefę wykluczoną"
          style={{
            borderColor: isZoneMode ? 'var(--accent-orange)' : undefined,
            color: isZoneMode ? 'var(--accent-orange)' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Slash size={14} />
          {isZoneMode ? 'Rysuj...' : 'Strefa'}
        </button>
        <div style={{ width: 1, background: 'var(--border-light)', margin: '0 2px', alignSelf: 'stretch' }} />
        <button className="btn-secondary" style={{ padding: '6px 10px' }}
          onClick={() => setScale(s => Math.min(s * 1.2, 5))}>
          <ZoomIn size={16} />
        </button>
        <button className="btn-secondary" style={{ padding: '6px 10px' }}
          onClick={() => setScale(s => Math.max(s / 1.2, 0.1))}>
          <ZoomOut size={16} />
        </button>
        <button className="btn-secondary" style={{ padding: '6px 10px' }}
          onClick={() => { setScale(0.8); setPosition({ x: 50, y: 50 }); }}>
          <Maximize size={16} />
        </button>
      </div>

      {/* Hint rysowania */}
      {isZoneMode && (
        <div style={{
          position: 'absolute', top: 64, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(249,115,22,0.92)', color: '#fff',
          padding: '6px 18px', borderRadius: 6, fontSize: 12, fontWeight: 700,
          zIndex: 20, pointerEvents: 'none',
        }}>
          Zaznacz myszką obszar legendy
        </div>
      )}

      {/* Canvas */}
      <div
        className="canvas-wrapper"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isZoneMode ? 'crosshair' : (isDragging ? 'grabbing' : 'grab') }}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            position: 'absolute',
            transition: isDragging || isDrawing ? 'none' : 'transform 0.25s ease-out',
            transformOrigin: '0 0',
          }}
        >
          <img src={imageSrc} alt="Plan view" className="plan-image" draggable={false} />

          {/* Strefy Wykluczone (lista) */}
          {excludedZones.map((zone, idx) => (
            <div
              key={idx}
              className="excluded-zone"
              style={{ left: zone.x, top: zone.y, width: zone.width, height: zone.height }}
            >
              <div className="excluded-zone-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                WYKLUCZONA #{idx + 1}
                {onRemoveExcludedZone && (
                  <button
                    onClick={e => { e.stopPropagation(); onRemoveExcludedZone(idx); }}
                    style={{
                      background: 'rgba(0,0,0,0.4)',
                      border: 'none', cursor: 'pointer',
                      color: '#fff', display: 'flex', padding: 2, borderRadius: 3,
                    }}
                    title="Usuń strefę"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            </div>
          ))}

          {/* Detection Boxes */}
          {boxes.map(box => {
            const isFocused = focusedBoxId === box.id || pulsingId === box.id;
            const isPulsing = pulsingId === box.id;
            const isLowConf = box.confidence < 0.65;
            const color = isFocused ? BOX_FOCUS_COLOR : (isLowConf ? BOX_LOW_COLOR : BOX_COLOR);

            return (
              <div
                key={box.id}
                onClick={e => { e.stopPropagation(); onBoxClick?.(box.id); }}
                title={`Pewność: ${(box.confidence * 100).toFixed(0)}%`}
                style={{
                  position: 'absolute',
                  left: box.x,
                  top: box.y,
                  width: box.width,
                  height: box.height,
                  border: `${isFocused ? 3 : 2}px solid ${color}`,
                  backgroundColor: isFocused ? color + '25' : 'transparent',
                  boxShadow: isFocused ? `0 0 14px ${color}99` : 'none',
                  cursor: 'pointer',
                  boxSizing: 'border-box',
                  animation: isPulsing ? 'boxPulse 0.5s ease-in-out 4' : 'none',
                }}
              />
            );
          })}

          {/* Drawing preview */}
          {isZoneMode && isDrawing && (
            <div style={{
              position: 'absolute',
              left: Math.min(drawStart.x, drawCurrent.x),
              top: Math.min(drawStart.y, drawCurrent.y),
              width: Math.abs(drawCurrent.x - drawStart.x),
              height: Math.abs(drawCurrent.y - drawStart.y),
              border: '2px dashed var(--accent-orange)',
              backgroundColor: 'rgba(249,115,22,0.12)',
              pointerEvents: 'none',
            }} />
          )}
        </div>
      </div>

      {/* CSS dla pulse animacji */}
      <style>{`
        @keyframes boxPulse {
          0%   { opacity: 1; }
          50%  { opacity: 0.3; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};
