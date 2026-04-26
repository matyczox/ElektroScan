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
  verificationScore?: number;
  symbolName: string;
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

interface ExcludedZone {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface AnalysisContext {
  analysisId?: string;
  generatedAtUtc?: string;
  sessionId?: string;
  sourcePdf?: string;
  hiddenLayersUsed?: string[];
  excludedZonesUsed?: Array<[number, number, number, number]>;
  hiddenLayerDebug?: {
    matched?: string[];
    unmatched?: string[];
    requested?: Array<{
      value?: string;
      repr?: string;
      length?: number;
      normalized?: string;
      matches?: string[];
    }>;
  };
}

interface CanvasViewProps {
  imageSrc: string | null;
  boxes?: Box[];
  analysisContext?: AnalysisContext | null;
  onBoxClick?: (id: string) => void;
  focusedBoxId?: string | null;
  excludedZones?: ExcludedZone[];
  onAddExcludedZone?: (x: number, y: number, w: number, h: number) => void;
  onRemoveExcludedZone?: (index: number) => void;
  symbolNames?: string[];
  onAddManualBox?: (box: Omit<Box, 'id' | 'color'> & { symbolName: string }) => void;
}

export const CanvasView: React.FC<CanvasViewProps> = ({
  imageSrc,
  boxes = [],
  analysisContext,
  onBoxClick,
  focusedBoxId,
  excludedZones = [],
  onAddExcludedZone,
  onRemoveExcludedZone,
  symbolNames = [],
  onAddManualBox,
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

  // Tryb ręcznego dodawania symbolu
  const [isManualMode, setIsManualMode] = useState(false);
  const [manualPos, setManualPos] = useState<{ x: number, y: number } | null>(null);
  const [manualSymbol, setManualSymbol] = useState('');
  const [manualSize, setManualSize] = useState(40);

  // Pulsowanie wybranej ramki
  const [pulsingId, setPulsingId] = useState<string | null>(null);
  const [copiedBoxId, setCopiedBoxId] = useState<string | null>(null);

  useEffect(() => {
    if (imageSrc) {
      setScale(0.8);
      setPosition({ x: 50, y: 50 });
    }
  }, [imageSrc]);

  useEffect(() => {
    setPulsingId(null);
    setCopiedBoxId(null);
  }, [analysisContext?.analysisId]);

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
    if (isManualMode) {
      const coords = getCanvasCoordinates(e.clientX, e.clientY);
      setManualPos(coords);
      if (symbolNames.length > 0 && !manualSymbol) setManualSymbol(symbolNames[0]);
      // Nie starujemy drag
      return;
    }
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

  const BOX_FOCUS_COLOR = '#f97316';
  const BOX_LOW_COLOR = '#f59e0b';

  const formatDebugValue = (value?: number, digits = 3) =>
    typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';

  const buildDebugPayload = (box: Box) => {
    const nearbyBoxes = boxes
      .filter(candidate => {
        const centerDx = Math.abs((candidate.x + candidate.width / 2) - (box.x + box.width / 2));
        const centerDy = Math.abs((candidate.y + candidate.height / 2) - (box.y + box.height / 2));
        return centerDx <= 80 && centerDy <= 80;
      })
      .map(candidate =>
        `${candidate.symbolName}@${candidate.x},${candidate.y},${candidate.width},${candidate.height}#${candidate.analysisId ?? analysisContext?.analysisId ?? 'n/a'}`
      );
    const hiddenLayers = box.hiddenLayersUsed?.length
      ? box.hiddenLayersUsed.join(" | ")
      : analysisContext?.hiddenLayersUsed?.length
      ? analysisContext.hiddenLayersUsed.join(" | ")
      : "(none)";
    const hiddenLayerUnmatched = analysisContext?.hiddenLayerDebug?.unmatched?.join(" | ") || "(none)";
    const hiddenLayerReprs = analysisContext?.hiddenLayerDebug?.requested
      ?.map(entry => `${entry.value ?? ''}=>${entry.repr ?? 'n/a'}|norm=${entry.normalized ?? 'n/a'}|len=${entry.length ?? 0}|matches=${(entry.matches ?? []).join('&') || '(none)'}`)
      .join(" || ") || "(none)";
    const excludedZones = analysisContext?.excludedZonesUsed?.length
      ? analysisContext.excludedZonesUsed.map(zone => zone.join(",")).join(" | ")
      : "(none)";
    const lines = [
      `symbol=${box.symbolName}`,
      `bbox=${box.x},${box.y},${box.width},${box.height}`,
      `match=${formatDebugValue(box.confidence)}`,
      `verification=${formatDebugValue(box.verificationScore)}`,
      `coverage=${formatDebugValue(box.coverage)}`,
      `purity=${formatDebugValue(box.purity)}`,
      `context_purity=${formatDebugValue(box.contextPurity)}`,
      `color_similarity=${formatDebugValue(box.colorSimilarity)}`,
      `rotation=${box.rotation ?? 0}`,
      `scale=${formatDebugValue(box.scale)}`,
      `mirrored=${box.mirrored ? 'true' : 'false'}`,
      `source=${box.source ?? 'template'}`,
      `analysis_id=${box.analysisId ?? analysisContext?.analysisId ?? 'n/a'}`,
      `analysis_generated_utc=${box.analysisGeneratedUtc ?? analysisContext?.generatedAtUtc ?? 'n/a'}`,
      `analysis_session=${box.analysisSession ?? analysisContext?.sessionId ?? 'n/a'}`,
      `source_pdf=${box.sourcePdf ?? analysisContext?.sourcePdf ?? 'n/a'}`,
      `hidden_layers_used=${hiddenLayers}`,
      `excluded_zones_used=${excludedZones}`,
      `hidden_layers_unmatched=${hiddenLayerUnmatched}`,
      `hidden_layers_repr=${hiddenLayerReprs}`,
      `frontend_boxes_count=${boxes.length}`,
      `frontend_nearby_boxes=${nearbyBoxes.join(' || ') || '(none)'}`,
      `box_id=${box.id}`,
    ];
    return lines.join('\n');
  };

  const copyBoxDebug = async (box: Box) => {
    const payload = buildDebugPayload(box);

    try {
      await navigator.clipboard.writeText(payload);
      setCopiedBoxId(box.id);
      window.setTimeout(() => {
        setCopiedBoxId(current => (current === box.id ? null : current));
      }, 1600);
    } catch (error) {
      console.error('Nie udało się skopiować debug info boxa', error);
    }
  };

  const confirmManualBox = () => {
    if (!manualPos || !manualSymbol) return;
    onAddManualBox?.({
      symbolName: manualSymbol,
      x: Math.round(manualPos.x - manualSize / 2),
      y: Math.round(manualPos.y - manualSize / 2),
      width: manualSize,
      height: manualSize,
      confidence: 1.0,
    });
    setManualPos(null);
    setIsManualMode(false);
  };

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
            const displayConfidence = box.verificationScore ?? box.confidence;
            const isLowConf = displayConfidence < 0.55;
            const color = isFocused ? BOX_FOCUS_COLOR : (isLowConf ? BOX_LOW_COLOR : box.color);

            return (
              <div
                key={`${box.analysisId ?? analysisContext?.analysisId ?? 'na'}:${box.id}`}
                onClick={e => {
                  e.stopPropagation();
                  onBoxClick?.(box.id);
                  void copyBoxDebug(box);
                }}
                title={`Weryfikacja: ${(displayConfidence * 100).toFixed(0)}%\nMatch template: ${(box.confidence * 100).toFixed(0)}%\nWzorzec: ${box.symbolName}\nKlik kopiuje debug`}
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
              >
                {/* Etykieta symbolu widoczna po najechaniu lub ciągle */}
                <div style={{
                  position: 'absolute',
                  top: -16,
                  left: -2,
                  background: color,
                  color: '#fff',
                  fontSize: 9,
                  fontWeight: 'bold',
                  padding: '1px 4px',
                  borderRadius: 2,
                  whiteSpace: 'nowrap',
                  pointerEvents: 'none',
                  opacity: 0.8,
                }}>
                  {/* Ucinamy za długie nazwy (np. z '.png' lub długie) */}
                  {(box as any).symbolName ? (box as any).symbolName.split('_')[0].substring(0, 15) : 'Symbol'}
                </div>
                {copiedBoxId === box.id && (
                  <div style={{
                    position: 'absolute',
                    top: box.height + 4,
                    left: 0,
                    background: 'rgba(15, 23, 42, 0.92)',
                    color: '#fff',
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                    zIndex: 5,
                  }}>
                    Debug skopiowany
                  </div>
                )}
              </div>
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

          {/* Ręczne dodawanie - Modal */}
          {manualPos && (
            <div
              style={{
                position: 'absolute',
                left: manualPos.x,
                top: manualPos.y,
                transform: `scale(${1 / scale}) translate(10px, 10px)`,
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-light)',
                borderRadius: 8,
                padding: 12,
                zIndex: 100,
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
                width: 200,
                cursor: 'default',
              }}
              onClick={e => e.stopPropagation()}
              onMouseDown={e => e.stopPropagation()}
            >
              <h4 style={{ margin: 0, fontSize: 12, color: 'var(--accent-gold)' }}>Dodaj symbol</h4>
              <select 
                value={manualSymbol} 
                onChange={e => setManualSymbol(e.target.value)}
                style={{ width: '100%', padding: '4px', background: 'var(--bg-primary)', color: 'white', border: '1px solid var(--border-light)', borderRadius: 4 }}
              >
                {symbolNames.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Rozmiar:</span>
                <input 
                  type="number" 
                  value={manualSize} 
                  onChange={e => setManualSize(Number(e.target.value))}
                  style={{ width: 50, padding: 2, background: 'var(--bg-primary)', color: 'white', border: '1px solid var(--border-light)' }}
                />
              </div>
              <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                <button 
                  onClick={confirmManualBox}
                  style={{ flex: 1, background: 'var(--accent-gold)', color: 'black', border: 'none', padding: '4px', borderRadius: 4, fontWeight: 'bold', cursor: 'pointer' }}
                >
                  Dodaj
                </button>
                <button 
                  onClick={() => setManualPos(null)}
                  style={{ flex: 1, background: 'transparent', border: '1px solid var(--border-light)', color: 'white', padding: '4px', borderRadius: 4, cursor: 'pointer' }}
                >
                  Anuluj
                </button>
              </div>
            </div>
          )}

          {/* Podgląd dodawanego boxa */}
          {manualPos && (
             <div style={{
                position: 'absolute',
                left: manualPos.x - manualSize / 2,
                top: manualPos.y - manualSize / 2,
                width: manualSize,
                height: manualSize,
                border: '2px dashed var(--accent-gold)',
                backgroundColor: 'rgba(198,168,124,0.2)',
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
