import React, { useRef, useState, useEffect } from 'react';
import { AlertTriangle, Layers, Plus, Trash2, ZoomIn, ZoomOut, Maximize, Move, Slash, X } from 'lucide-react';
import { useMemo } from 'react';
import { formatSymbolLabel } from '../symbolLabels';

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
  reason?: string;
  relatedFinal?: {
    symbolName?: string;
    bbox?: [number, number, number, number];
    verificationScore?: number;
    source?: string;
  };
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
  manualExcludedZonesUsed?: Array<[number, number, number, number]>;
  legendZoneUsed?: [number, number, number, number] | null;
  planZoneUsed?: [number, number, number, number] | null;
  planZoneOutsideExcluded?: Array<[number, number, number, number]>;
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

interface GrayDebugInfo {
  zoneThreshold: number;
  evidenceThreshold: number;
  zonePixels: number;
  evidencePixels: number;
  roiCount: number;
  roiRefs: number;
  templates: number;
}

interface CanvasViewProps {
  imageSrc: string | null;
  imageSize?: { width: number; height: number } | null;
  imageResetKey?: string | null;
  boxes?: Box[];
  analysisContext?: AnalysisContext | null;
  onBoxClick?: (id: string) => void;
  focusedBoxId?: string | null;
  excludedZones?: ExcludedZone[];
  legendZone?: ExcludedZone | null;
  planZone?: ExcludedZone | null;
  onAddExcludedZone?: (x: number, y: number, w: number, h: number) => void;
  onRemoveExcludedZone?: (index: number) => void;
  onSetLegendZone?: (x: number, y: number, w: number, h: number) => void;
  onClearLegendZone?: () => void;
  onSetPlanZone?: (x: number, y: number, w: number, h: number) => void;
  onClearPlanZone?: () => void;
  onInspectZone?: (x: number, y: number, w: number, h: number) => void;
  grayDebugOverlayImage?: string | null;
  grayDebugInfo?: GrayDebugInfo | null;
  onToggleGrayDebugZones?: () => void;
  isGrayDebugLoading?: boolean;
  symbolNames?: string[];
  onAddManualBox?: (box: Omit<Box, 'id' | 'color'> & { symbolName: string }) => void;
  onRejectBox?: (id: string) => void;
  legendTemplateCropTarget?: { id: string; name: string } | null;
  onLegendTemplateCrop?: (x: number, y: number, w: number, h: number) => void;
  onCancelLegendTemplateCrop?: () => void;
}

export const CanvasView: React.FC<CanvasViewProps> = ({
  imageSrc,
  imageSize = null,
  imageResetKey = null,
  boxes = [],
  analysisContext,
  onBoxClick,
  focusedBoxId,
  excludedZones = [],
  legendZone = null,
  planZone = null,
  onAddExcludedZone,
  onRemoveExcludedZone,
  onSetLegendZone,
  onClearLegendZone,
  onSetPlanZone,
  onClearPlanZone,
  onInspectZone,
  grayDebugOverlayImage = null,
  grayDebugInfo = null,
  onToggleGrayDebugZones,
  isGrayDebugLoading = false,
  symbolNames = [],
  onAddManualBox,
  onRejectBox,
  legendTemplateCropTarget = null,
  onLegendTemplateCrop,
  onCancelLegendTemplateCrop,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const scaleRef = useRef(1);
  const positionRef = useRef({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isSpaceDown, setIsSpaceDown] = useState(false);

  // Tryb rysowania strefy wykluczonej, legendy, planu albo inspektora.
  type DrawMode = 'none' | 'exclude' | 'legend' | 'plan' | 'inspect' | 'legend-template';
  const [drawMode, setDrawMode] = useState<DrawMode>('none');
  const activeDrawMode: DrawMode = legendTemplateCropTarget ? 'legend-template' : drawMode;
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
  const [isZooming, setIsZooming] = useState(false);
  const zoomTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);

  useEffect(() => {
    positionRef.current = position;
  }, [position]);

  const resetKey = imageResetKey ?? imageSrc;

  useEffect(() => {
    if (imageSrc) {
      setScale(0.8);
      setPosition({ x: 50, y: 50 });
    }
  }, [resetKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, button, [data-wheel-ui="true"]')) return;
      e.preventDefault();
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
        e.preventDefault();
        setIsSpaceDown(true);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') setIsSpaceDown(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  useEffect(() => {
    setPulsingId(null);
    setCopiedBoxId(null);
  }, [analysisContext?.analysisId]);

  useEffect(() => {
    if (!focusedBoxId || !onRejectBox) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Delete' && event.key !== 'Backspace') return;

      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName?.toLowerCase();
      if (tagName === 'input' || tagName === 'textarea' || tagName === 'select' || target?.isContentEditable) {
        return;
      }

      event.preventDefault();
      onRejectBox(focusedBoxId);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [focusedBoxId, onRejectBox]);

  // Kiedy focusedBoxId się zmienia, animuj pan do tej ramki.
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
    if (!imageSrc || !containerRef.current) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest('input, textarea, select, button, [data-wheel-ui="true"]')) return;
    e.preventDefault();

    if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);
    setIsZooming(true);
    zoomTimerRef.current = setTimeout(() => setIsZooming(false), 150);

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const deltaUnit =
      e.deltaMode === 1
        ? 16
        : e.deltaMode === 2
        ? containerRef.current.clientHeight
        : 1;
    const deltaY = e.deltaY * deltaUnit;

    if (e.shiftKey && !e.ctrlKey) {
      const nextPosition = {
        x: positionRef.current.x - deltaY,
        y: positionRef.current.y,
      };
      positionRef.current = nextPosition;
      setPosition(nextPosition);
      return;
    }

    const oldScale = scaleRef.current;
    const oldPosition = positionRef.current;
    const factor = Math.exp(-deltaY * 0.0012);
    const newScale = Math.min(Math.max(0.1, oldScale * factor), 5);
    const imageX = (mouseX - oldPosition.x) / oldScale;
    const imageY = (mouseY - oldPosition.y) / oldScale;
    const nextPosition = {
      x: mouseX - imageX * newScale,
      y: mouseY - imageY * newScale,
    };

    scaleRef.current = newScale;
    positionRef.current = nextPosition;
    setScale(newScale);
    setPosition(nextPosition);
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
    // Środkowy przycisk lub Space + LPM → pan
    if (e.button === 1 || (e.button === 0 && isSpaceDown)) {
      e.preventDefault();
      setIsDragging(true);
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
      return;
    }
    if (isManualMode && !legendTemplateCropTarget) {
      const coords = getCanvasCoordinates(e.clientX, e.clientY);
      setManualPos(coords);
      if (symbolNames.length > 0 && !manualSymbol) setManualSymbol(symbolNames[0]);
      return;
    }
    if (activeDrawMode !== 'none') {
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
    if (activeDrawMode !== 'none' && isDrawing) {
      setDrawCurrent(getCanvasCoordinates(e.clientX, e.clientY));
    } else if (isDragging) {
      setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    if (activeDrawMode !== 'none' && isDrawing) {
      setIsDrawing(false);
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      if (w > 5 && h > 5) {
        if (activeDrawMode === 'legend') onSetLegendZone?.(x, y, w, h);
        else if (activeDrawMode === 'plan') onSetPlanZone?.(x, y, w, h);
        else if (activeDrawMode === 'inspect') onInspectZone?.(x, y, w, h);
        else if (activeDrawMode === 'legend-template') onLegendTemplateCrop?.(x, y, w, h);
        else onAddExcludedZone?.(x, y, w, h);
      }
      setDrawMode('none');
    }
    setIsDragging(false);
  };

  const overlapGroupsByBoxId = useMemo(() => {
    const groups = new Map<string, Box[]>();
    const boxArea = (box: Box) => Math.max(1, box.width * box.height);
    const highOverlap = (left: Box, right: Box) => {
      const x1 = Math.max(left.x, right.x);
      const y1 = Math.max(left.y, right.y);
      const x2 = Math.min(left.x + left.width, right.x + right.width);
      const y2 = Math.min(left.y + left.height, right.y + right.height);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      if (inter <= 0) return false;

      const leftArea = boxArea(left);
      const rightArea = boxArea(right);
      const iou = inter / Math.max(1, leftArea + rightArea - inter);
      const iom = inter / Math.max(1, Math.min(leftArea, rightArea));
      const centerDistance = Math.hypot(
        (left.x + left.width / 2) - (right.x + right.width / 2),
        (left.y + left.height / 2) - (right.y + right.height / 2),
      );
      const referenceDiagonal = Math.max(
        1,
        Math.hypot(Math.min(left.width, right.width), Math.min(left.height, right.height)),
      );
      const almostSameBbox =
        Math.abs(left.x - right.x) <= 4 &&
        Math.abs(left.y - right.y) <= 4 &&
        Math.abs(left.width - right.width) <= 5 &&
        Math.abs(left.height - right.height) <= 5;

      return almostSameBbox || iou >= 0.82 || (iom >= 0.92 && centerDistance / referenceDiagonal <= 0.18);
    };

    for (const box of boxes) {
      const group = boxes
        .filter(candidate => candidate.id === box.id || highOverlap(box, candidate))
        .sort((left, right) => {
          if (left.symbolName !== right.symbolName) return left.symbolName.localeCompare(right.symbolName);
          return (right.verificationScore ?? right.confidence) - (left.verificationScore ?? left.confidence);
        });
      if (group.length > 1) groups.set(box.id, group);
    }
    return groups;
  }, [boxes]);

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
  const focusedBox = focusedBoxId ? boxes.find(box => box.id === focusedBoxId) : null;
  const drawableBoxes = [...boxes].sort((left, right) => {
    const rightArea = right.width * right.height;
    const leftArea = left.width * left.height;
    if (rightArea !== leftArea) return rightArea - leftArea;
    return (left.confidence ?? 0) - (right.confidence ?? 0);
  });

  const formatOverlapGroup = (group?: Box[]) =>
    group?.length ? group.map(item => `${item.symbolName}@${item.x},${item.y},${item.width},${item.height}`).join(' + ') : '';

  const formatDebugValue = (value?: number, digits = 3) =>
    typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';

  const toggleDrawMode = (mode: Exclude<DrawMode, 'none' | 'legend-template'>) => {
    if (legendTemplateCropTarget) onCancelLegendTemplateCrop?.();
    setDrawMode(current => current === mode ? 'none' : mode);
  };

  const drawModeAccent =
    activeDrawMode === 'legend'
      ? '#38bdf8'
      : activeDrawMode === 'plan'
      ? '#22c55e'
      : activeDrawMode === 'inspect'
      ? '#a78bfa'
      : activeDrawMode === 'legend-template'
      ? 'var(--accent-gold)'
      : 'var(--accent-orange)';

  const drawModeBackground =
    activeDrawMode === 'legend'
      ? 'rgba(14,165,233,0.92)'
      : activeDrawMode === 'plan'
      ? 'rgba(34,197,94,0.92)'
      : activeDrawMode === 'inspect'
      ? 'rgba(124,58,237,0.92)'
      : activeDrawMode === 'legend-template'
      ? 'rgba(198,168,124,0.94)'
      : 'rgba(249,115,22,0.92)';

  const drawModeFill =
    activeDrawMode === 'legend'
      ? 'rgba(14,165,233,0.12)'
      : activeDrawMode === 'plan'
      ? 'rgba(34,197,94,0.10)'
      : activeDrawMode === 'inspect'
      ? 'rgba(124,58,237,0.12)'
      : activeDrawMode === 'legend-template'
      ? 'rgba(198,168,124,0.16)'
      : 'rgba(249,115,22,0.12)';

  const drawModeHint =
    activeDrawMode === 'legend'
      ? 'Zaznacz obszar legendy'
      : activeDrawMode === 'plan'
      ? 'Zaznacz glowny obszar planu'
      : activeDrawMode === 'inspect'
      ? 'Zaznacz symbol do inspekcji'
      : activeDrawMode === 'legend-template'
      ? `Zaznacz wzorzec: ${legendTemplateCropTarget?.name ?? ''}`
      : 'Zaznacz strefe wykluczona';

  const buildDebugPayload = (box: Box) => {
    const overlapGroup = overlapGroupsByBoxId.get(box.id);
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
    const planZoneUsed = analysisContext?.planZoneUsed
      ? analysisContext.planZoneUsed.join(",")
      : "(none)";
    const planZoneOutside = analysisContext?.planZoneOutsideExcluded?.length
      ? analysisContext.planZoneOutsideExcluded.map(zone => zone.join(",")).join(" | ")
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
      `reason=${box.reason ?? 'accepted_detection'}`,
      `related_final=${box.relatedFinal ? `${box.relatedFinal.symbolName ?? 'n/a'}@${(box.relatedFinal.bbox ?? []).join(',')}#${formatDebugValue(box.relatedFinal.verificationScore)}` : '(none)'}`,
      `analysis_id=${box.analysisId ?? analysisContext?.analysisId ?? 'n/a'}`,
      `analysis_generated_utc=${box.analysisGeneratedUtc ?? analysisContext?.generatedAtUtc ?? 'n/a'}`,
      `analysis_session=${box.analysisSession ?? analysisContext?.sessionId ?? 'n/a'}`,
      `source_pdf=${box.sourcePdf ?? analysisContext?.sourcePdf ?? 'n/a'}`,
      `hidden_layers_used=${hiddenLayers}`,
      `excluded_zones_used=${excludedZones}`,
      `plan_zone_used=${planZoneUsed}`,
      `plan_zone_outside_excluded=${planZoneOutside}`,
      `hidden_layers_unmatched=${hiddenLayerUnmatched}`,
      `hidden_layers_repr=${hiddenLayerReprs}`,
      `frontend_boxes_count=${boxes.length}`,
      `frontend_nearby_boxes=${nearbyBoxes.join(' || ') || '(none)'}`,
      `frontend_overlap_boxes=${formatOverlapGroup(overlapGroup) || '(none)'}`,
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
      <div data-wheel-ui="true" style={{ position: 'absolute', top: 16, right: 16, zIndex: 20, display: 'flex', gap: 6 }}>
        <button
          className="btn-secondary"
          onClick={() => toggleDrawMode('exclude')}
          title="Dodaj strefę wykluczoną"
          style={{
            borderColor: drawMode === 'exclude' ? 'var(--accent-orange)' : undefined,
            color: drawMode === 'exclude' ? 'var(--accent-orange)' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Slash size={14} />
          {drawMode === 'exclude' ? 'Rysuj...' : 'Strefa'}
        </button>
        <button
          className="btn-secondary"
          onClick={() => toggleDrawMode('legend')}
          title="Zaznacz strefe legendy"
          style={{
            borderColor: drawMode === 'legend' ? '#38bdf8' : undefined,
            color: drawMode === 'legend' ? '#38bdf8' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Layers size={14} />
          {drawMode === 'legend' ? 'Legenda...' : 'Legenda'}
        </button>
        <button
          className="btn-secondary"
          onClick={() => toggleDrawMode('plan')}
          title="Zaznacz glowny obszar planu do analizy"
          style={{
            borderColor: drawMode === 'plan' ? '#22c55e' : undefined,
            color: drawMode === 'plan' ? '#22c55e' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Maximize size={14} />
          {drawMode === 'plan' ? 'Plan...' : 'Plan'}
        </button>
        <button
          className="btn-secondary"
          onClick={() => toggleDrawMode('inspect')}
          title="Sprawdz, co silnik widzi w zaznaczonym fragmencie"
          style={{
            borderColor: drawMode === 'inspect' ? '#a78bfa' : undefined,
            color: drawMode === 'inspect' ? '#a78bfa' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <AlertTriangle size={14} />
          {drawMode === 'inspect' ? 'ROI...' : 'Inspektor'}
        </button>
        <button
          className="btn-secondary"
          onClick={onToggleGrayDebugZones}
          title="Pokaz czarne strefy i ROI skanowania gray"
          style={{
            borderColor: grayDebugOverlayImage ? '#f97316' : undefined,
            color: grayDebugOverlayImage ? '#f97316' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Layers size={14} />
          {isGrayDebugLoading ? 'Licze...' : grayDebugOverlayImage ? 'Ukryj strefy' : 'Strefy'}
        </button>
        <button
          className="btn-secondary"
          onClick={() => setIsManualMode(value => !value)}
          title="Dodaj symbol ręcznie"
          style={{
            borderColor: isManualMode ? 'var(--accent-gold)' : undefined,
            color: isManualMode ? 'var(--accent-gold)' : undefined,
            padding: '6px 10px',
            fontSize: 11,
            fontWeight: 700,
          }}
        >
          <Plus size={14} />
          {isManualMode ? 'Kliknij plan' : 'Dodaj'}
        </button>
        {focusedBox && onRejectBox && (
          <button
            className="btn-secondary"
            onClick={() => onRejectBox(focusedBox.id)}
            title="Usun zaznaczone wykrycie (Delete)"
            style={{
              borderColor: '#ef4444',
              color: '#ef4444',
              padding: '6px 10px',
              fontSize: 11,
              fontWeight: 700,
            }}
          >
            <Trash2 size={14} />
            Usun box
          </button>
        )}
        {legendTemplateCropTarget && (
          <button
            className="btn-secondary"
            onClick={onCancelLegendTemplateCrop}
            title="Anuluj korektę wzorca"
            style={{
              borderColor: 'var(--accent-gold)',
              color: 'var(--accent-gold)',
              padding: '6px 10px',
              fontSize: 11,
              fontWeight: 700,
              maxWidth: 170,
            }}
          >
            <X size={14} />
            {legendTemplateCropTarget.name}
          </button>
        )}
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
      {activeDrawMode !== 'none' && (
        <div style={{
          position: 'absolute', top: 64, left: '50%', transform: 'translateX(-50%)',
          background: drawModeBackground, color: '#fff',
          padding: '6px 18px', borderRadius: 6, fontSize: 12, fontWeight: 700,
          zIndex: 20, pointerEvents: 'none',
        }}>
          {drawModeHint}
        </div>
      )}

      {/* Canvas */}
      <div
        className="canvas-wrapper"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: activeDrawMode !== 'none' ? 'crosshair' : isDragging ? 'grabbing' : isSpaceDown ? 'grab' : 'default' }}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            position: 'absolute',
            transition: isDragging || isDrawing || isZooming ? 'none' : 'transform 0.25s ease-out',
            transformOrigin: '0 0',
          }}
        >
          <img
            src={imageSrc}
            alt="Plan view"
            className="plan-image"
            draggable={false}
            style={imageSize ? { width: imageSize.width, height: imageSize.height } : undefined}
          />

          {grayDebugOverlayImage && (
            <img
              src={grayDebugOverlayImage}
              alt="Gray debug zones overlay"
              draggable={false}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                imageRendering: 'auto',
              }}
            />
          )}

          {grayDebugOverlayImage && grayDebugInfo && (
            <div
              style={{
                position: 'absolute',
                left: 12,
                top: 12,
                background: 'rgba(15,23,42,0.88)',
                color: '#e5e7eb',
                border: '1px solid rgba(249,115,22,0.45)',
                borderRadius: 8,
                padding: '7px 9px',
                fontSize: 11,
                lineHeight: 1.35,
                pointerEvents: 'none',
                maxWidth: 340,
              }}
            >
              <strong style={{ color: '#fb923c' }}>Gray strefy</strong>
              <div>zielone: zone &lt;{grayDebugInfo.zoneThreshold}, pomaranczowe: evidence &lt;{grayDebugInfo.evidenceThreshold}</div>
              <div>ROI: {grayDebugInfo.roiCount} unikalnych / {grayDebugInfo.roiRefs} lacznie, templates {grayDebugInfo.templates}</div>
              <div>piksele: zone {grayDebugInfo.zonePixels}, evidence {grayDebugInfo.evidencePixels}</div>
            </div>
          )}

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

          {/* Manual plan zone */}
          {planZone && (
            <div
              style={{
                position: 'absolute',
                left: planZone.x,
                top: planZone.y,
                width: planZone.width,
                height: planZone.height,
                border: '3px dashed #22c55e',
                backgroundColor: 'rgba(34,197,94,0.05)',
                boxSizing: 'border-box',
                pointerEvents: 'auto',
              }}
            >
              <div style={{
                position: 'absolute',
                top: -22,
                left: 0,
                background: '#16a34a',
                color: '#fff',
                fontSize: 10,
                fontWeight: 800,
                padding: '2px 6px',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                whiteSpace: 'nowrap',
              }}>
                PLAN
                {onClearPlanZone && (
                  <button
                    onClick={e => { e.stopPropagation(); onClearPlanZone(); }}
                    style={{
                      background: 'rgba(0,0,0,0.35)',
                      border: 'none',
                      cursor: 'pointer',
                      color: '#fff',
                      display: 'flex',
                      padding: 2,
                      borderRadius: 3,
                    }}
                    title="Usun strefe planu"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Manual legend zone */}
          {legendZone && (
            <div
              style={{
                position: 'absolute',
                left: legendZone.x,
                top: legendZone.y,
                width: legendZone.width,
                height: legendZone.height,
                border: '3px dashed #38bdf8',
                backgroundColor: 'rgba(14,165,233,0.08)',
                boxSizing: 'border-box',
                pointerEvents: 'auto',
              }}
            >
              <div style={{
                position: 'absolute',
                top: -22,
                left: 0,
                background: '#0284c7',
                color: '#fff',
                fontSize: 10,
                fontWeight: 800,
                padding: '2px 6px',
                borderRadius: 4,
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                whiteSpace: 'nowrap',
              }}>
                LEGENDA
                {onClearLegendZone && (
                  <button
                    onClick={e => { e.stopPropagation(); onClearLegendZone(); }}
                    style={{
                      background: 'rgba(0,0,0,0.35)',
                      border: 'none',
                      cursor: 'pointer',
                      color: '#fff',
                      display: 'flex',
                      padding: 2,
                      borderRadius: 3,
                    }}
                    title="Usun strefe legendy"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Detection Boxes */}
          {drawableBoxes.map(box => {
            const isFocused = focusedBoxId === box.id || pulsingId === box.id;
            const isPulsing = pulsingId === box.id;
            const displayConfidence = box.verificationScore ?? box.confidence;
            const isLowConf = displayConfidence < 0.55;
            const color = isFocused ? BOX_FOCUS_COLOR : (isLowConf ? BOX_LOW_COLOR : box.color);
            const overlapGroup = overlapGroupsByBoxId.get(box.id);
            const overlapSummary = formatOverlapGroup(overlapGroup);

            return (
              <div
                key={`${box.analysisId ?? analysisContext?.analysisId ?? 'na'}:${box.id}`}
                onClick={e => {
                  e.stopPropagation();
                  onBoxClick?.(box.id);
                  void copyBoxDebug(box);
                }}
                title={`Weryfikacja: ${(displayConfidence * 100).toFixed(0)}%\nMatch template: ${(box.confidence * 100).toFixed(0)}%\nWzorzec: ${box.symbolName}${overlapSummary ? `\nNakladki: ${overlapSummary}` : ''}\nKlik kopiuje debug`}
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
                  {box.symbolName ? box.symbolName.split('_')[0].substring(0, 15) : 'Symbol'}
                </div>
                {overlapGroup && (
                  <div style={{
                    position: 'absolute',
                    top: -18,
                    right: -8,
                    minWidth: 22,
                    height: 16,
                    background: '#7c3aed',
                    color: '#fff',
                    border: '1px solid rgba(255,255,255,0.72)',
                    borderRadius: 999,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 9,
                    fontWeight: 900,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.28)',
                    pointerEvents: 'none',
                    zIndex: 6,
                  }}>
                    {overlapGroup.length}x
                  </div>
                )}
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
          {activeDrawMode !== 'none' && isDrawing && (
            <div style={{
              position: 'absolute',
              left: Math.min(drawStart.x, drawCurrent.x),
              top: Math.min(drawStart.y, drawCurrent.y),
              width: Math.abs(drawCurrent.x - drawStart.x),
              height: Math.abs(drawCurrent.y - drawStart.y),
              border: `2px dashed ${drawModeAccent}`,
              backgroundColor: drawModeFill,
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
                {symbolNames.map(s => <option key={s} value={s}>{formatSymbolLabel(s)}</option>)}
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
