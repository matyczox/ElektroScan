import { useState, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { CanvasView } from './components/CanvasView';
import { ResultsPanel } from './components/ResultsPanel';
import { LegendReviewPanel, type LegendReviewItem } from './components/LegendReviewPanel';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000';
const withNoCache = (path: string) => `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;

type DetectorProfile = 'auto' | 'color' | 'gray';

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
  detectorProfileRequested?: DetectorProfile;
  detectorProfileUsed?: 'color' | 'gray';
  pdfDiagnostics?: PdfDiagnostics;
}

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
  sessionId?: string;
  analysisId?: string | null;
  stage?: string;
  percent?: number;
  detail?: string;
  done?: boolean;
  error?: string | null;
  updatedAtUtc?: string | null;
}

interface DetectionBox {
  id: string;
  symbolName: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  color: string;
  verificationScore?: number;
  source?: string;
  rotation?: number;
  scale?: number;
  mirrored?: boolean;
  coverage?: number;
  purity?: number;
  contextPurity?: number;
  colorSimilarity?: number;
  reason?: string;
  analysisId?: string;
  analysisGeneratedUtc?: string;
  analysisSession?: string;
  sourcePdf?: string;
  hiddenLayersUsed?: string[];
}

interface RoiCandidate {
  symbolName: string;
  accepted: boolean;
  reason: string;
  match: number;
  threshold?: number;
  verification: number;
  coverage: number;
  purity: number;
  contextPurity: number;
  scale: number;
  rotation: number;
  mirrored: boolean;
  bbox: { x: number; y: number; width: number; height: number };
  scanMask?: string;
}

interface RoiInspection {
  roi: { x: number; y: number; width: number; height: number };
  profile: 'color' | 'gray';
  usedScales: number[];
  templates: number;
  variantsChecked: number;
  rawHitsByScale: Record<string, number>;
  rejectedByReason: Record<string, number>;
  roiInkPixels: number;
  roiScanPixels: number;
  roiDarkInkPixels?: number;
  roiDarkScanPixels?: number;
  grayDarkInkThreshold?: number;
  roiImage?: string;
  roiRawMask?: string;
  roiScanMask?: string;
  roiDarkRawMask?: string;
  roiDarkScanMask?: string;
  candidates: RoiCandidate[];
}

interface GrayDebugZones {
  overlayImage: string;
  imageWidth: number;
  imageHeight: number;
  zoneThreshold: number;
  evidenceThreshold: number;
  zonePixels: number;
  evidencePixels: number;
  roiCount: number;
  roiRefs: number;
  templates: number;
}

interface Pattern {
  id: string;
  name: string;
  imgBase64: string;
  status?: string;
  correctedBBoxPx?: [number, number, number, number];
}

interface ResultGroup {
  name: string;
  count: number;
  color: string;
}

interface LegendCorrectionTarget {
  id: string;
  name: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [legendReviewItems, setLegendReviewItems] = useState<LegendReviewItem[]>([]);
  const [isLegendReviewOpen, setIsLegendReviewOpen] = useState(false);
  const [legendCorrectionTarget, setLegendCorrectionTarget] = useState<LegendCorrectionTarget | null>(null);
  const [results, setResults] = useState<ResultGroup[]>([]);
  const [boxes, setBoxes] = useState<DetectionBox[]>([]);
  const [focusedBoxId, setFocusedBoxId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [excludedZones, setExcludedZones] = useState<ExcludedZone[]>([]);
  const [legendZone, setLegendZone] = useState<ExcludedZone | null>(null);
  const [planZone, setPlanZone] = useState<ExcludedZone | null>(null);
  const [layers, setLayers] = useState<{name: string, visible: boolean}[]>([]);
  const [analysisContext, setAnalysisContext] = useState<AnalysisContext | null>(null);
  const [detectorProfile, setDetectorProfile] = useState<DetectorProfile>('auto');
  const [pdfDiagnostics, setPdfDiagnostics] = useState<PdfDiagnostics | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null);
  const [roiInspection, setRoiInspection] = useState<RoiInspection | null>(null);
  const [isInspectingRoi, setIsInspectingRoi] = useState(false);
  const [grayDebugZones, setGrayDebugZones] = useState<GrayDebugZones | null>(null);
  const [isLoadingGrayZones, setIsLoadingGrayZones] = useState(false);
  const detectRequestSeqRef = useRef(0);
  const detectAbortRef = useRef<AbortController | null>(null);

  const legendReviewCompleted = legendReviewItems.filter(item => item.status !== 'pending').length;
  const hasLegendReview = legendReviewItems.length > 0;
  const isLegendReviewComplete = !hasLegendReview || legendReviewCompleted === legendReviewItems.length;

  const replacePattern = (oldId: string, pattern: Pattern) => {
    setPatterns(prev => {
      const index = prev.findIndex(item => (item.id ?? item.name) === oldId);
      if (index === -1) return [...prev, pattern];
      const next = [...prev];
      next[index] = pattern;
      return next;
    });
  };

  const fetchTemplates = async () => {
    try {
      const response = await fetch(withNoCache('/api/templates'), { cache: 'no-store' });
      if (response.ok) {
        const data = await response.json() as { patterns?: Pattern[] };
        setPatterns(data.patterns || []);
      }
    } catch (e) {
      console.error('Nie udało się pobrać szablonów', e);
    }
  };

  // ── Handlery ────────────────────────────────────────────

  const handleFileSelect = async (selectedFile: File) => {
    setFile(selectedFile);
    setPdfPreview(null);
    setPatterns([]);
    setResults([]);
    setBoxes([]);
    setLegendReviewItems([]);
    setIsLegendReviewOpen(false);
    setLegendCorrectionTarget(null);
    setSessionId(null);
    setExcludedZones([]);
    setLegendZone(null);
    setPlanZone(null);
    setFocusedBoxId(null);
    setAnalysisContext(null);
    setPdfDiagnostics(null);
    setAnalysisProgress(null);
    setRoiInspection(null);
    setGrayDebugZones(null);
    
    setIsProcessing(true);
    setProgressText('Ładowanie podglądu PDF...');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const res = await fetch(withNoCache('/api/preview'), {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });
      if (!res.ok) throw new Error('Błąd podglądu');
      const data = await res.json();
      setPdfPreview(data.planPreview);
      setSessionId(data.sessionId);
      setPdfDiagnostics(data.pdfDiagnostics || null);
      
      // Fetch layers
      fetch(withNoCache(`/api/layers?session_id=${data.sessionId}`), { cache: 'no-store' })
        .then(r => r.json())
        .then(d => setLayers(d.layers || []))
        .catch(err => console.error("Layers fetch error", err));
        
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleToggleLayer = async (layerName: string) => {
    if (!sessionId) return;
    const newLayers = layers.map(l => l.name === layerName ? { ...l, visible: !l.visible } : l);
    setLayers(newLayers);
    
    setIsProcessing(true);
    setProgressText('Przeliczanie warstw...');
    try {
      const hiddenLayers = newLayers.filter(l => !l.visible).map(l => l.name);
      const res = await fetch(withNoCache(`/api/render-preview?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({ hidden_layers: hiddenLayers })
      });
      if (!res.ok) throw new Error('Błąd odświeżania podglądu');
      const data = await res.json();
      setPdfPreview(data.planPreview);
      setPdfDiagnostics(data.pdfDiagnostics || null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleExtractLegend = async () => {
    if (!sessionId) return;
    if (!legendZone) {
      alert('Zaznacz strefę legendy na planie przed ekstrakcją (tryb Legenda na canvasie).');
      return;
    }
    setIsProcessing(true);
    setProgressText('Ekstrakcja legendy (300 DPI)...');
    try {
      const response = await fetch(withNoCache(`/api/extract-legend?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({
          excluded_zones: excludedZones.map(z => ({
            x: Math.round(z.x),
            y: Math.round(z.y),
            width: Math.round(z.width),
            height: Math.round(z.height),
          })),
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
          detector_profile: detectorProfile,
          legend_zone: legendZone ? {
            page: 0,
            x: Math.round(legendZone.x),
            y: Math.round(legendZone.y),
            width: Math.round(legendZone.width),
            height: Math.round(legendZone.height),
          } : undefined,
        })
      });
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json() as { patterns?: Pattern[]; pdfDiagnostics?: PdfDiagnostics };
      const nextPatterns = data.patterns || [];
      setPatterns(nextPatterns);
      setLegendReviewItems(
        nextPatterns.map(pattern => ({
          id: pattern.id ?? pattern.name,
          name: pattern.name,
          imgBase64: pattern.imgBase64,
          status: 'pending',
        }))
      );
      setIsLegendReviewOpen(nextPatterns.length > 0);
      setLegendCorrectionTarget(null);
      setPdfDiagnostics(data.pdfDiagnostics || pdfDiagnostics);
    } catch (error) {
      console.error(error);
      alert('Wystąpił błąd podczas ekstrakcji legendy. Upewnij się, że backend działa.');
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleDetect = async () => {
    if (!sessionId) return;
    if (!isLegendReviewComplete) {
      setIsLegendReviewOpen(true);
      alert('Sprawdź wszystkie wzorce legendy przed analizą.');
      return;
    }
    const detectStartedAt = performance.now();
    detectRequestSeqRef.current += 1;
    const requestSeq = detectRequestSeqRef.current;
    detectAbortRef.current?.abort();
    const controller = new AbortController();
    detectAbortRef.current = controller;

    setIsProcessing(true);
    setProgressText('Analiza hybrydowa (HSV + Complexity Sorting)...');
    setAnalysisProgress({ sessionId, stage: 'start', percent: 1, detail: 'Start analizy', done: false });
    setResults([]);
    setBoxes([]);
    setAnalysisContext(null);
    setFocusedBoxId(null);
    let progressTimer: number | null = null;
    try {
      progressTimer = window.setInterval(async () => {
        try {
          const progressResponse = await fetch(
            withNoCache(`/api/analysis-progress?session_id=${sessionId}`),
            { cache: 'no-store' },
          );
          if (!progressResponse.ok || requestSeq !== detectRequestSeqRef.current) return;
          const progressData = await progressResponse.json();
          const progress = progressData.progress as AnalysisProgress | undefined;
          if (!progress) return;
          setAnalysisProgress(progress);
          if (progress.detail) setProgressText(progress.detail);
          if (progress.done && progressTimer !== null) {
            window.clearInterval(progressTimer);
            progressTimer = null;
          }
        } catch {
          // Progress polling is best-effort; the analysis request remains authoritative.
        }
      }, 700);

      const fetchStartedAt = performance.now();
      const response = await fetch(withNoCache(`/api/analyze?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        cache: 'no-store',
        body: JSON.stringify({
          excluded_zones: excludedZones.map(z => ({
            x: Math.round(z.x),
            y: Math.round(z.y),
            width: Math.round(z.width),
            height: Math.round(z.height),
          })),
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
          include_image: false,
          include_debug: true,
          detector_profile: detectorProfile,
          legend_zone: legendZone ? {
            page: 0,
            x: Math.round(legendZone.x),
            y: Math.round(legendZone.y),
            width: Math.round(legendZone.width),
            height: Math.round(legendZone.height),
          } : undefined,
          plan_zone: planZone ? {
            page: 0,
            x: Math.round(planZone.x),
            y: Math.round(planZone.y),
            width: Math.round(planZone.width),
            height: Math.round(planZone.height),
          } : undefined,
        })
      });
      const responseReceivedAt = performance.now();
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json();
      const jsonParsedAt = performance.now();
      if (requestSeq !== detectRequestSeqRef.current) return;
      setResults(data.results);
      setBoxes(data.boxes || []);
      setAnalysisContext(data.analysisContext || null);
      setAnalysisProgress({
        sessionId,
        analysisId: data.analysisContext?.analysisId,
        stage: 'done',
        percent: 100,
        detail: 'Analiza zakonczona',
        done: true,
      });
      setPdfDiagnostics(data.analysisContext?.pdfDiagnostics || pdfDiagnostics);
      if (data.resultImage) setPdfPreview(data.resultImage);
      setFocusedBoxId(null);
      window.setTimeout(() => {
        const uiSettledAt = performance.now();
        console.info('[ElektroScan timing]', {
          totalMs: Math.round(uiSettledAt - detectStartedAt),
          requestWaitMs: Math.round(responseReceivedAt - fetchStartedAt),
          jsonParseMs: Math.round(jsonParsedAt - responseReceivedAt),
          reactApplyApproxMs: Math.round(uiSettledAt - jsonParsedAt),
          backendMs: Math.round(data.performance?.backendTimingsMs?.total ?? 0),
          detectorMs: Math.round(data.performance?.backendTimingsMs?.detectSymbolsTotal ?? 0),
          boxes: data.boxes?.length ?? 0,
        });
      }, 0);
    } catch (error) {
      if ((error as Error).name === 'AbortError') return;
      console.error(error);
      alert('Błąd podczas analizy planu.');
    } finally {
      if (requestSeq === detectRequestSeqRef.current) {
        if (progressTimer !== null) window.clearInterval(progressTimer);
        setIsProcessing(false);
        setProgressText('');
      }
    }
  };

  const handleClear = async () => {
    try {
      await fetch(withNoCache('/api/clear'), { method: 'POST', cache: 'no-store' });
    } catch { /* ignore */ }
    setFile(null);
    setPdfPreview(null);
    setPatterns([]);
    setLegendReviewItems([]);
    setIsLegendReviewOpen(false);
    setLegendCorrectionTarget(null);
    setResults([]);
    setBoxes([]);
    setLayers([]);
    setSessionId(null);
    setExcludedZones([]);
    setLegendZone(null);
    setPlanZone(null);
    setFocusedBoxId(null);
    setAnalysisContext(null);
    setPdfDiagnostics(null);
    setAnalysisProgress(null);
    setRoiInspection(null);
    setGrayDebugZones(null);
  };

  const handleToggleGrayZones = async () => {
    if (grayDebugZones) {
      setGrayDebugZones(null);
      return;
    }
    if (!sessionId) return;
    setIsLoadingGrayZones(true);
    setProgressText('Liczenie czarnych stref...');
    try {
      const response = await fetch(withNoCache(`/api/gray-debug-zones?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
          detector_profile: detectorProfile,
          excluded_zones: excludedZones,
          legend_zone: legendZone ? { page: 0, x: legendZone.x, y: legendZone.y, width: legendZone.width, height: legendZone.height } : null,
          plan_zone: planZone ? { page: 0, x: planZone.x, y: planZone.y, width: planZone.width, height: planZone.height } : null,
        }),
      });
      if (!response.ok) throw new Error('Blad podgladu stref gray');
      const data = await response.json();
      setGrayDebugZones(data);
    } catch (error) {
      console.error(error);
      alert('Nie udalo sie policzyc czarnych stref.');
    } finally {
      setIsLoadingGrayZones(false);
      setProgressText('');
    }
  };

  const handleInspectRoi = async (x: number, y: number, width: number, height: number) => {
    if (!sessionId) return;
    setIsInspectingRoi(true);
    setProgressText('Inspektor ROI liczy dopasowania...');
    try {
      const response = await fetch(withNoCache(`/api/inspect-roi?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
          detector_profile: detectorProfile,
          top_n: 18,
          roi: {
            page: 0,
            x: Math.round(x),
            y: Math.round(y),
            width: Math.round(width),
            height: Math.round(height),
          },
        }),
      });
      if (!response.ok) throw new Error('Blad inspektora ROI');
      const data = await response.json();
      setRoiInspection(data.inspection || null);
      setPdfDiagnostics(data.pdfDiagnostics || pdfDiagnostics);
    } catch (error) {
      console.error(error);
      alert('Nie udalo sie sprawdzic ROI.');
    } finally {
      setIsInspectingRoi(false);
      setProgressText('');
    }
  };

  const handleClearTemplates = async () => {
    try {
      await fetch(withNoCache('/api/templates'), { method: 'DELETE', cache: 'no-store' });
      setPatterns([]);
      setLegendReviewItems([]);
      setIsLegendReviewOpen(false);
      setLegendCorrectionTarget(null);
    } catch (e) {
      console.error('Błąd podczas czyszczenia bazy wiedzy', e);
    }
  };

  const handleUpdatePattern = (index: number, newName: string) => {
    const updated = [...patterns];
    updated[index] = { ...updated[index], name: newName };
    setPatterns(updated);
  };

  const handleDeletePattern = async (index: number) => {
    const pattern = patterns[index];
    if (!pattern) return;

    try {
      const templateId = pattern.id ?? pattern.name;
      const response = await fetch(withNoCache(`/api/templates/${encodeURIComponent(templateId)}`), {
        method: 'DELETE',
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error('Nie udało się usunąć wzorca');
      }

      setPatterns(prev => prev.filter((_, currentIndex) => currentIndex !== index));
      setLegendReviewItems(prev => prev.filter(item => item.id !== templateId));
      if (legendCorrectionTarget?.id === templateId) setLegendCorrectionTarget(null);
    } catch (error) {
      console.error('Błąd podczas usuwania wzorca', error);
      alert('Nie udało się usunąć wzorca z bazy wiedzy.');
    }
  };

  const handleAcceptLegendItem = (id: string) => {
    setLegendReviewItems(prev =>
      prev.map(item => item.id === id ? { ...item, status: 'accepted' } : item)
    );
  };

  const handleRejectLegendItem = async (id: string) => {
    const reviewItem = legendReviewItems.find(item => item.id === id);
    if (reviewItem && !reviewItem.imgBase64) {
      setLegendReviewItems(prev =>
        prev.map(item => item.id === id ? { ...item, status: 'rejected' } : item)
      );
      if (legendCorrectionTarget?.id === id) setLegendCorrectionTarget(null);
      return;
    }

    try {
      const response = await fetch(withNoCache(`/api/templates/${encodeURIComponent(id)}`), {
        method: 'DELETE',
        cache: 'no-store',
      });
      if (!response.ok) throw new Error('Nie udało się odrzucić wzorca');

      setPatterns(prev => prev.filter(pattern => (pattern.id ?? pattern.name) !== id));
      setLegendReviewItems(prev =>
        prev.map(item => item.id === id ? { ...item, status: 'rejected' } : item)
      );
      if (legendCorrectionTarget?.id === id) setLegendCorrectionTarget(null);
    } catch (error) {
      console.error('Błąd podczas odrzucania wzorca', error);
      alert('Nie udało się odrzucić wzorca.');
    }
  };

  const handleRenameLegendItem = async (id: string, name: string) => {
    if (!name.trim()) {
      alert('Nazwa wzorca nie może być pusta.');
      return;
    }

    try {
      const response = await fetch(withNoCache(`/api/templates/${encodeURIComponent(id)}`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({ name: name.trim() }),
      });
      if (!response.ok) throw new Error('Nie udało się zmienić nazwy wzorca');
      const data = await response.json() as { pattern?: Pattern };
      const pattern = data.pattern;
      if (!pattern) throw new Error('Brak wzorca w odpowiedzi backendu');

      replacePattern(id, pattern);
      setLegendReviewItems(prev =>
        prev.map(item =>
          item.id === id
            ? { ...item, id: pattern.id, name: pattern.name, imgBase64: pattern.imgBase64 }
            : item
        )
      );
      setLegendCorrectionTarget(current =>
        current?.id === id ? { id: pattern.id, name: pattern.name } : current
      );
    } catch (error) {
      console.error('Błąd podczas zmiany nazwy wzorca', error);
      alert('Nie udało się zmienić nazwy wzorca.');
    }
  };

  const handleStartLegendCrop = (item: LegendReviewItem) => {
    setIsLegendReviewOpen(true);
    setLegendCorrectionTarget({ id: item.id, name: item.name });
  };

  const handleAddMissingLegendItem = () => {
    const name = window.prompt('Nazwa brakującego wzorca');
    if (!name?.trim()) return;

    const id = `manual_${Date.now()}`;
    const item: LegendReviewItem = {
      id,
      name: name.trim(),
      imgBase64: '',
      status: 'pending',
    };
    setLegendReviewItems(prev => [...prev, item]);
    setIsLegendReviewOpen(true);
    setLegendCorrectionTarget({ id, name: item.name });
  };

  const handleLegendTemplateCrop = async (x: number, y: number, width: number, height: number) => {
    if (!sessionId || !legendCorrectionTarget) return;

    const target = legendCorrectionTarget;
    setIsProcessing(true);
    setProgressText(`Zapisywanie wzorca ${target.name}...`);
    try {
      const response = await fetch(withNoCache(`/api/templates/${encodeURIComponent(target.id)}/crop`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({
          session_id: sessionId,
          x: Math.round(x),
          y: Math.round(y),
          width: Math.round(width),
          height: Math.round(height),
          name: target.id.startsWith('manual_') ? target.name : undefined,
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
        }),
      });
      if (!response.ok) throw new Error('Nie udało się zapisać ręcznego cropa');
      const data = await response.json() as { pattern?: Pattern };
      const pattern = data.pattern;
      if (!pattern) throw new Error('Brak wzorca w odpowiedzi backendu');

      replacePattern(target.id, pattern);
      setLegendReviewItems(prev => {
        const nextItem: LegendReviewItem = {
          id: pattern.id,
          name: pattern.name,
          imgBase64: pattern.imgBase64,
          status: 'fixed',
          correctedBBoxPx: pattern.correctedBBoxPx,
        };
        const exists = prev.some(item => item.id === target.id);
        if (!exists) return [...prev, nextItem];
        return prev.map(item => item.id === target.id ? nextItem : item);
      });
      setLegendCorrectionTarget(null);
      setIsLegendReviewOpen(true);
    } catch (error) {
      console.error('Błąd podczas zapisu ręcznego cropa', error);
      alert('Nie udało się zapisać poprawionego wzorca.');
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleRejectBox = (id: string) => {
    setBoxes(prev => prev.filter(b => b.id !== id));
    if (focusedBoxId === id) setFocusedBoxId(null);
  };

  const handleChangeBoxSymbol = (id: string, symbolName: string) => {
    setBoxes(prev => prev.map(box => box.id === id ? { ...box, symbolName } : box));
    setResults(prev => {
      if (prev.some(result => result.name === symbolName)) return prev;
      return [...prev, { name: symbolName, count: 0, color: '#22c55e' }];
    });
  };

  const handleAddManualBox = (box: Omit<DetectionBox, 'id' | 'color'>) => {
    // Kolor z backendu lub domyślny złoty, id losowe
    const newBox = {
      ...box,
      id: `manual_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      color: '#c6a87c', // Zostanie zaktualizowany przy ew. ponownej analizie, ale Canvas zignoruje bo używa wbudowanego
    };
    
    // ResultsPanel potrzebuje kolorów w grupach, ale grupy bierze z "results".
    // Musimy upewnić się, że symbol istnieje w results, jeśli nie, to go dodać.
    setResults(prev => {
      const exists = prev.find(r => r.name === box.symbolName);
      if (exists) return prev;
      
      // Prosty hash koloru, jeśli go nie było
      const hash = box.symbolName.split('').reduce((a,b)=>{a=((a<<5)-a)+b.charCodeAt(0);return a&a},0);
      const r = (hash >> 16) & 0xFF | 0x40;
      const g = (hash >> 8) & 0xFF | 0x40;
      const b = hash & 0xFF | 0x40;
      const color = `#${Math.min(r,255).toString(16).padStart(2,'0')}${Math.min(g,255).toString(16).padStart(2,'0')}${Math.min(b,255).toString(16).padStart(2,'0')}`;

      return [...prev, { name: box.symbolName, count: 0, color }];
    });

    setBoxes(prev => [...prev, newBox]);
  };

  // ── Render ──────────────────────────────────────────────

  return (
    <div className="app-container">
      {/* Lewy panel */}
      <Sidebar
        fileName={file?.name || null}
        onFileSelect={handleFileSelect}
        onExtractLegend={handleExtractLegend}
        onDetect={handleDetect}
        onClear={handleClear}
        onClearTemplates={handleClearTemplates}
        isProcessing={isProcessing || isInspectingRoi || isLoadingGrayZones}
        progressText={progressText}
        analysisProgress={analysisProgress}
        patterns={patterns}
        legendReviewTotal={legendReviewItems.length}
        legendReviewCompleted={legendReviewCompleted}
        isLegendReviewComplete={isLegendReviewComplete}
        onOpenLegendReview={() => setIsLegendReviewOpen(true)}
        onUpdatePattern={handleUpdatePattern}
        onDeletePattern={handleDeletePattern}
        layers={layers}
        onToggleLayer={handleToggleLayer}
        detectorProfile={detectorProfile}
        onDetectorProfileChange={setDetectorProfile}
        pdfDiagnostics={pdfDiagnostics}
        hasLegendZone={Boolean(legendZone)}
        onClearLegendZone={() => setLegendZone(null)}
        hasPlanZone={Boolean(planZone)}
        onClearPlanZone={() => setPlanZone(null)}
      />

      {/* Środek: Canvas */}
      <CanvasView
        key={analysisContext?.analysisId ?? sessionId ?? 'canvas-empty'}
        imageSrc={pdfPreview}
        boxes={boxes}
        analysisContext={analysisContext}
        focusedBoxId={focusedBoxId}
        onBoxClick={id => setFocusedBoxId(prev => prev === id ? null : id)}
        excludedZones={excludedZones}
        legendZone={legendZone}
        planZone={planZone}
        onAddExcludedZone={(x, y, w, h) => setExcludedZones(prev => [...prev, { x, y, width: w, height: h }])}
        onRemoveExcludedZone={idx => setExcludedZones(prev => prev.filter((_, i) => i !== idx))}
        onSetLegendZone={(x, y, w, h) => setLegendZone({ x, y, width: w, height: h })}
        onClearLegendZone={() => setLegendZone(null)}
        onSetPlanZone={(x, y, w, h) => setPlanZone({ x, y, width: w, height: h })}
        onClearPlanZone={() => setPlanZone(null)}
        symbolNames={patterns.map(p => p.name)}
        onAddManualBox={handleAddManualBox}
        onRejectBox={handleRejectBox}
        onInspectZone={handleInspectRoi}
        grayDebugOverlayImage={grayDebugZones?.overlayImage ?? null}
        grayDebugInfo={grayDebugZones}
        onToggleGrayDebugZones={handleToggleGrayZones}
        isGrayDebugLoading={isLoadingGrayZones}
        legendTemplateCropTarget={legendCorrectionTarget}
        onLegendTemplateCrop={handleLegendTemplateCrop}
        onCancelLegendTemplateCrop={() => setLegendCorrectionTarget(null)}
      />

      {isLegendReviewOpen && (
        <LegendReviewPanel
          items={legendReviewItems}
          activeCorrectionId={legendCorrectionTarget?.id ?? null}
          isProcessing={isProcessing}
          onAccept={handleAcceptLegendItem}
          onReject={handleRejectLegendItem}
          onStartCrop={handleStartLegendCrop}
          onCancelCrop={() => setLegendCorrectionTarget(null)}
          onRename={handleRenameLegendItem}
          onAddMissing={handleAddMissingLegendItem}
          onClose={() => setIsLegendReviewOpen(false)}
        />
      )}

      {roiInspection && (
        <div
          style={{
            position: 'fixed',
            right: 360,
            bottom: 18,
            width: 430,
            maxHeight: '72vh',
            overflow: 'auto',
            zIndex: 80,
            background: 'rgba(15, 23, 42, 0.96)',
            border: '1px solid rgba(198,168,124,0.45)',
            borderRadius: 12,
            padding: 14,
            color: 'var(--text-main)',
            boxShadow: '0 18px 50px rgba(0,0,0,0.45)',
          }}
        >
          <div className="flex-row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontWeight: 900, color: 'var(--accent-gold)' }}>Inspektor ROI</div>
              <div className="text-xs text-muted">
                {roiInspection.profile} | ROI {roiInspection.roi.x},{roiInspection.roi.y},{roiInspection.roi.width},{roiInspection.roi.height}
              </div>
            </div>
            <button className="btn-icon" onClick={() => setRoiInspection(null)}>x</button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
            {roiInspection.roiImage && <img src={roiInspection.roiImage} alt="roi" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
            {roiInspection.roiRawMask && <img src={roiInspection.roiRawMask} alt="raw mask" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
            {roiInspection.roiScanMask && <img src={roiInspection.roiScanMask} alt="scan mask" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
            {roiInspection.roiDarkScanMask && <img src={roiInspection.roiDarkScanMask} alt="dark scan mask" title="dark scan mask" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
          </div>

          <div className="text-xs text-muted" style={{ marginTop: 8, lineHeight: 1.45 }}>
            <div>Skale: {roiInspection.usedScales.map(scale => scale.toFixed(2)).join(', ')}</div>
            <div>Tusz ROI: raw {roiInspection.roiInkPixels}, scan {roiInspection.roiScanPixels}</div>
            {roiInspection.roiDarkInkPixels !== undefined && (
              <div>
                Czarny tusz (&lt;{roiInspection.grayDarkInkThreshold ?? '?'}): raw {roiInspection.roiDarkInkPixels}, scan {roiInspection.roiDarkScanPixels ?? 0}
              </div>
            )}
            <div>Peaki/skala: {Object.entries(roiInspection.rawHitsByScale).map(([scale, count]) => `${scale}:${count}`).join(' | ') || '(brak)'}</div>
            <div>Odrzuty: {Object.entries(roiInspection.rejectedByReason).map(([reason, count]) => `${reason}:${count}`).join(' | ') || '(brak)'}</div>
          </div>

          <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {roiInspection.candidates.length === 0 ? (
              <div className="text-sm text-muted">Brak kandydatow w zaznaczeniu.</div>
            ) : roiInspection.candidates.map((candidate, index) => (
              <div
                key={`${candidate.symbolName}_${index}_${candidate.scale}_${candidate.rotation}`}
                style={{
                  border: `1px solid ${candidate.accepted ? 'rgba(34,197,94,0.65)' : 'rgba(239,68,68,0.45)'}`,
                  borderRadius: 8,
                  padding: 8,
                  background: candidate.accepted ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.06)',
                }}
              >
                <div className="flex-row" style={{ justifyContent: 'space-between', gap: 8 }}>
                  <strong style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {index + 1}. {candidate.symbolName}
                  </strong>
                  <span style={{ fontSize: 11, color: candidate.accepted ? '#22c55e' : '#f87171', fontWeight: 900 }}>
                    {candidate.accepted ? 'PASS' : candidate.reason}
                  </span>
                </div>
                <div className="text-xs text-muted" style={{ marginTop: 4 }}>
                  match {candidate.match.toFixed(3)}
                  {candidate.threshold !== undefined ? ` / thr ${candidate.threshold.toFixed(3)}` : ''}
                  {' | '}ver {candidate.verification.toFixed(3)} | cov {candidate.coverage.toFixed(3)} | pur {candidate.purity.toFixed(3)} | ctx {candidate.contextPurity.toFixed(3)}
                </div>
                <div className="text-xs text-muted">
                  scale {candidate.scale.toFixed(2)} | rot {candidate.rotation} | mirror {candidate.mirrored ? 'yes' : 'no'} | mask {candidate.scanMask ?? '?'} | bbox {candidate.bbox.x},{candidate.bbox.y},{candidate.bbox.width},{candidate.bbox.height}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prawy panel */}
      {results.length > 0 && (
        <ResultsPanel
          results={results}
          boxes={boxes}
          focusedBoxId={focusedBoxId}
          onFocusBox={id => setFocusedBoxId(prev => prev === id ? null : id)}
          onRejectBox={handleRejectBox}
          onChangeBoxSymbol={handleChangeBoxSymbol}
          symbolNames={patterns.map(p => p.name)}
          onTemplateUploaded={fetchTemplates}
        />
      )}
    </div>
  );
}

export default App;
