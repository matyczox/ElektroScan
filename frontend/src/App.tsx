import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { CanvasView } from './components/CanvasView';
import { ResultsPanel } from './components/ResultsPanel';
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
  includeDebugCandidates?: boolean;
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

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [patterns, setPatterns] = useState<any[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [boxes, setBoxes] = useState<DetectionBox[]>([]);
  const [debugCandidates, setDebugCandidates] = useState<DetectionBox[]>([]);
  const [focusedBoxId, setFocusedBoxId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [excludedZones, setExcludedZones] = useState<ExcludedZone[]>([]);
  const [legendZone, setLegendZone] = useState<ExcludedZone | null>(null);
  const [layers, setLayers] = useState<{name: string, visible: boolean}[]>([]);
  const [analysisContext, setAnalysisContext] = useState<AnalysisContext | null>(null);
  const [detectorProfile, setDetectorProfile] = useState<DetectorProfile>('auto');
  const [showDebugCandidates, setShowDebugCandidates] = useState(false);
  const [pdfDiagnostics, setPdfDiagnostics] = useState<PdfDiagnostics | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null);
  const detectRequestSeqRef = useRef(0);
  const detectAbortRef = useRef<AbortController | null>(null);

  const fetchTemplates = async () => {
    try {
      const response = await fetch(withNoCache('/api/templates'), { cache: 'no-store' });
      if (response.ok) {
        const data = await response.json();
        setPatterns(data.patterns || []);
      }
    } catch (e) {
      console.error('Nie udało się pobrać szablonów', e);
    }
  };

  useEffect(() => {
    fetchTemplates();
  }, []);

  useEffect(() => {
    if (!showDebugCandidates) setDebugCandidates([]);
  }, [showDebugCandidates]);

  // ── Handlery ────────────────────────────────────────────

  const handleFileSelect = async (selectedFile: File) => {
    setFile(selectedFile);
    setPdfPreview(null);
    setResults([]);
    setBoxes([]);
    setDebugCandidates([]);
    setSessionId(null);
    setExcludedZones([]);
    setLegendZone(null);
    setFocusedBoxId(null);
    setAnalysisContext(null);
    setPdfDiagnostics(null);
    setAnalysisProgress(null);
    
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
    setIsProcessing(true);
    setProgressText(legendZone ? 'Ekstrakcja zaznaczonej legendy (300 DPI)...' : 'Ekstrakcja legendy (300 DPI)...');
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
      const data = await response.json();
      setPatterns(data.patterns);
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
    setDebugCandidates([]);
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
        } catch (_error) {
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
          include_debug_candidates: showDebugCandidates,
          detector_profile: detectorProfile,
        })
      });
      const responseReceivedAt = performance.now();
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json();
      const jsonParsedAt = performance.now();
      if (requestSeq !== detectRequestSeqRef.current) return;
      setResults(data.results);
      setBoxes(data.boxes || []);
      setDebugCandidates(showDebugCandidates ? (data.debugCandidates || []) : []);
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
    } catch (_e) { /* ignore */ }
    setFile(null);
    setPdfPreview(null);
    setPatterns([]);
    setResults([]);
    setBoxes([]);
    setDebugCandidates([]);
    setLayers([]);
    setSessionId(null);
    setExcludedZones([]);
    setLegendZone(null);
    setFocusedBoxId(null);
    setAnalysisContext(null);
    setPdfDiagnostics(null);
    setAnalysisProgress(null);
  };

  const handleClearTemplates = async () => {
    try {
      await fetch(withNoCache('/api/templates'), { method: 'DELETE', cache: 'no-store' });
      setPatterns([]);
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
    } catch (error) {
      console.error('Błąd podczas usuwania wzorca', error);
      alert('Nie udało się usunąć wzorca z bazy wiedzy.');
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

  const handleDismissDebugCandidate = (id: string) => {
    setDebugCandidates(prev => prev.filter(candidate => candidate.id !== id));
  };

  const handleAcceptDebugCandidate = (candidate: DetectionBox) => {
    const symbolName = candidate.symbolName === 'possible_missed'
      ? (patterns[0]?.name ?? candidate.symbolName)
      : candidate.symbolName;
    handleAddManualBox({
      symbolName,
      x: candidate.x,
      y: candidate.y,
      width: candidate.width,
      height: candidate.height,
      confidence: candidate.confidence || 1.0,
      verificationScore: candidate.verificationScore,
      source: `hitl_${candidate.reason ?? 'debug_candidate'}`,
      rotation: candidate.rotation,
      scale: candidate.scale,
      mirrored: candidate.mirrored,
      coverage: candidate.coverage,
      purity: candidate.purity,
      contextPurity: candidate.contextPurity,
      colorSimilarity: candidate.colorSimilarity,
    });
    handleDismissDebugCandidate(candidate.id);
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
        isProcessing={isProcessing}
        progressText={progressText}
        analysisProgress={analysisProgress}
        patterns={patterns}
        onUpdatePattern={handleUpdatePattern}
        onDeletePattern={handleDeletePattern}
        layers={layers}
        onToggleLayer={handleToggleLayer}
        detectorProfile={detectorProfile}
        onDetectorProfileChange={setDetectorProfile}
        showDebugCandidates={showDebugCandidates}
        onShowDebugCandidatesChange={setShowDebugCandidates}
        pdfDiagnostics={pdfDiagnostics}
        hasLegendZone={Boolean(legendZone)}
        onClearLegendZone={() => setLegendZone(null)}
      />

      {/* Środek: Canvas */}
      <CanvasView
        key={analysisContext?.analysisId ?? sessionId ?? 'canvas-empty'}
        imageSrc={pdfPreview}
        boxes={boxes}
        debugCandidates={debugCandidates}
        analysisContext={analysisContext}
        focusedBoxId={focusedBoxId}
        onBoxClick={id => setFocusedBoxId(prev => prev === id ? null : id)}
        onAcceptDebugCandidate={handleAcceptDebugCandidate}
        onDismissDebugCandidate={handleDismissDebugCandidate}
        excludedZones={excludedZones}
        legendZone={legendZone}
        onAddExcludedZone={(x, y, w, h) => setExcludedZones(prev => [...prev, { x, y, width: w, height: h }])}
        onRemoveExcludedZone={idx => setExcludedZones(prev => prev.filter((_, i) => i !== idx))}
        onSetLegendZone={(x, y, w, h) => setLegendZone({ x, y, width: w, height: h })}
        onClearLegendZone={() => setLegendZone(null)}
        symbolNames={patterns.map(p => p.name)}
        onAddManualBox={handleAddManualBox}
      />

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
          debugCandidates={debugCandidates}
          onAcceptDebugCandidate={handleAcceptDebugCandidate}
          onDismissDebugCandidate={handleDismissDebugCandidate}
          onTemplateUploaded={fetchTemplates}
        />
      )}
    </div>
  );
}

export default App;
