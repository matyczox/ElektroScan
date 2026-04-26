import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { CanvasView } from './components/CanvasView';
import { ResultsPanel } from './components/ResultsPanel';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000';
const withNoCache = (path: string) => `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;

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

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [patterns, setPatterns] = useState<any[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [boxes, setBoxes] = useState<any[]>([]);
  const [focusedBoxId, setFocusedBoxId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [excludedZones, setExcludedZones] = useState<ExcludedZone[]>([]);
  const [layers, setLayers] = useState<{name: string, visible: boolean}[]>([]);
  const [analysisContext, setAnalysisContext] = useState<AnalysisContext | null>(null);
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

  // ── Handlery ────────────────────────────────────────────

  const handleFileSelect = async (selectedFile: File) => {
    setFile(selectedFile);
    setPdfPreview(null);
    setResults([]);
    setBoxes([]);
    setSessionId(null);
    setExcludedZones([]);
    setFocusedBoxId(null);
    setAnalysisContext(null);
    
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
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name)
        })
      });
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json();
      setPatterns(data.patterns);
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
    detectRequestSeqRef.current += 1;
    const requestSeq = detectRequestSeqRef.current;
    detectAbortRef.current?.abort();
    const controller = new AbortController();
    detectAbortRef.current = controller;

    setIsProcessing(true);
    setProgressText('Analiza hybrydowa (HSV + Complexity Sorting)...');
    setResults([]);
    setBoxes([]);
    setAnalysisContext(null);
    setFocusedBoxId(null);
    try {
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
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name)
        })
      });
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json();
      if (requestSeq !== detectRequestSeqRef.current) return;
      setResults(data.results);
      setBoxes(data.boxes || []);
      setAnalysisContext(data.analysisContext || null);
      setPdfPreview(data.resultImage);
      setFocusedBoxId(null);
    } catch (error) {
      if ((error as Error).name === 'AbortError') return;
      console.error(error);
      alert('Błąd podczas analizy planu.');
    } finally {
      if (requestSeq === detectRequestSeqRef.current) {
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
    setLayers([]);
    setSessionId(null);
    setExcludedZones([]);
    setFocusedBoxId(null);
    setAnalysisContext(null);
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
      const response = await fetch(withNoCache(`/api/templates/${encodeURIComponent(pattern.name)}`), {
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

  const handleAddManualBox = (box: Omit<any, 'id' | 'color'>) => {
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
        patterns={patterns}
        onUpdatePattern={handleUpdatePattern}
        onDeletePattern={handleDeletePattern}
        layers={layers}
        onToggleLayer={handleToggleLayer}
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
        onAddExcludedZone={(x, y, w, h) => setExcludedZones(prev => [...prev, { x, y, width: w, height: h }])}
        onRemoveExcludedZone={idx => setExcludedZones(prev => prev.filter((_, i) => i !== idx))}
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
          onTemplateUploaded={fetchTemplates}
        />
      )}
    </div>
  );
}

export default App;
