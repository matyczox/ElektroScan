import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { CanvasView } from './components/CanvasView';
import { ResultsPanel } from './components/ResultsPanel';
import './index.css';

interface ExcludedZone {
  x: number;
  y: number;
  width: number;
  height: number;
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

  const fetchTemplates = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/templates');
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
    
    setIsProcessing(true);
    setProgressText('Ładowanie podglądu PDF...');
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const res = await fetch('http://localhost:8000/api/preview', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Błąd podglądu');
      const data = await res.json();
      setPdfPreview(data.planPreview);
      setSessionId(data.sessionId);
      
      // Fetch layers
      fetch(`http://localhost:8000/api/layers?session_id=${data.sessionId}`)
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
      const res = await fetch(`http://localhost:8000/api/render-preview?session_id=${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      const response = await fetch(`http://localhost:8000/api/extract-legend?session_id=${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
    setIsProcessing(true);
    setProgressText('Analiza hybrydowa (HSV + Complexity Sorting)...');
    try {
      const response = await fetch(`http://localhost:8000/api/analyze?session_id=${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      setResults(data.results);
      setBoxes(data.boxes || []);
      setPdfPreview(data.resultImage);
      setFocusedBoxId(null);
    } catch (error) {
      console.error(error);
      alert('Błąd podczas analizy planu.');
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleClear = async () => {
    try {
      await fetch('http://localhost:8000/api/clear', { method: 'POST' });
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
  };

  const handleClearTemplates = async () => {
    try {
      await fetch('http://localhost:8000/api/templates', { method: 'DELETE' });
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

  const handleDeletePattern = (index: number) => {
    const updated = [...patterns];
    updated.splice(index, 1);
    setPatterns(updated);
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
        imageSrc={pdfPreview}
        boxes={boxes}
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
