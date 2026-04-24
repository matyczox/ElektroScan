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

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setPdfPreview(null);
    setResults([]);
    setBoxes([]);
    setSessionId(null);
    setExcludedZones([]);
    setFocusedBoxId(null);
  };

  const handleExtractLegend = async () => {
    if (!file) return;
    setIsProcessing(true);
    setProgressText('Przesyłanie i konwersja PDF (300 DPI)...');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('http://localhost:8000/api/extract-legend', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Błąd serwera');
      const data = await response.json();
      setPatterns(data.patterns);
      setPdfPreview(data.planPreview);
      setSessionId(data.sessionId);
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
          }))
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
