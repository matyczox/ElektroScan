import { useEffect, useRef, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { apiFetch, projectApiPath, readApiError } from './api';
import { AuthScreen, type AuthUser } from './components/AuthScreen';
import {
  ProjectDashboard,
  type AuthSession,
  type ProjectAnalysisRun,
  type ProjectSummary,
} from './components/ProjectDashboard';
import { Sidebar } from './components/Sidebar';
import { CanvasView } from './components/CanvasView';
import { ResultsPanel } from './components/ResultsPanel';
import {
  LegendReviewPanel,
  type LegendReviewItem,
  type LegendReviewStatus,
} from './components/LegendReviewPanel';
import './index.css';

const getPatternKey = (pattern: Pick<Pattern, 'id' | 'name'>) => pattern.id ?? pattern.name;

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

interface LegendEngineStatus {
  engineRequested?: 'auto' | 'raster' | 'vector_first';
  engineUsed?: 'raster' | 'vector_first';
  fallbackReason?: string | null;
  pageKind?: string;
  patternCount: number;
}

interface PreviewMeta {
  previewDpi?: number;
  analysisDpi?: number;
  analysisSize?: { width: number; height: number };
  isFullResolution?: boolean;
  renderCacheHit?: boolean;
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

interface AnalysisSnapshot {
  analysisContext?: AnalysisContext;
  results?: ResultGroup[];
  boxes?: DetectionBox[];
}

interface ExtractLegendResponse {
  patterns?: Pattern[];
  legendExtractedCount?: number;
  legendAddedIds?: string[];
  legendZoneUsed?: [number, number, number, number];
  pdfDiagnostics?: PdfDiagnostics;
  legendEngineRequested?: LegendEngineStatus['engineRequested'];
  legendEngineUsed?: LegendEngineStatus['engineUsed'];
  legendFallbackReason?: string | null;
  legendPageProfile?: {
    page_kind?: string;
    pageKind?: string;
  } | null;
}

interface DetectionBox {
  id: string;
  symbolName: string;
  x: number;
  y: number;
  width: number;
  height: number;
  visualBBox?: [number, number, number, number] | null;
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
  note?: string;
  reviewStatus?: 'unchecked' | 'accepted' | 'wrong' | 'manual_check';
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
  roiColorScanPixels?: number;
  roiColorScanTemplate?: {
    templateId: number;
    symbolName: string;
    scanMask: string;
    dominantHsv?: [number, number, number] | null;
    maskBBox?: [number, number, number, number] | null;
  } | null;
  roiDarkInkPixels?: number;
  roiDarkScanPixels?: number;
  grayDarkInkThreshold?: number;
  roiImage?: string;
  roiRawMask?: string;
  roiScanMask?: string;
  roiColorScanMask?: string;
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

const legendEngineLabel = (engine?: string) => {
  if (engine === 'vector_first') return 'vector-first';
  if (engine === 'raster') return 'raster';
  return 'auto';
};

const legendFallbackLabel = (reason?: string | null) => {
  if (!reason) return null;
  if (reason === 'gray_mask_mode') return 'gray path';
  if (reason === 'legend_image_dominant') return 'scan/image PDF';
  if (reason === 'insufficient_vector_primitives') return 'za malo wektorow';
  if (reason === 'insufficient_text_primitives') return 'za malo tekstu PDF';
  if (reason === 'insufficient_row_anchors') return 'brak stabilnych wierszy';
  if (reason === 'insufficient_vector_drafts') return 'za malo draftow';
  if (reason === 'low_vector_draft_confidence') return 'niska pewnosc';
  if (reason === 'profile_not_vector_ready') return 'profil niegotowy';
  if (reason.startsWith('vector_exception')) return 'blad vector path';
  return reason.replaceAll('_', ' ');
};

const isLegendReviewStatus = (value?: string): value is LegendReviewStatus =>
  value === 'pending' || value === 'accepted' || value === 'fixed' || value === 'rejected';

const mergeLegendReviewItems = (
  nextPatterns: Pattern[],
  previousItems: LegendReviewItem[],
): LegendReviewItem[] => {
  const previousById = new Map(previousItems.map(item => [item.id, item]));

  return nextPatterns.map(pattern => {
    const id = getPatternKey(pattern);
    const previous = previousById.get(id);
    const rawStatus = pattern.status;
    const status: LegendReviewStatus = rawStatus === 'pending'
      ? 'pending'
      : previous?.status ?? (isLegendReviewStatus(rawStatus) ? rawStatus : 'pending');

    return {
      id,
      name: pattern.name,
      imgBase64: pattern.imgBase64,
      status,
      correctedBBoxPx: pattern.correctedBBoxPx ?? previous?.correctedBBoxPx,
    };
  });
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [pdfPreview, setPdfPreview] = useState<string | null>(null);
  const [previewMeta, setPreviewMeta] = useState<PreviewMeta | null>(null);
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
  const [legendEngineStatus, setLegendEngineStatus] = useState<LegendEngineStatus | null>(null);
  const [pdfDiagnostics, setPdfDiagnostics] = useState<PdfDiagnostics | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress | null>(null);
  const [roiInspection, setRoiInspection] = useState<RoiInspection | null>(null);
  const [isInspectingRoi, setIsInspectingRoi] = useState(false);
  const [grayDebugZones, setGrayDebugZones] = useState<GrayDebugZones | null>(null);
  const [isLoadingGrayZones, setIsLoadingGrayZones] = useState(false);
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [isProjectsLoading, setIsProjectsLoading] = useState(false);
  const [activeProject, setActiveProject] = useState<ProjectSummary | null>(null);
  const [authSessions, setAuthSessions] = useState<AuthSession[]>([]);
  const [isSessionsLoading, setIsSessionsLoading] = useState(false);
  const [analysisRuns, setAnalysisRuns] = useState<ProjectAnalysisRun[]>([]);
  const [historyProjectId, setHistoryProjectId] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [workspaceProjectId, setWorkspaceProjectId] = useState<string | null>(null);
  const detectRequestSeqRef = useRef(0);
  const detectAbortRef = useRef<AbortController | null>(null);
  const previewRenderSeqRef = useRef(0);

  const legendReviewCompleted = legendReviewItems.filter(item => item.status !== 'pending').length;
  const hasLegendReview = legendReviewItems.length > 0;
  const isLegendReviewComplete = !hasLegendReview || legendReviewCompleted === legendReviewItems.length;
  const activeProjectId = activeProject?.id ?? null;
  const patternLabelMap = Object.fromEntries(
    patterns.map(pattern => [pattern.id ?? pattern.name, pattern.name])
  );

  const projectPath = (path: string) => {
    if (!activeProjectId) throw new Error('Nie wybrano projektu.');
    return projectApiPath(activeProjectId, path);
  };

  const resetWorkspaceState = () => {
    previewRenderSeqRef.current += 1;
    setIsProcessing(false);
    setProgressText('');
    setFile(null);
    setPdfPreview(null);
    setPreviewMeta(null);
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
    setLegendEngineStatus(null);
    setAnalysisProgress(null);
    setRoiInspection(null);
    setGrayDebugZones(null);
  };

  const cancelActiveAnalysisRequest = () => {
    detectRequestSeqRef.current += 1;
    detectAbortRef.current?.abort();
    detectAbortRef.current = null;
  };

  const clearWorkspaceProject = () => {
    cancelActiveAnalysisRequest();
    setWorkspaceProjectId(null);
    resetWorkspaceState();
  };

  const loadProjects = async () => {
    setIsProjectsLoading(true);
    try {
      const response = await apiFetch('/api/projects');
      if (!response.ok) throw new Error(await readApiError(response, 'Nie udało się pobrać projektów.'));
      const payload = (await response.json()) as { projects?: ProjectSummary[] };
      setProjects(payload.projects || []);
    } finally {
      setIsProjectsLoading(false);
    }
  };

  const loadAuthSessions = async () => {
    setIsSessionsLoading(true);
    try {
      const response = await apiFetch('/api/auth/sessions');
      if (!response.ok) throw new Error(await readApiError(response, 'Nie udało się pobrać sesji.'));
      const payload = (await response.json()) as { sessions?: AuthSession[] };
      setAuthSessions(payload.sessions || []);
    } finally {
      setIsSessionsLoading(false);
    }
  };

  const loadAnalysisRuns = async (project: ProjectSummary) => {
    setHistoryProjectId(project.id);
    setIsHistoryLoading(true);
    try {
      const response = await apiFetch(projectApiPath(project.id, '/analysis-runs'));
      if (!response.ok) throw new Error(await readApiError(response, 'Nie udało się pobrać historii analiz.'));
      const payload = (await response.json()) as { analysisRuns?: ProjectAnalysisRun[] };
      setAnalysisRuns(payload.analysisRuns || []);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;

    const loadSession = async () => {
      try {
        const response = await apiFetch('/api/auth/me');
        if (!isMounted) return;
        if (!response.ok) {
          setAuthUser(null);
          setProjects([]);
          setAuthSessions([]);
          return;
        }
        const payload = (await response.json()) as { user?: AuthUser };
        if (!payload.user) {
          setAuthUser(null);
          setProjects([]);
          setAuthSessions([]);
          return;
        }
        setAuthUser(payload.user);
        await Promise.all([loadProjects(), loadAuthSessions()]);
      } catch {
        if (isMounted) {
          setAuthUser(null);
          setAuthSessions([]);
        }
      } finally {
        if (isMounted) setIsAuthLoading(false);
      }
    };

    loadSession();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleAuthenticated = async (user: AuthUser) => {
    setAuthUser(user);
    await Promise.all([loadProjects(), loadAuthSessions()]);
  };

  const handleCreateProject = async (name: string, description: string) => {
    const response = await apiFetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się utworzyć projektu.'));
    }
    const payload = (await response.json()) as { project?: ProjectSummary };
    if (!payload.project) throw new Error('Backend nie zwrócił projektu.');
    setProjects(prev => [payload.project!, ...prev.filter(project => project.id !== payload.project!.id)]);
    cancelActiveAnalysisRequest();
    setWorkspaceProjectId(payload.project.id);
    setActiveProject(payload.project);
    resetWorkspaceState();
  };

  const restoreProjectWorkspace = async (project: ProjectSummary) => {
    if (!project.latestSessionId) return;
    const renderSeq = ++previewRenderSeqRef.current;
    setIsProcessing(true);
    setProgressText('Ładowanie ostatniego podglądu...');
    try {
      const latestSessionId = project.latestSessionId;
      setSessionId(latestSessionId);
      setPreviewMeta(null);
      const [previewResponse, layersResponse] = await Promise.all([
        apiFetch(projectApiPath(project.id, `/render-preview?session_id=${latestSessionId}`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ hidden_layers: [], preview: true }),
        }),
        apiFetch(projectApiPath(project.id, `/layers?session_id=${latestSessionId}`)),
      ]);

      if (renderSeq !== previewRenderSeqRef.current) return;
      if (previewResponse.ok) {
        const previewData = await previewResponse.json();
        setPdfPreview(previewData.planPreview || null);
        setPreviewMeta(readPreviewMeta(previewData));
        setPdfDiagnostics(previewData.pdfDiagnostics || null);
      }
      if (layersResponse.ok) {
        const layersData = await layersResponse.json();
        setLayers(layersData.layers || []);
      }
      void apiFetch(projectApiPath(project.id, `/render-preview?session_id=${latestSessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden_layers: [] }),
      })
        .then(response => response.ok ? response.json() : null)
        .then(fullData => {
          if (!fullData || renderSeq !== previewRenderSeqRef.current) return;
          setPdfPreview(fullData.planPreview || null);
          setPreviewMeta(readPreviewMeta(fullData));
          setPdfDiagnostics(fullData.pdfDiagnostics || null);
        })
        .catch(err => console.error('Full project preview warmup error', err));
      await restoreLatestProjectAnalysis(project.id, latestSessionId, renderSeq);
    } catch (error) {
      console.error('Nie udało się odtworzyć ostatniego podglądu projektu', error);
    } finally {
      if (renderSeq === previewRenderSeqRef.current) {
        setIsProcessing(false);
        setProgressText('');
      }
    }
  };

  const restoreLatestProjectAnalysis = async (
    projectId: string,
    currentSessionId: string,
    renderSeq?: number,
  ) => {
    try {
      const runsResponse = await apiFetch(projectApiPath(projectId, '/analysis-runs'));
      if (!runsResponse.ok) return;
      const runsPayload = (await runsResponse.json()) as { analysisRuns?: ProjectAnalysisRun[] };
      const latestRun = (runsPayload.analysisRuns || []).find(
        run => run.sessionId === currentSessionId
      );
      if (!latestRun) return;

      const snapshotResponse = await apiFetch(
        projectApiPath(projectId, `/analysis-runs/${encodeURIComponent(latestRun.id)}`)
      );
      if (!snapshotResponse.ok) return;
      const snapshotPayload = (await snapshotResponse.json()) as { snapshot?: AnalysisSnapshot | null };
      const snapshot = snapshotPayload.snapshot;
      if (!snapshot?.analysisContext) return;
      if (renderSeq !== undefined && renderSeq !== previewRenderSeqRef.current) return;

      setResults(snapshot.results || []);
      setBoxes(snapshot.boxes || []);
      setAnalysisContext(snapshot.analysisContext);
      setPdfDiagnostics(snapshot.analysisContext.pdfDiagnostics || null);
      setAnalysisProgress({
        sessionId: snapshot.analysisContext.sessionId,
        analysisId: snapshot.analysisContext.analysisId,
        stage: 'done',
        percent: 100,
        detail: 'Analiza zakonczona',
        done: true,
        updatedAtUtc: snapshot.analysisContext.generatedAtUtc,
      });
    } catch (error) {
      console.error('Nie udało się odtworzyć ostatniej analizy projektu', error);
    }
  };

  const handleUpdateProject = async (projectId: string, name: string, description: string) => {
    const response = await apiFetch(projectApiPath(projectId, ''), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się zapisać projektu.'));
    }
    const payload = (await response.json()) as { project?: ProjectSummary };
    if (!payload.project) throw new Error('Backend nie zwrócił projektu.');
    setProjects(prev =>
      prev.map(project => project.id === projectId ? payload.project! : project)
    );
    setActiveProject(current => current?.id === projectId ? payload.project! : current);
    return payload.project;
  };

  const handleArchiveProject = async (projectId: string) => {
    const response = await apiFetch(projectApiPath(projectId, ''), { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się zarchiwizować projektu.'));
    }
    setProjects(prev => prev.filter(project => project.id !== projectId));
    if (historyProjectId === projectId) {
      setHistoryProjectId(null);
      setAnalysisRuns([]);
    }
    if (activeProject?.id === projectId) {
      setActiveProject(null);
      clearWorkspaceProject();
    }
    if (workspaceProjectId === projectId) clearWorkspaceProject();
  };

  const handleSelectProject = (project: ProjectSummary) => {
    setActiveProject(project);
    if (workspaceProjectId !== project.id) {
      cancelActiveAnalysisRequest();
      resetWorkspaceState();
      setWorkspaceProjectId(project.id);
    }
    fetchTemplates(project.id);
    if (workspaceProjectId !== project.id || (!pdfPreview && project.latestSessionId)) {
      void restoreProjectWorkspace(project);
    }
  };

  const handleUpdateProfile = async (name: string) => {
    const response = await apiFetch('/api/auth/me', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się zapisać profilu.'));
    }
    const payload = (await response.json()) as { user?: AuthUser };
    if (!payload.user) throw new Error('Backend nie zwrócił użytkownika.');
    setAuthUser(payload.user);
  };

  const handleDeleteSession = async (sessionIdToDelete: string) => {
    const response = await apiFetch(`/api/auth/sessions/${encodeURIComponent(sessionIdToDelete)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się zamknąć sesji.'));
    }
    const payload = (await response.json()) as { deletedCurrentSession?: boolean };
    if (payload.deletedCurrentSession) {
      setAuthUser(null);
      setProjects([]);
      setAuthSessions([]);
      setActiveProject(null);
      clearWorkspaceProject();
      return;
    }
    await loadAuthSessions();
  };

  const handleLogoutAll = async () => {
    await apiFetch('/api/auth/logout-all', { method: 'POST' });
    setAuthUser(null);
    setProjects([]);
    setAuthSessions([]);
    setActiveProject(null);
    clearWorkspaceProject();
  };

  const handleLogout = async () => {
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' });
    } catch {
      // Local state reset is authoritative for the UI.
    }
    setAuthUser(null);
    setProjects([]);
    setAuthSessions([]);
    setHistoryProjectId(null);
    setAnalysisRuns([]);
    setActiveProject(null);
    clearWorkspaceProject();
  };

  const replacePattern = (oldId: string, pattern: Pattern) => {
    setPatterns(prev => {
      const index = prev.findIndex(item => (item.id ?? item.name) === oldId);
      if (index === -1) return [...prev, pattern];
      const next = [...prev];
      next[index] = pattern;
      return next;
    });
  };

  const readPreviewMeta = (data: Partial<PreviewMeta>): PreviewMeta | null => {
    if (!data.analysisSize) return null;
    return {
      previewDpi: data.previewDpi,
      analysisDpi: data.analysisDpi,
      analysisSize: data.analysisSize,
      isFullResolution: data.isFullResolution,
      renderCacheHit: data.renderCacheHit,
    };
  };

  const fetchTemplates = async (projectId = activeProjectId) => {
    if (!projectId) return;
    try {
      const response = await apiFetch(projectApiPath(projectId, '/templates'));
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
    if (!activeProjectId) return;
    const currentProjectId = activeProjectId;
    previewRenderSeqRef.current += 1;
    setFile(selectedFile);
    setPdfPreview(null);
    setPreviewMeta(null);
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
      const res = await apiFetch(projectApiPath(currentProjectId, '/preview'), {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Błąd podglądu');
      const data = await res.json();
      setPdfPreview(data.planPreview);
      setSessionId(data.sessionId);
      setPreviewMeta(readPreviewMeta(data));
      setPdfDiagnostics(data.pdfDiagnostics || null);

      const renderSeq = ++previewRenderSeqRef.current;
      const loadedSessionId = data.sessionId;
      void apiFetch(projectApiPath(currentProjectId, `/render-preview?session_id=${loadedSessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden_layers: [] }),
      })
        .then(response => response.ok ? response.json() : null)
        .then(fullData => {
          if (!fullData || renderSeq !== previewRenderSeqRef.current) return;
          setPdfPreview(fullData.planPreview);
          setPreviewMeta(readPreviewMeta(fullData));
          setPdfDiagnostics(fullData.pdfDiagnostics || null);
        })
        .catch(err => console.error('Full preview warmup error', err));
      const uploadAtUtc = new Date().toISOString();
      setActiveProject(current =>
        current
          ? {
              ...current,
              latestSessionId: data.sessionId,
              latestSourcePdf: data.sourcePdf || selectedFile.name,
              latestUploadAtUtc: uploadAtUtc,
              updatedAtUtc: uploadAtUtc,
            }
          : current
      );
      setProjects(prev =>
        prev.map(project =>
          project.id === currentProjectId
            ? {
                ...project,
                latestSessionId: data.sessionId,
                latestSourcePdf: data.sourcePdf || selectedFile.name,
                latestUploadAtUtc: uploadAtUtc,
                updatedAtUtc: uploadAtUtc,
              }
            : project
        )
      );
      
      // Fetch layers
      apiFetch(projectApiPath(currentProjectId, `/layers?session_id=${data.sessionId}`))
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
    if (!sessionId || !activeProjectId) return;
    const newLayers = layers.map(l => l.name === layerName ? { ...l, visible: !l.visible } : l);
    setLayers(newLayers);
    const renderSeq = ++previewRenderSeqRef.current;
    
    setIsProcessing(true);
    setProgressText('Przeliczanie warstw...');
    try {
      const hiddenLayers = newLayers.filter(l => !l.visible).map(l => l.name);
      const res = await apiFetch(projectPath(`/render-preview?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden_layers: hiddenLayers })
      });
      if (!res.ok) throw new Error('Błąd odświeżania podglądu');
      const data = await res.json();
      if (renderSeq !== previewRenderSeqRef.current) return;
      setPdfPreview(data.planPreview);
      setPreviewMeta(readPreviewMeta(data));
      setPdfDiagnostics(data.pdfDiagnostics || null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsProcessing(false);
      setProgressText('');
    }
  };

  const handleExtractLegend = async () => {
    if (!sessionId || !activeProjectId) return;
    if (!legendZone) {
      alert('Zaznacz strefę legendy na planie przed ekstrakcją (tryb Legenda na canvasie).');
      return;
    }
    setIsProcessing(true);
    setProgressText('Ekstrakcja legendy (300 DPI)...');
    try {
      const response = await apiFetch(projectPath(`/extract-legend?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          excluded_zones: excludedZones.map(z => ({
            x: Math.round(z.x),
            y: Math.round(z.y),
            width: Math.round(z.width),
            height: Math.round(z.height),
          })),
          hidden_layers: layers.filter(l => !l.visible).map(l => l.name),
          detector_profile: detectorProfile,
          legend_engine: 'auto',
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
      const data = await response.json() as ExtractLegendResponse;
      const nextPatterns = data.patterns || [];
      setLegendEngineStatus({
        engineRequested: data.legendEngineRequested,
        engineUsed: data.legendEngineUsed,
        fallbackReason: data.legendFallbackReason ?? null,
        pageKind: data.legendPageProfile?.page_kind ?? data.legendPageProfile?.pageKind,
        patternCount: nextPatterns.length,
      });
      const extractedCount = data.legendExtractedCount ?? nextPatterns.length;
      if (extractedCount === 0) {
        setLegendCorrectionTarget(null);
        if (nextPatterns.length > 0) {
          setPatterns(nextPatterns);
          setLegendReviewItems(prev => mergeLegendReviewItems(nextPatterns, prev));
        }
        setPdfDiagnostics(data.pdfDiagnostics || pdfDiagnostics);
        alert('Nie znaleziono nowych poprawnych wzorców w zaznaczonej legendzie. Już zaakceptowane wzorce zostały zachowane.');
        return;
      }
      setPatterns(nextPatterns);
      setLegendReviewItems(prev => mergeLegendReviewItems(nextPatterns, prev));
      setIsLegendReviewOpen(true);
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
    if (!sessionId || !activeProjectId) return;
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
          const progressResponse = await apiFetch(`/api/analysis-progress?session_id=${sessionId}`);
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
      const response = await apiFetch(projectPath(`/analyze?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
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
    previewRenderSeqRef.current += 1;
    try {
      if (activeProjectId) await apiFetch(projectPath('/clear'), { method: 'POST' });
    } catch { /* ignore */ }
    setFile(null);
    setPdfPreview(null);
    setPreviewMeta(null);
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
    if (!sessionId || !activeProjectId) return;
    setIsLoadingGrayZones(true);
    setProgressText('Liczenie czarnych stref...');
    try {
      const response = await apiFetch(projectPath(`/gray-debug-zones?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
    if (!sessionId || !activeProjectId) return;
    setIsInspectingRoi(true);
    setProgressText('Inspektor ROI liczy dopasowania...');
    try {
      const response = await apiFetch(projectPath(`/inspect-roi?session_id=${sessionId}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      if (!activeProjectId) return;
      await apiFetch(projectPath('/templates'), { method: 'DELETE' });
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
      const response = await apiFetch(projectPath(`/templates/${encodeURIComponent(templateId)}`), {
        method: 'DELETE',
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

  const handleAcceptAllLegendItems = () => {
    setLegendReviewItems(prev =>
      prev.map(item =>
        item.status !== 'rejected' && item.imgBase64
          ? { ...item, status: 'accepted' }
          : item
      )
    );
    setLegendCorrectionTarget(null);
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
      const response = await apiFetch(projectPath(`/templates/${encodeURIComponent(id)}`), {
        method: 'DELETE',
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
      const response = await apiFetch(projectPath(`/templates/${encodeURIComponent(id)}`), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
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

  const handleRenameResultSymbol = async (currentName: string, name: string) => {
    if (!name.trim()) {
      throw new Error('Nazwa symbolu nie może być pusta.');
    }

    const response = await apiFetch(projectPath(`/templates/${encodeURIComponent(currentName)}`), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, 'Nie udało się zmienić nazwy symbolu.'));
    }

    const data = await response.json() as { pattern?: Pattern };
    const pattern = data.pattern;
    if (!pattern) throw new Error('Backend nie zwrócił wzorca po zmianie nazwy.');

    const nextName = pattern.id ?? pattern.name;
    replacePattern(currentName, pattern);
    setLegendReviewItems(prev =>
      prev.map(item =>
        item.id === currentName
          ? { ...item, id: nextName, name: pattern.name, imgBase64: pattern.imgBase64 }
          : item
      )
    );
    setLegendCorrectionTarget(current =>
      current?.id === currentName ? { id: nextName, name: pattern.name } : current
    );
    setBoxes(prev =>
      prev.map(box =>
        box.symbolName === currentName ? { ...box, symbolName: nextName } : box
      )
    );
    setResults(prev =>
      prev.map(result =>
        result.name === currentName ? { ...result, name: nextName } : result
      )
    );

    return nextName;
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
    if (!sessionId || !activeProjectId || !legendCorrectionTarget) return;

    const target = legendCorrectionTarget;
    setIsProcessing(true);
    setProgressText(`Zapisywanie wzorca ${target.name}...`);
    try {
      const response = await apiFetch(projectPath(`/templates/${encodeURIComponent(target.id)}/crop`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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

  const ensureResultGroup = (symbolName: string, fallbackColor = '#22c55e') => {
    setResults(prev => {
      if (prev.some(result => result.name === symbolName)) return prev;
      return [...prev, { name: symbolName, count: 0, color: fallbackColor }];
    });
  };

  const handleChangeBoxSymbol = (id: string, symbolName: string) => {
    setBoxes(prev => prev.map(box => box.id === id ? { ...box, symbolName } : box));
    ensureResultGroup(symbolName);
  };

  const handleUpdateBox = (id: string, patch: Partial<DetectionBox>) => {
    const geometryChanged = ['x', 'y', 'width', 'height'].some(key => key in patch);
    setBoxes(prev =>
      prev.map(box =>
        box.id === id
          ? {
              ...box,
              ...patch,
              ...(geometryChanged ? { visualBBox: null } : {}),
            }
          : box
      )
    );
    if (patch.symbolName) ensureResultGroup(patch.symbolName);
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

  if (isAuthLoading) {
    return (
      <div className="auth-shell">
        <div className="auth-panel">
          <div className="auth-brand">ElektroScan AI</div>
          <div className="text-sm text-muted">Ładowanie sesji...</div>
        </div>
      </div>
    );
  }

  if (!authUser) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  if (!activeProject) {
    return (
      <ProjectDashboard
        user={authUser}
        projects={projects}
        sessions={authSessions}
        analysisRuns={analysisRuns}
        historyProjectId={historyProjectId}
        isLoading={isProjectsLoading}
        isSessionsLoading={isSessionsLoading}
        isHistoryLoading={isHistoryLoading}
        onCreateProject={handleCreateProject}
        onUpdateProject={handleUpdateProject}
        onArchiveProject={handleArchiveProject}
        onSelectProject={handleSelectProject}
        onOpenHistory={loadAnalysisRuns}
        onUpdateProfile={handleUpdateProfile}
        onRefreshSessions={loadAuthSessions}
        onDeleteSession={handleDeleteSession}
        onLogoutAll={handleLogoutAll}
        onLogout={handleLogout}
      />
    );
  }

  const workspaceFileName = file?.name || (sessionId ? activeProject.latestSourcePdf || null : null);

  return (
    <div className="workspace-shell">
      <div className="project-workspace-bar">
        <button
          className="btn-secondary"
          style={{ width: 'auto' }}
          onClick={() => {
            previewRenderSeqRef.current += 1;
            setIsProcessing(false);
            setProgressText('');
            setActiveProject(null);
            loadProjects();
          }}
        >
          <ArrowLeft size={16} />
          Projekty
        </button>
        <div>
          <b>{activeProject.name}</b>
          <span>{workspaceFileName || 'brak wgranego PDF'}</span>
        </div>
        {legendEngineStatus && (
          <div
            className={`legend-engine-notice ${
              legendEngineStatus.engineUsed === 'vector_first' ? 'is-vector' : 'is-raster'
            }`}
            title={legendEngineStatus.fallbackReason || undefined}
          >
            <b>Legenda: {legendEngineLabel(legendEngineStatus.engineUsed)}</b>
            {legendEngineStatus.fallbackReason && (
              <span>{legendFallbackLabel(legendEngineStatus.fallbackReason)}</span>
            )}
            <span>{legendEngineStatus.patternCount} wzorcow</span>
            {legendEngineStatus.pageKind && <span>{legendEngineStatus.pageKind}</span>}
          </div>
        )}
      </div>

      <div className="app-container">
      {/* Lewy panel */}
      <Sidebar
        fileName={workspaceFileName}
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
        imageSize={previewMeta?.analysisSize ?? null}
        imageResetKey={sessionId}
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
        symbolNames={patterns.map(p => p.id ?? p.name)}
        onAddManualBox={handleAddManualBox}
        onUpdateBox={handleUpdateBox}
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
          onAcceptAll={handleAcceptAllLegendItems}
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
            {roiInspection.roiRawMask && <img src={roiInspection.roiRawMask} alt="raw all-ink mask" title="raw all-ink mask" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
            {(roiInspection.roiColorScanMask || roiInspection.roiScanMask) && (
              <img
                src={roiInspection.roiColorScanMask || roiInspection.roiScanMask}
                alt={roiInspection.profile === 'color' ? 'color scan mask' : 'scan mask'}
                title={roiInspection.profile === 'color' ? 'color scan mask for top candidate' : 'scan mask'}
                style={{ width: '100%', borderRadius: 6, background: '#fff' }}
              />
            )}
            {roiInspection.profile === 'gray' && roiInspection.roiDarkScanMask && <img src={roiInspection.roiDarkScanMask} alt="dark scan mask" title="dark scan mask" style={{ width: '100%', borderRadius: 6, background: '#fff' }} />}
          </div>

          <div className="text-xs text-muted" style={{ marginTop: 8, lineHeight: 1.45 }}>
            <div>Skale: {roiInspection.usedScales.map(scale => scale.toFixed(2)).join(', ')}</div>
            <div>
              {roiInspection.profile === 'color'
                ? `Tusz ROI: all-ink ${roiInspection.roiInkPixels}, color-scan ${roiInspection.roiColorScanPixels ?? roiInspection.roiScanPixels}`
                : `Tusz ROI: raw ${roiInspection.roiInkPixels}, scan ${roiInspection.roiScanPixels}`}
            </div>
            {roiInspection.profile === 'color' && roiInspection.roiColorScanTemplate && (
              <div>
                Color mask: {roiInspection.roiColorScanTemplate.symbolName} · {roiInspection.roiColorScanTemplate.scanMask}
              </div>
            )}
            {roiInspection.profile === 'gray' && roiInspection.roiDarkInkPixels !== undefined && (
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
          analysisContext={analysisContext}
          focusedBoxId={focusedBoxId}
          onFocusBox={id => setFocusedBoxId(prev => prev === id ? null : id)}
          onRejectBox={handleRejectBox}
          onChangeBoxSymbol={handleChangeBoxSymbol}
          onUpdateBox={handleUpdateBox}
          onRenameSymbol={handleRenameResultSymbol}
          symbolNames={patterns.map(p => p.id ?? p.name)}
          symbolLabels={patternLabelMap}
          projectId={activeProject.id}
          onTemplateUploaded={() => fetchTemplates(activeProject.id)}
        />
      )}
      </div>
    </div>
  );
}

export default App;
