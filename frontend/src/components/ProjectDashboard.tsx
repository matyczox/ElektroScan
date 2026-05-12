import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import {
  ArrowDownAZ,
  ChevronDown,
  Clock3,
  FolderOpen,
  History,
  LogOut,
  MonitorX,
  Pencil,
  Plus,
  RotateCw,
  Save,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import type { AuthUser } from './AuthScreen';

export interface ProjectSummary {
  id: string;
  name: string;
  description?: string;
  createdAtUtc: string;
  updatedAtUtc: string;
  latestSessionId?: string | null;
  latestSourcePdf?: string | null;
  latestUploadAtUtc?: string | null;
  latestAnalysisAtUtc?: string | null;
  analysisCount?: number;
}

export interface AuthSession {
  id: string;
  createdAtUtc: string;
  lastSeenAtUtc: string;
  expiresAtUtc: string;
  isCurrent?: boolean;
}

export interface ProjectAnalysisRun {
  id: string;
  projectId: string;
  sessionId: string;
  sourcePdf: string;
  generatedAtUtc: string;
  hasSnapshot?: boolean;
}

type ProjectSort = 'updated' | 'created' | 'name' | 'analysis';

const PROJECT_SORT_OPTIONS: Array<{ value: ProjectSort; label: string }> = [
  { value: 'updated', label: 'Ostatnia aktywność' },
  { value: 'analysis', label: 'Ostatnia analiza' },
  { value: 'created', label: 'Data utworzenia' },
  { value: 'name', label: 'Nazwa' },
];

interface ProjectDashboardProps {
  user: AuthUser;
  projects: ProjectSummary[];
  sessions: AuthSession[];
  analysisRuns: ProjectAnalysisRun[];
  historyProjectId: string | null;
  isLoading: boolean;
  isSessionsLoading: boolean;
  isHistoryLoading: boolean;
  onCreateProject: (name: string, description: string) => Promise<void>;
  onUpdateProject: (projectId: string, name: string, description: string) => Promise<ProjectSummary>;
  onArchiveProject: (projectId: string) => Promise<void>;
  onSelectProject: (project: ProjectSummary) => void;
  onOpenHistory: (project: ProjectSummary) => Promise<void>;
  onUpdateProfile: (name: string) => Promise<void>;
  onRefreshSessions: () => Promise<void>;
  onDeleteSession: (sessionId: string) => Promise<void>;
  onLogoutAll: () => Promise<void>;
  onLogout: () => void;
}

const formatDate = (value?: string | null) => {
  if (!value) return 'brak danych';
  return new Intl.DateTimeFormat('pl-PL', {
    dateStyle: 'short',
    timeStyle: 'short',
  }).format(new Date(value));
};

const dateValue = (value?: string | null) => (value ? Date.parse(value) || 0 : 0);

export const ProjectDashboard = ({
  user,
  projects,
  sessions,
  analysisRuns,
  historyProjectId,
  isLoading,
  isSessionsLoading,
  isHistoryLoading,
  onCreateProject,
  onUpdateProject,
  onArchiveProject,
  onSelectProject,
  onOpenHistory,
  onUpdateProfile,
  onRefreshSessions,
  onDeleteSession,
  onLogoutAll,
  onLogout,
}: ProjectDashboardProps) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [profileName, setProfileName] = useState(user.name);
  const [securityMessage, setSecurityMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [sessionBusyId, setSessionBusyId] = useState<string | null>(null);
  const [isLogoutAllBusy, setIsLogoutAllBusy] = useState(false);
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<ProjectSort>('updated');
  const [isSortMenuOpen, setIsSortMenuOpen] = useState(false);
  const sortMenuRef = useRef<HTMLDivElement | null>(null);
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');

  const historyProject = projects.find(project => project.id === historyProjectId) || null;
  const sortLabel = PROJECT_SORT_OPTIONS.find(option => option.value === sort)?.label || PROJECT_SORT_OPTIONS[0].label;

  useEffect(() => {
    if (!isSortMenuOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && sortMenuRef.current?.contains(target)) return;
      setIsSortMenuOpen(false);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsSortMenuOpen(false);
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isSortMenuOpen]);

  const visibleProjects = useMemo(() => {
    const query = search.trim().toLowerCase();
    const filtered = query
      ? projects.filter(project =>
          [project.name, project.description, project.latestSourcePdf]
            .filter(Boolean)
            .some(value => String(value).toLowerCase().includes(query))
        )
      : projects;

    return [...filtered].sort((left, right) => {
      if (sort === 'name') return left.name.localeCompare(right.name, 'pl');
      if (sort === 'created') return dateValue(right.createdAtUtc) - dateValue(left.createdAtUtc);
      if (sort === 'analysis') {
        return dateValue(right.latestAnalysisAtUtc) - dateValue(left.latestAnalysisAtUtc);
      }
      return dateValue(right.updatedAtUtc) - dateValue(left.updatedAtUtc);
    });
  }, [projects, search, sort]);

  const handleCreate = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    setError(null);
    setIsCreating(true);
    try {
      await onCreateProject(name.trim(), description.trim());
      setName('');
      setDescription('');
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setIsCreating(false);
    }
  };

  const startEditing = (project: ProjectSummary) => {
    setEditingProjectId(project.id);
    setEditName(project.name);
    setEditDescription(project.description || '');
  };

  const handleSaveProject = async (event: FormEvent, project: ProjectSummary) => {
    event.preventDefault();
    if (!editName.trim()) return;
    setError(null);
    try {
      await onUpdateProject(project.id, editName.trim(), editDescription.trim());
      setEditingProjectId(null);
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const handleArchive = async (project: ProjectSummary) => {
    if (!window.confirm(`Zarchiwizować projekt "${project.name}"?`)) return;
    setError(null);
    try {
      await onArchiveProject(project.id);
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const handleSaveProfile = async (event: FormEvent) => {
    event.preventDefault();
    setSecurityMessage(null);
    setError(null);
    setIsSavingProfile(true);
    try {
      await onUpdateProfile(profileName.trim());
      setSecurityMessage('Profil zapisany.');
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    setSessionBusyId(sessionId);
    setError(null);
    try {
      await onDeleteSession(sessionId);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setSessionBusyId(null);
    }
  };

  const handleLogoutAll = async () => {
    setIsLogoutAllBusy(true);
    setError(null);
    try {
      await onLogoutAll();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setIsLogoutAllBusy(false);
    }
  };

  return (
    <div className="project-shell">
      <header className="project-topbar">
        <div>
          <div className="auth-brand">ElektroScan AI</div>
          <div className="text-sm text-muted">Zalogowano jako {user.email}</div>
        </div>
        <button className="btn-secondary" style={{ width: 'auto' }} onClick={onLogout}>
          <LogOut size={16} />
          Wyloguj
        </button>
      </header>

      <main className="project-main project-main-wide">
        <section className="project-create">
          <h2>Nowy projekt</h2>
          <form onSubmit={handleCreate}>
            <label className="form-field">
              Nazwa projektu
              <input
                value={name}
                onChange={event => setName(event.target.value)}
                placeholder="np. Bronisze E8"
                required
              />
            </label>
            <label className="form-field">
              Opis
              <textarea
                value={description}
                onChange={event => setDescription(event.target.value)}
                placeholder="Opcjonalna notatka"
                rows={3}
              />
            </label>
            {error && <div className="form-error">{error}</div>}
            <button className="btn-primary" type="submit" disabled={isCreating}>
              <Plus size={18} />
              {isCreating ? 'Tworzenie...' : 'Utwórz projekt'}
            </button>
          </form>

          <div className="account-panel">
            <div className="project-section-header">
              <h2>Konto</h2>
              <span className="badge badge-gold">{user.email}</span>
            </div>

            <form className="account-form" onSubmit={handleSaveProfile}>
              <label className="form-field">
                Nazwa
                <input
                  value={profileName}
                  onChange={event => setProfileName(event.target.value)}
                  required
                />
              </label>
              <button className="btn-secondary" type="submit" disabled={isSavingProfile}>
                <Save size={15} />
                {isSavingProfile ? 'Zapisywanie...' : 'Zapisz profil'}
              </button>
            </form>

            {securityMessage && <div className="form-success">{securityMessage}</div>}

            <div className="sessions-header">
              <h2>Sesje</h2>
              <button className="btn-icon" type="button" onClick={onRefreshSessions} title="Odśwież sesje">
                <RotateCw size={15} />
              </button>
            </div>
            {isSessionsLoading ? (
              <div className="project-empty">Ładowanie sesji...</div>
            ) : sessions.length === 0 ? (
              <div className="project-empty">Brak aktywnych sesji.</div>
            ) : (
              <div className="session-list">
                {sessions.map(session => (
                  <div className="session-row" key={session.id}>
                    <MonitorX size={16} color="var(--accent-blue)" />
                    <span>
                      <b>{session.isCurrent ? 'Ta sesja' : 'Aktywna sesja'}</b>
                      <small>Ostatnio: {formatDate(session.lastSeenAtUtc)}</small>
                    </span>
                    <button
                      className="btn-icon"
                      type="button"
                      onClick={() => handleDeleteSession(session.id)}
                      disabled={sessionBusyId === session.id}
                      title="Zamknij sesję"
                    >
                      <X size={15} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <button className="btn-secondary btn-danger" type="button" onClick={handleLogoutAll} disabled={isLogoutAllBusy}>
              <LogOut size={15} />
              Wyloguj wszędzie
            </button>
          </div>
        </section>

        <section className="project-list-section">
          <div className="project-section-header">
            <h2>Projekty</h2>
            <span className="badge badge-gold">{visibleProjects.length}</span>
          </div>

          <div className="project-toolbar">
            <label className="form-field project-search">
              Szukaj
              <span>
                <Search size={15} />
                <input
                  value={search}
                  onChange={event => setSearch(event.target.value)}
                  placeholder="Nazwa, opis, PDF"
                />
              </span>
            </label>
            <label className="form-field project-sort">
              Sortuj
              <div className="project-sort-control" ref={sortMenuRef}>
                <ArrowDownAZ size={15} />
                <button
                  type="button"
                  className="project-sort-trigger"
                  aria-haspopup="listbox"
                  aria-expanded={isSortMenuOpen}
                  onClick={() => setIsSortMenuOpen(open => !open)}
                >
                  <span>{sortLabel}</span>
                  <ChevronDown className="project-sort-caret" size={15} aria-hidden="true" />
                </button>
                {isSortMenuOpen && (
                  <div className="project-sort-menu" role="listbox" aria-label="Sortuj projekty">
                    {PROJECT_SORT_OPTIONS.map(option => (
                      <button
                        key={option.value}
                        type="button"
                        role="option"
                        aria-selected={sort === option.value}
                        className={sort === option.value ? 'is-active' : ''}
                        onClick={() => {
                          setSort(option.value);
                          setIsSortMenuOpen(false);
                        }}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </label>
          </div>

          {isLoading ? (
            <div className="project-empty">Ładowanie projektów...</div>
          ) : visibleProjects.length === 0 ? (
            <div className="project-empty">Nie znaleziono projektów.</div>
          ) : (
            <div className="project-list">
              {visibleProjects.map(project => {
                const isEditing = editingProjectId === project.id;
                return (
                  <div className="project-row" key={project.id}>
                    {isEditing ? (
                      <form className="project-edit-form" onSubmit={event => handleSaveProject(event, project)}>
                        <label className="form-field">
                          Nazwa
                          <input
                            value={editName}
                            onChange={event => setEditName(event.target.value)}
                            required
                          />
                        </label>
                        <label className="form-field">
                          Opis
                          <textarea
                            value={editDescription}
                            onChange={event => setEditDescription(event.target.value)}
                            rows={2}
                          />
                        </label>
                        <div className="inline-actions">
                          <button className="btn-secondary" type="submit">
                            <Save size={15} />
                            Zapisz
                          </button>
                          <button className="btn-secondary" type="button" onClick={() => setEditingProjectId(null)}>
                            <X size={15} />
                            Anuluj
                          </button>
                        </div>
                      </form>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="project-row-main"
                          onClick={() => onSelectProject(project)}
                        >
                          <FolderOpen size={20} color="var(--accent-gold)" />
                          <span>
                            <b>{project.name}</b>
                            <small>{project.description || 'Bez opisu'}</small>
                            <small>
                              {project.latestSourcePdf
                                ? `${project.latestSourcePdf} · upload ${formatDate(project.latestUploadAtUtc)}`
                                : `utworzono ${formatDate(project.createdAtUtc)}`}
                            </small>
                          </span>
                        </button>
                        <div className="project-row-meta">
                          <span>
                            <Clock3 size={13} />
                            {project.latestAnalysisAtUtc
                              ? formatDate(project.latestAnalysisAtUtc)
                              : 'brak analiz'}
                          </span>
                          <span>{project.analysisCount || 0} analiz</span>
                        </div>
                        <div className="project-row-actions">
                          <button className="btn-icon" type="button" onClick={() => onOpenHistory(project)} title="Historia analiz">
                            <History size={16} />
                          </button>
                          <button className="btn-icon" type="button" onClick={() => startEditing(project)} title="Edytuj projekt">
                            <Pencil size={16} />
                          </button>
                          <button className="btn-icon btn-danger" type="button" onClick={() => handleArchive(project)} title="Archiwizuj projekt">
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="project-history-section">
          <div className="project-section-header">
            <h2>Historia analiz</h2>
            <span className="badge badge-blue">{analysisRuns.length}</span>
          </div>
          {!historyProject ? (
            <div className="project-empty">Wybierz ikonę historii przy projekcie.</div>
          ) : isHistoryLoading ? (
            <div className="project-empty">Ładowanie historii dla {historyProject.name}...</div>
          ) : analysisRuns.length === 0 ? (
            <div className="project-empty">Projekt nie ma jeszcze zapisanych analiz.</div>
          ) : (
            <div className="analysis-list">
              {analysisRuns.map(run => (
                <div className="analysis-row" key={run.id}>
                  <History size={16} color="var(--accent-blue)" />
                  <span>
                    <b>{run.sourcePdf}</b>
                    <small>{formatDate(run.generatedAtUtc)}</small>
                    <small>ID: {run.id.slice(0, 8)}</small>
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};
