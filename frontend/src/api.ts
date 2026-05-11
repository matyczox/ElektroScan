export const API_BASE = `http://${window.location.hostname || '127.0.0.1'}:8000`;

export const apiUrl = (path: string) =>
  `${API_BASE}${path}${path.includes('?') ? '&' : '?'}_ts=${Date.now()}`;

export const projectApiPath = (projectId: string, path: string) =>
  `/api/projects/${encodeURIComponent(projectId)}${path}`;

export const apiFetch = (path: string, init: RequestInit = {}) =>
  fetch(apiUrl(path), {
    ...init,
    credentials: 'include',
    cache: 'no-store',
  });

export const readApiError = async (response: Response, fallback: string) => {
  try {
    const payload = await response.json();
    return typeof payload?.detail === 'string' ? payload.detail : fallback;
  } catch {
    return fallback;
  }
};
