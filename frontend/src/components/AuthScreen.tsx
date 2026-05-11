import { useState } from 'react';
import type { FormEvent } from 'react';
import { KeyRound, LogIn, UserPlus } from 'lucide-react';
import { apiFetch, readApiError } from '../api';

export interface AuthUser {
  id: string;
  email: string;
  name: string;
  createdAtUtc?: string;
}

interface AuthScreenProps {
  onAuthenticated: (user: AuthUser) => void;
}

export const AuthScreen = ({ onAuthenticated }: AuthScreenProps) => {
  const [mode, setMode] = useState<'login' | 'register' | 'reset'>('login');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isRegister = mode === 'register';
  const isReset = mode === 'reset';

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);
    setIsSubmitting(true);
    try {
      if (isReset) {
        if (resetToken.trim()) {
          const response = await apiFetch('/api/auth/password-reset/confirm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              token: resetToken.trim(),
              new_password: newPassword,
            }),
          });
          if (!response.ok) {
            throw new Error(await readApiError(response, 'Nie udało się zmienić hasła.'));
          }
          setMode('login');
          setPassword('');
          setNewPassword('');
          setResetToken('');
          setMessage('Hasło zostało zmienione. Zaloguj się ponownie.');
          return;
        }

        const response = await apiFetch('/api/auth/password-reset/request', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email.trim() }),
        });
        if (!response.ok) {
          throw new Error(await readApiError(response, 'Nie udało się rozpocząć resetu hasła.'));
        }
        const payload = (await response.json()) as {
          message?: string;
          passwordReset?: { resetToken?: string } | null;
        };
        setResetToken(payload.passwordReset?.resetToken || '');
        setMessage(payload.message || 'Jeśli konto istnieje, wysłano instrukcję resetu hasła.');
        return;
      }

      const response = await apiFetch(`/api/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: email.trim(),
          password,
          ...(isRegister ? { name: name.trim() || undefined } : {}),
        }),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response, 'Nie udało się zalogować.'));
      }
      const payload = (await response.json()) as { user?: AuthUser };
      if (!payload.user) throw new Error('Backend nie zwrócił użytkownika.');
      onAuthenticated(payload.user);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-shell">
      <form className="auth-panel" onSubmit={handleSubmit}>
        <div>
          <div className="auth-brand">ElektroScan AI</div>
          <div className="text-sm text-muted">
            {isReset
              ? 'Zresetuj hasło do konta.'
              : isRegister
                ? 'Utwórz konto do pracy na projektach.'
                : 'Zaloguj się do swojego workspace.'}
          </div>
        </div>

        <div className="auth-tabs" role="tablist" aria-label="Tryb logowania">
          <button
            type="button"
            className={mode === 'login' ? 'active' : ''}
            onClick={() => {
              setMode('login');
              setError(null);
              setMessage(null);
            }}
          >
            Logowanie
          </button>
          <button
            type="button"
            className={mode === 'register' ? 'active' : ''}
            onClick={() => {
              setMode('register');
              setError(null);
              setMessage(null);
            }}
          >
            Rejestracja
          </button>
        </div>

        <label className="form-field">
          E-mail
          <input
            type="email"
            autoComplete="email"
            value={email}
            onChange={event => setEmail(event.target.value)}
            required
          />
        </label>

        {isRegister && (
          <label className="form-field">
            Nazwa
            <input
              type="text"
              autoComplete="name"
              value={name}
              onChange={event => setName(event.target.value)}
              placeholder="np. Jan Kowalski"
            />
          </label>
        )}

        {!isReset && (
          <label className="form-field">
            Hasło
            <input
              type="password"
              autoComplete={isRegister ? 'new-password' : 'current-password'}
              value={password}
              onChange={event => setPassword(event.target.value)}
              minLength={8}
              required
            />
          </label>
        )}

        {isReset && resetToken && (
          <>
            <label className="form-field">
              Token resetu
              <input
                value={resetToken}
                onChange={event => setResetToken(event.target.value)}
                required
              />
            </label>
            <label className="form-field">
              Nowe hasło
              <input
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={event => setNewPassword(event.target.value)}
                minLength={8}
                required
              />
            </label>
          </>
        )}

        {message && <div className="form-success">{message}</div>}
        {error && <div className="form-error">{error}</div>}

        <button className="btn-primary" type="submit" disabled={isSubmitting}>
          {isReset ? <KeyRound size={18} /> : isRegister ? <UserPlus size={18} /> : <LogIn size={18} />}
          {isSubmitting
            ? 'Przetwarzanie...'
            : isReset
              ? resetToken
                ? 'Zmień hasło'
                : 'Wyślij reset'
              : isRegister
                ? 'Utwórz konto'
                : 'Zaloguj'}
        </button>

        {mode === 'login' && (
          <button
            type="button"
            className="btn-text"
            onClick={() => {
              setMode('reset');
              setError(null);
              setMessage(null);
            }}
          >
            Nie pamiętasz hasła?
          </button>
        )}

        {mode === 'reset' && (
          <button
            type="button"
            className="btn-text"
            onClick={() => {
              setMode('login');
              setResetToken('');
              setNewPassword('');
              setError(null);
              setMessage(null);
            }}
          >
            Wróć do logowania
          </button>
        )}
      </form>
    </div>
  );
};
