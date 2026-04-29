import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PatternModal } from '../components/PatternModal';

const mockPattern = {
  name: 'symbol_06_test',
  imgBase64: 'data:image/png;base64,abc123',
};

describe('PatternModal', () => {
  it('renders pattern name in the input', () => {
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    const input = screen.getByRole('textbox') as HTMLInputElement;
    expect(input.value).toBe('symbol_06_test');
  });

  it('renders the pattern image', () => {
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    const img = screen.getByRole('img') as HTMLImageElement;
    expect(img.src).toContain('base64');
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={onClose}
        onSave={vi.fn()}
        onDelete={vi.fn()}
      />
    );
    const buttons = screen.getAllByRole('button');
    const closeBtn = buttons.find((btn) => btn.querySelector('svg'));
    closeBtn?.click();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('calls onDelete when delete button is clicked', () => {
    const onDelete = vi.fn();
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={vi.fn()}
        onSave={vi.fn()}
        onDelete={onDelete}
      />
    );
    fireEvent.click(screen.getByText('Usuń'));
    expect(onDelete).toHaveBeenCalledOnce();
  });

  it('calls onSave with original name when save is clicked without editing', () => {
    const onSave = vi.fn();
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={vi.fn()}
        onSave={onSave}
        onDelete={vi.fn()}
      />
    );
    fireEvent.click(screen.getByText('Zapisz zmiany'));
    expect(onSave).toHaveBeenCalledWith('symbol_06_test');
  });

  it('calls onSave with updated name after editing the input', () => {
    const onSave = vi.fn();
    render(
      <PatternModal
        pattern={mockPattern}
        onClose={vi.fn()}
        onSave={onSave}
        onDelete={vi.fn()}
      />
    );
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'nowa_nazwa' } });
    fireEvent.click(screen.getByText('Zapisz zmiany'));
    expect(onSave).toHaveBeenCalledWith('nowa_nazwa');
  });
});
