import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { LegendReviewPanel, type LegendReviewItem } from '../components/LegendReviewPanel';

const baseItems: LegendReviewItem[] = [
  {
    id: '01_C1',
    name: 'C1',
    imgBase64: 'data:image/png;base64,abc123',
    status: 'pending',
  },
  {
    id: '02_D1',
    name: 'D1',
    imgBase64: 'data:image/png;base64,def456',
    status: 'accepted',
  },
];

const renderPanel = (items = baseItems, overrides = {}) => {
  const props = {
    items,
    activeCorrectionId: null,
    isProcessing: false,
    onAccept: vi.fn(),
    onAcceptAll: vi.fn(),
    onReject: vi.fn(),
    onStartCrop: vi.fn(),
    onCancelCrop: vi.fn(),
    onRename: vi.fn(),
    onAddMissing: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };

  render(<LegendReviewPanel {...props} />);
  return props;
};

describe('LegendReviewPanel', () => {
  it('shows review progress', () => {
    renderPanel();
    expect(screen.getByText('1/2 gotowe')).toBeInTheDocument();
  });

  it('accepts a pending template', () => {
    const props = renderPanel();
    fireEvent.click(screen.getAllByTitle('Akceptuj')[0]);
    expect(props.onAccept).toHaveBeenCalledWith('01_C1');
  });

  it('does not allow accepting a missing template before crop', () => {
    renderPanel([
      {
        id: 'manual_1',
        name: 'AW1',
        imgBase64: '',
        status: 'pending',
      },
    ]);

    expect(screen.getByTitle('Akceptuj')).toBeDisabled();
  });
});
