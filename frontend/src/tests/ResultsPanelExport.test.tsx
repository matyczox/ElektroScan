import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ResultsPanel } from '../components/ResultsPanel';

describe('ResultsPanel export', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('exports current visible detections with display labels', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Blob(['xlsx']), {
        status: 200,
        headers: {
          'Content-Disposition': "attachment; filename*=UTF-8''wyniki.xlsx",
        },
      })
    );
    vi.stubGlobal('fetch', fetchMock);

    const createObjectURL = vi.fn(() => 'blob:wyniki');
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    render(
      <ResultsPanel
        projectId="project-1"
        analysisContext={{ analysisId: 'analysis-1', sourcePdf: 'plan.pdf' }}
        results={[
          { name: '01_TM', count: 10, color: '#ef4444' },
          { name: '02_TAB', count: 5, color: '#f97316' },
        ]}
        boxes={[
          {
            id: 'box-1',
            symbolName: '01_TM',
            x: 10,
            y: 10,
            width: 20,
            height: 20,
            confidence: 0.9,
            color: '#ef4444',
          },
          {
            id: 'box-2',
            symbolName: '02_TAB',
            x: 40,
            y: 10,
            width: 20,
            height: 20,
            confidence: 0.8,
            color: '#f97316',
          },
        ]}
        symbolLabels={{
          '01_TM': 'rozdzielnica glowna mieszkaniowa',
          '02_TAB': 'rozdzielnica administracyjna budynku',
        }}
        onRejectBox={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Eksport'));
    fireEvent.click(screen.getByText('Eksportuj XLSX'));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0];
    const payload = JSON.parse(String(init?.body));

    expect(String(url)).toContain('/api/projects/project-1/analysis-export');
    expect(init?.method).toBe('POST');
    expect(payload.boxes).toHaveLength(2);
    expect(payload.results[0].count).toBe(10);
    expect(payload.symbolLabels['01_TM']).toBe('rozdzielnica glowna mieszkaniowa');
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:wyniki');
  });

  it('exports review JSON without calling the backend', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const createObjectURL = vi.fn(() => 'blob:review-json');
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    render(
      <ResultsPanel
        projectId="project-1"
        analysisContext={{ analysisId: 'analysis-1', sourcePdf: 'plan.pdf' }}
        results={[
          { name: '01_TM', count: 1, color: '#ef4444' },
        ]}
        boxes={[
          {
            id: 'box-1',
            symbolName: '01_TM',
            x: 10,
            y: 10,
            width: 20,
            height: 20,
            confidence: 0.9,
            color: '#ef4444',
            source: 'manual',
            note: 'roiInspectorTop=01_TM PASS match=0.900',
            reviewStatus: 'accepted',
          },
        ]}
        symbolLabels={{ '01_TM': 'rozdzielnica glowna mieszkaniowa' }}
        onRejectBox={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Eksport'));
    fireEvent.click(screen.getByText('Eksportuj JSON review'));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:review-json');
  });
});
