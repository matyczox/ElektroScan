import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { CostPanel } from '../components/CostPanel';

const mockResults = [
  { name: 'Symbol A', count: 3 },
  { name: 'Symbol B', count: 5 },
];

describe('CostPanel', () => {
  it('renders empty state when results array is empty', () => {
    render(<CostPanel results={[]} />);
    expect(screen.getByText('Brak danych do kosztorysu.')).toBeInTheDocument();
  });

  it('renders all symbol names from results', () => {
    render(<CostPanel results={mockResults} />);
    expect(screen.getByText('Symbol A')).toBeInTheDocument();
    expect(screen.getByText('Symbol B')).toBeInTheDocument();
  });

  it('shows zero total initially', () => {
    render(<CostPanel results={mockResults} />);
    expect(screen.getByText('0.00 PLN')).toBeInTheDocument();
  });

  it('count inputs are read-only', () => {
    render(<CostPanel results={mockResults} />);
    const allInputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    const readOnlyInputs = allInputs.filter((el) => el.readOnly);
    expect(readOnlyInputs).toHaveLength(2);
    expect(readOnlyInputs[0].value).toBe('3');
    expect(readOnlyInputs[1].value).toBe('5');
  });

  it('calculates total correctly after setting price for one symbol', () => {
    render(<CostPanel results={mockResults} />);
    const allInputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    const priceInputs = allInputs.filter((el) => !el.readOnly);
    // set price for Symbol A to 10 → 3 * 10 = 30
    fireEvent.change(priceInputs[0], { target: { value: '10' } });
    expect(screen.getByText('30.00 PLN')).toBeInTheDocument();
  });

  it('calculates total correctly with prices for both symbols', () => {
    render(<CostPanel results={mockResults} />);
    const allInputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    const priceInputs = allInputs.filter((el) => !el.readOnly);
    // Symbol A: 3 * 10 = 30, Symbol B: 5 * 4 = 20 → total 50
    fireEvent.change(priceInputs[0], { target: { value: '10' } });
    fireEvent.change(priceInputs[1], { target: { value: '4' } });
    expect(screen.getByText('50.00 PLN')).toBeInTheDocument();
  });

  it('treats non-numeric price input as zero', () => {
    render(<CostPanel results={mockResults} />);
    const allInputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    const priceInputs = allInputs.filter((el) => !el.readOnly);
    fireEvent.change(priceInputs[0], { target: { value: 'abc' } });
    expect(screen.getByText('0.00 PLN')).toBeInTheDocument();
  });

  it('renders single result correctly', () => {
    render(<CostPanel results={[{ name: 'Tylko jeden', count: 7 }]} />);
    expect(screen.getByText('Tylko jeden')).toBeInTheDocument();
    const allInputs = screen.getAllByRole('spinbutton') as HTMLInputElement[];
    const readOnly = allInputs.filter((el) => el.readOnly);
    expect(readOnly[0].value).toBe('7');
  });
});
