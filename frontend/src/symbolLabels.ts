const stripDiacritics = (value: string) =>
  value.normalize('NFD').replace(/\p{Diacritic}/gu, '');

const containsAny = (value: string, needles: string[]) =>
  needles.some(needle => value.includes(needle));

export const formatSymbolLabel = (name: string) => {
  const label = name.replace(/^\d+_+/, '').replace(/_+/g, ' ').trim() || name;
  const plain = stripDiacritics(label).toUpperCase().replace(/[^A-Z0-9+\-/ ]+/g, ' ');
  const compact = plain.replace(/\s+/g, ' ').trim();
  const voltage = compact.match(/\b(230|400)\s*V\b/)?.[1];
  const ip = compact.match(/\bIP\s*(20|44|54|65)\b/)?.[1];
  const phaseRaw = compact.match(/\b([135])\s*-?\s*F\b/)?.[1];
  const phase = phaseRaw === '5' ? '3' : phaseRaw;

  if (compact.includes('ROZDZ')) {
    return 'ROZDZIELNICA';
  }

  if (
    containsAny(compact, ['WYPUST', 'WYPUS', 'WYFUS', 'WYIFUS'])
    && containsAny(compact, ['SCIANY', 'SCLANY', 'SC1ANY'])
  ) {
    return `WYPUST ZE SCIANY${voltage ? ` ${voltage}V` : ''}`;
  }

  if (
    containsAny(compact, ['ZESTAW', 'SOCKET KIT'])
    && (compact.includes('2X16') || compact.includes('SOCKET KIT'))
  ) {
    return 'ZESTAW GNIAZD 2x16A 3f 2x16A 1f';
  }

  if (containsAny(compact, ['BOLCEM', 'ROICEM', 'OCHRONNYM', 'OCHRONNY'])) {
    return [
      'GNIAZDO',
      phase ? `${phase}-F` : '',
      'Z BOLCEM OCHRONNYM',
      compact.includes('16A') || compact.includes('I6A') ? '16A' : '',
      ip ? `IP${ip}` : '',
    ].filter(Boolean).join(' ');
  }

  return label;
};
