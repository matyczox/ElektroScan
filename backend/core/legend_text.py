"""Legend text cleanup, file-name sanitizing and short-code token helpers."""

from __future__ import annotations

import re
import unicodedata

MAX_FILENAME_LENGTH = 80


def _sanitize_filename(text: str) -> str:
    """Clean text to a safe ASCII-ish filename fragment."""
    text = text.strip().replace("\n", "_")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    _pl = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    text = text.translate(_pl)
    return text[:MAX_FILENAME_LENGTH].strip("_")


def _clean_ocr_label_text(text: str) -> str | None:
    """Turn noisy OCR from a legend description into a stable human label."""

    raw = " ".join(str(text or "").replace("_", " ").split())
    if sum(1 for char in raw if char.isalnum()) < 2:
        return None

    upper = unicodedata.normalize("NFKC", raw).upper()
    upper = upper.translate(str.maketrans("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ"))
    upper = re.sub(r"[^A-Z0-9+\-/ ]+", " ", upper)
    upper = re.sub(r"\s+", " ", upper).strip()
    compact = re.sub(r"[^A-Z0-9]+", " ", upper)

    def contains_any(*needles: str) -> bool:
        return any(needle in compact for needle in needles)

    voltage_match = re.search(r"\b(230|400)\s*V\b", compact)
    voltage = f"{voltage_match.group(1)}V" if voltage_match else None
    ip_match = re.search(r"\bIP\s*(20|44|54|65)\b", compact)
    ip = f"IP{ip_match.group(1)}" if ip_match else None
    phase_match = re.search(r"\b([135])\s*[-]?\s*F\b", compact)
    phase = phase_match.group(1) if phase_match else None
    if phase == "5":
        # Tesseract commonly misreads the 3-phase row as 5-F on this CAD font.
        phase = "3"

    if "ROZDZ" in compact:
        has_descriptor = any(
            token in compact
            for token in (
                "GLOWNA",
                "ADMINISTRACYJNA",
                "MIESZKANIOWA",
                "BUDYNKU",
                "LOKALNA",
                "PIETROWA",
            )
        )
        if has_descriptor:
            readable = re.sub(
                r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż+\-/ ]+",
                " ",
                raw,
            )
            readable = re.sub(r"\s+", " ", readable).strip()
            if readable:
                return readable
        return "ROZDZIELNICA"

    if contains_any("WYPUST", "WYPUS", "WYFUS", "WYIFUS") and contains_any(
        "SCIANY", "SCLANY", "SC1ANY"
    ):
        suffix = f" {voltage}" if voltage else ""
        return f"WYPUST ZE SCIANY{suffix}".strip()

    if contains_any("ZESTAW", "SOCKET KIT") and ("2X16" in compact or "SOCKET KIT" in compact):
        return "ZESTAW GNIAZD 2x16A 3f 2x16A 1f"

    if contains_any("BOLCEM", "ROICEM", "OCHRONNYM", "OCHRONNY"):
        parts = ["GNIAZDO"]
        if phase:
            parts.append(f"{phase}-F")
        parts.extend(["Z", "BOLCEM", "OCHRONNYM"])
        if "16A" in compact or "I6A" in compact:
            parts.append("16A")
        if ip:
            parts.append(ip)
        return " ".join(parts)

    readable = re.sub(r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż+\-/ ]+", " ", raw)
    readable = re.sub(r"\s+", " ", readable).strip()
    tokens = readable.split()
    if len(tokens) > 12:
        tokens = tokens[:12]
    cleaned = " ".join(tokens)
    if sum(1 for char in cleaned if char.isalnum()) < 2:
        return None
    return cleaned


def _clean_table_description_ocr_text(text: str) -> str | None:
    """Keep OCR table descriptions readable without forcing electrical shortcuts."""

    raw = " ".join(str(text or "").replace("_", " ").split())
    if sum(1 for char in raw if char.isalnum()) < 2:
        return None

    readable = re.sub(
        r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż+\-/()., ]+",
        " ",
        raw,
    )
    readable = re.sub(r"\s+", " ", readable).strip(" .,-")
    if not readable:
        return None

    compact = _sanitize_filename(readable).casefold().strip("_")
    ignored = {
        "legenda",
        "legend",
        "symbol",
        "opis",
        "nazwa",
        "nazwa_artykulu",
        "nazwaartykułu",
        "indeks",
        "producent",
    }
    if compact in ignored:
        return None

    tokens = readable.split()
    if len(tokens) > 28:
        tokens = tokens[:28]
    cleaned = " ".join(tokens)
    return cleaned if sum(1 for char in cleaned if char.isalnum()) >= 2 else None


def _symbol_text_token(text: str) -> str | None:
    """Return a short alphanumeric symbol token from PDF text, if it looks like one."""

    token = _sanitize_filename(str(text or "")).upper()
    if not re.fullmatch(r"[A-Z0-9_]{2,12}", token):
        return None

    compact = token.replace("_", "")
    if not (2 <= len(compact) <= 10):
        return None
    if not re.search(r"[A-Z]", compact):
        return None
    if not (re.search(r"\d", compact) or len(compact) <= 4):
        return None
    if re.fullmatch(r"\d+(?:X\d+)+", compact):
        return None
    return compact
