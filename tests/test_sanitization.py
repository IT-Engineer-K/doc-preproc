"""Repository-wide sanitization tests to guarantee public-repo hygiene."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".json",
    ".yml",
    ".yaml",
    ".gitignore",
}


def _iter_repo_text_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "py.typed"}:
            files.append(path)
    return sorted(files)


def _build_forbidden_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Build sanitization regexes using generic PII checks and optional env terms."""
    url_scheme = "".join(["ht", "tp", "s?://"])
    patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            "url_literal",
            re.compile(url_scheme, re.IGNORECASE),
        ),
        (
            "phone_number",
            re.compile(r"(?<!\d)(?:\+?\d{1,3}-)?0\d{1,4}-\d{1,4}-\d{3,4}(?!\d)"),
        ),
        (
            "invoice_registration_id",
            re.compile(r"\bT\d{13}\b"),
        ),
    ]
    extra_terms = [
        re.escape(tok.strip())
        for tok in os.environ.get("SANITIZATION_FORBIDDEN_TERMS", "").split(",")
        if tok.strip()
    ]
    if extra_terms:
        patterns.append(
            (
                "forbidden_term",
                re.compile("|".join(extra_terms), re.IGNORECASE),
            )
        )
    return patterns


class TestRepositorySanitization(unittest.TestCase):
    def test_no_forbidden_patterns_in_repo(self) -> None:
        files = _iter_repo_text_files()
        self.assertGreater(len(files), 0)
        patterns = _build_forbidden_patterns()

        violations: list[str] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            rel = file_path.relative_to(REPO_ROOT)
            for label, regex in patterns:
                for match in regex.finditer(content):
                    violations.append(f"{rel}: matched {label} ({match.group(0)!r})")

        self.assertEqual(violations, [], "Sanitization violations found:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
