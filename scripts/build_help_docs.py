#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - tooling dependency
    yaml = None


HELP_EXTS = {".yaml", ".yml", ".json"}
DEFAULT_HTML_CSS = "help.css"
INTRO_SECTION = "Introduction"
INTRO_PAGE = "Introduction"
SECTION_LABELS = {
    "10_basics": "Basics",
    "20_features": "Features",
}
FEATURE_ORDER = [
    "data",
    "selections",
    "statistics",
    "charts",
    "som",
    "regression",
    "forecasting",
    "log",
    "chat",
]
TITLE_OVERRIDES = {
    "som": "SOM",
    "ui": "UI",
    "api": "API",
}


def _load_help_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load .yaml help files.")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        return {}
    return data


def _extract_entries(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries = data.get("entries")
    if entries is None:
        non_meta = {k: v for k, v in data.items() if k not in {"version", "metadata"}}
        entries = non_meta or {}
    if not isinstance(entries, dict):
        return {}
    return {k: v for k, v in entries.items() if isinstance(v, dict)}


def _titleize(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"^[0-9]+[\\s_-]*", "", raw)
    key = raw.lower()
    if key in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[key]
    return raw.replace("_", " ").replace("-", " ").title()


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "section"


def _anchor(section: str, subsection: Optional[str], page: str) -> str:
    if subsection:
        return _slug(f"{section}-{subsection}-{page}")
    return _slug(f"{section}-{page}")


def _section_for_path(rel_parts: Tuple[str, ...]) -> Tuple[str, Optional[str]]:
    if not rel_parts or len(rel_parts) == 1:
        return INTRO_SECTION, None
    top = rel_parts[0]
    if top in SECTION_LABELS:
        section = SECTION_LABELS[top]
    else:
        section = _titleize(top)
    subsection = None
    if top == "20_features" and len(rel_parts) > 1:
        subsection = _titleize(rel_parts[1])
    return section, subsection


def _page_title(rel_parts: Tuple[str, ...]) -> str:
    if not rel_parts:
        return INTRO_PAGE
    if len(rel_parts) == 1:
        return INTRO_PAGE
    stem = Path(rel_parts[-1]).stem
    if stem in {"tab", "overview"}:
        return INTRO_PAGE
    return _titleize(stem)


def _iter_help_files(help_dir: Path) -> Iterable[Path]:
    return sorted(
        p for p in help_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in HELP_EXTS
    )


def _is_intro_page(page: str) -> bool:
    return page == INTRO_PAGE


def _should_show_entry_title(
    section: str,
    subsection: Optional[str],
    page: str,
    entries: List[Tuple[str, Dict[str, Any]]],
    entry_title: str,
) -> bool:
    if not _is_intro_page(page):
        return True
    if len(entries) != 1:
        return True

    normalized_title = entry_title.strip().lower()
    candidate_labels = {section.strip().lower()}
    if subsection:
        candidate_labels.add(subsection.strip().lower())

    return normalized_title not in candidate_labels


# @ai(gpt-5, codex, refactor, 2026-03-10)
def build_help_doc(help_dir: Path, output_path: Path) -> None:
    files = list(_iter_help_files(help_dir))
    if not files:
        raise RuntimeError(f"No help files found in {help_dir}")

    metadata: Dict[str, Any] = {}
    version = None
    tree: Dict[str, Dict[Optional[str], Dict[str, List[Tuple[str, Dict[str, Any]]]]]] = {}
    section_order: List[str] = []

    for path in files:
        data = _load_help_file(path)
        if data.get("version") is not None:
            version = data.get("version")
        meta = data.get("metadata")
        if isinstance(meta, dict):
            metadata.update(meta)

        entries = _extract_entries(data)
        if not entries:
            continue

        rel = path.relative_to(help_dir)
        section, subsection = _section_for_path(rel.parts)
        page_title = _page_title(rel.parts)
        if section not in tree:
            tree[section] = {}
            section_order.append(section)
        if subsection not in tree[section]:
            tree[section][subsection] = {}
        if page_title not in tree[section][subsection]:
            tree[section][subsection][page_title] = []
        for key, entry in entries.items():
            tree[section][subsection][page_title].append((key, entry))

    title = metadata.get("app") or "Help"
    description = metadata.get("description") or ""
    updated = date.today().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title} Help\n\n")
        if description:
            handle.write(f"{description}\n\n")
        if updated:
            handle.write(f"Updated: {updated}\n\n")
        if version is not None:
            handle.write(f"Version: {version}\n\n")

        for section in section_order:
            handle.write(f"## {section}\n\n")
            subsections = tree.get(section, {})
            subsection_items = list(subsections.items())
            if section == "Features":
                order_index = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
                subsection_items.sort(
                    key=lambda item: order_index.get((item[0] or "").lower(), 999)
                )
            for subsection, pages in subsection_items:
                if subsection is not None:
                    handle.write(f"### {subsection}\n\n")
                for page, entries in sorted(
                    pages.items(),
                    key=lambda item: (0 if item[0] == INTRO_PAGE else 1, item[0]),
                ):
                    if not _is_intro_page(page):
                        handle.write(f"#### {page}\n\n")
                    for key, entry in entries:
                        title_text = entry.get("title") or key
                        short = entry.get("short")
                        body = entry.get("body")
                        show_entry_title = _should_show_entry_title(
                            section,
                            subsection,
                            page,
                            entries,
                            str(title_text),
                        )
                        if show_entry_title:
                            handle.write(f"##### {title_text}\n\n")
                        if short:
                            handle.write(f"{short}\n\n")
                        if body:
                            handle.write(f"{body}\n\n")


# @ai(gpt-5, codex, refactor, 2026-03-10)
def build_help_html(help_dir: Path, output_path: Path) -> None:
    files = list(_iter_help_files(help_dir))
    if not files:
        raise RuntimeError(f"No help files found in {help_dir}")

    metadata: Dict[str, Any] = {}
    version = None

    tree: Dict[str, Dict[Optional[str], Dict[str, List[Tuple[str, Dict[str, Any]]]]]] = {}
    section_order: List[str] = []

    for path in files:
        data = _load_help_file(path)
        if data.get("version") is not None:
            version = data.get("version")
        meta = data.get("metadata")
        if isinstance(meta, dict):
            metadata.update(meta)

        entries = _extract_entries(data)
        if not entries:
            continue

        rel = path.relative_to(help_dir)
        section, subsection = _section_for_path(rel.parts)
        page_title = _page_title(rel.parts)
        if section not in tree:
            tree[section] = {}
            section_order.append(section)
        if subsection not in tree[section]:
            tree[section][subsection] = {}
        if page_title not in tree[section][subsection]:
            tree[section][subsection][page_title] = []
        for key, entry in entries.items():
            tree[section][subsection][page_title].append((key, entry))

    title = metadata.get("app") or "Help"
    description = metadata.get("description") or ""
    updated = date.today().isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("<!doctype html>\n")
        handle.write("<html lang=\"en\">\n")
        handle.write("<head>\n")
        handle.write("  <meta charset=\"utf-8\" />\n")
        handle.write(f"  <title>{html.escape(str(title))} Help</title>\n")
        handle.write("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n")
        handle.write(f"  <link rel=\"stylesheet\" href=\"{DEFAULT_HTML_CSS}\" />\n")
        handle.write("</head>\n")
        handle.write("<body>\n")
        handle.write("<div class=\"layout\">\n")
        handle.write("<nav>\n")
        handle.write(f"  <h2>{html.escape(str(title))}</h2>\n")
        if description:
            handle.write(f"  <div class=\"meta\">{html.escape(str(description))}</div>\n")
        if updated:
            handle.write(f"  <div class=\"meta\">Updated: {html.escape(str(updated))}</div>\n")
        if version is not None:
            handle.write(f"  <div class=\"meta\">Version: {html.escape(str(version))}</div>\n")

        for section in section_order:
            handle.write("  <details open>\n")
            handle.write(f"    <summary>{html.escape(section)}</summary>\n")
            subsections = tree.get(section, {})
            subsection_items = list(subsections.items())
            if section == "Features":
                order_index = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
                subsection_items.sort(
                    key=lambda item: order_index.get((item[0] or "").lower(), 999)
                )
            for subsection, pages in subsection_items:
                if subsection is None:
                    for page in sorted(
                        pages,
                        key=lambda name: (0 if name == INTRO_PAGE else 1, name),
                    ):
                        anchor = _anchor(section, None, page)
                        handle.write(f"    <a class=\"nav-page\" href=\"#{anchor}\">{html.escape(page)}</a>\n")
                    continue
                handle.write(f"    <details class=\"nav-group\" open>\n")
                handle.write(f"      <summary>{html.escape(subsection)}</summary>\n")
                has_overview = INTRO_PAGE in pages
                for page in sorted(
                    pages,
                    key=lambda name: (
                        0 if name == INTRO_PAGE else 1,
                        name,
                    ),
                ):
                    if page == INTRO_PAGE and not has_overview:
                        continue
                    anchor = _anchor(section, subsection, page)
                    handle.write(f"      <a class=\"nav-page\" href=\"#{anchor}\">{html.escape(page)}</a>\n")
                handle.write("    </details>\n")
            handle.write("  </details>\n")
        handle.write("</nav>\n")

        handle.write("<main>\n")
        handle.write(f"<h1>{html.escape(str(title))} Help</h1>\n")
        if description:
            handle.write(f"<p>{html.escape(str(description))}</p>\n")

        for section in section_order:
            handle.write(f"<h2>{html.escape(section)}</h2>\n")
            subsections = tree.get(section, {})
            subsection_items = list(subsections.items())
            if section == "Features":
                order_index = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
                subsection_items.sort(
                    key=lambda item: order_index.get((item[0] or "").lower(), 999)
                )
            for subsection, pages in subsection_items:
                if subsection is not None:
                    handle.write(f"<h3>{html.escape(subsection)}</h3>\n")
                for page, entries in sorted(
                    pages.items(),
                    key=lambda item: (0 if item[0] == INTRO_PAGE else 1, item[0]),
                ):
                    anchor = _anchor(section, subsection, page)
                    if not _is_intro_page(page):
                        handle.write(f"<h4 id=\"{anchor}\">{html.escape(page)}</h4>\n")
                    else:
                        handle.write(f"<div id=\"{anchor}\" class=\"page-anchor\"></div>\n")
                    for key, entry in entries:
                        title_text = entry.get("title") or key
                        short = entry.get("short") or ""
                        body = entry.get("body") or ""
                        show_entry_title = _should_show_entry_title(
                            section,
                            subsection,
                            page,
                            entries,
                            str(title_text),
                        )
                        if show_entry_title:
                            handle.write(f"<h5>{html.escape(str(title_text))}</h5>\n")
                        if short:
                            handle.write(f"<p>{html.escape(str(short))}</p>\n")
                        if body:
                            handle.write(f"{body}\n")
        handle.write("</main>\n")
        handle.write("</div>\n")
        handle.write("</body>\n")
        handle.write("</html>\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build markdown docs from help YAML files.")
    parser.add_argument(
        "--help-dir",
        default="resources/help",
        help="Path to help directory (default: resources/help)",
    )
    parser.add_argument(
        "--output",
        default="resources/help/docs/help.md",
        help="Output markdown path (default: resources/help/docs/help.md)",
    )
    parser.add_argument(
        "--html",
        default="resources/help/docs/help.html",
        help="Output HTML path (default: resources/help/docs/help.html)",
    )
    args = parser.parse_args()

    help_dir = Path(args.help_dir).resolve()
    output_path = Path(args.output).resolve()
    html_path = Path(args.html).resolve()

    if not help_dir.exists():
        print(f"Help directory not found: {help_dir}", file=sys.stderr)
        return 1

    try:
        build_help_doc(help_dir, output_path)
        build_help_html(help_dir, html_path)
    except Exception as exc:
        print(f"Failed to build help docs: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote help documentation to {output_path}")
    print(f"Wrote help documentation to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
