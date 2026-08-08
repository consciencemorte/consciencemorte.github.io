#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

THEME_VARS = ("paper", "paper-deep", "ink", "graphite", "line", "oxide", "moss", "blue")
VAR_RE = re.compile(r"var\(\s*--([A-Za-z0-9_-]+)\s*(?:,\s*([^\)]+?)\s*)?\)")
ROOT_BLOCK_RE = re.compile(r"(?ms)(^|\n)([ \t]*):root\s*\{(?P<body>.*?)\}\s*")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_light_palette(scss_path: Path) -> dict[str, str]:
    text = read_text(scss_path)
    # First plain :root block, excluding :root[data-theme=...]
    m = re.search(r"(?ms)^\s*:root\s*\{(?P<body>.*?)^\s*\}", text)
    if not m:
        raise SystemExit(f"Impossible de trouver le bloc :root dans {scss_path}")
    body = m.group("body")
    out: dict[str, str] = {}
    for name, value in re.findall(r"--([A-Za-z0-9_-]+)\s*:\s*([^;]+);", body):
        out[name] = value.strip()
    missing = [n for n in THEME_VARS if n not in out]
    if missing:
        raise SystemExit(f"Variables de thème absentes de {scss_path}: {', '.join(missing)}")
    bad = [f"--{n}={out[n]}" for n in THEME_VARS if not HEX_RE.match(out[n])]
    if bad:
        raise SystemExit("Les couleurs du thème doivent être des #RRGGBB: " + ", ".join(bad))
    return {n: out[n].lower() for n in THEME_VARS}


def strip_local_self_root(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        body = m.group("body")
        # Ne retire que le bloc local qui redéfinit les variables via var(--x,...)
        if any(re.search(rf"--{re.escape(name)}\s*:\s*var\(\s*--{re.escape(name)}\b", body)
               for name in THEME_VARS):
            return m.group(1)
        return m.group(0)
    return ROOT_BLOCK_RE.sub(repl, text)


def resolve_for_inkscape(text: str, palette: dict[str, str], src: Path) -> str:
    text = strip_local_self_root(text)
    # Inkscape 1.4 interprète parfois fill="transparent" comme du noir opaque.
    # fill="none" est l’équivalent visuel portable pour un SVG.
    text = re.sub(r'(?i)fill\s*=\s*(["\'])transparent\1', r'fill=\1none\1', text)
    text = re.sub(r'(?i)(fill|stroke)\s*:\s*transparent\b', lambda m: f"{m.group(1)}: none", text)
    unresolved: list[str] = []

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        fallback = (m.group(2) or "").strip()
        if name in palette:
            return palette[name]
        if fallback:
            return fallback
        unresolved.append(name)
        return m.group(0)

    text = VAR_RE.sub(repl, text)
    if "var(" in text:
        names = sorted(set(unresolved or re.findall(r"var\(\s*--([A-Za-z0-9_-]+)", text)))
        raise ValueError(f"{src}: variables CSS non résolues: {', '.join('--'+n for n in names)}")
    return text


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def restore_web_vars(text: str, palette: dict[str, str]) -> str:
    # Remplace les couleurs canoniques du thème, y compris les formes rgb() qu'Inkscape peut écrire.
    # Les nouvelles couleurs personnalisées choisies dans Inkscape restent donc fixes volontairement.
    for name in THEME_VARS:
        color = palette[name]
        fallback = color
        cssvar = f"var(--{name}, {fallback})"
        text = re.sub(re.escape(color), cssvar, text, flags=re.IGNORECASE)
        r, g, b = hex_to_rgb(color)
        rgb_patterns = [
            rf"rgb\(\s*{r}\s*,\s*{g}\s*,\s*{b}\s*\)",
            rf"rgb\(\s*{r}\s+{g}\s+{b}\s*\)",
        ]
        for pat in rgb_patterns:
            text = re.sub(pat, cssvar, text, flags=re.IGNORECASE)
    return text


def svg_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.svg") if p.is_file())


def prepare(source: Path, edit: Path, palette: dict[str, str]) -> int:
    files = svg_files(source)
    if not files:
        print(f"Aucun SVG dans {source}")
        return 1
    errors = 0
    for src in files:
        rel = src.relative_to(source)
        dst = edit / rel
        try:
            out = resolve_for_inkscape(read_text(src), palette, src)
            write_text(dst, out)
            print(f"PREPARE  {rel}")
        except Exception as e:
            errors += 1
            print(f"ERREUR   {e}", file=sys.stderr)
    print(f"\n{len(files)-errors}/{len(files)} SVG préparés dans {edit}")
    return 1 if errors else 0


def publish(source: Path, edit: Path, palette: dict[str, str], write: bool) -> int:
    files = svg_files(edit)
    if not files:
        print(f"Aucun SVG édité dans {edit}")
        return 1
    errors = 0
    changed = 0
    for edited in files:
        rel = edited.relative_to(edit)
        dst = source / rel
        if not dst.exists():
            print(f"SKIP     {rel} (pas de source correspondante)")
            continue
        try:
            out = restore_web_vars(read_text(edited), palette)
            # Le fichier publié ne doit pas contenir un bloc local auto-référent.
            out = strip_local_self_root(out)
            old = read_text(dst)
            if out != old:
                changed += 1
                if write:
                    shutil.copy2(dst, dst.with_suffix(dst.suffix + ".bak"))
                    write_text(dst, out)
                    print(f"PUBLISH  {rel}  (backup: {dst.name}.bak)")
                else:
                    print(f"WOULD    {rel}")
        except Exception as e:
            errors += 1
            print(f"ERREUR   {rel}: {e}", file=sys.stderr)
    if write:
        print(f"\n{changed} SVG publiés. Vérifie maintenant: git diff -- _includes/figures")
    else:
        print(f"\n{changed} SVG seraient publiés. Relance avec --write pour écrire.")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pont SVG web <-> Inkscape sans casser le thème CSS du site.")
    ap.add_argument("mode", choices=("prepare", "publish"))
    ap.add_argument("--repo", default=".", help="racine du dépôt")
    ap.add_argument("--source", default="_includes/figures", help="dossier SVG de production")
    ap.add_argument("--edit", default=".svg-inkscape", help="miroir éditable par Inkscape")
    ap.add_argument("--scss", default="assets/main.scss", help="feuille contenant le :root du thème")
    ap.add_argument("--write", action="store_true", help="avec publish, écrase les sources (avec .bak)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = (repo / args.source).resolve()
    edit = (repo / args.edit).resolve()
    scss = (repo / args.scss).resolve()
    palette = extract_light_palette(scss)

    if args.mode == "prepare":
        return prepare(source, edit, palette)
    return publish(source, edit, palette, args.write)


if __name__ == "__main__":
    raise SystemExit(main())
