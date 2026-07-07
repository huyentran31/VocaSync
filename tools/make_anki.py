"""
make_anki.py — Tool #7 (write-local, medium cost). Mostly deterministic.

Turn vocab units into a .apkg deck with three card types per item:
  • Basic     — term  -> definition / example / collocations (+ media)
  • Cloze     — the sentence with the target blanked: "We must {{c1::reduce}} ..."
  • Dictation — audio on the front, type-what-you-hear on the back (only if audio)

Media is cut SHORT (sentence ±1 line) with ffmpeg (REUSES legacy/ffmpeg helpers):
audio .mp3 + a mid-frame screenshot .jpg, ASCII filenames `anki_<id>.*`
(clip_extraction.gherkin, AGENTS.md §6). Fast seek = -ss before -i; re-encode is
fine for short clips.

Error model: per-card failures are clip-errors — set status='fail' + a note, log,
and CONTINUE (never crash the deck). Missing ffmpeg just means text-only cards.
GUIDs are stable (genanki.guid_for) so re-running updates cards instead of dupes.
"""

from __future__ import annotations

import os
import re
import zlib

import genanki

from _common import ascii_safe, log_tool_call, new_run_id, run_dir

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {"description": "Node dicts, or {'node':..., 'clip':{video,start,end}} drafts."},
        "deck_name": {"type": "string"},
        "run_id": {"type": "string"},
        "gen_dictation": {"type": "boolean",
                          "description": "Generate Dictation cards too (default true)."},
    },
    "required": ["units"],
}

# --- stable model ids (fixed so decks merge cleanly across runs) -------------- #
_BASIC_ID = 1607392319
_CLOZE_ID = 1607392320
_DICT_ID = 1607392321

# SHARED CSS (S18 #8) — Supabase-dark theme, attached via css= to ALL THREE models so every
# card type shares one look. Fields / model_id / guid / model_type are UNCHANGED (invariant:
# changing any of those would DOUBLE existing decks on re-import). Only presentation changes.
_SHARED_CSS = (
    ".card{font-family:system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,Helvetica,Arial,sans-serif;font-size:16px;color:#ededed;background-color:#1c1c1c;text-align:left;line-height:1.6;padding:16px;max-width:600px;margin:0 auto}\n"
    ".vs-container{background-color:#2a2a2a;border:1px solid #333;border-radius:8px;padding:24px;box-shadow:0 4px 12px rgba(0,0,0,.3)}\n"
    ".vs-term{font-size:2rem;font-weight:700;color:#fff;letter-spacing:-.02em;margin-bottom:16px;line-height:1.2}\n"
    ".vs-def{font-size:1.15rem;color:#ededed;margin-bottom:16px}\n"
    ".vs-example{font-style:italic;color:#b0b0b0;border-left:3px solid #3ecf8e;padding-left:12px;margin:16px 0}\n"
    ".vs-divider{border:0;height:1px;background:#333;margin:24px 0}\n"
    ".vs-extra{font-size:.95rem;color:#b0b0b0;background-color:#1c1c1c;padding:16px;border-radius:6px;border:1px solid #2e2e2e}\n"
    ".vs-media{margin-top:20px}\n"
    ".vs-media img{max-width:100%;height:auto;border-radius:6px;margin-top:12px;border:1px solid #333}\n"
    ".vs-media audio,.vs-media .audio-player,.vs-media .replay-button{display:block;text-align:left;margin-right:auto}\n"
    ".cloze{color:#3ecf8e;font-weight:600;border-bottom:2px solid #3ecf8e;padding-bottom:2px}\n"
    ".vs-prompt{font-size:.85rem;text-transform:uppercase;letter-spacing:.05em;color:#8e8e8e;margin-bottom:12px}\n"
    # S18 Phase 2: Source line always visible; the heavier info folds into a native <details>
    # (no JS -> works on Anki desktop + mobile). "Also seen in" lists extra occurrences.
    ".vs-src{font-size:.85rem;color:#8e8e8e;margin-top:8px}\n"
    ".vs-more{margin-top:8px}\n"
    ".vs-more>summary{cursor:pointer;color:#3ecf8e;font-size:.9rem;font-weight:600;list-style:revert}\n"
    ".vs-more[open]>summary{margin-bottom:8px}\n"
    ".vs-occ{margin-top:10px;font-size:.9rem;color:#b0b0b0}\n"
    ".vs-occ .vs-occ-h{color:#8e8e8e;text-transform:uppercase;letter-spacing:.05em;font-size:.75rem;margin-bottom:4px}\n"
    ".nightMode .card,.card.nightMode{background-color:#1c1c1c;color:#ededed}\n"
    ".nightMode .vs-container{background-color:#2a2a2a}\n"
)

_BASIC_MODEL = genanki.Model(
    _BASIC_ID, "VocaSync Basic",
    fields=[{"name": "Term"}, {"name": "Definition"}, {"name": "Example"},
            {"name": "Extra"}, {"name": "Media"}],
    css=_SHARED_CSS,
    templates=[{
        "name": "Card 1",
        "qfmt": '<div class="vs-container"><div class="vs-term">{{Term}}</div></div>',
        "afmt": '<div class="vs-container">'
                '<div class="vs-term">{{Term}}</div>'
                '<div class="vs-def">{{Definition}}</div>'
                '{{#Example}}<div class="vs-example">{{Example}}</div>{{/Example}}'
                '{{#Extra}}<hr class="vs-divider"><div class="vs-extra">{{Extra}}</div>{{/Extra}}'
                '{{#Media}}<div class="vs-media">{{Media}}</div>{{/Media}}'
                '</div>',
    }],
)

_CLOZE_MODEL = genanki.Model(
    _CLOZE_ID, "VocaSync Cloze",
    model_type=genanki.Model.CLOZE,
    fields=[{"name": "Text"}, {"name": "Extra"}, {"name": "Media"}],
    css=_SHARED_CSS,
    templates=[{
        "name": "Cloze",
        "qfmt": '<div class="vs-container"><div class="vs-def">{{cloze:Text}}</div></div>',
        "afmt": '<div class="vs-container">'
                '<div class="vs-def">{{cloze:Text}}</div>'
                '{{#Extra}}<hr class="vs-divider"><div class="vs-extra">{{Extra}}</div>{{/Extra}}'
                '{{#Media}}<div class="vs-media">{{Media}}</div>{{/Media}}'
                '</div>',
    }],
)

_DICT_MODEL = genanki.Model(
    _DICT_ID, "VocaSync Dictation",
    fields=[{"name": "Audio"}, {"name": "Answer"}, {"name": "Extra"}],
    css=_SHARED_CSS,
    templates=[{
        "name": "Dictation",
        # S18 P0-2: {{type:Answer}} renders Anki's type-in box so the learner can transcribe
        # the WHOLE sentence (Answer = full sentence); Anki diffs the typed text on flip.
        # Field / model_id / guid UNCHANGED — Answer already exists, only the template grows.
        "qfmt": '<div class="vs-container">'
                '<div class="vs-prompt">Type what you hear</div>'
                '<div class="vs-media">{{Audio}}</div>'
                '<div style="margin-top:16px">{{type:Answer}}</div>'
                '</div>',
        "afmt": '<div class="vs-container">'
                '<div class="vs-prompt">Answer</div>'
                '<div class="vs-def" style="font-weight:600;color:#fff">{{Answer}}</div>'
                '{{#Extra}}<hr class="vs-divider"><div class="vs-extra">{{Extra}}</div>{{/Extra}}'
                '<div class="vs-media" style="margin-top:16px">{{Audio}}</div>'
                '</div>',
    }],
)


# S18 Phase 2 (1c): NEW note type "Definition -> guess the word". model_id is NEW and
# distinct from the 3 legacy ids (additive — legacy decks are untouched, no history loss).
# Reuses the Basic field set so no new node data is needed. Front shows the definition (+POS
# hint); back reveals the term + the full info block (same as Basic's back, with <details>).
_DEF_ID = 1607392322
_DEF_MODEL = genanki.Model(
    _DEF_ID, "VocaSync Definition",
    fields=[{"name": "Term"}, {"name": "Definition"}, {"name": "Example"},
            {"name": "Extra"}, {"name": "Media"}],
    css=_SHARED_CSS,
    templates=[{
        "name": "Definition",
        "qfmt": '<div class="vs-container">'
                '<div class="vs-prompt">Which word?</div>'
                '<div class="vs-def">{{Definition}}</div>'
                '</div>',
        "afmt": '<div class="vs-container">'
                '<div class="vs-term">{{Term}}</div>'
                '<div class="vs-def">{{Definition}}</div>'
                '{{#Example}}<div class="vs-example">{{Example}}</div>{{/Example}}'
                '{{#Extra}}<hr class="vs-divider"><div class="vs-extra">{{Extra}}</div>{{/Extra}}'
                '{{#Media}}<div class="vs-media">{{Media}}</div>{{/Media}}'
                '</div>',
    }],
)


def _deck_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def _guid(*parts) -> str:
    return genanki.guid_for("::".join(str(p) for p in parts))


def _clean_source(src: str) -> str:
    """A readable source label for a subdeck: drop the media extension. '' -> 'Unsorted'."""
    base = re.sub(r"\.(srt|vtt|ass|mp4|mkv|mov|mp3|wav|m4a)$", "", (src or "").strip(), flags=re.I)
    return base.strip() or "Unsorted"


def _tag(text: str) -> str:
    """Anki tag token — tags are space-separated, so spaces/punctuation collapse to '_'."""
    return re.sub(r"\s+", "_", re.sub(r"[^\w\s-]", "", (text or "").strip())).strip("_") or "Unsorted"


def _edge_targets(nd: dict, types: tuple) -> list[str]:
    """Distinct edge targets of the given relation type(s), order-preserving, with the
    "#sense" suffix stripped (edge targets are node keys like "vehicle#vehicle.n.01").
    Used by P0-3 to surface synonyms / antonyms / is_a / part_of on the card."""
    out: list[str] = []
    for e in (nd.get("edges") or []):
        if isinstance(e, dict) and e.get("type") in types:
            t = (e.get("target") or "").split("#")[0].strip()
            if t and t not in out:
                out.append(t)
    return out


def _sources_of(nd: dict) -> list[str]:
    """Distinct source filenames this node was seen in (order-preserving), from occurrences."""
    seen, out = set(), []
    for o in (nd.get("occurrences") or []):
        s = (o.get("source") or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _node_of(u) -> dict:
    if hasattr(u, "model_dump"):
        return u.model_dump()
    if isinstance(u, dict) and "node" in u:
        return u["node"]
    return u


def _clip_of(u) -> dict | None:
    if isinstance(u, dict) and isinstance(u.get("clip"), dict):
        return u["clip"]
    return None


def _cloze_sentence(sentence: str, surface: str) -> str | None:
    """Blank the FIRST occurrence of `surface` in `sentence` as a c1 cloze."""
    if not sentence or not surface:
        return None
    pat = re.compile(re.escape(surface), re.IGNORECASE)
    if not pat.search(sentence):
        return None
    return pat.sub(lambda m: "{{c1::" + m.group(0) + "}}", sentence, count=1)


def _cut_media(clip: dict, out_dir: str, stem: str) -> tuple[str | None, str | None, str]:
    """Cut short audio + mid screenshot. Returns (mp3_path|None, jpg_path|None, note).

    A bad timestamp / missing ffmpeg / out-of-bounds end is a clip-error: return
    (None, None, note) so the caller falls back to a text-only card.
    """
    import config
    try:
        import ffmpeg as legacy_ffmpeg  # legacy/ffmpeg.py (on sys.path via _common)
    except Exception as e:
        return None, None, f"ffmpeg module unavailable: {e}"

    video = clip.get("video", "")
    start = clip.get("start", "")
    end = clip.get("end", "")
    if not video or not os.path.exists(video):
        return None, None, f"source media not found: {video!r}"
    # S18 P0-1: accept FLOAT seconds (preferred — keeps millisecond precision) OR a legacy
    # "HH:MM:SS[.mmm]" string (backward-compat; older callers / tests). The Mine + commit
    # paths now thread the segment's float start_sec/end_sec into the clip.
    def _to_sec(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?", v.strip()):
            a, b, c = v.strip().split(":")
            return int(a) * 3600 + int(b) * 60 + float(c)
        return None

    start_s, end_s = _to_sec(start), _to_sec(end)
    if start_s is None or end_s is None:
        return None, None, f"bad timestamps {start!r}/{end!r}"
    if end_s <= start_s:
        return None, None, f"end {end!r} not after start {start!r}"

    # S18 P0-1: pad each edge so short cues aren't clipped mid-word / into silence.
    pad = getattr(config, "ANKI_CLIP_PAD", 0.2)
    start_s = max(0.0, start_s - pad)
    end_s = end_s + pad

    dur = legacy_ffmpeg.get_clip_duration(video)
    # S18 P0-1 (id B): only clamp the upper edge when duration is KNOWN (dur > 0). On this
    # box ffprobe is absent -> get_clip_duration()==0.0; clamping end to 0 would yield a
    # zero-length (silent) clip. With dur unknown, let ffmpeg read to the true end.
    if dur and dur > 0:
        if start_s >= dur:                              # start past EOF -> clip-error
            return None, None, f"start {start!r} beyond media duration {dur:.1f}s"
        end_s = min(dur, end_s)

    mp3 = os.path.join(out_dir, f"{stem}.mp3")
    jpg = os.path.join(out_dir, f"{stem}.jpg")
    length = end_s - start_s
    mid = start_s + length / 2.0
    try:
        # audio: fast seek (-ss before -i), re-encode to mp3, mono
        legacy_ffmpeg._run_ffmpeg(["-y", "-ss", str(start_s), "-i", video, "-t", str(length),
                                   "-vn", "-ac", "1", "-ar", "44100", mp3])
        # screenshot: single mid frame
        legacy_ffmpeg._run_ffmpeg(["-y", "-ss", str(mid), "-i", video, "-frames:v", "1",
                                   "-q:v", "2", jpg])
    except Exception as e:
        for p in (mp3, jpg):
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        return None, None, f"ffmpeg failed: {e}"

    has_mp3 = os.path.exists(mp3)
    has_jpg = os.path.exists(jpg)
    return (mp3 if has_mp3 else None), (jpg if has_jpg else None), ""


def make_anki(units, deck_name: str = "VocaSync", run_id: str | None = None,
              gen_dictation: bool = True) -> dict:
    """Build a .apkg from units. Returns {apkg, n_cards, run_id, failures:[...]}.

    Never raises on a single bad card (clip-error → fail + continue).

    gen_dictation (S18 #9): when False, Dictation cards are not generated (the sidebar toggle
    "Generate Dictation cards"). ADDITIVE — a keyword with a default, so existing callers and
    the tool contract are unchanged. Basic + Cloze are always produced.
    """
    args = {"n_units": len(units or []), "deck_name": deck_name, "run_id": run_id,
            "gen_dictation": gen_dictation}
    run_id = run_id or new_run_id()
    out = run_dir(run_id)
    media_dir = os.path.join(out, "media")
    os.makedirs(media_dir, exist_ok=True)

    # Per-source SUBDECKS (S12): each note lands in "VocaSync::<source>" (Anki renders the
    # "::" as a subdeck tree), so a learner sees words grouped by the film/episode they came
    # from instead of one merged pile. A node seen in several sources goes to its FIRST source's
    # subdeck but is TAGGED with every source (see below), so it is still findable under each.
    decks: dict[str, genanki.Deck] = {}

    def _deck_for(name: str) -> genanki.Deck:
        if name not in decks:
            decks[name] = genanki.Deck(_deck_id(name), name)
        return decks[name]

    media_files: list[str] = []
    failures: list[dict] = []
    n_cards = 0

    for u in units or []:
        nd = _node_of(u)
        try:
            term = (nd.get("term") or "").strip()
            if not term:
                failures.append({"unit": str(nd)[:60], "status": "fail",
                                 "system_note": "missing term"})
                continue
            key = nd.get("key") or term
            definition = nd.get("definition") or nd.get("category") or ""
            collocations = ", ".join(nd.get("collocations") or [])
            mnemonic = nd.get("mnemonic") or ""
            occ = (nd.get("occurrences") or [{}])
            sentence = occ[0].get("sentence", "") if occ else ""
            surface = (u.get("surface") if isinstance(u, dict) else "") or term

            # --- subdeck + tags from provenance (deterministic; source lives in occurrences) ---
            sources = _sources_of(nd)
            subdeck = f"{deck_name}::{_clean_source(sources[0])}" if sources else deck_name
            target_deck = _deck_for(subdeck)
            # Hierarchical tags: filter by film (source::Charade_25min) or by kind (type::idiom).
            # Anki's own "Added" date already answers "what's new"; these answer "from where".
            tags = [f"source::{_tag(_clean_source(s))}" for s in sources]
            if (nd.get("word_type") or "").strip():
                tags.append(f"type::{_tag(nd['word_type'])}")
            # S18 P0-3: enrich Extra with data the node ALREADY carries (no new Anki field).
            # Only render a line when its value exists (pattern of extra_bits) -> no empty labels.
            pos = (nd.get("pos") or "").strip()
            synonyms = ", ".join(_edge_targets(nd, ("synonym",)))
            antonyms = ", ".join(_edge_targets(nd, ("antonym",)))
            _rel_labels = {"is_a": "is a", "part_of": "part of"}
            relations = "; ".join(
                f"{_rel_labels[e['type']]} {(e.get('target') or '').split('#')[0].strip()}"
                for e in (nd.get("edges") or [])
                if isinstance(e, dict) and e.get("type") in _rel_labels
                and (e.get("target") or "").split("#")[0].strip()
            )
            node_tags = ", ".join(t for t in (nd.get("tags") or []) if t)
            extra_bits = [b for b in (
                f"POS: {pos}" if pos else "",
                f"Synonyms: {synonyms}" if synonyms else "",
                f"Antonyms: {antonyms}" if antonyms else "",
                f"Related: {relations}" if relations else "",
                f"Collocations: {collocations}" if collocations else "",
                f"Mnemonic: {mnemonic}" if mnemonic else "",
                f"Tags: {node_tags}" if node_tags else "",
            ) if b]

            # S18 Phase 2 (1a/1b): the card back ALWAYS shows Definition + Example (own fields)
            # + a Source@time line; the heavier info + any OTHER occurrences fold into a native
            # <details> (no JS). Extra holds: [source line] + <details>[bits][also-seen-in]</details>.
            def _esc(s: str) -> str:
                return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            occs = nd.get("occurrences") or []
            src_label = _clean_source(sources[0]) if sources else ""
            occ0_time = (occs[0].get("start") if occs else "") or ""
            src_line = ""
            if src_label:
                src_line = (f'<div class="vs-src">Source: {_esc(src_label)}'
                            + (f' @ {_esc(occ0_time)}' if occ0_time else "") + "</div>")
            occ_items = []
            for o in occs[1:]:
                s2 = (o.get("sentence") or "").strip()
                if not s2:
                    continue
                film = _clean_source(o.get("source") or "")
                occ_items.append(f'<div>· "{_esc(s2)}"'
                                 + (f" — {_esc(film)}" if film else "") + "</div>")
                if len(occ_items) >= 6:                 # cap so a very common word won't bloat
                    break
            occ_block = ('<div class="vs-occ"><div class="vs-occ-h">Also seen in</div>'
                         + "".join(occ_items) + "</div>") if occ_items else ""
            bits_html = "<br>".join(extra_bits)
            details = ""
            if bits_html or occ_block:
                details = ('<details class="vs-more"><summary>More ▾</summary>'
                           + (f"<div>{bits_html}</div>" if bits_html else "")
                           + occ_block + "</details>")
            extra = src_line + details
            # Cloze/Dictation have NO dedicated Definition field. Surface term+definition inside
            # their Extra so EVERY card type shows the meaning. History-safe: no model_id/guid/
            # field/template change — only the CONTENT passed to the existing Extra field.
            def_line = (f'<div class="vs-def">{_esc(term)}: {_esc(definition)}</div>'
                        if definition else "")

            # --- media (best-effort; clip-error -> text-only) ---
            media_html = ""
            audio_field = ""
            clip = _clip_of(u)
            stem = "anki_" + ascii_safe(key)
            if clip:
                mp3, jpg, note = _cut_media(clip, media_dir, stem)
                if note:
                    failures.append({"term": term, "status": "fail", "system_note": note})
                if mp3:
                    media_files.append(mp3)
                    audio_field = f"[sound:{os.path.basename(mp3)}]"
                    media_html += audio_field
                if jpg:
                    media_files.append(jpg)
                    media_html += f'<br><img src="{os.path.basename(jpg)}">'

            # --- Basic ---
            target_deck.add_note(genanki.Note(
                model=_BASIC_MODEL,
                fields=[term, definition, sentence, extra, media_html],
                guid=_guid(key, "basic"), tags=tags))
            n_cards += 1

            # --- Definition -> guess the word (S18 Phase 2, NEW note type; needs a definition
            # to pose the front). Additive: its own model_id/guid, legacy cards untouched. ---
            if definition:
                target_deck.add_note(genanki.Note(
                    model=_DEF_MODEL,
                    fields=[term, definition, sentence, extra, media_html],
                    guid=_guid(key, "definition"), tags=tags))
                n_cards += 1

            # --- Cloze (only if we can blank the surface in the sentence) ---
            cloze_text = _cloze_sentence(sentence, surface)
            if cloze_text:
                target_deck.add_note(genanki.Note(
                    model=_CLOZE_MODEL,
                    fields=[cloze_text, def_line + extra, media_html],
                    guid=_guid(key, "cloze"), tags=tags))
                n_cards += 1

            # --- Dictation (only if we have audio; S18 #9: skipped when toggle is off) ---
            if gen_dictation and audio_field and sentence:
                target_deck.add_note(genanki.Note(
                    model=_DICT_MODEL,
                    fields=[audio_field, sentence, def_line + extra],
                    guid=_guid(key, "dictation"), tags=tags))
                n_cards += 1

        except Exception as e:               # one card blew up -> fail + continue
            failures.append({"term": nd.get("term", "?"), "status": "fail",
                             "system_note": f"card build error: {e}"})
            continue

    apkg_path = os.path.join(out, f"{ascii_safe(deck_name)}.apkg")
    # Package ALL subdecks together (genanki accepts a list of decks). Empty run -> keep a
    # single empty base deck so the .apkg is still valid.
    pkg = genanki.Package(list(decks.values()) or [genanki.Deck(_deck_id(deck_name), deck_name)])
    if media_files:
        pkg.media_files = media_files
    pkg.write_to_file(apkg_path)

    result = {"apkg": apkg_path, "n_cards": n_cards, "run_id": run_id,
              "decks": sorted(decks.keys()), "failures": failures}
    log_tool_call("make_anki", args,
                  result={"apkg": apkg_path, "n_cards": n_cards,
                          "decks": len(decks), "failures": len(failures)})
    return result


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import load_graph
    g = load_graph()
    units = [n.model_dump() for n in g.nodes.values()]
    print(json.dumps(make_anki(units, deck_name="Demo"), ensure_ascii=False, indent=2))
