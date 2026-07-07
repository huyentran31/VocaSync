"""
app.py — VocaSync Streamlit UI (GĐ4, task 2 + 3).

A THIN shell over the existing agent/tools (HANDOVER §3.1: Streamlit stays thin).
It does NOT reimplement any tool — it dispatches through agent.loop (run_agent /
run_intent) and agent.registry (call_tool), exactly like the MCP server.

Design contract (docs/UI_DESIGN.md + the 4 review fixes):
  • App chrome is ENGLISH; the agent conversation can be any language.
  • Disk-as-truth (HANDOVER §3.2): data/personal_graph.json is the source of truth;
    stats are recomputed by load_graph() on every rerun; session_state holds only light
    pointers (run id, graph.html path, a small last-result summary).
  • Free text box  -> run_agent  (the LLM CHOOSES tools)              [fix 3]
  • Mine/Explain/Expand buttons -> run_intent (fixed fallback sequences) [fix 3]
  • Trajectory is rendered from the REAL tool calls, never scripted.     [fix 1]
  • SINGLE commit point = the "Commit Approved" button (in-app review df -> graph). [§3.2]
  • Native widgets only; whitespace via st.write("")/st.empty(), no unsafe HTML. [fix 4]
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pandas as pd
import streamlit as st

# --- make the project importable the same way the agent does ------------------ #
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "agent"), os.path.join(_ROOT, "tools"),
           os.path.join(_ROOT, "legacy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from loop import run_agent, run_intent, build_final_exports
from registry import call_tool
from _common import (GRAPH_PATH, TRAJECTORY_PATH, SystemError_, ascii_safe,
                     load_graph, save_graph)
from schema import Node, normalize_collocations
import review_io

UPLOAD_DIR = os.path.join(_ROOT, "data", "uploads")
SAMPLE_HINT = os.path.join("data", "samples", "climate.srt")


# --------------------------------------------------------------------------- #
# Disk-as-truth helpers
# --------------------------------------------------------------------------- #

def graph_stats() -> tuple[int, int, int]:
    """(#nodes, #edges, #sources) recomputed from disk every rerun — always fresh."""
    try:
        g = load_graph(GRAPH_PATH)
        nodes = len(g.nodes)
        edges = sum(len(n.edges) for n in g.nodes.values())
        sources = len({o.source for n in g.nodes.values() for o in n.occurrences if o.source})
        return nodes, edges, sources
    except Exception:
        return 0, 0, 0


def _traj_count() -> int:
    try:
        with open(TRAJECTORY_PATH, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _traj_since(n: int) -> list[dict]:
    """The REAL tool calls appended to the trajectory log since line `n` (fix 1)."""
    try:
        with open(TRAJECTORY_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()[n:]
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
                out.append({"tool": e.get("tool", "?"), "args": e.get("args", {})})
            except Exception:
                continue
        return out
    except Exception:
        return []


def _save_upload(uploaded) -> str:
    """Persist an uploaded media/transcript file with an ASCII-safe name; return path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    stem, ext = os.path.splitext(uploaded.name)
    safe = ascii_safe(stem) + (ext.lower() if ext else "")
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _drafts_preview(drafts: list[dict]) -> list[dict]:
    """A light, session-safe preview of enrich drafts (no heavy node payloads)."""
    rows = []
    for d in drafts or []:
        node = d.get("node", {})
        rows.append({
            "term": node.get("term", ""),
            "sense_id": node.get("sense_id") or "",
            "definition": (node.get("definition") or "")[:80],
            "confidence": round(float(d.get("confidence", 0.0) or 0.0), 2),
            "needs_review": bool(d.get("needs_review")),
        })
    return rows


# --------------------------------------------------------------------------- #
# Actions (each returns a light result dict stored in session_state)
# --------------------------------------------------------------------------- #

def _run_with_trajectory(fn):
    """Run an action, capturing the REAL trajectory delta from the log (fix 1)."""
    before = _traj_count()
    out = fn()
    out["_trajectory"] = _traj_since(before)
    return out


def do_ask(query: str, source: str = "", prior_scratch: list | None = None) -> dict:
    """Free query -> the LLM picks tools (run_agent). Real in-memory trajectory.

    `source` (optional): an attached local .srt/media path so the agent can answer
    questions ABOUT that movie/audio (ingest + extract_vocab + lookups), not only the graph.
    `prior_scratch` (optional): the scratch returned by a previous turn, so a reply CONTINUES
    the same conversation (multi-turn, task #3) instead of restarting.
    """
    out = run_agent(query, source=source, prior_scratch=prior_scratch)
    return {
        "kind": "ask",
        "summary": out.get("answer", ""),
        "trajectory": out.get("trajectory", []),
        "asked": out.get("asked", False),
        "drafts": _drafts_preview(out.get("drafts", [])),
        "scratch": out.get("scratch", []),
    }


def do_mine(source: str, focus: str, media: str = "") -> dict:
    res = _run_with_trajectory(lambda: run_intent("mine", source=source, focus=focus, media=media))
    drafts = res.get("drafts", [])
    # HITL: stash full nodes into the review queue (pending_drafts.json) with clip + srt_path
    # for the final exports, which are generated ONLY at Commit — GATE-EXPORT task #1. The
    # learner reviews/approves in-app (st.data_editor). No deck/vault/.ass yet.
    review_io.export_review(drafts, srt_path=res.get("srt_path", ""), source=os.path.basename(source))
    n_review = sum(1 for d in drafts if d.get("needs_review"))
    recalled = res.get("recalled_terms") or []
    known_note = (f" **{len(recalled)}** of these are already in your graph "
                  f"({', '.join(recalled[:5])}{'…' if len(recalled) > 5 else ''}) — "
                  "committing ADDs this film's occurrences to them (no duplicates)."
                  if recalled else "")
    return {
        "kind": "mine",
        "summary": (f"Mined **{len(drafts)}** candidates from "
                    f"`{os.path.basename(source)}` ({n_review} need review) → review → "
                    "Commit to generate the deck / Obsidian / .ass." + known_note),
        "trajectory": res.get("_trajectory", []),
        "drafts": _drafts_preview(drafts),
        "graph_html": res.get("graph", "") or "",
        "deck": "",                        # nothing final at Mine time (HITL gate)
        "run_id": res.get("run_id", ""),
    }


def do_explain(query: str) -> dict:
    res = _run_with_trajectory(lambda: run_intent("explain", query=query))
    return {"kind": "explain", "summary": res.get("answer", ""),
            "trajectory": res.get("_trajectory", []), "drafts": []}


def do_expand(term: str) -> dict:
    res = _run_with_trajectory(lambda: run_intent("expand", term=term))
    senses = res.get("senses", []) or []
    hit = res.get("recall", {}) or {}
    known = " (already in your graph)" if hit.get("found") else ""
    lines = [f"**{term}** — {len(senses)} WordNet sense(s){known}:"]
    for s in senses[:6]:
        lines.append(f"- `{s.get('sense_id','')}` — {s.get('definition','')}")
    return {"kind": "expand", "summary": "\n".join(lines),
            "trajectory": res.get("_trajectory", []), "drafts": []}


def _apply_ask(action: dict) -> None:
    """Fold an Ask turn into the multi-turn state. The scratch is always kept so the learner
    can keep the conversation going (whether the agent asked a question or gave a final answer);
    run_agent has no turn cap — the loop ends when the agent returns a final answer."""
    scratch = action.get("scratch") or []
    st.session_state["ask_scratch"] = scratch
    st.session_state["agent_pending"] = scratch if action.get("asked") else None
    st.session_state["chat"].append({"role": "assistant", "text": action.get("summary", "")})
    st.session_state["result"] = action


_VALID_WORD_TYPES = {"word", "phrasal_verb", "idiom", "collocation", "slang"}


def approve_undecided(df, confident_only: bool = False):
    """Bulk-approve review rows WITHOUT overriding human decisions (S14 T5).

    Only rows whose `status` is blank are touched; `confident_only=True` further
    restricts to rows whose `needs_review` flag cell is EMPTY (the cell now holds the
    machine's flag REASONS — "polysemy, definition(ai)" — so empty = no concern).
    Legacy "FALSE" is treated as empty for old sessions. Returns (new_df, n_marked).
    """
    df = df.copy()
    pick = df["status"].fillna("").astype(str).str.strip() == ""
    if confident_only:
        flags = df["needs_review"].fillna("").astype(str).str.strip()
        pick &= (flags == "") | (flags.str.upper() == "FALSE")
    df.loc[pick, "status"] = "approved"
    return df, int(pick.sum())


def validate_edits(node_dict: dict) -> list[str]:
    """DOUBLE-CHECK a to-be-committed node — DETERMINISTIC, NO AI (S12 T4).

    A second, independent gate after human review: catches a fabricated sense_id or a
    mistyped word_type BEFORE it reaches the graph. Returns a list of human-readable
    reasons; an EMPTY list means the node is valid. Never crashes (WordNet unavailable ->
    sense check is skipped, not failed — degrade, don't block).
    """
    reasons: list[str] = []

    wt = (node_dict.get("word_type") or "").strip().lower().replace(" ", "_")
    if wt not in _VALID_WORD_TYPES:
        reasons.append(f"word_type '{node_dict.get('word_type')}' not in {sorted(_VALID_WORD_TYPES)}")

    if not (node_dict.get("definition") or "").strip():
        reasons.append("definition is empty")

    # S16 T2: backstop the grounding gate at commit — a node with no occurrence carrying a
    # non-empty source `sentence` has no source in the transcript and must not be committed
    # (the "flyback" case: an invented/mis-recalled word that slipped through). Deterministic.
    if not any(str((o or {}).get("sentence", "")).strip()
               for o in (node_dict.get("occurrences") or [])):
        reasons.append("no grounded occurrence in source — cannot commit an unsourced word")

    tags = node_dict.get("tags")
    if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        reasons.append("tags do not parse to a list of strings")

    sense_id = (node_dict.get("sense_id") or "").strip()
    if sense_id:
        try:
            from nltk.corpus import wordnet as wn
            try:
                wn.synset(sense_id)                      # resolves -> real WordNet sense
            except Exception:
                term = (node_dict.get("term") or "").replace(" ", "_")
                names = {s.name() for s in wn.synsets(term)} if term else set()
                if sense_id not in names:
                    reasons.append(f"sense_id '{sense_id}' does not resolve in WordNet")
        except Exception:
            pass   # WordNet unavailable -> cannot verify; do not block (no-crash degrade)

    return reasons


def _rederive_on_sense_change(node_dict: dict, new_sense: str) -> dict:
    """S19 OPEN-8: refresh every field DERIVED from a synset after the learner picked a new
    WordNet sense, so the Anki card can't keep the OLD meaning's synonyms / "is a X" / gloss /
    mnemonic (the 'put down' bug: def said "put in a horizontal position" but Related still
    said "is a kill"). Deterministic (0 LLM): pulls the new sense's edges/pos/category/gloss
    straight from wordnet_lookup. A learner-typed definition/mnemonic/collocations
    (source_map=='user') always wins. Also rekeys (key = term#sense_id) so the guid + graph
    dedup follow the new sense. No-crash: any failure leaves the node with its edited sense_id
    but old derived fields (never worse than before)."""
    term = str(node_dict.get("term", "")).strip()
    smap = node_dict.setdefault("source_map", {})
    try:
        wl = call_tool("wordnet_lookup", {"term": term}) if term else {}
        match = next((s for s in (wl.get("senses") or [])
                      if s.get("sense_id") == new_sense), None)
    except Exception:
        match = None
    # conceptnet edges are sense-agnostic + human-vetted -> keep them; swap the WordNet ones.
    cn = [e for e in (node_dict.get("edges") or [])
          if isinstance(e, dict) and e.get("source") == "conceptnet"]
    if match:
        node_dict["edges"] = list(match.get("edges") or []) + cn
        if match.get("pos"):
            node_dict["pos"] = match["pos"]
        if match.get("category"):
            node_dict["category"] = match["category"]
        if smap.get("definition") != "user" and match.get("definition"):
            node_dict["definition"] = match["definition"]
            smap["definition"] = "wordnet"
    else:
        # sense not resolvable (OOV / AI-authored def): drop stale WordNet edges, keep the
        # learner's definition ("thà thiếu còn hơn sai").
        node_dict["edges"] = cn
    # AI fields written for the OLD meaning -> clear unless the learner just typed them.
    if smap.get("mnemonic") != "user":
        node_dict["mnemonic"] = None
    if smap.get("collocations") != "user":
        node_dict["collocations"] = []
    if term and new_sense:
        node_dict["key"] = f"{term.lower()}#{new_sense}"
    return node_dict


def _ai_resolve_ungrounded(ung_df, sents) -> None:
    """S19 (owner P2) — SUGGEST a real film line for every ungrounded row and FILL it into the
    `sentence` cell (no copy-paste, no double-typing). Python's lemma ranking is the reliable
    suggester (live-proven: "give a shot" -> "give the kid a shot", "wash out" -> the puke line);
    the AI is used ONLY when its pick is VERBATIM-in-transcript (Flash-Lite is inconsistent, so it
    never overrides a real line with a guess). Every filled line is FLAGGED for the learner to
    verify before Commit. Works even with no AI key (Python-only). A term with no lemma overlap
    anywhere stays ungrounded."""
    import json as _json
    from extract_vocab import _content_lemmas, _norm_match
    terms = [str(r.get("term", "")).strip() for _, r in ung_df.iterrows() if str(r.get("term", "")).strip()]
    if not terms:
        return
    _lines = [s for s in (sents or []) if s][:400]
    _norm_lines = {_norm_match(s): s for s in _lines}
    # AI picks (optional — only trusted when verbatim-real below)
    picks = {}
    if config.has_ai_key():
        try:
            from ai_client import call_ai
            system = "You match vocabulary terms to the EXACT transcript line they occur in. Reply ONLY with JSON."
            prompt = ("Transcript lines (verbatim):\n" + "\n".join(f"- {s}" for s in _lines)
                      + "\n\nFor each term below, return the SINGLE transcript line above that "
                        "actually contains it, copied EXACTLY/verbatim. If none, use null.\n"
                        "Terms: " + ", ".join(terms) + "\n"
                        'Reply JSON only: {"<term>": "<exact line or null>", ...}')
            raw = call_ai(prompt, system)
            _m = re.search(r"\{.*\}", str(raw), re.S)
            picks = _json.loads(_m.group(0)) if _m else {}
        except Exception:
            picks = {}
    rdf = st.session_state.get("review_df")
    by_ai, by_py, none = [], [], []
    for _, r in ung_df.iterrows():
        term = str(r.get("term", "")).strip()
        key = str(r.get("key", "")).strip()
        if not term:
            continue
        chosen, via = None, None
        # 1) AI pick — accept ONLY if it is a REAL verbatim transcript line (exact or substring).
        ai_line = picks.get(term) if isinstance(picks, dict) else None
        if ai_line:
            nz = _norm_match(str(ai_line))
            real = _norm_lines.get(nz) or next((v for k, v in _norm_lines.items() if nz and nz in k), None)
            if real:
                chosen, via = real, "ai"
        # 2) fall back to Python's top lemma-overlap candidate (reliable).
        if not chosen:
            tl = _content_lemmas(term)
            ranked = sorted(((len(tl & _content_lemmas(s)), s) for s in _lines), key=lambda x: -x[0])
            top = next((s for n, s in ranked if n), None)
            if top:
                chosen, via = top, "py"
        if chosen and rdf is not None and (rdf["key"] == key).any():
            rdf.loc[rdf["key"] == key, "sentence"] = chosen
            (by_ai if via == "ai" else by_py).append(term)
        else:
            none.append(term)
    if by_ai or by_py:
        st.session_state["review_df"] = rdf
        st.session_state["editor_nonce"] += 1
    flash = {"success": [], "warning": [], "error": []}
    if by_ai or by_py:
        flash["success"].append(
            f"🪄 Suggested a line for {len(by_ai) + len(by_py)} word(s) — filled into `sentence`, "
            f"VERIFY before Commit"
            + (f" · AI-picked: {', '.join(by_ai)}" if by_ai else "")
            + (f" · best-guess: {', '.join(by_py)}" if by_py else "") + ".")
    if none:
        flash["warning"].append(
            f"{len(none)} word(s) had no candidate line — edit `sentence` by hand: {', '.join(none)}.")
    st.session_state["commit_flash"] = flash
    st.rerun()


def commit_approved(review_df) -> dict:
    """The SINGLE commit point (HANDOVER §3.2): the in-app review df -> personal graph.

    NON-EXCEL: reads the learner's edited review rows from `review_df` (the st.data_editor
    DataFrame held in session_state), NOT review.xlsx. Each row with status='approved' is
    reconstructed from the full pending node, has its human edits applied, is double-checked
    by validate_edits (T4) and PARTITIONED (invalid rows held back with a reason), then valid
    rows upsert into the graph — the only graph write.
    """
    pending = review_io.load_pending()
    if review_df is None or len(review_df) == 0:
        return {"committed": [], "missing": [], "invalid": [],
                "error": "No review rows yet — run Mine first."}
    rows = review_df.to_dict("records") if hasattr(review_df, "to_dict") else list(review_df)
    graph = load_graph(GRAPH_PATH)
    committed, missing, invalid = [], [], []
    rekeyed_from = {}             # S19 OPEN-8: new_key -> old_key (sense changed) to drop orphans
    approved_drafts = []          # {node(with edits), surface, clip} -> final exports (task #1)
    for i, r in enumerate(rows):
        if str(r.get("status", "")).strip().lower() != "approved":
            continue
        key = str(r.get("key", "")).strip()
        pend = pending.get(key)
        if not pend:
            missing.append(key or str(r.get("term", "")))
            continue
        node_dict = dict(pend["node"])
        old_key_before_rekey = node_dict.get("key") or key
        # S19 OPEN-5: the term column is now editable — the learner may fix a distorted headword
        # (e.g. "be all over the place" -> "all over the place"). Changing the term changes the
        # key (term#sense_id, which drives the Anki guid + graph dedup), so it rekeys + drops the
        # old node, exactly like a sense change (shared cleanup via rekeyed_from below).
        new_term = str(r.get("term", "")).strip()
        term_changed = bool(new_term and new_term != (node_dict.get("term") or ""))
        if term_changed:
            node_dict["term"] = new_term
            node_dict.setdefault("source_map", {})["term"] = "user"
        # apply human edits (definition / sense_id) -> mark provenance 'user'
        new_def = str(r.get("definition", "")).strip()
        if new_def and new_def != (node_dict.get("definition") or ""):
            node_dict["definition"] = new_def
            node_dict.setdefault("source_map", {})["definition"] = "user"
        new_sense = str(r.get("sense_id", "")).strip()
        sense_changed = bool(new_sense and new_sense != (node_dict.get("sense_id") or ""))
        if sense_changed:
            node_dict["sense_id"] = new_sense
            node_dict.setdefault("source_map", {})["sense_id"] = "user"
        # apply edited word_type (idiom/phrasal_verb/...) -> 'user' if changed
        new_wt = str(r.get("word_type", "")).strip().lower().replace(" ", "_")
        if new_wt and new_wt != (node_dict.get("word_type") or ""):
            node_dict["word_type"] = new_wt
            node_dict.setdefault("source_map", {})["word_type"] = "user"
        # apply edited sentence (S17): the learner may supply the REAL source line for an
        # agent-staged word that arrived ungrounded — human input is the grounding authority
        # here (HITL). Applied to the first occurrence; lemmas recomputed so recall keeps
        # working. AI never touches this field — only the learner's edit lands.
        new_sent = str(r.get("sentence", "")).strip()
        occs = node_dict.get("occurrences") or []
        old_sent = (occs[0].get("sentence", "") if occs and isinstance(occs[0], dict) else "")
        if new_sent and new_sent != old_sent:
            if occs and isinstance(occs[0], dict):
                occs[0]["sentence"] = new_sent
                # S19 (B3): drop the OLD cue's timing — the learner just re-grounded this word to
                # a DIFFERENT line, so the previous start/end belong to the wrong cue. Commit's
                # media step re-locates from the edited sentence only when timing is absent
                # (app: "only re-locate when the occurrence carries none"); without this the card
                # would keep audio cut from the old (wrong) line even though the sentence is fixed.
                for _k in ("start", "end", "start_sec", "end_sec"):
                    occs[0].pop(_k, None)
                try:
                    from extract_vocab import _content_lemmas
                    occs[0]["lemmas"] = sorted(_content_lemmas(new_sent))
                except Exception:
                    pass
            node_dict.setdefault("source_map", {})["sentence"] = "user"
        # apply edited tags (topic/exam labels, ";"- or ","-separated) -> 'user' if changed
        raw_tags = str(r.get("tags", "") or "").replace(",", ";")
        edited_tags = [t.strip().lower() for t in raw_tags.split(";") if t.strip()]
        if edited_tags != [t.lower() for t in (node_dict.get("tags") or [])]:
            node_dict["tags"] = edited_tags
            node_dict.setdefault("source_map", {})["tags"] = "user"
        # apply edited mnemonic (S19 OPEN-8): now an editable column so the learner can retype
        # it after changing the sense (the AI one was written for the OLD meaning).
        new_mnem = str(r.get("mnemonic", "") or "").strip()
        if new_mnem and new_mnem != (node_dict.get("mnemonic") or ""):
            node_dict["mnemonic"] = new_mnem
            node_dict.setdefault("source_map", {})["mnemonic"] = "user"
        # apply edited collocations (";"- or ","-separated) -> 'user' if changed
        # S19 BUG-1: normalize_collocations handles str and repairs any char-broken list.
        edited_coll = normalize_collocations(r.get("collocations", ""))
        if edited_coll and edited_coll != (node_dict.get("collocations") or []):
            node_dict["collocations"] = edited_coll
            node_dict.setdefault("source_map", {})["collocations"] = "user"
        # S19 OPEN-8 (owner "put down" bug): a CHANGED sense must refresh every field DERIVED
        # from the old synset, else the card keeps the old sense's synonyms / "is a X" relation /
        # mnemonic (factually wrong). Re-derive deterministically (0 LLM) from the newly-chosen
        # WordNet sense; a learner-typed definition/mnemonic/collocations (source_map=='user')
        # always wins. rekey so Anki guid + graph dedup stay consistent with the new sense.
        if sense_changed:
            node_dict = _rederive_on_sense_change(node_dict, new_sense)
        # S19 OPEN-5/OPEN-8: recompute the key from the FINAL term + sense whenever either
        # changed, then register the old->new mapping so the stale node is dropped after commit.
        if term_changed or sense_changed:
            _t = str(node_dict.get("term", "")).strip().lower()
            _s = str(node_dict.get("sense_id", "")).strip()
            node_dict["key"] = f"{_t}#{_s}" if _s else f"{_t}#nowordnet"
            if node_dict["key"] != old_key_before_rekey:
                rekeyed_from[node_dict["key"]] = old_key_before_rekey
        # DOUBLE-CHECK (S12 T4): deterministic validation AFTER edits, BEFORE the write.
        # Partition — an invalid row is held back (with reasons) for the human to fix and
        # re-Commit; valid rows in the same batch still commit (idempotent by key).
        problems = validate_edits(node_dict)
        if problems:
            # `num` = the '#' column shown in the table (S16 T3/T4) so the held-back message
            # points at exactly the row the learner sees; fall back to 1-based position.
            num = r.get("#")
            try:
                num = int(num)
            except (TypeError, ValueError):
                num = i + 1
            invalid.append({"num": num, "key": key,
                            "term": node_dict.get("term", ""), "reason": "; ".join(problems)})
            continue
        try:
            graph.upsert(Node(**node_dict))
            committed.append(key)
            approved_drafts.append({"node": node_dict, "surface": pend.get("surface"),
                                    "clip": pend.get("clip")})
        except Exception as e:
            missing.append(f"{key} ({e})")
    exports = {}
    media_warn = ""
    if committed:
        # S17: agent-staged words arrive with NO clip (only Mine attaches one), so their
        # cards used to come out text-only even with a video attached. If the session has a
        # transcript + video, locate each missing clip's timestamps DETERMINISTICALLY from
        # the occurrence sentence (same _locate_timestamp the Mine path uses — AI never
        # touches timestamps). A sentence not found in the transcript simply gets no media.
        # S18 #2/#3: recover the srt from the ① transcript cache when the sidebar last_srt is
        # empty (mine-via-chat), run the lookup PER-WORD, and surface a clear warning instead
        # of degrading to text-only silently.
        # Drafts that arrive WITHOUT a clip are the agent-staged ones — only these are the
        # subject of media recovery + the text-only warning (pure Mine drafts already have a clip).
        agent_staged = [d for d in approved_drafts if not d.get("clip")]
        try:
            _video = str(st.session_state.get("last_video") or "").strip().strip('"')
            _src_label = os.path.basename(pending.get("_meta", {}).get("source", "")
                                          or pending.get("_meta", {}).get("srt_path", ""))
            _srt = (pending.get("_meta", {}).get("srt_path", "")
                    or str(st.session_state.get("last_srt") or "")).strip().strip('"')
            if not (_srt and os.path.exists(_srt)):
                from _common import load_cached_srt      # S18 #2: recover from ① cache
                _srt = load_cached_srt(_src_label) or _srt
            # askfix (owner V1): mine-via-chat never sets a sidebar video, so cards were ALWAYS
            # text-only even when the companion video sits right next to the srt with the same
            # name (Charlie S1 E1.srt + Charlie S1 E1.mp4). Fall back to that sibling file.
            if not (_video and os.path.exists(_video)) and _srt and os.path.exists(_srt):
                _base = os.path.splitext(_srt)[0]
                for _ext in (".mp4", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".wav"):
                    if os.path.exists(_base + _ext):
                        _video = _base + _ext
                        break
            if _video and os.path.exists(_video) and _srt and os.path.exists(_srt) and agent_staged:
                from loop import _locate_timestamp, _expand_to_neighbor, _sec_of
                _ing = call_tool("ingest_transcript", {"source": _srt})
                _segs = _ing.get("segments", []) if isinstance(_ing, dict) else []
                for d in agent_staged:                    # PER-WORD: each word matched on its own
                    if not _segs:
                        continue
                    n = d.get("node") or {}
                    occ = (n.get("occurrences") or [{}])[0]
                    if not isinstance(occ, dict):
                        occ = {}
                    # askfix (owner V3/V4): USE the occurrence's OWN grounded timestamp — stage
                    # already snapped the sentence to its real cue and recorded start/end. The old
                    # path RE-LOCATED here via _locate_timestamp, which (a) redid the match on a
                    # possibly-different srt and (b) mis-hit a short cue ("for crying out loud" ->
                    # 06:43 "Out!"), so the audio came from the wrong line. Trust the grounded time;
                    # only re-locate when the occurrence carries none (ungrounded / legacy row).
                    s0 = _sec_of(occ.get("start_sec") if occ.get("start_sec") not in ("", None)
                                 else occ.get("start"))
                    e0 = _sec_of(occ.get("end_sec") if occ.get("end_sec") not in ("", None)
                                 else occ.get("end"))
                    if s0 is None or e0 is None or e0 <= s0:
                        sent = occ.get("sentence", "")
                        s0, e0 = _locate_timestamp(_segs, d.get("surface") or n.get("term", ""), sent)
                    # S18 P0-1: seconds may be 0.0 (falsy but valid) -> test against "" / None.
                    if s0 != "" and s0 is not None and e0 != "" and e0 is not None:
                        t0, t1 = _expand_to_neighbor(_segs, s0, e0)   # ±1 subtitle context
                        d["clip"] = {"video": _video, "start": t0, "end": t1}
        except Exception:
            pass                                            # media is best-effort — never block commit
        # S18 #3 — no silent degrade: if agent-staged words still ended up text-only, say WHY.
        if agent_staged:
            still = [(d.get("node") or {}).get("term", "") for d in agent_staged if not d.get("clip")]
            still = [t for t in still if t]
            if still:
                _video_ok = bool(_video and os.path.exists(_video))
                _srt_ok = bool(_srt and os.path.exists(_srt))
                if not _video_ok:
                    media_warn = (f"{len(still)} card(s) are TEXT-ONLY (no clip): no companion video. "
                                  "Companion video must be a LOCAL .mp4 file (not a link); attach it "
                                  "in the sidebar before Commit to get audio + screenshot.")
                elif not _srt_ok:
                    media_warn = (f"{len(still)} card(s) are TEXT-ONLY (no clip): no transcript found "
                                  "to locate clip timings. Ingest the media (or attach its .srt) first.")
                else:
                    media_warn = (f"{len(still)} card(s) are TEXT-ONLY (no clip): the sentence didn't "
                                  f"match a transcript line: {', '.join(still[:6])}.")
        # S19 OPEN-8: when a learner re-sensed a word, the node committed under its NEW key
        # (term#new_sense). Drop the stale OLD-key node so no duplicate card / orphan sense
        # lingers in the graph (owner: "rekey + dọn node cũ"). Only removes a key we rekeyed
        # away from, and never the new key itself.
        for _new_key, _old_key in rekeyed_from.items():
            if _old_key and _old_key != _new_key and _old_key in graph.nodes:
                graph.nodes.pop(_old_key, None)
        save_graph(graph, GRAPH_PATH)                       # <-- the only graph write
        # `recent` = the keys committed THIS session -> gold ring on the cumulative graph so
        # the learner sees what the latest Mine added to the growing graph (and where).
        html = call_tool("build_render_graph",
                         {"units": graph, "run_id": "committed", "recent": committed})
        st.session_state["graph_html"] = html
        # GATE-EXPORT (task #1): the final deliverables are generated HERE, from the approved
        # subset only — never at Mine. The .ass needs the run's srt_path (stashed in _meta).
        # S17: per-session output folder named <source>_<MMDD_HHMM> so a new commit never
        # overwrites the previous session's deck/infolog/vault. Inside Anki the deck/subdeck
        # names stay IDENTICAL across sessions, so importing each .apkg merges by stable
        # GUID into the same per-film subdeck (review "all of film X" just works).
        import datetime
        import re as _re
        _src = os.path.basename(pending.get("_meta", {}).get("source", "") or
                                pending.get("_meta", {}).get("srt_path", "") or "session")
        _slug = _re.sub(r"[^A-Za-z0-9]+", "", os.path.splitext(_src)[0])[:24] or "session"
        _run = f"committed_{_slug}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        # S18 P1-3: the highlighted .ass needs an srt with timings. The agent-staged path
        # (mine-via-chat) leaves _meta.srt_path empty — recover it from the ① transcript cache
        # (same recovery the media/clip block above uses) so the .ass is produced for that path
        # too, not just pure Mine. Additive: no srt found -> build_final_exports yields "".
        _ass_srt = pending.get("_meta", {}).get("srt_path", "")
        if not (_ass_srt and os.path.exists(_ass_srt)):
            try:
                from _common import load_cached_srt
                _lbl = os.path.basename(pending.get("_meta", {}).get("source", "")
                                        or pending.get("_meta", {}).get("srt_path", ""))
                _ass_srt = load_cached_srt(_lbl) or _ass_srt
            except Exception:
                pass
        exports = build_final_exports(
            approved_drafts, run_id=_run, srt_path=_ass_srt,
            gen_dictation=bool(st.session_state.get("gen_dictation", True)))   # S18 #9 toggle
    return {"committed": committed, "missing": missing, "invalid": invalid,
            "exports": exports, "media_warn": media_warn, "error": ""}


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="VocaSync", layout="wide")
HAS_KEY = config.has_ai_key()

# ---- Static theme (dark, Supabase × Linear). CSS ONLY — no dynamic content interpolated. ---- #
st.markdown(
    """
    <style>
      :root{ --vg-base:#090D16; --vg-surface:#111827; --vg-accent:#10B981;
             --vg-text:#F3F4F6; --vg-muted:#94A3B8; --vg-border:rgba(255,255,255,0.05); }
      .stApp{ background:var(--vg-base); }
      #masthead{ font-weight:800; letter-spacing:.24em; color:var(--vg-accent);
                 font-size:1.35rem; text-transform:uppercase; margin:0 0 .05rem; }
      #masthead + .vg-sub{ color:var(--vg-muted); letter-spacing:.04em; font-size:.78rem;
                 margin-bottom:.45rem; padding-bottom:.45rem; border-bottom:1px solid var(--vg-border); }
      /* Text input / textarea — surface card with thin border */
      textarea, .stTextInput input, [data-baseweb="input"]{
          background:var(--vg-surface)!important; border:1px solid var(--vg-border)!important;
          border-radius:12px!important; color:var(--vg-text)!important; }
      textarea:focus, .stTextInput input:focus{ border-color:var(--vg-accent)!important;
          box-shadow:0 0 0 2px rgba(16,185,129,.18)!important; }
      /* Buttons — Linear-style: pill, thin border, hover lift */
      .stButton>button{ border-radius:10px; border:1px solid var(--vg-border);
          background:var(--vg-surface); color:var(--vg-text); font-weight:600;
          transition:transform .05s ease, border-color .15s ease, background .15s ease; }
      .stButton>button:hover{ border-color:rgba(16,185,129,.5); transform:translateY(-1px); }
      /* Primary (Ask) = filled emerald */
      .stButton>button[kind="primary"]{ background:var(--vg-accent); color:#04231a;
          border:1px solid rgba(16,185,129,.5); font-weight:700; font-size:1.0rem; padding:.42rem 0; }
      .stButton>button[kind="primary"]:hover{ filter:brightness(1.08); }
      /* Export items in the sidebar expander — ALL FOUR share one uniform bar: same bg,
         border, radius, weight, size, left-aligned. Covers download buttons (Anki/Infolog),
         the plain button (Obsidian) and the anchor link (Graph). Scoped to the expander so
         other sidebar buttons (Commit/Approve/Reload) keep their own styling. */
      section[data-testid="stSidebar"] [data-testid="stExpander"] .stDownloadButton>button,
      section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton>button{
          border-radius:10px; border:1px solid var(--vg-border); background:var(--vg-base);
          color:var(--vg-text); font-weight:600; font-size:.9rem;
          justify-content:flex-start; text-align:left; padding-left:.7rem; }
      a.vg-open{ display:block; width:100%; box-sizing:border-box; padding:.42rem .7rem;
          margin:.15rem 0; border:1px solid var(--vg-border); border-radius:10px;
          background:var(--vg-base); color:var(--vg-text); text-decoration:none;
          font-weight:600; font-size:.9rem; }
      a.vg-open:hover{ border-color:rgba(16,185,129,.5); }
      /* Obsidian path shown compactly under its bar */
      section[data-testid="stSidebar"] .stCode{ margin:.1rem 0 .3rem; }
      /* Trim Streamlit's default top padding for a tighter header */
      .block-container{ padding-top:1.0rem; }
      /* Compact vertical rhythm so the Review table sits near the fold */
      [data-testid="stElementContainer"]:has(> hr){ margin:.5rem 0; }
      hr{ margin:.5rem 0 !important; }
      /* Flat Linear-style tabs */
      button[data-baseweb="tab"]{ background:transparent; border-radius:8px 8px 0 0; }
      div[data-baseweb="tab-list"]{ border-bottom:1px solid var(--vg-border); gap:2px; }
      /* Sticky tab bar (S16-followup): a long Ask conversation scrolls the page, but the
         Conversation | Review & Approve switcher stays pinned at the top — no scrolling
         back up to reach Review. Chat itself stays inline/full-height (see tab_conv_top
         comment: a fixed-height chat box clipped long answers, so we pin the TABS, not
         the chat). Solid bg so scrolled content doesn't bleed through. */
      div[data-testid="stTabs"] div[data-baseweb="tab-list"]{
          position:sticky; top:0; z-index:99;
          background:var(--vg-base, #090D16); }
      /* Fallback jump link (S17): sticky can be defeated by an overflow ancestor in some
         Streamlit builds — a fixed bottom-right "back to tabs" chip always works. */
      a.vg-jump{ position:fixed; right:18px; z-index:999;
          padding:.4rem .7rem; border-radius:999px; font-weight:700; font-size:.78rem;
          background:var(--vg-surface); color:var(--vg-accent)!important;
          border:1px solid rgba(16,185,129,.45); text-decoration:none;
          box-shadow:0 2px 8px rgba(0,0,0,.45); }
      a.vg-jump:hover{ background:rgba(16,185,129,.12); }
      a.vg-up{ bottom:58px; }     /* ⬆ to tabs / review */
      a.vg-down{ bottom:18px; }   /* ⬇ to the latest message */
      /* Thin borders + soft shadow on cards/expanders/forms */
      [data-testid="stExpander"], [data-testid="stForm"]{
          border:1px solid var(--vg-border); border-radius:12px;
          box-shadow:0 1px 3px rgba(0,0,0,.35); }
      /* Chat bubbles — theme via the native testid, NOT a custom class */
      [data-testid="stChatMessage"]{ background:var(--vg-surface);
          border:1px solid var(--vg-border); border-radius:12px; margin-bottom:.4rem;
          border-left:3px solid var(--vg-border); }
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
          border-left:3px solid var(--vg-accent); background:rgba(16,185,129,0.02); }
      [data-testid="stChatMessageContent"] p{ line-height:1.65; }
      /* Trajectory args read as monospace */
      [data-testid="stCaptionContainer"]{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
      /* S18 P1-1: export bars (Anki/Graph/Obsidian/Infolog) all share the NEUTRAL bordered
         look defined above (a.vg-open + .stDownloadButton>button) — no per-item accent/color
         so the 2×2 grid reads uniform. div.vg-open (Obsidian, non-link) matches the anchors. */
      div.vg-open{ display:block; width:100%; box-sizing:border-box; padding:.42rem .7rem;
          margin:.15rem 0; border:1px solid var(--vg-border); border-radius:10px;
          background:var(--vg-base); color:var(--vg-text); font-weight:600; font-size:.9rem;
          cursor:default; }
      section[data-testid="stSidebar"]{ background:var(--vg-surface);
          border-right:1px solid var(--vg-border); }
      /* Sidebar section headers — one consistent hierarchy (S14 T15) */
      .vg-side-h{ display:block; text-transform:uppercase; letter-spacing:.12em;
          font-size:.72rem; font-weight:700; color:var(--vg-muted); margin:.2rem 0 .1rem; }
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
          font-size:.62rem; opacity:.7; }
      /* Transparent header (S14 T17): keep the sidebar-toggle button, lose the solid bar */
      [data-testid="stHeader"]{ background:transparent!important; min-height:0!important; }
      /* Thin scrollbars app-wide (S14 T17) */
      ::-webkit-scrollbar{ width:4px; height:4px; }
      ::-webkit-scrollbar-thumb{ background:rgba(255,255,255,.14); border-radius:2px; }
      ::-webkit-scrollbar-thumb:hover{ background:rgba(16,185,129,.5); }
      ::-webkit-scrollbar-track{ background:transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.session_state.setdefault("result", None)
st.session_state.setdefault("graph_html", "")
st.session_state.setdefault("commit_exports", {})
st.session_state.setdefault("chat", [])            # multi-turn Ask history [{role,text}]
st.session_state.setdefault("agent_pending", None)  # scratch when the agent is awaiting a reply
st.session_state.setdefault("ask_scratch", None)    # scratch kept to continue the conversation
st.session_state.setdefault("review_df", None)      # edited review rows (st.data_editor truth)
st.session_state.setdefault("editor_nonce", 0)      # bump to reseed the data_editor widget
st.session_state.setdefault("held_back_keys", [])   # keys held back at the last Commit (S16 T3)

# ---- Sidebar: STATS / ATTACH MEDIA / WORKFLOW / EXPORT ---- #
nodes, edges, n_sources = graph_stats()
# One-line stats caption at the very top of the sidebar (S14 T15 — no framed stats block).
st.sidebar.caption(f"{nodes} nodes · {edges} edges · {n_sources} sources")

# Attach Media lives in a collapsed expander; its label shows the picked file name
# (computed from the previous run's widget state — S14 T15).
_prev_attach = st.session_state.get("attach_on", False)
_prev_sel = st.session_state.get("attach_src_sel", "")
_prev_other = (st.session_state.get("attach_src_other", "") or "").strip().strip('"')
_OTHER = "Other… (enter a local path)"
_picked = _prev_other if _prev_sel == _OTHER else _prev_sel
_attach_label = (f"📎 {os.path.basename(_picked)}"
                 if (_prev_attach and _picked and _picked != _OTHER) else "📎 ATTACH MEDIA")
source_path = ""
media_path = ""
with st.sidebar.expander(_attach_label, expanded=False):
    attach = st.checkbox("Attach a source (for Ask or Mine)", key="attach_on")
    if attach:
        # Local-file flow (the app runs on your machine, so it reads files by PATH — no
        # upload/copy). Discover ready .srt sources in the project folders; "Other…" lets
        # you point at any local path (an .srt, or a video/audio file for Whisper).
        _TRANSCRIPT_EXT = (".srt", ".vtt", ".txt", ".md")
        found = []
        for d in (os.path.join(_ROOT, "video_script_sample"), os.path.join(_ROOT, "data", "samples")):
            if os.path.isdir(d):
                found += [os.path.join(d, f) for f in sorted(os.listdir(d))
                          if f.lower().endswith(_TRANSCRIPT_EXT)]
        choice = st.selectbox(
            "Source file", found + [_OTHER], key="attach_src_sel",
            help="Pick a discovered transcript (.srt · .vtt · .txt · .md), or 'Other…' to type "
                 "any local path (a transcript, or a video/audio file for Whisper).",
            format_func=lambda p: os.path.basename(p) if p != _OTHER else p)
        source_path = choice if choice != _OTHER else st.text_input(
            "Local path", key="attach_src_other",
            placeholder="C:\\path\\to\\transcript.srt (.srt · .vtt · .txt · .md)")
        media_path = st.text_input(
            "Companion video (optional)", placeholder="C:\\path\\to\\movie.mp4",
            help="A .mp4/.mp3 to cut short audio + a screenshot onto each Anki card.")
        # S17: remember the attached video AS SOON as it is valid — not only on Mine.
        # Commit uses it to cut media for agent-staged words and for the .ass copy button
        # (previously an Ask-path session lost the video and exported text-only cards).
        _mp = (media_path or "").strip().strip('"')
        if _mp and os.path.exists(_mp):
            st.session_state["last_video"] = _mp
        _sp = (source_path or "").strip().strip('"')
        if _sp and _sp != _OTHER and os.path.exists(_sp):
            st.session_state["last_srt"] = _sp    # transcript for commit-time clip locating
st.sidebar.divider()

# Flash messages stashed BEFORE st.rerun() (a success/warning drawn right before rerun
# is swallowed by it). Rendered once here at the top of the sidebar, then popped.
_flash = st.session_state.pop("commit_flash", None)
if isinstance(_flash, dict):
    for msg in _flash.get("success", []) or []:
        st.sidebar.success(msg)
    for msg in _flash.get("warning", []) or []:
        st.sidebar.warning(msg)
    for msg in _flash.get("error", []) or []:
        st.sidebar.error(msg)

st.sidebar.markdown('<span class="vg-side-h">Workflow actions</span>', unsafe_allow_html=True)
if st.sidebar.button("Approve All", width='stretch',
                     help="Mark all *undecided* rows approved (rows you already set are kept)."):
    # Bulk-mark every UNDECIDED candidate approved IN the review df — a row the human
    # already set (rejected / needs_revision / approved) is never overridden. Still goes
    # through the single Commit gate below.
    df = st.session_state.get("review_df")
    flash = {"success": [], "warning": [], "error": []}
    if df is None or len(df) == 0:
        flash["warning"].append("No rows to approve — run Mine first.")
    else:
        df, n = approve_undecided(df)
        st.session_state["review_df"] = df
        st.session_state["editor_nonce"] += 1     # reseed the editor so it shows 'approved'
        flash["success"].append(
            f"Marked {n} undecided row(s) approved — now click Commit Approved.")
    st.session_state["commit_flash"] = flash
    st.rerun()
if st.sidebar.button("Approve confident", width='stretch',
                     help="Approve only undecided rows with needs_review = FALSE. "
                          "Does NOT commit — click Commit Approved after."):
    df = st.session_state.get("review_df")
    flash = {"success": [], "warning": [], "error": []}
    if df is None or len(df) == 0:
        flash["warning"].append("No rows to approve — run Mine first.")
    else:
        df, n = approve_undecided(df, confident_only=True)
        st.session_state["review_df"] = df
        st.session_state["editor_nonce"] += 1
        flash["success"].append(
            f"Marked {n} confident undecided row(s) approved — now click Commit Approved.")
    st.session_state["commit_flash"] = flash
    st.rerun()
# S18 #9: toggle Dictation card generation (default ON). Read in commit_approved ->
# build_final_exports -> make_anki(gen_dictation=...). Rendered before Commit so its state
# is set on the same run the learner clicks Commit.
st.sidebar.checkbox("Generate Dictation cards", value=True, key="gen_dictation",
                    help="Also make 'type what you hear' audio cards (needs a companion video).")
if st.sidebar.button("Commit Approved", type="primary", width='stretch'):
    res = commit_approved(st.session_state.get("review_df"))
    flash = {"success": [], "warning": [], "error": []}
    if res["error"]:
        flash["error"].append(res["error"])
    else:
        st.session_state["commit_exports"] = res.get("exports", {}) or {}
        if res.get("invalid"):
            # Report each held-back row by its '#' + term (S16 T3), e.g.
            #   "#1 'make a walk' — definition is empty · #9 'be in' — no grounded occurrence".
            # The '#' matches the table's `#` column exactly. Keep the invalid keys so the
            # table can surface just those rows (held-back toggle).
            bad = " · ".join(
                f"#{e.get('num', '?')} '{e['term']}' — "
                f"{e.get('reason') or ('status=' + str(e.get('status', '')))}"
                for e in res["invalid"][:8])
            flash["warning"].append(
                f"{len(res['invalid'])} row(s) held back (fix and re-Commit): {bad}")
            st.session_state["held_back_keys"] = [
                e.get("key", "") for e in res["invalid"] if e.get("key")]
        else:
            st.session_state["held_back_keys"] = []
        flash["success"].append(f"Committed {len(res['committed'])} approved item(s).")
        if res.get("media_warn"):
            flash["warning"].append(res["media_warn"])     # S18 #3: no silent text-only degrade
        if res["missing"]:
            flash["warning"].append(f"Skipped {len(res['missing'])}: {', '.join(res['missing'][:5])}")
    st.session_state["commit_flash"] = flash
    st.rerun()
st.sidebar.caption("Set each row's status in the Review & Approve tab, then click Commit.")
st.sidebar.divider()
st.sidebar.markdown('<span class="vg-side-h">Export data</span>', unsafe_allow_html=True)
# GATE-EXPORT (task #1): downloads appear ONLY after Commit, from the approved subset —
# never at Mine. Read the artefacts build_final_exports() produced at commit time.
exports = st.session_state.get("commit_exports") or {}


def _dl(container, label: str, path: str, mime: str) -> None:
    with open(path, "rb") as f:
        container.download_button(label, f.read(), file_name=os.path.basename(path),
                                  mime=mime, width='stretch')


def _open_folder(path: str) -> None:
    """Open a folder in the OS file manager. Works because this app runs LOCALLY (Streamlit
    server = the learner's own machine). Windows: os.startfile; macOS: open; Linux: xdg-open."""
    import subprocess
    import sys as _sys
    if os.name == "nt":
        os.startfile(path)                        # noqa: S606 (local desktop app)
    elif _sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _stage_graph_static(src_html: str) -> bool:
    """Copy the committed graph.html into ./static/ so Streamlit's static server can serve it
    (a file:// link is blocked by the browser). Returns True if staged."""
    if not (src_html and os.path.exists(src_html)):
        return False
    os.makedirs(os.path.join(_ROOT, "static"), exist_ok=True)
    shutil.copyfile(src_html, os.path.join(_ROOT, "static", "graph.html"))
    return True


deck_path = (exports.get("deck") or {}).get("apkg", "")
ass_path = exports.get("highlighted_ass", "")
vault_path = exports.get("obsidian_vault", "")
infolog_path = exports.get("infolog", "")
graph_committed = st.session_state.get("graph_html", "")
has_deck = bool(deck_path and os.path.exists(deck_path))
has_vault = bool(vault_path and os.path.isdir(vault_path))
has_graph = _stage_graph_static(graph_committed)
has_infolog = bool(infolog_path and os.path.exists(infolog_path))
any_export = has_deck or has_vault or has_graph or has_infolog

if not any_export:
    st.sidebar.caption("Approve rows and click Commit to generate the deck / graph / "
                       "Obsidian / infolog / highlighted script — they will appear here as links.")
else:
    # S17: the whole export block folds into ONE expander so it stops eating the sidebar —
    # open by default right after a Commit, collapsible any time. Inside: the 2×2 grid
    # (Anki · Graph · Obsidian · Infolog) + the .ass action.
    with st.sidebar.expander("📤 Export data — ready", expanded=True):
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        # S18 P1-1: the 4 export controls share ONE look (neutral bordered bar, no emoji).
        if has_deck:
            _dl(r1c1, "Anki", deck_path, "application/octet-stream")
        if has_graph:
            r1c2.markdown('<a href="app/static/graph.html" target="_blank" rel="noopener" '
                          'class="vg-open">Graph</a>', unsafe_allow_html=True)
        if has_vault:
            # S18: real button — opens the vault FOLDER in the OS file manager so the learner
            # opens it in Obsidian directly (app runs locally). Path stays in the tooltip only.
            if r2c1.button("Obsidian", width='stretch',
                           help=f"Open the Obsidian vault folder:\n{vault_path}"):
                try:
                    _open_folder(vault_path)
                    st.toast(f"Opened: {vault_path}")
                except Exception as e:
                    st.warning(f"Could not open the folder ({e}). Path: {vault_path}")
        if has_infolog:
            _dl(r2c2, "Infolog", infolog_path, "text/plain")

        # Copy the highlighted subtitle next to the attached video (same basename) so the media
        # player auto-loads it. This is the ONE write outside the project dir — allowed because
        # it is an explicit user action (a button click), per the S16 brief invariants.
        video = (st.session_state.get("last_video") or "").strip().strip('"')
        if ass_path and os.path.exists(ass_path) and video and os.path.exists(video):
            vdir = os.path.dirname(os.path.abspath(video))
            dest = os.path.join(vdir, os.path.splitext(os.path.basename(video))[0] + ".ass")
            if st.button(
                    ".ass → video folder", width='stretch',
                    help=("Copies the highlighted subtitle next to your video file so the "
                          "player auto-loads it. File: " + os.path.basename(dest))):
                try:
                    shutil.copyfile(ass_path, dest)
                    st.success(f"Copied subtitle → {dest}")
                except Exception as e:
                    st.warning(f"Could not copy the .ass next to the video: {e}")
        elif ass_path and os.path.exists(ass_path):
            # .ass exists but no video attached -> still let the learner download it directly.
            _dl(st, "Highlighted script (.ass)", ass_path, "text/plain")

# ---- Main (full-width workspace) ---- #
st.markdown('<div id="masthead">VocaSync</div>'
            '<div class="vg-sub">Sync your words into one graph. Anchor them in Anki.</div>'
            '<a class="vg-jump vg-up" href="#masthead">⬆ Tabs</a>'
            '<a class="vg-jump vg-down" href="#vg-end">⬇ Latest</a>',
            unsafe_allow_html=True)
if not HAS_KEY:
    st.info("No AI key configured (.env). Deterministic **Expand** works; "
            "**Ask / Mine / Explain** need an API key.")

# Composer collapses once an Ask conversation is open, so the chat owns the screen
# (S14 T16); open the expander to fire a new request at any time.
_chat_open = bool(st.session_state.get("chat"))
with st.expander("✏️ New request", expanded=not _chat_open):
    query = st.text_area(
        "query",
        placeholder=("Ask the agent a question, or describe what to Mine "
                     "(e.g. 'only phrasal verbs and fixed expressions', or a topic). "
                     "Leave blank to mine the most useful items."),
        height=68, label_visibility="collapsed")

    b1, b2, b3, b4 = st.columns(4)
    ask_clicked = b1.button("💬 Ask Agent", width='stretch', disabled=not HAS_KEY, type="primary")
    mine_clicked = b2.button("⛏ Mine", width='stretch', disabled=not HAS_KEY)
    explain_clicked = b3.button("📖 Explain", width='stretch', disabled=not HAS_KEY)
    expand_clicked = b4.button("🌱 Expand", width='stretch')

# ---- dispatch ---- #
action = None
try:
    if ask_clicked:
        if not query.strip():
            st.warning("Type a request first.")
        else:
            ask_src = (source_path or "").strip().strip('"')
            ask_src = ask_src if (ask_src and os.path.exists(ask_src)) else ""
            # A new Ask starts a fresh conversation (drop any prior context).
            st.session_state["chat"] = [{"role": "user", "text": query.strip()}]
            st.session_state["agent_pending"] = None
            st.session_state["ask_scratch"] = None
            with st.spinner("Agent is analyzing..."):
                action = do_ask(query, ask_src)
            _apply_ask(action)
            action = None            # already folded into chat; skip generic result assign
    elif mine_clicked:
        src = (source_path or "").strip().strip('"')
        media = (media_path or "").strip().strip('"')
        if not src:
            st.warning(f"Pick or enter a source .srt/media file to mine (sample: {SAMPLE_HINT}).")
        elif not os.path.exists(src):
            st.warning(f"File not found: {src}")
        else:
            media = media if (media and os.path.exists(media)) else ""
            st.session_state["last_video"] = media   # for the ".ass → video folder" copy (T7)
            with st.spinner("Mining vocabulary..."):
                action = do_mine(src, query.strip(), media)
            # seed the Review & Approve editor from the freshly stashed pending drafts
            st.session_state["review_df"] = pd.DataFrame(
                review_io.pending_to_rows(review_io.load_pending()), columns=review_io.COLUMNS)
            st.session_state["editor_nonce"] += 1
            # a new Mine clears the held-back state + toggle (S16 T3.3)
            st.session_state["held_back_keys"] = []
            st.session_state["show_held_back"] = False
    elif explain_clicked:
        if not query.strip():
            st.warning("Type a word or sentence to explain.")
        else:
            with st.spinner("Explaining..."):
                action = do_explain(query)
    elif expand_clicked:
        if not query.strip():
            st.warning("Type a term to expand.")
        else:
            with st.spinner("Looking up senses..."):
                action = do_expand(query)
except SystemError_ as e:                       # system-error -> HALT + report
    st.error(f"System error (halted): {e}")
except Exception as e:
    st.error(f"Something went wrong: {e}")

if action is not None:
    st.session_state["result"] = action

# ---- render (Conversation + Review & Approve as top-level tabs) ---- #
# S16 T5b: only two tabs now — the old "Anki Drafts" sub-tab is gone; its card preview
# moved to a "Card preview (last run)" expander at the end of Review & Approve.
st.divider()
tab_conv_top, tab_review = st.tabs(["💬 Conversation", "✅ Review & Approve"])

with tab_conv_top:
    result = st.session_state.get("result")
    if result:
        # Trajectory (REAL tool calls) — st.status, auto-collapse when done (fix 1).
        # ONLY the Ask path (kind='ask') is LLM-chosen; Mine/Explain/Expand run a FIXED
        # deterministic pipeline. Label them differently so the two are not confused.
        traj = result.get("trajectory", [])
        if traj:
            chain = " → ".join(t.get("tool", "?") for t in traj)
            is_agent = result.get("kind") == "ask"
            verb = "chosen by the LLM" if is_agent else "run as a fixed pipeline"
            done = (f"Agent chose {len(traj)} tool(s):  {chain}" if is_agent
                    else f"Pipeline ran {len(traj)} step(s):  {chain}")
            with st.status(f"{'Agent trajectory' if is_agent else 'Pipeline trace'} — "
                           f"{len(traj)} tool(s) {verb}", expanded=True) as s:
                for i, step in enumerate(traj, 1):
                    line = f"**{i}. `{step.get('tool','?')}`**"
                    if step.get("thought"):
                        line += f" — 💭 {step['thought']}"
                    st.write(line)
                    if step.get("args"):
                        st.caption(f"args: {step.get('args')}")
                s.update(label=done, state="complete", expanded=True)

        # Ask = multi-turn conversation; Mine/Explain/Expand = one-shot markdown summary.
        if result.get("kind") == "ask" and st.session_state["chat"]:
            # Render the conversation INLINE (no fixed-height inner scrollbar). A cramped
            # st.container(height=…) box gave every turn its own scrollbar, so a long agent
            # answer was clipped and the learner had to scroll a tiny inner pane. Inline lets
            # each message show in full and the learner scrolls the page normally.
            for m in st.session_state["chat"]:
                st.chat_message(m["role"]).markdown(m["text"])
            st.markdown('<div id="vg-end"></div>', unsafe_allow_html=True)  # ⬇ jump target

            # S19 — deterministic action shortcuts under the conversation (NOT floating per bubble;
            # one row reflecting the CURRENT session, next to the reply box). Each button just feeds
            # the right phrase to do_ask, so it reuses the EXISTING Python-driven paths (continuation
            # / bulk-save) — 0 LLM decisions, exact counts, no new agent logic. A SPECIFIC few-word
            # save is still done by typing it to the agent. Terms are read from the carried scratch.
            _sc = st.session_state.get("ask_scratch") or []

            def _scratch_terms(pfx):
                ln = next((s for s in reversed(_sc)
                           if isinstance(s, str) and s.startswith(pfx)), "")
                return [t.strip() for t in ln.split(":", 1)[1].split(",") if t.strip()] if ln else []
            _allf = _scratch_terms("ALL_FOUND:")
            _remf = _scratch_terms("REMAINING_UNEXPLAINED:")
            if _allf or _remf:
                # Explain remaining = read the rest; Save all found = keep everything discovered.
                # (No "save remaining" — you read unexplained phrases first, you don't save unseen ones.)
                bx1, bx2 = st.columns(2)
                _do_rest = bx1.button(f"📖 Explain remaining ({len(_remf)})", key="btn_explain_rest",
                                      disabled=not _remf, width='stretch')
                _do_all = bx2.button(f"💾 Save all found ({len(_allf)})", key="btn_save_all",
                                     disabled=not _allf, width='stretch')
                _phrase = ("explain the rest" if _do_rest
                           else "save all the found phrases to my queue" if _do_all else "")
                if _phrase:
                    st.session_state["chat"].append({"role": "user", "text": _phrase})
                    with st.spinner("Working..."):
                        act = do_ask(_phrase, prior_scratch=_sc)
                    _apply_ask(act)
                    st.rerun()
            # Reply box — ALWAYS available while an Ask conversation is open, so the learner
            # can keep talking whether the agent asked a question or gave a final answer. No
            # turn cap inside run_agent — the loop ends when the agent returns a final answer;
            # a fresh "Ask Agent" click starts a new conversation.
            with st.form("reply_form", clear_on_submit=True):
                reply = st.text_input("Continue the conversation", key="reply_text",
                                      placeholder="Type your reply and press Send…")
                sent = st.form_submit_button("Send", type="primary")
            if sent and reply.strip():
                st.session_state["chat"].append({"role": "user", "text": reply.strip()})
                with st.spinner("Agent is thinking..."):
                    act = do_ask(reply.strip(),
                                 prior_scratch=st.session_state.get("ask_scratch"))
                _apply_ask(act)
                st.rerun()
        elif result.get("summary"):
            st.markdown(result["summary"])
        else:
            st.caption("Nothing to show yet.")
    else:
        st.caption("Ask, Mine, Explain, or Expand above to get started.")

with tab_review:
    # NON-EXCEL HITL checkpoint: review / edit / approve candidates in-app. The edited df is
    # the single source of truth read by Commit Approved (sidebar). Nothing is written to the
    # graph here — that happens only in commit_approved (the single write point).
    # S17 fix (stuck-queue bug): ALWAYS reconcile the in-session edit df with disk-truth
    # (pending_drafts.json) so words the AGENT staged mid-conversation show up immediately,
    # while the learner's in-progress edits on existing rows are preserved. The pending file
    # is the single review queue; the table must mirror it, not a stale Mine snapshot.
    # S18 P2-1: reconcile already rebuilds from disk each render, but if a session's cached
    # review_df ever diverges (e.g. after switching sessions), this button drops the cache so
    # the table reloads PURELY from pending_drafts.json. Cheap escape hatch, no logic change.
    if st.button("↻ Reload review from disk", key="reload_review"):
        st.session_state["review_df"] = None
        st.rerun()
    review_df = review_io.reconcile_rows(
        st.session_state.get("review_df"), review_io.load_pending())
    st.session_state["review_df"] = review_df
    if review_df is None or len(review_df) == 0:
        st.caption("No candidates to review yet — run Mine or ask the agent to save words.")
    else:
        # S19 (owner: hint chiếm nhiều diện tích) — fold the how-to into a COLLAPSED expander so
        # the table + warnings sit higher; the learner opens it only when needed.
        with st.expander("ℹ️ How to use the Review table", expanded=False):
            st.caption("Set each row's **status**, edit term / definition / sense_id / word_type / "
                       "tags / mnemonic as needed, then click **Commit Approved** in the sidebar.")
            st.caption("Tip: mark the few bad rows (rejected / needs_revision), leave good rows "
                       "blank, then Approve All · in a status cell type 'a'+Enter → approved · "
                       "Ctrl+C/V works across cells.")

        # S19 (#4b): a YELLOW banner surfacing ungrounded rows up-front, so the learner never has
        # to scan the whole ⚠ column. Yellow (not red): these rows are held back from Commit, but
        # good rows still commit — it needs attention, it does not block the batch. Each term shows
        # the suggested REAL transcript line (from _transcript_hint via the flag reason, S19 #4a)
        # so it can be pasted straight into the `sentence` cell — no digging through the script.
        try:
            _af = review_df["ai_fields"].fillna("").astype(str)
            _rs = review_df["needs_review"].fillna("").astype(str)
            _ung = review_df[_af.str.contains("ungrounded", case=False)]
        except Exception:
            _ung = None
        if _ung is not None and len(_ung):
            _items = []
            for _, _r in _ung.iterrows():
                _term = str(_r.get("term", "")).strip()
                _m = re.search(r"Real transcript line\(s\):\s*(.+)$", str(_r.get("needs_review", "")))
                _hint = f" → maybe: _{_m.group(1).strip()}_" if _m else ""
                _items.append(f"**{_term}**{_hint}")
            st.warning(
                f"⚠️ **{len(_ung)} ungrounded word(s)** — fix the `sentence` cell with the real film "
                "line before Commit (other rows still commit normally):\n\n- "
                + "\n- ".join(_items))

            # S19 OPEN-4: a deterministic (0-LLM) candidate PAGER — page through REAL transcript
            # lines ranked for each ungrounded term and drop the right one straight into its
            # `sentence` cell. Replaces the agent's useless "Not found" loop and the manual .srt
            # dig. Bridges chat/Review: writes into review_df (session) + reseeds the editor.
            _sents = []
            try:
                _meta4 = review_io.load_pending().get("_meta", {})
                _meta4 = _meta4 if isinstance(_meta4, dict) else {}
                _srt4 = (_meta4.get("srt_path", "") or str(st.session_state.get("last_srt") or "")).strip().strip('"')
                if _srt4 and os.path.exists(_srt4):
                    _seg4 = call_tool("ingest_transcript", {"source": _srt4}).get("segments", [])
                    _sents = [str(s.get("text", "")).strip() for s in _seg4 if str(s.get("text", "")).strip()]
            except Exception:
                _sents = []
            if _sents:
                from extract_vocab import _content_lemmas
                st.session_state.setdefault("ung_hint_page", {})
                with st.expander("🔎 Resolve ungrounded — find the real film line (click to page)",
                                 expanded=False):
                    # S19 (owner P2): one click FILLS a suggested real film line into every
                    # ungrounded row's `sentence` cell (no copy-paste, no double-typing). Python's
                    # lemma ranking is the reliable suggester; the AI is used only when its pick is
                    # verbatim-real. Every fill is flagged to verify before Commit.
                    if st.button("🪄 Suggest a line for all ungrounded", key="ai_resolve_ung"):
                        _ai_resolve_ungrounded(_ung, _sents)
                    for _, _ur in _ung.iterrows():
                        _ut = str(_ur.get("term", "")).strip()
                        _uk = str(_ur.get("key", "")).strip()
                        if not _ut:
                            continue
                        _utl = _content_lemmas(_ut)
                        _scored = sorted(((len(_utl & _content_lemmas(s)), s) for s in _sents),
                                         key=lambda x: -x[0])
                        _cands = [s for _n, s in _scored if _n][:8]
                        if not _cands:
                            st.caption(f"**{_ut}** — no candidate line found in the transcript.")
                            continue
                        _pg = st.session_state["ung_hint_page"].get(_ut, 0) % len(_cands)
                        _cur = _cands[_pg]
                        st.markdown(f"**{_ut}** — candidate {_pg + 1}/{len(_cands)}: _{_cur}_")
                        _c1, _c2 = st.columns(2)
                        if _c1.button("🔎 another line", key=f"ungnext_{_uk}"):
                            st.session_state["ung_hint_page"][_ut] = _pg + 1
                            st.rerun()
                        if _c2.button("✓ use this line", key=f"unguse_{_uk}"):
                            _rdf = st.session_state.get("review_df")
                            if _rdf is not None and (_rdf["key"] == _uk).any():
                                _rdf.loc[_rdf["key"] == _uk, "sentence"] = _cur
                                st.session_state["review_df"] = _rdf
                                st.session_state["editor_nonce"] += 1
                                st.session_state["commit_flash"] = {
                                    "success": [f"Set the line for '{_ut}' — review it, then Commit."],
                                    "warning": [], "error": []}
                                st.rerun()

        # S19 OPEN-10: remind the learner to verify POLYSEMOUS words' sense before approving —
        # these carry many WordNet senses and the AI may have picked the wrong one (the 'put
        # down' case). Changing sense_id auto-refreshes synonyms/relations/definition (OPEN-8).
        try:
            _poly = review_df[_rs.str.contains("polysemy", case=False)]
        except Exception:
            _poly = None
        if _poly is not None and len(_poly):
            _pt = [str(_r.get("term", "")).strip()
                   for _, _r in _poly.iterrows() if str(_r.get("term", "")).strip()]
            # S19 (owner: save space) — one compact caption; the term list folds away.
            st.caption(f"🔀 **{len(_pt)} polysemous word(s)** — check `sense_id` (open 🔎 Sense browser) "
                       "before Approve; changing the sense auto-refreshes synonyms/definition at Commit.")
            with st.expander(f"Show {len(_pt)} polysemous word(s)", expanded=False):
                st.markdown(", ".join(f"**{t}**" for t in _pt))

        # Sense browser (S15 T3) — moved ABOVE the table (S16 T6) so the reviewer can look up
        # the right sense_id BEFORE editing the row below. Read-only WordNet lookup (data_editor
        # can't do a per-row dropdown); goes through the registry so policy gates it; offline.
        with st.expander("🔎 Sense browser — look up WordNet senses", expanded=False):
            terms = [t for t in dict.fromkeys(review_df["term"].tolist()) if str(t).strip()]
            if not terms:
                st.caption("No terms to look up yet.")
            else:
                sel = st.selectbox("Term", terms, key="sense_browser_term")
                # the sense_id currently set for that term's row (bold it in the list)
                cur_rows = review_df[review_df["term"] == sel]
                cur_sense = (str(cur_rows.iloc[0]["sense_id"]).strip()
                             if len(cur_rows) else "")
                try:
                    senses = call_tool("wordnet_lookup", {"term": sel}).get("senses", [])
                except Exception as e:
                    senses = []
                    st.caption(f"WordNet lookup unavailable: {e}")
                if senses:
                    for s in senses:
                        sid = s.get("sense_id", "")
                        line = (f"`{sid}` — ({s.get('pos', '')}) {s.get('definition', '')}"
                                if sid else s.get("definition", ""))
                        exs = s.get("examples") or []
                        if exs:
                            line += f"  — _“{exs[0]}”_"
                        if sid and sid == cur_sense:
                            line = f"**➡ {line}**"      # the row's current sense_id, highlighted
                        st.markdown(line)
                else:
                    st.caption("No WordNet senses — the definition is AI-authored (flagged); "
                               "edit it directly in the table.")

        # Held-back rows (S16 T3): after a Commit that partitioned some rows, offer to sort
        # them to the TOP so the learner can find and fix them. We SORT (not filter) so every
        # row stays in the editor and no edit outside the view can be lost.
        held = [k for k in (st.session_state.get("held_back_keys") or []) if k]
        show_held = False
        if held:
            show_held = st.toggle(
                f"Show held-back rows first ({len(held)})", key="show_held_back",
                help="Sort the rows held back at the last Commit to the top of the table "
                     "so you can fix them, then Commit again.")
        display_df = review_df
        if show_held:
            held_set = set(held)
            rank = review_df["key"].map(lambda k: 0 if k in held_set else 1)
            display_df = (review_df.assign(_hb=rank.values)
                          .sort_values("_hb", kind="stable").drop(columns="_hb"))

        # Table height grows with the row count (~35px/row + header, capped) so a long batch
        # fills the screen instead of scrolling inside a short pane (S16 T4).
        table_h = min(64 + 35 * len(display_df), 1400)
        edited = st.data_editor(
            display_df,
            key=f"review_editor_{st.session_state['editor_nonce']}_{int(show_held)}",
            width='stretch', height=table_h, hide_index=True, num_rows="fixed",
            column_order=review_io.COLUMNS,
            column_config={
                "#": st.column_config.NumberColumn(
                    "#", disabled=True, width="small", pinned=True,
                    help="Row number shown here and in the held-back message."),
                "status": st.column_config.SelectboxColumn(
                    "status", options=list(review_io.STATUS_VALUES), required=False,
                    pinned=True, width="small"),
                "term": st.column_config.TextColumn(
                    "term", pinned=True, width="small",
                    help="EDITABLE (S19): fix a distorted headword (e.g. 'be all over the place' "
                         "→ 'all over the place'). Commit rekeys the card + drops the old node."),
                "sentence": st.column_config.TextColumn(
                    "sentence", width="large",
                    help="The source line the word appeared in. EDITABLE: if a row is "
                         "flagged 'ungrounded' (agent-staged with an unverified sentence), "
                         "paste the real transcript line here — commit applies your edit."),
                "definition": st.column_config.TextColumn("definition", width="large"),
                "word_type": st.column_config.SelectboxColumn(
                    "word_type", options=sorted(_VALID_WORD_TYPES), required=False, width="small"),
                "sense_id": st.column_config.TextColumn("sense_id", width="small"),
                "tags": st.column_config.TextColumn(
                    "tags", width="small", help="'; '-separated, strongest first"),
                "mnemonic": st.column_config.TextColumn(
                    "mnemonic", width="medium",
                    help="Memory hook. EDITABLE — retype it after changing the sense (the AI "
                         "one was written for the old meaning), or use the Re-enrich button."),
                "collocations": st.column_config.TextColumn(
                    "collocations", width="small", help="'; '-separated natural phrases. EDITABLE."),
                "needs_review": st.column_config.TextColumn(
                    "⚠ ai flag", disabled=True,
                    help="WHY the machine flags this row — the cell lists the reasons "
                         "(polysemy · definition(ai) · sense_id · low confidence …). "
                         "EMPTY = no concern; that is what 'Approve confident' trusts. "
                         "Not a decision — your `status` column is."),
                "ai_fields": st.column_config.TextColumn(
                    "🤖 ai fields", disabled=True, width="small",
                    help="PROVENANCE: which fields the AI authored (collocations · tags · "
                         "mnemonic · pattern · definition …). Shown even on confident rows — "
                         "independent of the ⚠ ai flag gate."),
                "confidence": st.column_config.TextColumn("confidence", disabled=True, width="small"),
                "comment": st.column_config.TextColumn("comment", width="small"),
                "key": st.column_config.TextColumn("key", disabled=True, width="small"),
            },
        )
        st.session_state["review_df"] = edited

    # Card preview (S16 T5b): the old "Anki Drafts" tab folded into an expander at the end of
    # Review & Approve. Shows the drafts from the last run (preview only — the real deck is
    # generated at Commit from the approved subset).
    _res = st.session_state.get("result") or {}
    _drafts = _res.get("drafts", [])
    with st.expander("🃏 Card preview — what each Anki card will hold", expanded=False):
        # S17: preview WHAT THE CARD WILL SAY (front/back/cloze sentence), not raw draft
        # dicts. Falls back to the pending queue so agent-staged words (which don't come
        # from a Mine "last run") show up too.
        _pending = review_io.load_pending()
        # S19 OPEN-11 (media honesty): a card gets audio+screenshot when it has a clip OR a
        # companion video is resolvable at Commit (mine-via-chat attaches no clip, yet Commit
        # still cuts audio from the .mp4 sitting next to the .srt). Resolve the video ONCE the
        # same way commit_approved does so the preview stops always saying "text only".
        _video_ok = False
        try:
            _meta = _pending.get("_meta", {}) if isinstance(_pending.get("_meta"), dict) else {}
            _pv = str(st.session_state.get("last_video") or "").strip().strip('"')
            _ps = (_meta.get("srt_path", "") or str(st.session_state.get("last_srt") or "")).strip().strip('"')
            if not (_pv and os.path.exists(_pv)) and _ps and os.path.exists(_ps):
                _pb = os.path.splitext(_ps)[0]
                for _pe in (".mp4", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".wav"):
                    if os.path.exists(_pb + _pe):
                        _pv = _pb + _pe
                        break
            _video_ok = bool(_pv and os.path.exists(_pv))
        except Exception:
            _video_ok = False

        def _media_label(has_clip: bool, has_sentence: bool) -> str:
            if has_clip:
                return "🎬 audio + screenshot"
            if _video_ok and has_sentence:
                return "🎬 audio + screenshot (cut at Commit)"
            return "📝 text only"

        _cards = []
        # S19 OPEN-11: build the preview from review_df (SESSION — reflects the learner's live
        # edits: definition / sentence / collocations) joined with pending for the clip/media.
        # Falls back to the raw drafts/pending when there is no in-session table yet.
        _rdf = st.session_state.get("review_df")
        if _rdf is not None and len(_rdf):
            for r in _rdf.to_dict("records"):
                key = str(r.get("key", "")).strip()
                pend = _pending.get(key) if isinstance(_pending.get(key), dict) else {}
                sent = str(r.get("sentence", "") or "")
                _cards.append({
                    "front (term)": r.get("term", ""),
                    "back (definition)": r.get("definition") or "",
                    "cloze sentence": sent,
                    "extra": r.get("collocations", "") or "",
                    "media": _media_label(bool(pend.get("clip")), bool(sent.strip())),
                })
        else:
            _drafts2 = _drafts or [v for k, v in _pending.items()
                                   if k != "_meta" and isinstance(v, dict)]
            for d in _drafts2:
                n = d.get("node", d) if isinstance(d, dict) else {}
                occ = (n.get("occurrences") or [{}])
                sent = (occ[0].get("sentence", "") if occ and isinstance(occ[0], dict) else "")
                _cards.append({
                    "front (term)": n.get("term", ""),
                    "back (definition)": n.get("definition") or "",
                    "cloze sentence": sent,
                    "extra": ", ".join(normalize_collocations(n.get("collocations"))),
                    "media": _media_label(bool(isinstance(d, dict) and d.get("clip")), bool(str(sent).strip())),
                })
        if _cards:
            st.caption("Preview reflects your table edits. Synonyms/relations + a WordNet "
                       "definition are re-derived from `sense_id` at Commit. 'media' shows "
                       "audio+screenshot when a companion video is attached or sits next to the .srt.")
            st.dataframe(_cards, width='stretch')
        else:
            st.caption("No card drafts yet — run Mine (or ask the agent to save words) first.")
