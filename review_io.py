"""
review_io.py — HITL stash for the VOCAB review (GĐ4, task 3; NON-EXCEL flow).

This is the human checkpoint (AGENTS.md §5, execution_policy hitl): enrich drafts are
stashed to data/pending_drafts.json (the FULL Nodes, keyed by Node.key), and the learner
reviews / edits / approves them IN THE APP via st.data_editor (app.py). ONLY rows they
mark `status=approved` are later committed to the personal graph — the single commit point
lives in app.commit_approved.

NON-EXCEL (S12 non-excel flow): the old review.xlsx round-trip (openpyxl write +
apply_highlight + read_review) is gone. Review state now lives in the st.data_editor
DataFrame (session_state), built from `pending_to_rows(pending)` and read back at Commit.

Disk-as-truth (HANDOVER §3.2): pending_drafts.json holds the full draft Nodes (+ per-draft
confidence / needs_review / ai_fields / surface / clip) so the commit step can reconstruct
each Node (edges/occurrences/collocations) and merge it. `pending["_meta"]` carries
{srt_path, source} for the run (merged, never overwritten with empty).
"""

from __future__ import annotations

import json
import os

from schema import normalize_collocations

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
PENDING_PATH = os.path.join(DATA_DIR, "pending_drafts.json")

# Column order for the editable review rows (S16 T4). `#` is a 1-based display index (the
# data_editor hides the real index, so this is how a learner/message points at "row N");
# `status` + `term` lead so the decision and the word are always visible. Two DISTINCT
# machine columns (separated per owner decision, S16-followup):
#   `needs_review` (shown as "⚠ ai flag") = the GATE — REASONS the machine wants a human
#     look ("polysemy, definition(ai)"; generic "low confidence" fallback). An EMPTY cell
#     means no concern — that emptiness is what "Approve confident" trusts.
#   `ai_fields` = PROVENANCE — which fields the AI authored (collocations, tags, mnemonic,
#     …), shown even on confident rows so AI-written content is never invisible.
# `tags` is AI-proposed + learner-editable ("; "-separated, strongest first);
# `key` (= Node.key) is the round-trip id, last.
COLUMNS = ["#", "status", "term", "sentence", "definition", "word_type", "sense_id",
           "tags", "mnemonic", "collocations", "needs_review", "ai_fields", "confidence",
           "comment", "key"]

# Editable by the human in st.data_editor; everything else is informational/disabled.
# `sentence` is editable (S17): an agent-staged word may carry an unverified sentence —
# the LEARNER supplies the real source line; commit applies it to the occurrence.
EDITABLE = {"word_type", "definition", "sense_id", "tags", "status", "comment", "sentence",
            "mnemonic", "collocations",   # S19 OPEN-8: retypeable after a sense change
            "term"}                       # S19 OPEN-5: fix a distorted headword (rekeys at commit)

# ai_fields entries that name a FIELD the AI authored (provenance) — everything else in the
# list is a GATE reason (polysemy, conceptnet_edges, ungrounded, …). The two review columns
# split along this line (S17, owner decision: "different function -> different column").
AI_FIELD_NAMES = {"definition", "collocations", "mnemonic", "pattern", "tags", "usage_pos",
                  "sense_id"}

TAGS_SEP = "; "   # how a tag list is serialised into the single cell (strongest first)
STATUS_VALUES = ("approved", "rejected", "needs_revision")


def _norm_sent(s: str) -> str:
    """Same [a-z0-9']+ normalization the grounding path uses — one canon for 'same sentence'."""
    import re
    return " ".join(re.findall(r"[a-z0-9']+", str(s).lower()))


def _snap_same_sentence_senses(new_pending: dict, existing_pending: dict) -> None:
    """S18 'one sentence, one sense' (owner decision 2026-07-04): the SAME source sentence
    cannot ground TWO different senses of the same term — a film line has ONE meaning.
    (Seen in real data: `find out#determine.v.08` AND `find out#learn.v.02` both citing
    "How did you find out?" — fine graph-mechanically, wrong for understanding the film.)

    For each INCOMING draft whose term + normalized occurrence sentence already ground an
    EXISTING sense (in the committed graph or the current queue), SNAP the draft onto that
    existing sense (key / sense_id / definition / word_type), so downstream dedup-by-key and
    commit-upsert MERGE the occurrence instead of forking a second node for the same line.
    A draft with a DIFFERENT sentence keeps its own sense — true polysemy stays possible
    (e.g. `go on` = continue.v.01 and continue.v.02, each grounded by its own line).

    Mutates `new_pending` in place (rekeys snapped entries). Deterministic, 0 AI, no-crash:
    any failure leaves the drafts untouched (prior behaviour)."""
    try:
        refs = {}                      # term(lower) -> [(key, {norm sentences}, node), ...]

        def _add_ref(key, node):
            t = str(node.get("term", "")).strip().lower()
            if not t or not key:
                return
            sents = {_norm_sent(o.get("sentence", ""))
                     for o in (node.get("occurrences") or []) if isinstance(o, dict)}
            refs.setdefault(t, []).append((key, sents - {""}, node))

        try:                                              # committed graph (may be absent)
            from _common import load_graph, GRAPH_PATH
            for k, n in load_graph(GRAPH_PATH).nodes.items():
                _add_ref(k, n.model_dump())
        except Exception:
            pass
        for k, v in (existing_pending or {}).items():     # current queue
            if k != "_meta" and isinstance(v, dict):
                _add_ref(k, v.get("node", {}) if isinstance(v.get("node"), dict) else {})

        for key in list(new_pending):
            v = new_pending[key]
            node = v.get("node", {}) if isinstance(v.get("node"), dict) else {}
            t = str(node.get("term", "")).strip().lower()
            sents = {_norm_sent(o.get("sentence", ""))
                     for o in (node.get("occurrences") or []) if isinstance(o, dict)} - {""}
            if not t or not sents:
                continue
            for rkey, rsents, rnode in refs.get(t, []):
                if rkey == key or not (sents & rsents):
                    continue
                # same term + same sentence, DIFFERENT sense -> the existing sense wins
                # (the learner already has/queued it; one line cannot mean two things).
                node["key"] = rkey
                node["sense_id"] = rnode.get("sense_id")
                if rnode.get("definition"):
                    node["definition"] = rnode["definition"]
                if rnode.get("word_type"):
                    node["word_type"] = rnode["word_type"]
                del new_pending[key]
                new_pending.setdefault(rkey, v)           # rekeyed (append-dedup then merges)
                break
    except Exception:
        pass


def _row_from_draft(d: dict) -> dict:
    """One review row from a draft/pending entry {node, confidence, needs_review, ai_fields}."""
    node = d.get("node", {}) if isinstance(d, dict) else {}
    occs = node.get("occurrences") or []
    sentence = (occs[0].get("sentence", "") if occs and isinstance(occs[0], dict) else "")
    return {
        "term": node.get("term", ""),
        "sentence": sentence,               # context line (display-only; commit ignores it)
        "word_type": node.get("word_type", "") or "",
        "sense_id": node.get("sense_id") or "",
        "definition": node.get("definition") or "",
        "confidence": f"{float(d.get('confidence', 0.0) or 0.0):.2f}",
        # Two DISTINCT cells (S17 split — they used to show the same joined list):
        #   ⚠ ai flag  = GATE reasons only (polysemy, conceptnet_edges, ungrounded, …;
        #                fallback "low confidence"). Empty = machine sees no concern.
        #   🤖 ai fields = FIELD provenance only (which columns the AI wrote), ALWAYS
        #                shown — a confident row can still carry AI-written content.
        # S19 OPEN-3: append the persisted ungrounded reason so the Review banner regex
        # ("Real transcript line(s): …") and the ⚠ column show the suggested real line.
        "needs_review": ((", ".join([f for f in (d.get("ai_fields") or [])
                                     if f not in AI_FIELD_NAMES] or ["low confidence"]))
                         + (f" — {d.get('ungrounded_reason', '').strip()}"
                            if d.get("ungrounded_reason") else ""))
                        if d.get("needs_review") else "",
        "ai_fields": ", ".join(f for f in (d.get("ai_fields") or []) if f in AI_FIELD_NAMES),
        "tags": TAGS_SEP.join(node.get("tags", []) or []),   # AI-proposed; learner edits
        # S19 OPEN-8: mnemonic/collocations editable so a re-sensed word can be re-annotated
        # (the AI ones were written for the previous meaning).
        "mnemonic": node.get("mnemonic") or "",
        # S19 BUG-1 defensive: repair any char-broken list before joining for display.
        "collocations": TAGS_SEP.join(normalize_collocations(node.get("collocations"))),
        "status": "",                       # blank = undecided; human sets a STATUS_VALUE
        "comment": "",
        "key": node.get("key", ""),
    }


def export_review(drafts: list[dict], pending_path: str = PENDING_PATH,
                  mode: str = "overwrite", srt_path: str = "", source: str = "") -> str:
    """Stash the full draft Nodes into pending_drafts.json (NON-EXCEL; no .xlsx written).

    mode="overwrite" (default): replace the pending queue with these drafts.
    mode="append": MERGE these drafts into the existing queue, deduping by `key` (a key
      already present is kept so the learner's prior review is never clobbered).

    The stash keeps, per key: the full node, its `surface`, its `clip` (-> final exports),
    and the draft-level confidence / needs_review / ai_fields (so pending_to_rows can rebuild
    the editable review rows). `pending["_meta"]` carries {srt_path, source}, merged and never
    overwritten with empty. NEVER writes personal_graph.json (the single commit point is
    app.commit_approved). Returns the pending_drafts.json path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(pending_path)), exist_ok=True)

    # full-node stash for THIS batch (keyed by Node.key)
    new_pending = {}
    for d in (drafts or []):
        node = d.get("node", {}) if isinstance(d, dict) else {}
        if node.get("key"):
            new_pending[node["key"]] = {
                "node": node,
                "surface": d.get("surface", ""),
                "clip": d.get("clip"),                     # clip -> #1 final exports
                "confidence": d.get("confidence", 0.0),
                "needs_review": bool(d.get("needs_review")),
                "ai_fields": d.get("ai_fields", []) or [],
                # S19 OPEN-3: keep the ungrounded hint ("Real transcript line(s): …") so the
                # Review banner + ⚠ column can suggest the real line, not just the flag.
                "ungrounded_reason": d.get("ungrounded_reason", "") or "",
            }

    existing_pending = load_pending(pending_path)
    existing_meta = existing_pending.get("_meta") if isinstance(existing_pending.get("_meta"), dict) else {}

    # S18 "one sentence, one sense": snap an incoming draft that re-grounds an already-known
    # sentence under a DIFFERENT sense onto the existing sense (see helper doc). Runs for both
    # modes — the committed graph is a reference even when the queue is being overwritten.
    _snap_same_sentence_senses(new_pending, existing_pending)

    if mode == "append":
        pending = {k: v for k, v in existing_pending.items() if k != "_meta"}
        for k, v in new_pending.items():
            if k not in pending:                           # dedup-by-key (keep prior review)
                pending[k] = v
            else:
                # S18 #1 — RE-STAGE with a corrected sentence. dedup-by-key used to keep the
                # OLD row verbatim, so an agent that re-staged an ungrounded word with the real
                # transcript line ("done, updated it") silently changed nothing — the queue kept
                # the fabricated sentence. Now: if the EXISTING row is still flagged `ungrounded`
                # and the INCOMING draft is grounded (cleared the flag via stage_for_review's
                # gate), replace it so the corrected sentence + cleared flag land in the queue.
                # Guarded so we never clobber a row the human already fixed by hand: only an
                # `ungrounded`-flagged existing row is eligible (a human fix clears that flag at
                # commit, and reconcile_rows keeps in-session edits regardless).
                old_ungrounded = "ungrounded" in (pending[k].get("ai_fields") or [])
                new_grounded = "ungrounded" not in (v.get("ai_fields") or [])
                if old_ungrounded and new_grounded:
                    pending[k] = v
    else:                                                  # overwrite: this batch replaces it
        pending = dict(new_pending)

    # _meta merged, never overwritten with empty (keeps a prior srt_path/source if none given)
    meta = dict(existing_meta)
    if srt_path:
        meta["srt_path"] = srt_path
    if source:
        meta["source"] = source
    pending["_meta"] = meta

    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    return pending_path


def pending_to_rows(pending: dict) -> list[dict]:
    """Build editable review rows from the pending stash (for st.data_editor).

    Each row carries COLUMNS; `status` starts blank (undecided) and the learner sets it in
    the app. Skips the `_meta` entry. Reuses `_row_from_draft` so the row shape matches the
    old Excel columns exactly.
    """
    rows = []
    for key, v in (pending or {}).items():
        if key == "_meta":
            continue
        if not isinstance(v, dict):
            continue
        rows.append(_row_from_draft(v))
    # 1-based display index (S16 T4), renumbered 1..N so it is always continuous — the
    # data_editor hides the real DataFrame index, so this column is what a learner (and the
    # commit message) points at as "row N".
    for i, r in enumerate(rows, 1):
        r["#"] = i
    return rows


def reconcile_rows(review_df, pending: dict):
    """Merge disk-truth (pending) into the in-session edit DataFrame (S17 fix).

    ROOT CAUSE of the "review table stuck at N rows" bug: `review_df` was cached in
    session_state and only rebuilt when None, so words the AGENT staged via
    stage_for_review (which appends to pending_drafts.json on disk during an Ask turn)
    never appeared — the agent truthfully said "staged", the disk truthfully grew, but the
    table showed the stale Mine snapshot. This reconciles on every render:

      • rows for pending keys NOT yet in the df are ADDED (fresh, undecided),
      • existing rows are KEPT AS-IS (the learner's in-progress status/edits survive),
      • rows whose key vanished from pending are dropped,
      • the `#` column is renumbered 1..N so the held-back messages still line up.

    Returns a new list-of-dict rows ready for a DataFrame (COLUMNS order).
    """
    import pandas as pd  # local: keep review_io importable without pandas for pure tests
    existing = {}
    if review_df is not None and len(review_df):
        for r in review_df.to_dict("records"):
            k = str(r.get("key", "")).strip()
            if k:
                existing[k] = r
    disk_rows = {r["key"]: r for r in pending_to_rows(pending) if r.get("key")}
    merged = []
    for key, disk_row in disk_rows.items():
        if key in existing:
            row = dict(existing[key])          # keep learner edits
            # refresh the machine-owned flag cells from disk (agent may have re-staged with
            # a fixed sentence -> ungrounded flag cleared); learner-editable cells untouched.
            for col in ("needs_review", "ai_fields"):
                row[col] = disk_row.get(col, row.get(col, ""))
            merged.append(row)
        else:
            merged.append(disk_row)            # NEW word from disk (agent- or mine-staged)
    for i, r in enumerate(merged, 1):
        r["#"] = i
    return pd.DataFrame(merged, columns=COLUMNS)


def load_pending(pending_path: str = PENDING_PATH) -> dict:
    """Load the stashed full draft nodes (key -> {node, surface, clip, ...} + _meta)."""
    if not os.path.exists(pending_path):
        return {}
    try:
        with open(pending_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
