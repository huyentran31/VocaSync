"""
build_render_graph.py — Tool #6 (write-local, cheap, DETERMINISTIC).

Render vocab Nodes into an interactive pyvis HTML graph, clustered with Louvain.
Adapted from legacy/_demo_graph_build.py (same colour language) but reads REAL Node
units instead of hardcoded senses, links satellite targets back to existing main nodes
so the graph CONNECTS as it grows, and writes under output/<run>/ (no hardcoded path).

  • node colour  = Louvain community (the "cluster" the rubric asks for)
  • edge colour  = relation type (synonym/antonym/is_a/hyponym/collocation)
  • main node    = bigger dot; satellite (WordNet neighbour) = small dot

Accepts a list of Node dicts, a list of enrich draft units ({"node": ...}), or a
PersonalGraph. On a bad node: skip + log, never crash (docs/TOOLS.md).
"""

from __future__ import annotations

import os

import networkx as nx

from _common import OUTPUT_DIR, log_tool_call, new_run_id, run_dir

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {"description": "List of Node dicts / enrich drafts, or a PersonalGraph."},
        "run_id": {"type": "string"},
        "title": {"type": "string"},
        "recent": {"description": "Optional list of Node keys added/grown in the latest "
                                  "session — drawn with a gold ring + 🆕 so the learner sees "
                                  "what THIS session contributed to the growing graph."},
    },
    "required": ["units"],
}

RECENT_BORDER = "#ffd700"   # gold ring marking nodes from the most recent session

# Dark canvas so the coloured clusters pop (set in Network() below).
NODE_DEFAULT = "#4e79a7"
# HIGH-CONTRAST, distinct hues — each relation type is instantly tellable apart and
# none of them clashes with the teal main-word colour.
E_COLOR = {
    "synonym":     "#2ecc71",  # green
    "antonym":     "#e74c3c",  # red
    "is_a":        "#3498db",  # blue   (parent)
    "hyponym":     "#e67e22",  # orange (child)
    "part_of":     "#e84393",  # magenta
    "used_for":    "#f1c40f",  # yellow  (ConceptNet)
    "has_context": "#9b59b6",  # purple  (ConceptNet)
    "collocation": "#e056fd",  # bright violet
    "category":    "#8a8f98",  # grey (not drawn as a node)
}
# The "life-context" layer (ConceptNet) is drawn DASHED + thin so the WordNet
# backbone stays readable (anti graph-explosion at a glance).
_DASHED_TYPES = {"used_for", "has_context"}
# Human-readable names for the on-canvas legend.
_REL_LABEL = {
    "synonym": "synonym", "antonym": "antonym", "is_a": "is a (parent)",
    "hyponym": "kind of (child)", "part_of": "part of", "used_for": "used for",
    "has_context": "context", "collocation": "collocation",
}
MAIN_COLOR = "#4ecdc4"  # ALL learner vocabulary nodes share this colour (uniform = no noise)
SAT_COLOR = "#3b4048"   # satellite (WordNet/ConceptNet neighbour) — dim so main words pop
MAIN_SHAPE = "dot"      # learner words = big circle (default)
# A learner word's SHAPE encodes its grammatical type (idioms/phrasal verbs stand out):
_MAIN_WT_SHAPE = {
    "word": "dot", "phrasal_verb": "diamond", "idiom": "star",
    "collocation": "hexagon", "slang": "triangleDown",
}
# Satellite SHAPE encodes the relation FAMILY (colour still encodes the exact relation):
#   triangle = hierarchy (is_a / hyponym / part_of) · square = life-context
#   (used_for / has_context / collocation) · dot = lexical (synonym / antonym)
_SHAPE = {
    "is_a": "triangle", "hyponym": "triangle", "part_of": "triangle",
    "used_for": "square", "has_context": "square", "collocation": "square",
    "synonym": "dot", "antonym": "dot",
}
#   => node colour carries NO meaning beyond "main word vs neighbour";
#      the MEANING (relation type) lives entirely in the EDGE colours (see legend).
# 10-colour palette for Louvain communities — bright enough for a dark background.
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#9c755f",
           "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#86bcb6"]


def _rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)' — translucent leaf nodes / silk-thread edges (S14 T13)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return hex_color


def _coerce_nodes(units) -> list[dict]:
    """Normalize the various accepted inputs into a flat list of Node dicts."""
    # PersonalGraph?
    if hasattr(units, "nodes") and isinstance(getattr(units, "nodes"), dict):
        return [n.model_dump() if hasattr(n, "model_dump") else n for n in units.nodes.values()]
    out = []
    for u in units or []:
        if hasattr(u, "model_dump"):           # a Node model
            out.append(u.model_dump())
        elif isinstance(u, dict) and "node" in u:   # enrich draft
            out.append(u["node"])
        elif isinstance(u, dict):
            out.append(u)
    return out


def _louvain(G: nx.Graph) -> dict:
    """Map node -> community index. Prefer Louvain; degrade gracefully."""
    if G.number_of_nodes() == 0:
        return {}
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G)
        except Exception:
            comms = [set(G.nodes())]
    mapping = {}
    for i, c in enumerate(comms):
        for n in c:
            mapping[n] = i
    return mapping


def _inject_tweaks(html: str) -> str:
    """Small-screen (13") friendliness + drag fix, applied to standalone AND embedded HTML.

      • CSS: full-viewport canvas (works with Network(height='100vh')) + a hard cap on the
        vis.js hover tooltip so a multi-line tip can't overrun the filter bar.
      • JS : FREEZE physics once the layout settles and keep it frozen. Dragging a node then moves
        ONLY that node (no re-simulation → the cluster/canvas is never flung around). dragNodes
        moves a node; dragView pans the canvas on empty space.
    """
    style = (
        "<style>"
        "html,body{height:100%;margin:0;background:#090D16}"
        # pyvis' template emits a duplicate <center><h1> heading + a light-gray bordered,
        # floated #mynetwork. Hide the heading and de-clutter the canvas so the standalone
        # file opens clean and full-bleed (esp. on a 13" screen).
        "center,h1{display:none!important}"
        "#mynetwork{height:100vh!important;width:100%!important;border:none!important;float:none!important}"
        ".card{height:100vh!important;border:none!important;background:#090D16!important}"
        ".vis-tooltip{max-width:340px!important;white-space:pre-line!important;"
        "word-break:break-word!important;font:13px/1.4 system-ui,sans-serif!important}"
        "</style>"
    )
    script = (
        "<script type=\"text/javascript\">"
        "(function(){function tune(){try{if(typeof network==='undefined')return setTimeout(tune,120);"
        "network.setOptions({"
        # Obsidian/Logseq aesthetic: size scales with connections, soft glow, rounded nodes,
        # labels grow with zoom (readable close, uncluttered far), gently curved translucent edges.
        "nodes:{shape:'dot',borderWidth:0,shadow:{enabled:true,color:'rgba(0,0,0,0.45)',size:14,x:0,y:3},"
        "scaling:{min:8,max:46,label:{enabled:false}},"
        "font:{color:'#e5e7eb',size:14,strokeWidth:4,strokeColor:'#090D16',face:'system-ui'}},"
        "edges:{smooth:{enabled:true,type:'continuous',roundness:0.6},hoverWidth:1.4,selectionWidth:2},"
        "physics:{enabled:true,stabilization:{enabled:true,iterations:400,fit:true},"
        "minVelocity:1.5,timestep:0.4},"
        "interaction:{dragNodes:true,dragView:true,zoomView:true,hover:true,tooltipDelay:120}});"
        # FREEZE physics after the initial stabilization and keep it frozen. Dragging a node then
        # moves ONLY that node (dragNodes) — it never re-simulates, so the cluster/canvas is never
        # flung around or rearranged ('sau khi kéo bị xô lệch' bug). Drag empty space = pan (dragView).
        "network.once('stabilizationIterationsDone',function(){network.setOptions({physics:false});});"
        "setTimeout(function(){try{network.setOptions({physics:false});}catch(e){}},5000);"
        # S17 stuck-drag watchdog: sometimes vis.js misses the mouseup after a plain click
        # (esp. inside the Streamlit iframe), leaving the pan 'glued' to the cursor with no
        # button held. If the pointer moves with NO buttons pressed while vis still thinks a
        # drag is on, force-release it. Physics untouched.
        # S18 P1-4: vis.js can miss the drag-END event entirely (mouseup landing OUTSIDE the
        # Streamlit iframe, a plain click with no move, or the window losing focus), leaving the
        # pan 'glued' to the cursor. One force-release helper, fired from EVERY end-of-drag
        # signal: mousemove-with-no-buttons (original), mouseup, pointerup, the pointer leaving
        # the document, and the window blurring. Physics/canvas/bgcolor untouched.
        "function vgRelease(e){try{if(network&&network.interactionHandler&&"
        "network.interactionHandler.drag&&network.interactionHandler.drag.dragging){"
        "var x=(e&&e.clientX)||0,y=(e&&e.clientY)||0;"
        "network.interactionHandler.onDragEnd({center:{x:x,y:y},srcEvent:e||{},pointers:[e||{}]});}}catch(err){}}"
        "document.addEventListener('mousemove',function(e){if(e.buttons===0){vgRelease(e);}},true);"
        "document.addEventListener('mouseup',vgRelease,true);"
        "document.addEventListener('pointerup',vgRelease,true);"
        "document.addEventListener('mouseleave',vgRelease,true);"
        "window.addEventListener('blur',function(){vgRelease(null);},true);"
        # T13: edges carry NO text by default; selecting a node reveals the relation name on
        # ITS edges only (edge data already carries `rel`), deselect hides them again. Pure
        # DataSet label updates — physics stays frozen/untouched.
        "network.on('selectNode',function(p){try{"
        "var es=network.body.data.edges;var ids=network.getConnectedEdges(p.nodes[0]);"
        "es.update(ids.map(function(id){var e=es.get(id);"
        "return {id:id,label:(e&&e.rel)||'',font:{color:'#e5e7eb',size:12,strokeWidth:4,"
        "strokeColor:'#090D16',align:'middle'}};}));}catch(e){}});"
        "network.on('deselectNode',function(){try{"
        "var es=network.body.data.edges;"
        "es.update(es.get().filter(function(e){return e.label;})"
        ".map(function(e){return {id:e.id,label:' '};}));}catch(e){}});"
        "}catch(e){console.warn('vgTune:',e);}}tune();})();"
        "</script>"
    )
    inject = style + script
    if "</head>" in html:
        html = html.replace("</head>", style + "</head>", 1)
        return html.replace("</body>", script + "</body>", 1) if "</body>" in html else html + script
    return html.replace("<body>", "<body>" + inject, 1) if "<body>" in html else inject + html


def _inject_legend(html: str, edge_types: dict, recent_note: bool = False) -> str:
    """Overlay a small fixed legend explaining edge colours + the two node colours."""
    present = []
    seen = set()
    for etype in edge_types.values():
        if etype not in seen:
            seen.add(etype)
            present.append(etype)
    # stable order following E_COLOR
    present.sort(key=lambda t: list(E_COLOR).index(t) if t in E_COLOR else 99)

    rows = []
    for et in present:
        col = E_COLOR.get(et, "#666c75")
        dash = "border-bottom:2px dashed" if et in _DASHED_TYPES else "border-bottom:3px solid"
        rows.append(
            f'<div style="margin:2px 0"><span style="display:inline-block;width:22px;'
            f'{dash} {col};vertical-align:middle"></span> '
            f'<span style="color:#e6e6e6">{_REL_LABEL.get(et, et)}</span></div>'
        )
    node_rows = (
        f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;height:14px;'
        f'border-radius:50%;background:{MAIN_COLOR};vertical-align:middle"></span>'
        f' <span style="color:#e6e6e6">your word — ● word · ◆ phrasal verb · ★ idiom</span></div>'
        f'<div style="margin:2px 0"><span style="color:#e6e6e6">related word — coloured by relation '
        f'(see below); the edge colour/dash tells the type</span></div>'
    )
    if recent_note:
        node_rows += (
            f'<div style="margin:2px 0"><span style="display:inline-block;width:14px;height:14px;'
            f'border-radius:50%;background:{MAIN_COLOR};border:3px solid {RECENT_BORDER};'
            f'vertical-align:middle"></span>'
            f' <span style="color:#e6e6e6">🆕 added in the latest session</span></div>'
        )
    # Collapsible: the container no longer blocks node clicks/drags (was pointer-events:none,
    # which hard-covered nodes underneath). The header is clickable (pointer-events:auto) and
    # toggles the body; the body itself stays pass-through so a node under it is still draggable.
    # MEANING SOURCE key (S16 T-G3): explains the tooltip's icon-only badge (matches the
    # Knowledge filter). Keeps the tooltip clean while the meaning is documented here.
    meaning_rows = (
        '<div style="margin:2px 0;color:#e6e6e6">Ⓦ WordNet dictionary · '
        '🤖 AI-defined (review-flagged) · ✍ edited by you</div>'
    )
    body = (
        '<div id="vgLegendBody" style="pointer-events:none;display:none">'   # S17: start collapsed
        '<div style="color:#9aa0a6;margin:6px 0 4px;font-weight:600">NODES</div>' + node_rows +
        '<div style="color:#9aa0a6;margin:6px 0 4px;font-weight:600">MEANING SOURCE</div>'
        + meaning_rows +
        '<div style="color:#9aa0a6;margin:6px 0 4px;font-weight:600">RELATIONS (arrow = direction)</div>'
        + "".join(rows) +
        '</div>'
    )
    legend = (
        '<div style="position:fixed;bottom:16px;left:16px;z-index:9998;pointer-events:none;'
        'background:rgba(9,13,22,0.96);border:1px solid #2b313a;border-radius:8px;'
        'padding:10px 14px;font:15px/1.6 system-ui,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,.4)">'
        '<div onclick="var b=document.getElementById(\'vgLegendBody\');'
        'var c=this.querySelector(\'span\');'
        'if(b.style.display===\'none\'){b.style.display=\'\';c.textContent=\'\\u25BE\';}'
        'else{b.style.display=\'none\';c.textContent=\'\\u25B8\';}" '
        'style="pointer-events:auto;cursor:pointer;color:#9aa0a6;font-weight:700;'
        'display:flex;justify-content:space-between;gap:14px;user-select:none">'
        '<span style="color:#e6e6e6">LEGEND</span><span>▸</span></div>'
        + body +
        '</div>'
    )
    return html.replace("<body>", "<body>" + legend, 1) if "<body>" in html else legend + html


def _chip(dim: str, val: str, label: str, col: str | None = None) -> str:
    """One filter chip. `dim`+`val` drive the JS state; colour follows the relation palette
    when given, else the neutral emerald surface."""
    if col:
        bg, fg, bd = _rgba(col, 0.16), col, _rgba(col, 0.55)
    else:
        bg, fg, bd = "#111827", "#F3F4F6", "rgba(16,185,129,0.5)"
    return (f'<span class="vgchip" data-dim="{dim}" data-val="{val}" '
            f'onclick="vgSet(\'{dim}\',\'{val}\')" style="cursor:pointer;padding:3px 10px;'
            f'border-radius:12px;background:{bg};color:{fg};border:1px solid {bd};'
            f'font-weight:600;margin:2px;display:inline-block">{label}</span>')


def _chip_row(title: str, dim: str, values: list) -> str:
    """A titled row: an 'All' chip + one chip per value (single-select in JS)."""
    if not values:
        return ""
    chips = [_chip(dim, "__all", "All")] + [_chip(dim, v, v) for v in values]
    return ('<div style="color:#94A3B8;font-weight:700;letter-spacing:.08em;font-size:12px;'
            f'margin:8px 0 2px">{title}</div><div>' + "".join(chips) + '</div>')


def _dd(title: str, dim: str, values: list, colors: dict | None = None,
        labels: dict | None = None) -> str:
    """One compact multiselect dropdown (<details> + checkboxes). Empty selection = All.
    Within a dimension the checked values OR together; across dimensions AND (unchanged)."""
    if not values:
        return ""
    rows = []
    for v in values:
        col = (colors or {}).get(v)
        lab = (labels or {}).get(v, v)
        swatch = (f'<span style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                  f'background:{col};margin-right:6px;vertical-align:middle"></span>' if col else "")
        rows.append(
            f'<label style="display:block;padding:2px 4px;cursor:pointer;white-space:nowrap">'
            f'<input type="checkbox" data-dim="{dim}" data-val="{v}" onchange="vgTog(this)" '
            f'style="accent-color:#10B981;margin-right:6px;vertical-align:middle">{swatch}{lab}'
            f'</label>')
    return (f'<details class="vgdd" style="display:inline-block;position:relative;margin:2px 4px 2px 0">'
            f'<summary id="vgdd_{dim}" style="cursor:pointer;list-style:none;padding:4px 10px;'
            f'border-radius:9px;background:#111827;color:#94A3B8;'
            f'border:1px solid rgba(255,255,255,0.08);font-size:13px;user-select:none">'
            f'{title} ▾</summary>'
            f'<div style="position:absolute;left:0;top:calc(100% + 4px);z-index:10000;'
            f'background:#0d1320;border:1px solid rgba(255,255,255,0.1);border-radius:10px;'
            f'padding:8px 10px;max-height:40vh;overflow:auto;box-shadow:0 4px 14px rgba(0,0,0,.6)">'
            + "".join(rows) + '</div></details>')


def _inject_controls(html: str, topics: list, relations: list,
                     types: list, sources: list, knowledge: list) -> str:
    """Inject the custom filter panel (top-left). Everything filters via the global vis
    `network` DataSets — the pyvis menus are off (T12), this is the single search/filter UI.

    S17 redesign (owner feedback: chips ate the screen): ONE search box that accepts
    MULTIPLE comma-separated words (prefix-per-word, e.g. "tr, jour" -> travel + journey)
    + one row of compact MULTISELECT dropdowns (Type/Tag/Source/Knowledge/Added/Relations).
    Within a dropdown: OR; across dropdowns: AND; nothing checked = All.
    Kept from T-G1/T-G2: the panel event guard (no pan from panel clicks) and the PAIR rule
    (an edge shows only with both endpoints; a relation view never shows orphans)."""
    tag_vals = sorted({t for t in topics if t and t != "vocab"})
    type_vals = sorted({t for t in types if t})
    src_vals = sorted({s for s in sources if s})
    kn_vals = [k for k in ("wordnet", "nowordnet", "conceptnet") if k in set(knowledge)]

    present = {r for r in relations if r and r != "category"}
    rels = [r for r in E_COLOR if r in present] + sorted(present - set(E_COLOR))

    # Palette matches the app (S14 T12): #090D16 base, hairline borders, emerald accent.
    control = (
        '<div id="vgPanel" style="position:fixed;top:12px;left:12px;z-index:9999;'
        'background:rgba(9,13,22,0.96);border:1px solid rgba(255,255,255,0.05);border-radius:12px;'
        'padding:10px 12px;max-width:520px;font:15px/1.5 system-ui,sans-serif;color:#F3F4F6;'
        'box-shadow:0 2px 12px rgba(0,0,0,.5)">'
        '<div style="margin-bottom:6px">'
        '<input id="vgQ" autocomplete="off" placeholder="🔍 word1, word2, … (prefix ok)" '
        'oninput="vgApply()" '
        'style="font:15px system-ui;padding:6px 10px;border-radius:10px;background:#111827;'
        'color:#F3F4F6;border:1px solid rgba(255,255,255,0.08);outline:none;width:260px">'
        '<button onclick="vgReset()" '
        'style="margin-left:8px;font:14px system-ui;padding:6px 11px;border-radius:10px;'
        'background:#111827;color:#94A3B8;border:1px solid rgba(255,255,255,0.08);'
        'cursor:pointer">reset</button></div><div>'
        + _dd("TYPE", "type", type_vals)
        + _dd("TAG", "tag", tag_vals)
        + _dd("SOURCE", "source", src_vals)
        + _dd("KNOWLEDGE", "knowledge", kn_vals)
        + _dd("ADDED", "added", ["today", "week"],
              labels={"today": "Today", "week": "This week"})
        + _dd("RELATIONS", "rel", rels, colors=E_COLOR,
              labels={r: _REL_LABEL.get(r, r) for r in rels})
        + '</div></div>'
    )
    script = (
        '<script type="text/javascript">'
        'var vgS={qs:[],type:[],tag:[],source:[],knowledge:[],added:[],rel:[]};var vgAdj=null;'
        'function vgAdjBuild(){var a={};network.body.data.edges.get().forEach(function(e){'
        '(a[e.from]=a[e.from]||[]).push(e.to);(a[e.to]=a[e.to]||[]).push(e.from);});return a;}'
        # multi-term search: "tr, jour" -> a node matches if ANY term prefix-matches ANY word
        # of its label (tr -> travel; jour -> journey).
        'function vgPre(v,q){return (""+v).toLowerCase().split(/\\s+/).some('
        'function(w){return w.indexOf(q)===0;});}'
        # match ANY node (main OR satellite) by label: a term matches if any search term
        # prefix-matches any word of the label (multiword ok: "get a divorce" matches "divorce")
        # OR the whole query is a substring (so "give up" as one query still hits).
        'function vgMatch(n){if(!vgS.qs.length)return true;var lab=(""+(n.label||"")).toLowerCase();'
        'return vgS.qs.some(function(q){return vgPre(n.label,q)||lab.indexOf(q)>=0;});}'
        # multiselect dims: nothing checked = All; checked values OR within a dim, AND across.
        'function vgDim(n){'
        'if(vgS.type.length&&vgS.type.indexOf(n.wtype||"")<0)return false;'
        'if(vgS.tag.length&&vgS.tag.indexOf(n.topic||"")<0)return false;'
        'if(vgS.source.length){var ss=(""+(n.srcs||"")).split(", ");'
        'if(!vgS.source.some(function(v){return ss.indexOf(v)>=0;}))return false;}'
        'if(vgS.knowledge.length){var ks=(""+(n.kn||"")).split(" ");'
        'if(!vgS.knowledge.some(function(v){return ks.indexOf(v)>=0;}))return false;}'
        'if(vgS.added.length){if(!n.added)return false;var dd=new Date(n.added);'
        'if(isNaN(dd))return false;var t=new Date();var okA=false;'
        'if(vgS.added.indexOf("today")>=0&&dd.toDateString()===t.toDateString())okA=true;'
        'if(vgS.added.indexOf("week")>=0){var df=(t-dd)/86400000;if(df<=7&&df>=-1)okA=true;}'
        'if(!okA)return false;}return true;}'
        'function vgApply(){try{'
        'vgS.qs=(document.getElementById("vgQ").value||"").toLowerCase().split(",")'
        '.map(function(s){return s.trim();}).filter(function(s){return s;});'
        'var ns=network.body.data.nodes,es=network.body.data.edges;'
        'var nodes=ns.get(),edges=es.get();if(!vgAdj)vgAdj=vgAdjBuild();'
        'var searching=vgS.qs.length>0,relOn=vgS.rel.length>0;'
        # dim filters gate MAIN words only (satellites inherit via the pair rule).
        'var dimP={};nodes.forEach(function(n){if(n.main===true)dimP[n.id]=vgDim(n);});'
        # S17: search now finds SATELLITE nodes too. A "seed" is any node the text matches
        # (a main must also pass the dropdown dims; a satellite just needs to match). With no
        # text, seeds = every dim-passing main.
        'var seed={};nodes.forEach(function(n){'
        'if(n.main===true){if(dimP[n.id]&&(!searching||vgMatch(n)))seed[n.id]=true;}'
        'else{if(searching&&vgMatch(n))seed[n.id]=true;}});'
        # while searching, pull in each seed\'s immediate neighbours for context (a matched
        # satellite shows its parent word; a matched word shows its relations).
        'var show={};Object.keys(seed).forEach(function(id){show[id]=true;'
        'if(searching)(vgAdj[id]||[]).forEach(function(nb){var nn=ns.get(nb);if(!nn)return;'
        'if(nn.main===true){if(dimP[nb])show[nb]=true;}else{show[nb]=true;}});});'
        # edges: both endpoints shown + relation filter. Touch marks satellites to keep in the
        # no-search pair-rule view.
        'var eVis={},touch={};edges.forEach(function(e){var relok=(!relOn||vgS.rel.indexOf(e.rel)>=0);'
        'var ok;if(searching){ok=relok&&!!show[e.from]&&!!show[e.to];}'
        'else{var ep=function(id){var n=ns.get(id);return n&&n.main===true?!!dimP[id]:true;};'
        'ok=relok&&ep(e.from)&&ep(e.to);}'
        'eVis[e.id]=ok;if(ok){touch[e.from]=true;touch[e.to]=true;}});'
        # finalize node visibility for the no-search view (pair rule); search view keeps `show`.
        'if(!searching){show={};nodes.forEach(function(n){if(n.main===true){'
        'show[n.id]=dimP[n.id]&&(!relOn||!!touch[n.id]);}else{show[n.id]=!!touch[n.id];}});}'
        'ns.update(nodes.map(function(n){return {id:n.id,hidden:!show[n.id]};}));'
        'es.update(edges.map(function(e){return {id:e.id,hidden:!eVis[e.id]};}));'
        '}catch(e){console.warn("vgApply:",e);}}'
        # multiselect plumbing: checkbox toggles push/remove the value; the dropdown's summary
        # shows the active count and turns emerald so an active filter is visible when closed.
        'function vgSync(){["type","tag","source","knowledge","added","rel"].forEach('
        'function(d){var s=document.getElementById("vgdd_"+d);if(!s)return;'
        'var n=vgS[d].length;var t=s.textContent.replace(/\\s*\\(\\d+\\)\\s*▾$/,"").replace(/\\s*▾$/,"");'
        's.textContent=t+(n?" ("+n+")":"")+" ▾";'
        's.style.color=n?"#10B981":"#94A3B8";'
        's.style.borderColor=n?"rgba(16,185,129,0.5)":"rgba(255,255,255,0.08)";});}'
        'function vgTog(el){var d=el.getAttribute("data-dim"),v=el.getAttribute("data-val");'
        'var a=vgS[d],i=a.indexOf(v);if(el.checked&&i<0)a.push(v);if(!el.checked&&i>=0)a.splice(i,1);'
        'vgSync();vgApply();}'
        'function vgReset(){document.getElementById("vgQ").value="";'
        '["type","tag","source","knowledge","added","rel"].forEach(function(d){vgS[d]=[];});'
        'var cb=document.querySelectorAll("#vgPanel input[type=checkbox]");'
        'for(var i=0;i<cb.length;i++)cb[i].checked=false;'
        'var dd=document.querySelectorAll("#vgPanel details");'
        'for(var j=0;j<dd.length;j++)dd[j].open=false;'
        'vgSync();vgApply();}'
        # close an open dropdown when another one opens (one at a time keeps the panel tidy)
        'document.addEventListener("toggle",function(e){if(!e.target.matches||'
        '!e.target.matches("#vgPanel details")||!e.target.open)return;'
        'var dd=document.querySelectorAll("#vgPanel details");'
        'for(var i=0;i<dd.length;i++){if(dd[i]!==e.target)dd[i].open=false;}},true);'
        # T-G1: keep pointer/mouse/touch events on the panel from reaching the vis canvas
        # (so clicking/holding a chip never pans/zooms the graph). stopPropagation only,
        # no preventDefault, so the input + chip clicks still work.
        'function vgGuard(){var p=document.getElementById("vgPanel");if(!p)return;'
        '["pointerdown","mousedown","touchstart","wheel","dblclick"].forEach(function(ev){'
        'p.addEventListener(ev,function(e){e.stopPropagation();},false);});}'
        'function vgInit(){if(typeof network==="undefined")return setTimeout(vgInit,120);'
        'vgAdj=vgAdjBuild();vgGuard();vgSync();}vgInit();'
        '</script>'
    )
    inject = control + script
    return html.replace("<body>", "<body>" + inject, 1) if "<body>" in html else inject + html


def build_render_graph(units, run_id: str | None = None, title: str = "Vocab graph",
                       recent: list | None = None) -> str:
    """Build + write the pyvis HTML. Returns the html path under output/<run>/.

    `recent` = Node keys added or grown in the latest session; those main nodes get a gold
    ring + a "🆕 added this session" tooltip line so the learner can see, on the cumulative
    graph, exactly what the most recent Mine contributed (the graph grows — this shows WHERE).
    """
    args = {"run_id": run_id, "title": title}
    recent_set = {str(k) for k in (recent or [])}
    try:
        from pyvis.network import Network
    except Exception as e:
        log_tool_call("build_render_graph", args, error=f"pyvis unavailable: {e}")
        raise

    nodes = _coerce_nodes(units)
    run_id = run_id or new_run_id()
    out = run_dir(run_id)
    html_path = os.path.join(out, "graph.html")

    # --- 1. assemble a networkx graph (for Louvain) ---
    G = nx.Graph()
    term_to_main = {}            # lowercased term -> main node id (for cross-linking)
    meta = {}                    # node id -> {label, size, shape, title}
    edge_types = {}              # (a,b) -> relation type
    node_topic = {}              # node id -> topic (from tags) — drives the clean Topic filter

    for nd in nodes:
        try:
            main_id = nd.get("key") or nd.get("term")
            if not main_id:
                continue
            term = nd.get("term", main_id)
            wtype = nd.get("word_type") or "word"
            is_recent = main_id in recent_set
            # Meaning-source badge (S16 T-G3) — ICON ONLY (legend explains it), by
            # source_map['definition'] (3 mutually-exclusive states). Other AI-authored
            # fields (mnemonic/tags/sense-choice) do NOT change this badge.
            smap = nd.get("source_map") or {}
            badge = {"wordnet": "Ⓦ", "ai": "🤖", "user": "✍"}.get(smap.get("definition", ""), "")
            # TOOLTIP = max 4 lines: term[wtype]·badge / definition / 📍source (phrases only) / 🆕.
            # sense_id + category deliberately dropped (looked up in the app's Sense browser).
            # S17: newline separator instead of "<br>" — some vis.js builds render the title
            # as TEXT (the learner saw literal "<br>"); "\n" + CSS white-space:pre-line is
            # correct in both modes. Word class always shown: multi-word type in brackets,
            # plus the WordNet pos (noun/verb/adj) when known.
            # Word class, always shown and CONSISTENT (S17 owner fix): a multi-word term with
            # a space that WordNet tagged as a verb is a phrasal verb / collocation — surface
            # that instead of a bare "(verb)" so 'give up'/'go on' don't read as plain verbs.
            pos = (nd.get("pos") or "").strip()
            wt = wtype
            if wt == "word" and " " in str(term).strip():
                wt = "phrasal verb" if pos == "verb" else "phrase"
            line1 = term
            if wt and wt != "word":
                line1 += f" [{wt.replace('_', ' ')}]"
            if pos:
                line1 += f" ({pos})"
            if badge:
                line1 += f" · {badge}"
            if is_recent:
                line1 = "🆕 " + line1
            tip = line1
            # CONSISTENT tooltip body (S17): every node shows BOTH a definition line AND an
            # example line when it has them (previously some showed one, some the other).
            definition = (nd.get("definition") or "").strip()
            if definition:
                tip += "\n📖 " + (definition[:140] + "…" if len(definition) > 140 else definition)
            example = ""
            for o in (nd.get("occurrences") or []):
                s = (o.get("sentence") or "").strip() if isinstance(o, dict) else ""
                if s:
                    example = s
                    break
            if example:
                tip += "\n💬 " + (example[:120] + "…" if len(example) > 120 else example)
            # DISTINCT occurrence sources (used by the tooltip AND the Source filter).
            srcs, _seen_src = [], set()
            for o in (nd.get("occurrences") or []):
                s = (o.get("source") or "").strip()
                if s and s not in _seen_src:
                    _seen_src.add(s)
                    srcs.append(s)
            # Provenance line: plain words occur everywhere (noise) — show 📍 only for
            # idioms/phrasal verbs/collocations/slang, max 2 (full ledger = infolog).
            if wtype != "word":
                for s in srcs[:2]:
                    tip += f"\n📍 {s}"
                if len(srcs) > 2:
                    tip += f"\n… (+{len(srcs) - 2} more)"
            # added_at of the first occurrence -> the Added (today/this week) filter (T-G2).
            added_at = ""
            for o in (nd.get("occurrences") or []):
                if o.get("added_at"):
                    added_at = str(o.get("added_at"))
                    break
            # Knowledge class (T-G2 filter): nowordnet if the key is a #nowordnet key, else
            # wordnet; plus 'conceptnet' when the node carries a ConceptNet life-context edge.
            nowordnet = "#nowordnet" in str(main_id).lower()
            has_cn = any(e.get("type") in ("used_for", "has_context")
                         for e in (nd.get("edges") or []))
            kn = ("nowordnet" if nowordnet else "wordnet") + (" conceptnet" if has_cn else "")
            # NOTE: the sentence text is deliberately NOT shown on the graph — it lives in
            # the infolog (infolog_export) to keep the graph light. Tooltip = where, not what.
            # Topic group (for the filter menu), strongest signal first:
            #   user/AI tags[0]  ->  ConceptNet has_context (top weight)  ->  WordNet category.
            group = None
            tags = nd.get("tags") or []
            if tags:
                group = tags[0]
            if not group:
                for e in nd.get("edges", []) or []:   # edges already weight-sorted by conceptnet_lookup
                    if e.get("type") == "has_context" and e.get("target"):
                        group = e["target"]
                        break
            group = group or nd.get("category") or "vocab"
            G.add_node(main_id)
            meta[main_id] = {"label": term, "size": 26, "shape": "dot", "title": tip,
                             "group": group, "wtype": wtype, "recent": is_recent,
                             "srcs": ", ".join(srcs), "kn": kn, "added": added_at}
            node_topic[main_id] = group          # a main word's topic = its strongest tag
            term_to_main[term.strip().lower()] = main_id
        except Exception as e:
            log_tool_call("build_render_graph", args, error=f"skip node: {e}")
            continue

    # second pass: edges (now that all main nodes/terms are known for cross-linking)
    for nd in nodes:
        main_id = nd.get("key") or nd.get("term")
        if main_id not in meta:
            continue
        for e in nd.get("edges", []):
            etype = e.get("type")
            target = (e.get("target") or "").strip()
            if not target or etype == "category":   # category is an attribute, not a visual node
                continue
            tgt_key = target.lower()
            if tgt_key in term_to_main:              # link to an existing main node (graph connects)
                tgt_id = term_to_main[tgt_key]
            else:
                tgt_id = f"{etype}::{tgt_key}"
                if tgt_id not in meta:
                    G.add_node(tgt_id)
                    meta[tgt_id] = {"label": target, "size": 12, "shape": "dot",
                                    "title": etype, "group": etype}
            if tgt_id == main_id:
                continue
            # a satellite inherits its parent word's topic, so filtering "finance" also
            # shows that word's relations (first parent wins for shared satellites).
            node_topic.setdefault(tgt_id, node_topic.get(main_id, "vocab"))
            G.add_edge(main_id, tgt_id)
            edge_types[(main_id, tgt_id)] = etype

    # --- 2. Louvain communities (computed for the topic filter; NOT used for colour) ---
    comm = _louvain(G)
    main_ids = set(term_to_main.values())     # the learner's vocabulary (uniform colour)
    deg = dict(G.degree())                     # Obsidian/Logseq look: node size scales with degree

    # --- 3. render with pyvis (dark canvas, spaced-out, DIRECTED) ---
    #   directed              -> arrows show relation direction (tusk --part_of--> elephant)
    #   select_menu/filter_menu OFF (S14 T12): the raw Bootstrap dropdowns duplicated the
    #   custom overlay (search box + relation chips in _inject_controls), which is now the
    #   single search/filter UI, styled to the app palette.
    #   neighborhood_highlight-> click a word: light up its related cluster, dim the rest
    net = Network(height="100vh", width="100%", bgcolor="#090D16",
                  font_color="#e6e6e6", notebook=False, cdn_resources="in_line", directed=True,
                  select_menu=False, filter_menu=False, neighborhood_highlight=True)
    # Compact, well-tested layout (the look the learner preferred). Physics is FROZEN after the
    # initial stabilization (see _inject_tweaks), so dragging a node moves only that node — it
    # never drags the whole cluster/canvas — and nothing drifts on its own.
    net.barnes_hut(gravity=-8000, central_gravity=0.25, spring_length=200,
                   spring_strength=0.02, overlap=1)
    for nid, m in meta.items():
        # Main words: ONE uniform colour. Related (satellite) words: SAME colour as the
        # relation edge pointing to them (satellite's group == its relation type), so each
        # relation type reads as a single colour across both its edge AND its target node.
        is_main = nid in main_ids
        if is_main:
            wtype = m.get("wtype", "word")
            color, shape, rel = MAIN_COLOR, _MAIN_WT_SHAPE.get(wtype, MAIN_SHAPE), ""
        else:
            wtype = ""
            et = m.get("group")
            # related/satellite nodes: colour (+ the edge's dash) already encodes the relation,
            # so shape adds no info — keep them all plain dots (learner's request).
            # Leaf nodes fade back (alpha ~0.55) so the learner's hub words carry the scene (T13).
            color, shape, rel = _rgba(E_COLOR.get(et, SAT_COLOR), 0.55), "dot", (et or "")
        # custom `topic`/`rel`/`wtype` attrs (NOT pyvis `group`, which would override our colour):
        # filter_menu can filter by `topic` (word level), `rel` (relation), or `wtype` (idiom/
        # phrasal verb). Label font is bumped globally after render (pyvis overrides per-node font).
        # A node from the LATEST session keeps its fill colour but gains a gold ring (req: show
        # what this session added to the growing graph).
        # `value` = degree drives Obsidian-style size scaling (well-connected words read bigger);
        # main words get a floor so a brand-new isolated word is still clearly visible.
        val = deg.get(nid, 1) + (2 if is_main else 0)
        # Obsidian look: the learner's HUB words get a big bright label; related (leaf) words get a
        # smaller, dimmer label so dozens of neighbours don't collide into an unreadable blob.
        font = ({"size": 20, "color": "#f1f5f9", "strokeWidth": 5, "strokeColor": "#090D16"}
                if is_main else
                {"size": 11, "color": "#94a3b8", "strokeWidth": 3, "strokeColor": "#090D16"})
        node_kwargs = dict(label=m["label"], title=m["title"], color=color, value=val, font=font,
                           shape=shape, topic=node_topic.get(nid, "vocab"), rel=rel, wtype=wtype)
        # Filter metadata (T-G2): main nodes carry their source list / knowledge class /
        # added date + a `main` flag so the JS filters act only on learner words (satellites
        # follow the pair rule). Additive vis.js attrs — physics untouched.
        if is_main:
            node_kwargs.update(main=True, srcs=m.get("srcs", ""), kn=m.get("kn", ""),
                               added=m.get("added", ""))
        if is_main and m.get("recent"):
            node_kwargs["color"] = {"background": color, "border": RECENT_BORDER}
            node_kwargs["borderWidth"] = 4
        # Hub glow (T13, Obsidian/Eltiverse look): recent/gold-ring hubs glow emerald,
        # ordinary hubs a faint white halo. Static per-node option — no physics touched.
        if is_main:
            glow = "rgba(16,185,129,0.35)" if m.get("recent") else "rgba(241,245,249,0.18)"
            node_kwargs["shadow"] = {"enabled": True, "color": glow, "size": 18, "x": 0, "y": 0}
        net.add_node(nid, **node_kwargs)
    for (a, b), etype in edge_types.items():
        # symmetric relations (synonym/antonym) get no arrowhead; directional ones do.
        # Thin, soft edges (Obsidian/Logseq look) — relation COLOUR is kept (pedagogy), just lighter.
        directional = etype not in ("synonym", "antonym")
        # Silk-thread edges (T13): width 0.5 + ~0.6 alpha; hover/selection widths stay
        # (set globally in _inject_tweaks). Relation colour is kept, just translucent.
        net.add_edge(a, b, color=_rgba(E_COLOR.get(etype, "#666c75"), 0.6), title=etype, rel=etype,
                     width=0.5, dashes=etype in _DASHED_TYPES, arrows="to" if directional else "")

    html = net.generate_html()
    # Bump every label to a big, readable, outlined font (pyvis ignores a per-node `font`,
    # so patch the global node-font it emitted).
    html = html.replace(
        '"font": {"color": "#e6e6e6"}',
        '"font": {"color": "#ffffff", "size": 26, "strokeWidth": 5, "strokeColor": "#090D16"}',
    )
    html = _inject_legend(html, edge_types, recent_note=bool(recent_set))   # colour/shape reference
    # Custom overlay = the ONLY search/filter UI (pyvis select/filter menus are off — T12).
    _types = {m.get("wtype") for nid, m in meta.items() if nid in main_ids and m.get("wtype")}
    _sources = {s for nid, m in meta.items() if nid in main_ids
                for s in (m.get("srcs", "").split(", ") if m.get("srcs") else [])}
    _knowledge = {tok for nid, m in meta.items() if nid in main_ids
                  for tok in (m.get("kn", "").split() if m.get("kn") else [])}
    html = _inject_controls(html, sorted(set(node_topic.values())),
                            sorted(set(edge_types.values())),
                            sorted(_types), sorted(_sources), sorted(_knowledge))
    html = _inject_tweaks(html)   # small-screen: responsive height, tooltip cap, drag/physics fix
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    log_tool_call("build_render_graph", args,
                  result={"html": html_path, "nodes": G.number_of_nodes(),
                          "edges": G.number_of_edges(), "communities": len(set(comm.values()))})
    return html_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import load_graph
    g = load_graph()
    print("graph ->", build_render_graph(g, title="Demo"))
