"""
Block Selector Panel (Dash + Plotly)

DATA CONTRACT (you provide this to the builder)
-----------------------------------------------
A list of dicts, one per file. All times are in **seconds** already.

records = [
    {
        "file_name": "rat01_day1.adicht",
        "blocks_s": [22134.0, 474.5, 2579.0],      # list[float] block lengths (seconds)
        "selected": 2,                              # int, 0-based selected block index
        "comments_by_block": {                      # dict[int, list[float]] (seconds RELATIVE to block start)
            0: [12.3, 250.0],
            2: [5.0, 90.1]
        }
    },
    ...
]

WHAT THIS BUILDER RETURNS
-------------------------
make_block_selector_panel(records, panel_id="blocksel") -> dash.html.Div

- A scrollable container with:
  * one dcc.Graph per file (id={'type':'blocksel_graph','file': <file_name>})
  * an "Accept" button  (id='blocksel_accept')
  * a "Cancel" button   (id='blocksel_cancel')
- Each graph:
  * draws blocks sequentially along x (seconds)
  * colors: light blue (unselected), yellow (selected)
  * draws small black vertical lines for comment times
  * carries customdata on each block bar: {'file': file_name, 'block': j}

WHAT YOU STILL WIRE (in app.py)
-------------------------------
1) A pattern-matching callback on Input({'type':'blocksel_graph','file': ALL}, 'clickData')
   - Read clicked block index from trace's customdata -> update your selection store
   - Rebuild ONLY that file's graph using build_block_row_figure(record, selected_override=<new_index>)

2) Accept/Cancel buttons:
   - '{panel_id}_accept' -> apply the selection to your pipeline (e.g., index_df["block"]) 
   - '{panel_id}_cancel' -> close panel / discard changes
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from dash import html, dcc
import plotly.graph_objects as go

# If you have this in your project, leave import; otherwise remove or adapt.
from backend.adi_parse import AdiParse

# ---------------------------
# Appearance (tweak as needed)
# ---------------------------
UNSELECTED_COLOR = "#77BBEB"   # light blue
SELECTED_COLOR   = "#FBC02D"   # yellow
BAR_EDGE_COLOR   = "#EFEFEF"   # light gray edge
COMMENT_LINE     = dict(color="#000000", width=3)  # black vertical ticks


# ---------------------------
# Helpers
# ---------------------------

def _normalize_comments_by_block(cb: Optional[dict]) -> Dict[int, List[float]]:
    """Ensure dict keys are ints (dcc.Store/JSON serializes keys to strings).

    Parameters
    ----------
    cb : dict or None
        Mapping of block index -> list of comment times (relative to that block).

    Returns
    -------
    dict[int, list[float]]
    """
    out: Dict[int, List[float]] = {}
    if not cb:
        return out
    for k, v in cb.items():
        try:
            ik = int(k)
        except Exception:
            continue
        if v is None:
            continue
        out[ik] = v
    return out


# ------------------------------------------
# Data loader (build records for the selector)
# ------------------------------------------

def get_block_data(folder_path: str, channel_structures: dict) -> List[Dict]:
    """
    Scan `folder_path` for .adicht files and build a list of block records
    ready for the block selector.

    Returns
    -------
    list[dict]
        Each record has:
          - file_name : str
          - blocks_s : list[float]  # block lengths in seconds
          - selected : int          # default block index to highlight
          - comments_by_block : dict[int, list[float]]  # comment times per block (in seconds, RELATIVE to block start)
    """
    records: List[Dict] = []

    for root, _, files in os.walk(folder_path):
        for file in (f for f in files if f.endswith(".adicht")):
            fpath = os.path.join(root, file)

            # Build block info using AdiParse
            adi_parse = AdiParse(fpath, channel_structures)
            block_info = adi_parse.get_block_coms()

            # Normalize comment keys early so everything downstream sees int keys
            cb = _normalize_comments_by_block(block_info.get("comments_by_block"))

            records.append({
                "file_name": os.path.basename(fpath),
                "blocks_s": block_info.get("blocks_s", []) or [],
                "selected": int(block_info.get("selected", 0) or 0),
                "comments_by_block": cb,
            })

    return records


# ------------------------------------
# Figure builder for a single file row
# ------------------------------------

def build_block_row_figure(
    record: Dict,
    selected_override: Optional[int] = None,
    row_height_px: int = 110,
    gap_s: Optional[float] = None,
) -> go.Figure:
    """
    Build a Plotly figure for one file's blocks.

    Parameters
    ----------
    record : dict
        One file record as specified in the DATA CONTRACT above.
    selected_override : int or None
        If given, overrides record["selected"] for highlighting.
    row_height_px : int
        Fixed height (px) for the figure.
    gap_s : float or None
        If given, fixed gap (seconds) between blocks; otherwise auto-computed.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    file_name: str = record.get("file_name", "")
    blocks: List[float] = list(record.get("blocks_s", []) or [])
    n = len(blocks)

    # Selected block index
    sel = int(record.get("selected", 0) or 0)
    if selected_override is not None:
        sel = int(selected_override)
    if n == 0 or sel < 0 or sel >= n:
        sel = 0

    # Spacing between blocks on x so they never stack on y
    total_no_gap = float(sum(blocks)) if blocks else 0.0
    if gap_s is None:
        gap_s = min(max(total_no_gap * 0.01, 2.0), 30.0)  # ~1%, clamped to 2..30s

    starts = [0.0]
    for j in range(1, n):
        starts.append(starts[-1] + blocks[j - 1] + gap_s)

    fig = go.Figure()

    # Bars: one y row; explicit bar width keeps all bars on same y=0 "lane"
    bar_width = 0.5
    for j in range(n):
        fig.add_bar(
            x=[blocks[j]],
            y=[0],
            base=starts[j],
            width=bar_width,
            orientation="h",
            marker=dict(
                color=SELECTED_COLOR if j == sel else UNSELECTED_COLOR,
                line=dict(color=BAR_EDGE_COLOR, width=2),
            ),
            opacity=0.80,  # Slight transparency so overlays are always visible
            hovertemplate=(
                f"<b>{file_name}</b><br>"
                f"Block: B{j}<br>"
                "Length: %{x:.6f} s<br>"
                "Start: %{base:.6f} s<extra></extra>"
            ),
            name=f"B{j}",
            offsetgroup=file_name,
            customdata=[{"file": file_name, "block": j}],
        )

    # Comment ticks: draw as a single scatter trace (last), thicker, tiny markers
    comments_by_block = _normalize_comments_by_block(record.get("comments_by_block"))

    xs: List[float] = []
    ys: List[float] = []
    for j in range(n):
        start_j = starts[j]
        for t in (comments_by_block.get(j, []) or []):
            try:
                x = start_j + float(t)  # t is RELATIVE to this block's start
            except Exception:
                continue
            # short vertical segment + separator
            xs.extend([x, x, None])
            ys.extend([-0.48, 0.48, None])

    if xs:  # only add the layer if comments exist for this file
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=COMMENT_LINE,
            marker=dict(size=4),
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,  # never clip at axes bounds
        )

    # Layout & axes
    total_with_gaps = (starts[-1] + (blocks[-1] if blocks else 0.0)) if n else 1.0
    fig.update_layout(
        height=row_height_px,
        margin=dict(l=10, r=20, t=26, b=10),
        title=dict(text=file_name, x=0.01, y=0.95, font=dict(size=12)),
        showlegend=False,
        barmode="overlay",
        bargap=0.0,
        transition_duration=0,
    )
    fig.update_xaxes(
        title_text="seconds",
        range=[0, total_with_gaps],
        showgrid=True,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        type="linear",
        range=[-0.5, 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )

    return fig


# ----------------------------------------
# Panel builder (graphs + Accept/Cancel UI)
# ----------------------------------------

def make_block_selector_panel(
    records: List[Dict],
    panel_id: str = "blocksel",
    max_height: str = "70vh",
    row_height_px: int = 110,
):
    """
    Create a scrollable panel with one graph per file and Accept/Cancel buttons.

    IDs you will use in callbacks:
      - Graph per row:   {'type': f'{panel_id}_graph', 'file': <file_name>}
      - Accept button:   f'{panel_id}_accept'
      - Cancel button:   f'{panel_id}_cancel'
    """
    rows: List[dcc.Graph] = []
    for rec in records:
        fname = rec.get("file_name", "")
        fig = build_block_row_figure(rec, row_height_px=row_height_px, gap_s=1.0)
        graph = dcc.Graph(
            id={"type": f"{panel_id}_graph", "file": fname},
            figure=fig,
            config={"displayModeBar": False},  # keep it clean; toggle if you want zoom tools visible
            style={"height": f"{row_height_px}px"},
        )
        rows.append(graph)

    controls = html.Div(
        [
            html.Button("Cancel", id=f"{panel_id}_cancel", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Accept", id=f"{panel_id}_accept", n_clicks=0),
        ],
        style={"display": "flex", "justifyContent": "flex-end", "gap": "8px", "marginTop": "8px"},
    )

    return html.Div(
        [
            html.Div(
                rows,
                id=f"{panel_id}_rows",
                style={"maxHeight": max_height, "overflowY": "auto", "paddingRight": "6px"},
            ),
            controls,
        ],
        id=f"{panel_id}_panel",
        style={"width": "100%"},
    )
