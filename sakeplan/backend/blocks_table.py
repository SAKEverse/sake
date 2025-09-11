# backend/block_selector.py

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
  * one dcc.Graph per file (id={'type':'blocksel-graph','file': <file_name>})
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
   - Rebuild ONLY that file's graph using build_block_row_figure(record, selected=<new>)

2) Accept/Cancel buttons:
   - 'blocksel-accept' -> apply the selection to your pipeline (index_df["block"])
   - 'blocksel-cancel' -> close panel / discard changes

"""

import os
from typing import Dict, List, Optional
from dash import html, dcc
import plotly.graph_objects as go
from backend.adi_parse import AdiParse

# similar to get_File data but uses record) AdiParse.get_block_coms()
def get_block_data(folder_path: str, channel_structures: dict):
    """
    Scan folder_path for .adicht files and build a list of block records ready for the block selector.

    Returns
    -------
    list[dict]
        Each record has:
          - file_name : str
          - blocks_s : list[float]  # block lengths in seconds
          - selected : int          # default block index to highlight
          - comments_by_block : dict[int, list[float]]  # comment times per block (in seconds)
    """
    records = []
    for root, dirs, files in os.walk(folder_path):
        filelist = [f for f in files if f.endswith(".adicht")]
        for file in filelist:
            fpath = os.path.join(root, file)

            # build block info using AdiParse
            adi_parse = AdiParse(fpath, channel_structures)
            block_info = adi_parse.get_block_coms()  # <-- must return dict with keys below

            records.append({
                "file_name": os.path.basename(fpath),
                "blocks_s": block_info.get("blocks_s", []),
                "selected": block_info.get("selected", 0),
                "comments_by_block": block_info.get("comments_by_block", {}),
            })

    return records

# ---- Appearance (change if you want different styling) ----
UNSELECTED_COLOR = "#77BBEB"   # light blue
SELECTED_COLOR   = "#FBC02D"   # yellow
BAR_EDGE_COLOR   = "#F2EFEF"   # black
COMMENT_LINE     = dict(color="#000000", width=1)  # black vertical ticks


def build_block_row_figure(
    record: Dict,
    selected_override: Optional[int] = None,
    row_height_px: int = 110,
    ) -> go.Figure:
    """
    Build a single-row horizontal bar figure for one file.

    Parameters
    ----------
    record : dict
        One of the records from the DATA CONTRACT above.
    selected_override : int or None
        If provided, use this index as the selected block (ignores record['selected']).
    row_height_px : int
        Fixed pixel height for this row figure.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    file_name: str = record["file_name"]
    blocks: List[float] = list(record.get("blocks_s", []))
    n = len(blocks)
    sel = int(record.get("selected", 0))
    if selected_override is not None:
        sel = int(selected_override)
    if n == 0:
        sel = 0
    elif sel < 0 or sel >= n:
        sel = 0

    # cumulative starts so blocks place sequentially
    starts = [0.0]
    for j in range(1, n):
        starts.append(starts[-1] + blocks[j - 1])

    fig = go.Figure()

    # one bar per block
    for j in range(n):
        fig.add_bar(
            x=[blocks[j]],
            y=[0],                    # single row at y=0 (continuous axis)
            base=starts[j],
            orientation="h",
            marker=dict(
                color=SELECTED_COLOR if j == sel else UNSELECTED_COLOR,
                line=dict(color=BAR_EDGE_COLOR, width=1),
            ),
            hovertemplate=(
                f"<b>{file_name}</b><br>"
                f"Block: B{j}<br>"
                "Length: %{x:.6f} s<br>"
                "Start: %{base:.6f} s<extra></extra>"
            ),
            name=f"B{j}",
            customdata=[{"file": file_name, "block": j}],
        )

    # comment vertical ticks (seconds are relative to block start)
    comments_by_block: Dict[int, List[float]] = record.get("comments_by_block", {}) or {}
    shapes = []
    for j in range(n):
        block_comments = comments_by_block.get(j, []) or []
        start_j = starts[j]
        for t in block_comments:
            x = start_j + float(t)
            shapes.append(
                dict(
                    type="line",
                    xref="x", yref="y",
                    x0=x, x1=x,
                    y0=-0.35, y1=0.35,   # spans the bar vertically
                    line=COMMENT_LINE,
                )
            )

    # layout
    total = sum(blocks) if blocks else 1.0
    fig.update_layout(
        height=row_height_px,
        margin=dict(l=10, r=20, t=26, b=10),
        title=dict(text=file_name, x=0.01, y=0.95, font=dict(size=12)),
        showlegend=False,
        shapes=shapes,
        bargap=0.0,
    )
    fig.update_xaxes(
        title_text="seconds",
        range=[0, total],
        showgrid=True,
        zeroline=False,
    )
    # continuous y centered at 0, hide ticks
    fig.update_yaxes(
        range=[-0.5, 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    # keep interactions simple: click to select; zoom optional per-row
    fig.update_layout(transition_duration=0)

    # disable zoom on x-axis
    fig.update_xaxes(fixedrange=True)

    return fig


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

    Parameters
    ----------
    records : list[dict]
        As specified in the DATA CONTRACT above.
    panel_id : str
        Namespace prefix for component IDs.
    max_height : str
        CSS max-height for the scroll container (e.g., "70vh").
    row_height_px : int
        Fixed height (px) for each row figure.

    Returns
    -------
    dash.html.Div
        Container ready to insert in your layout or modal.
    """
    rows = []
    for rec in records:
        fname = rec["file_name"]
        fig = build_block_row_figure(rec, row_height_px=row_height_px)
        graph = dcc.Graph(
            id={"type": f"{panel_id}_graph", "file": fname},
            figure=fig,
            config={"displayModeBar": False},  # keep it clean; enable if you want zoom tools visible
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
