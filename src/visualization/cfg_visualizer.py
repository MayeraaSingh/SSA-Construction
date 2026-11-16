"""Control Flow Graph visualizer.

Provides graphical visualization of control flow graphs using NetworkX
for graph structure and Matplotlib for rendering.  Supports both the
original (pre-SSA) CFG and the SSA-form CFG with phi functions.

Diagrams are saved as high-resolution PNG files suitable for inclusion in
reports.

Usage::

    from src.visualization.cfg_visualizer import visualize_cfg, visualize_ssa

    visualize_cfg(cfg, "output/cfg.png")
    visualize_ssa(ssa_builder, "output/ssa.png")
"""

from __future__ import annotations

from pathlib import Path

from src.ir.basic_block import BasicBlock, TerminationType
from src.ir.control_flow_graph import ControlFlowGraph
from src.ir.instructions import (
    AssignmentInstruction,
    ConditionalBranchInstruction,
    UnconditionalJumpInstruction,
)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output.
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    _VIZ_AVAILABLE = True
    _VIZ_ERROR = ""
except ImportError as exc:
    _VIZ_AVAILABLE = False
    _VIZ_ERROR = str(exc)

# ---------------------------------------------------------------------------
# SSA type imports
# ---------------------------------------------------------------------------

from src.ssa.ssa_builder import (
    SSAAssignment,
    SSAConditionalBranch,
    SSABuilder,
    SSAUnconditionalJump,
)
from src.ssa.phi_functions import PhiFunction


class VisualizationError(Exception):
    """Raised when visualization cannot proceed (e.g. missing dependencies)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def visualize_cfg(
    cfg: ControlFlowGraph,
    output_path: str | Path,
    title: str = "Control Flow Graph",
) -> Path:
    """Render the original (pre-SSA) CFG and save as a PNG file.

    Args:
        cfg: The control flow graph to render.
        output_path: Path where the PNG file will be saved.
        title: Title displayed on the figure.

    Returns:
        The resolved path of the saved file.

    Raises:
        VisualizationError: If matplotlib or networkx is not installed.
    """
    _check_deps()
    graph = _build_graph(cfg)
    pos = _compute_layout(cfg, graph)
    node_labels = {bid: _format_cfg_label(cfg.get_block(bid)) for bid in cfg.blocks}
    edge_labels = _get_cfg_edge_labels(cfg)
    node_colors = _get_node_colors(cfg)
    backedges = get_backedges(cfg)
    return _render(
        graph, pos, node_labels, edge_labels, node_colors, backedges,
        output_path, title, cfg,
    )


def visualize_ssa(
    builder: SSABuilder,
    output_path: str | Path,
    title: str = "SSA Control Flow Graph",
) -> Path:
    """Render the SSA-form CFG and save as a PNG file.

    Phi functions are displayed at the top of each block's label.

    Args:
        builder: The SSABuilder holding the SSA-form CFG.
        output_path: Path where the PNG file will be saved.
        title: Title displayed on the figure.

    Returns:
        The resolved path of the saved file.

    Raises:
        VisualizationError: If matplotlib or networkx is not installed.
    """
    _check_deps()
    cfg = builder._cfg
    graph = _build_graph(cfg)
    pos = _compute_layout(cfg, graph)
    node_labels = {
        bid: _format_ssa_label(cfg.get_block(bid), builder) for bid in cfg.blocks
    }
    edge_labels = _get_ssa_edge_labels(builder)
    node_colors = _get_node_colors(cfg)
    backedges = get_backedges(cfg)
    return _render(
        graph, pos, node_labels, edge_labels, node_colors, backedges,
        output_path, title, cfg,
    )


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _check_deps() -> None:
    """Raise VisualizationError if matplotlib or networkx is not available."""
    if not _VIZ_AVAILABLE:
        raise VisualizationError(
            f"Visualization dependencies not installed: {_VIZ_ERROR}\n"
            "Install with: pip install networkx matplotlib"
        )


# ---------------------------------------------------------------------------
# Backedge detection — DFS-based (correct)
# ---------------------------------------------------------------------------


def get_backedges(cfg: ControlFlowGraph) -> set[tuple[int, int]]:
    """Identify true loop backedges using DFS grey-node classification.

    An edge u -> v is a backedge when v is an *ancestor* of u in the DFS
    tree, i.e. v is currently on the DFS stack (grey) when the edge is
    first encountered.  This is the standard algorithm-textbook definition
    and is correct for any directed graph — it does not rely on block IDs.

    Returns:
        A set of (src_id, dst_id) pairs that are genuine backedges.
    """
    if cfg.entry_block is None:
        return set()

    backedges: set[tuple[int, int]] = set()
    # DFS colouring: white (unvisited), grey (on stack), black (done).
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[int, int] = {bid: WHITE for bid in cfg.blocks}

    def dfs(u: int) -> None:
        color[u] = GREY
        block = cfg.blocks[u]
        for v in block.successors:
            if v not in color:
                continue  # Safety: ignore edges to blocks not in CFG.
            if color[v] == GREY:
                backedges.add((u, v))   # Back edge: v is an ancestor.
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    dfs(cfg.entry_block.block_id)
    return backedges


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph(cfg: ControlFlowGraph) -> "nx.DiGraph":
    """Build a NetworkX directed graph from the CFG."""
    graph = nx.DiGraph()
    for block_id in cfg.blocks:
        graph.add_node(block_id)
    for block in cfg.blocks.values():
        for succ_id in block.successors:
            graph.add_edge(block.block_id, succ_id)
    return graph


# ---------------------------------------------------------------------------
# Layout — layered BFS, skipping backedges
# ---------------------------------------------------------------------------


def _compute_layout(
    cfg: ControlFlowGraph,
    graph: "nx.DiGraph",
) -> dict[int, tuple[float, float]]:
    """Compute node positions using a top-down layered layout.

    Depth (layer) is computed by BFS from the entry block following only
    forward edges (backedges excluded so loop headers sit above their bodies).
    Blocks at the same depth are spread horizontally with enough spacing
    to accommodate their label widths.
    """
    if not cfg.blocks:
        return {}

    entry_id = cfg.entry_block.block_id if cfg.entry_block else min(cfg.blocks)
    backedges = get_backedges(cfg)

    # BFS over non-backedge edges to assign depths.
    depths: dict[int, int] = {entry_id: 0}
    queue: list[int] = [entry_id]
    while queue:
        current = queue.pop(0)
        block = cfg.blocks[current]
        for succ_id in block.successors:
            if (current, succ_id) not in backedges and succ_id not in depths:
                depths[succ_id] = depths[current] + 1
                queue.append(succ_id)

    # Assign unreachable blocks to a layer below everything.
    max_known = max(depths.values(), default=0)
    for bid in cfg.blocks:
        if bid not in depths:
            depths[bid] = max_known + 1

    # Group blocks by depth.
    max_depth = max(depths.values(), default=0)
    layers: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for bid, d in depths.items():
        layers[d].append(bid)

    # Horizontal spacing: at least 4.0 units between block centres.
    H_GAP = 4.0
    # Vertical spacing: 3.5 units between layers.
    V_GAP = 3.5

    pos: dict[int, tuple[float, float]] = {}
    for d, layer in enumerate(layers):
        n = len(layer)
        for i, bid in enumerate(sorted(layer)):
            x = (i - (n - 1) / 2.0) * H_GAP
            y = -d * V_GAP
            pos[bid] = (x, y)

    return pos


# ---------------------------------------------------------------------------
# Edge labels
# ---------------------------------------------------------------------------


def _get_cfg_edge_labels(cfg: ControlFlowGraph) -> dict[tuple[int, int], str]:
    """Return True/False labels for conditional branch edges (pre-SSA CFG)."""
    labels: dict[tuple[int, int], str] = {}
    for block in cfg.blocks.values():
        for inst in block.instructions:
            if isinstance(inst, ConditionalBranchInstruction):
                labels[(block.block_id, inst.true_dest)] = "T"
                labels[(block.block_id, inst.false_dest)] = "F"
    return labels


def _get_ssa_edge_labels(builder: SSABuilder) -> dict[tuple[int, int], str]:
    """Return True/False labels for conditional branch edges (SSA CFG)."""
    labels: dict[tuple[int, int], str] = {}
    for block in builder._cfg.blocks.values():
        for inst in block.ssa_instructions:
            if isinstance(inst, SSAConditionalBranch):
                labels[(block.block_id, inst.true_dest)] = "T"
                labels[(block.block_id, inst.false_dest)] = "F"
    return labels


# ---------------------------------------------------------------------------
# Node colors
# ---------------------------------------------------------------------------


def _get_node_colors(cfg: ControlFlowGraph) -> dict[int, str]:
    """Entry=green, exit=coral, everything else=light blue."""
    colors: dict[int, str] = {}
    for bid in cfg.blocks:
        if cfg.entry_block is not None and bid == cfg.entry_block.block_id:
            colors[bid] = "#90EE90"   # light green
        elif cfg.exit_block is not None and bid == cfg.exit_block.block_id:
            colors[bid] = "#F08080"   # light coral
        else:
            colors[bid] = "#ADD8E6"   # light blue
    return colors


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

# Maximum characters per line before wrapping long instruction text.
_MAX_LINE = 32


def _truncate(text: str, limit: int = _MAX_LINE) -> str:
    """Hard-truncate *text* with an ellipsis if it exceeds *limit* chars."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_cfg_label(block: BasicBlock) -> str:
    """Format a pre-SSA block as a compact multiline label."""
    lines = [f"── Block {block.block_id} ──"]
    if not block.instructions:
        lines.append("(empty)")
    for inst in block.instructions:
        if isinstance(inst, AssignmentInstruction):
            lines.append(_truncate(f"{inst.target} = {inst.expr}"))
        elif isinstance(inst, ConditionalBranchInstruction):
            lines.append(_truncate(f"if {inst.condition}"))
            lines.append(f"  T→B{inst.true_dest}  F→B{inst.false_dest}")
        elif isinstance(inst, UnconditionalJumpInstruction):
            lines.append(f"→ B{inst.dest}")
        else:
            lines.append(_truncate(str(inst)))
    return "\n".join(lines)


def _format_ssa_label(block: BasicBlock, builder: SSABuilder) -> str:
    """Format an SSA block as a compact multiline label.

    Phi functions appear first (before other instructions).
    """
    lines = [f"── Block {block.block_id} ──"]

    for phi in block.phis:
        if not phi.eliminated:
            ops = ", ".join(
                f"{builder._resolve_alias(v)}[B{k}]"
                for k, v in phi.operands.items()
            )
            lines.append(_truncate(f"φ {phi.result}=({ops})"))

    has_content = any(not p.eliminated for p in block.phis)
    if not block.ssa_instructions and not has_content:
        lines.append("(empty)")

    for inst in block.ssa_instructions:
        if isinstance(inst, SSAAssignment):
            lines.append(_truncate(
                f"{inst.result} = {builder._render_ssa_expr(inst.expr)}"
            ))
        elif isinstance(inst, SSAConditionalBranch):
            lines.append(_truncate(
                f"if {builder._render_ssa_expr(inst.condition)}"
            ))
            lines.append(f"  T→B{inst.true_dest}  F→B{inst.false_dest}")
        elif isinstance(inst, SSAUnconditionalJump):
            lines.append(f"→ B{inst.dest}")
        else:
            lines.append(_truncate(str(inst)))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node size helpers
# ---------------------------------------------------------------------------


def _label_dimensions(label: str) -> tuple[int, int]:
    """Return (max_chars_per_line, num_lines) for a multiline label string."""
    lines = label.split("\n")
    return max(len(l) for l in lines), len(lines)


def _node_size_for_label(label: str, chars_per_pt: float = 7.5) -> float:
    """Estimate a square node area (pts²) large enough for *label*.

    The heuristic converts max line width in characters to a node edge
    length in points (at ~7.5 chars per pt at font_size=7) and squares it.
    A minimum size prevents trivially small nodes.
    """
    max_chars, num_lines = _label_dimensions(label)
    side = max(max_chars * chars_per_pt, num_lines * 14, 60)
    return side * side


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(
    graph: "nx.DiGraph",
    pos: dict[int, tuple[float, float]],
    node_labels: dict[int, str],
    edge_labels: dict[tuple[int, int], str],
    node_colors: dict[int, str],
    backedges: set[tuple[int, int]],
    output_path: str | Path,
    title: str,
    cfg: ControlFlowGraph,
) -> Path:
    """Render the graph to a PNG file and return its path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not pos:
        # Empty graph — produce a minimal blank figure.
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output

    # ------------------------------------------------------------------ #
    # Figure sizing                                                        #
    # ------------------------------------------------------------------ #
    layers = _group_by_layer(pos)
    max_width = max(len(layer) for layer in layers) if layers else 1
    n_layers = len(layers)

    # Minimum 3 inches per horizontal block column, 2.5 inches per layer row.
    fig_w = max(8.0, min(28.0, max_width * 3.2))
    fig_h = max(6.0, min(24.0, n_layers * 2.5 + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------------ #
    # Per-node sizes (dynamic, proportional to label content)             #
    # ------------------------------------------------------------------ #
    node_sizes = [
        _node_size_for_label(node_labels.get(bid, ""))
        for bid in sorted(cfg.blocks)
    ]

    color_list = [
        node_colors.get(bid, "#ADD8E6") for bid in sorted(cfg.blocks)
    ]

    # ------------------------------------------------------------------ #
    # Expand axis limits so nodes are never clipped by the data boundary  #
    #                                                                      #
    # NetworkX draws node squares in display-points; the positions are in #
    # data coordinates.  When both nodes share x=0 (single-column graph), #
    # Matplotlib's auto-limit sets xlim ≈ (-0.5, 0.5), which is far too  #
    # narrow for 100+ pt node squares.  We compute the largest half-side  #
    # in data units and pad all four sides of the axis explicitly.        #
    # ------------------------------------------------------------------ #
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Convert the largest node half-side from display-points to data units.
    # The figure renderer isn't set up yet, but we can estimate:
    #   data_range_x ≈ H_GAP * (max_width - 1) or at least H_GAP
    #   data_range_y ≈ V_GAP * (n_layers - 1) or at least V_GAP
    # We use a fixed padding in data units based on the gap constants.
    H_GAP = 4.0
    V_GAP = 3.5
    x_pad = H_GAP * 0.75   # ~3 data units of horizontal margin each side
    y_pad = V_GAP * 0.6    # ~2 data units of vertical margin each side

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ------------------------------------------------------------------ #
    # Draw nodes                                                           #
    # ------------------------------------------------------------------ #
    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=sorted(cfg.blocks),
        node_color=color_list,
        node_size=node_sizes,
        node_shape="s",
        linewidths=1.2,
        ax=ax,
    )

    # ------------------------------------------------------------------ #
    # Draw edges                                                           #
    # ------------------------------------------------------------------ #
    normal_edges = [e for e in graph.edges() if e not in backedges]
    if normal_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=normal_edges,
            edge_color="#444444",
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=22,
            min_target_margin=22,
            ax=ax,
        )

    if backedges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=list(backedges),
            edge_color="#CC0000",
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            style="dashed",
            connectionstyle="arc3,rad=0.45",
            min_source_margin=22,
            min_target_margin=22,
            ax=ax,
        )

    # ------------------------------------------------------------------ #
    # Draw node labels                                                     #
    # ------------------------------------------------------------------ #
    nx.draw_networkx_labels(
        graph, pos,
        labels=node_labels,
        font_size=7,
        font_family="monospace",
        ax=ax,
    )

    # ------------------------------------------------------------------ #
    # Draw edge labels (T / F only)                                        #
    # ------------------------------------------------------------------ #
    visible_edge_labels = {
        (u, v): lbl
        for (u, v), lbl in edge_labels.items()
        if graph.has_edge(u, v) and (u, v) not in backedges
    }
    if visible_edge_labels:
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=visible_edge_labels,
            font_size=8,
            font_color="#005599",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
            ax=ax,
        )

    # ------------------------------------------------------------------ #
    # Legend — anchored outside the axes so it never overlaps any node.   #
    # ------------------------------------------------------------------ #
    legend_handles = [
        mpatches.Patch(facecolor="#90EE90", edgecolor="#555", label="Entry block"),
        mpatches.Patch(facecolor="#F08080", edgecolor="#555", label="Exit block"),
        mpatches.Patch(facecolor="#ADD8E6", edgecolor="#555", label="Basic block"),
    ]
    if backedges:
        legend_handles.append(
            mpatches.Patch(facecolor="#CC0000", label="Backedge (loop)")
        )
    # Place the legend in the figure's bottom-right corner, outside the axes.
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),   # just right of the axes bounding box
        borderaxespad=0,
        fontsize=8,
        framealpha=0.9,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    # Use tight_layout with rect so the legend (outside axes) is included.
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_by_layer(
    pos: dict[int, tuple[float, float]],
) -> list[list[int]]:
    """Group node IDs by their y-coordinate (rounded to 2 dp)."""
    if not pos:
        return []
    by_y: dict[float, list[int]] = {}
    for bid, (_, y) in pos.items():
        key = round(y, 2)
        by_y.setdefault(key, []).append(bid)
    return [by_y[k] for k in sorted(by_y, reverse=True)]
