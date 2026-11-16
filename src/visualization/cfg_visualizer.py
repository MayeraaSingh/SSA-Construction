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
from typing import Any

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
    import networkx as nx

    _VIZ_AVAILABLE = True
    _VIZ_ERROR = ""
except ImportError as exc:
    _VIZ_AVAILABLE = False
    _VIZ_ERROR = str(exc)


# ---------------------------------------------------------------------------
# SSA type imports (deferred to avoid import cost when viz is unused)
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
    pos = _compute_layout(cfg)
    node_labels = {bid: _format_cfg_label(cfg.get_block(bid)) for bid in cfg.blocks}
    edge_labels = _get_cfg_edge_labels(cfg)
    node_colors = _get_node_colors(cfg)
    backedges = _get_backedges(cfg)

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
    pos = _compute_layout(cfg)
    node_labels = {
        bid: _format_ssa_label(cfg.get_block(bid), builder) for bid in cfg.blocks
    }
    edge_labels = _get_ssa_edge_labels(builder)
    node_colors = _get_node_colors(cfg)
    backedges = _get_backedges(cfg)

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
# Layout
# ---------------------------------------------------------------------------


def _compute_layout(cfg: ControlFlowGraph) -> dict[int, tuple[float, float]]:
    """Compute node positions using a layered (top-down) layout.

    Depth is the shortest path from the entry block (BFS).  Blocks at the
    same depth are spread horizontally.  Backedges are ignored when
    computing depth so loop headers appear above loop bodies.
    """
    entry_id = cfg.entry_block.block_id if cfg.entry_block else 0

    # BFS to compute depths (ignores backedges naturally — each node visited once).
    depths: dict[int, int] = {entry_id: 0}
    queue: list[int] = [entry_id]
    while queue:
        current = queue.pop(0)
        block = cfg.get_block(current)
        for succ_id in block.successors:
            if succ_id not in depths:
                depths[succ_id] = depths[current] + 1
                queue.append(succ_id)

    # Ensure every block has a depth (in case some aren't reachable).
    for bid in cfg.blocks:
        if bid not in depths:
            depths[bid] = max(depths.values(), default=0) + 1

    # Group blocks by depth.
    max_depth = max(depths.values()) if depths else 0
    layers: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for bid, d in depths.items():
        layers[d].append(bid)

    # Assign positions.
    pos: dict[int, tuple[float, float]] = {}
    for d, layer in enumerate(layers):
        n = len(layer)
        for i, bid in enumerate(sorted(layer)):
            x = (i - (n - 1) / 2) * 3.5
            y = -d * 3.0
            pos[bid] = (x, y)

    return pos


# ---------------------------------------------------------------------------
# Edge labels and styling
# ---------------------------------------------------------------------------


def _get_cfg_edge_labels(cfg: ControlFlowGraph) -> dict[tuple[int, int], str]:
    """Map edges to 'True'/'False' labels for conditional branches."""
    labels: dict[tuple[int, int], str] = {}
    for block in cfg.blocks.values():
        for inst in block.instructions:
            if isinstance(inst, ConditionalBranchInstruction):
                labels[(block.block_id, inst.true_dest)] = "True"
                labels[(block.block_id, inst.false_dest)] = "False"
    return labels


def _get_ssa_edge_labels(builder: SSABuilder) -> dict[tuple[int, int], str]:
    """Map SSA-form edges to 'True'/'False' labels."""
    labels: dict[tuple[int, int], str] = {}
    for block in builder._cfg.blocks.values():
        for inst in block.ssa_instructions:
            if isinstance(inst, SSAConditionalBranch):
                labels[(block.block_id, inst.true_dest)] = "True"
                labels[(block.block_id, inst.false_dest)] = "False"
    return labels


def _get_backedges(cfg: ControlFlowGraph) -> set[tuple[int, int]]:
    """Identify backedges using the heuristic dst_id < src_id.

    For structured CFGs produced by our builder, loop headers always have
    smaller block IDs than their bodies, so this correctly identifies
    backedges without computing dominators.
    """
    backedges: set[tuple[int, int]] = set()
    for block in cfg.blocks.values():
        for succ_id in block.successors:
            if succ_id < block.block_id:
                backedges.add((block.block_id, succ_id))
    return backedges


# ---------------------------------------------------------------------------
# Node colors
# ---------------------------------------------------------------------------


def _get_node_colors(cfg: ControlFlowGraph) -> dict[int, str]:
    """Assign colors to blocks: entry=green, exit=coral, normal=lightblue."""
    colors: dict[int, str] = {}
    for bid in cfg.blocks:
        if cfg.entry_block is not None and bid == cfg.entry_block.block_id:
            colors[bid] = "#90EE90"  # light green
        elif cfg.exit_block is not None and bid == cfg.exit_block.block_id:
            colors[bid] = "#F08080"  # light coral
        else:
            colors[bid] = "#ADD8E6"  # light blue
    return colors


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def _format_cfg_label(block: BasicBlock) -> str:
    """Format a block's Day 1 IR instructions as a multiline label."""
    lines = [f"Block {block.block_id}"]
    if not block.instructions:
        lines.append("(empty)")
    for inst in block.instructions:
        if isinstance(inst, AssignmentInstruction):
            lines.append(f"{inst.target} = {inst.expr}")
        elif isinstance(inst, ConditionalBranchInstruction):
            lines.append(f"if {inst.condition} -> B{inst.true_dest} else -> B{inst.false_dest}")
        elif isinstance(inst, UnconditionalJumpInstruction):
            lines.append(f"jump -> B{inst.dest}")
        else:
            lines.append(str(inst))
    return "\n".join(lines)


def _format_ssa_label(block: BasicBlock, builder: SSABuilder) -> str:
    """Format a block's SSA instructions and phi functions as a multiline label."""
    lines = [f"Block {block.block_id}"]

    # Phi functions first.
    for phi in block.phis:
        if not phi.eliminated:
            ops = ", ".join(
                f"{builder._resolve_alias(v)} [B{k}]"
                for k, v in phi.operands.items()
            )
            lines.append(f"{phi.result} = phi({ops})")

    # SSA instructions.
    if not block.ssa_instructions and not any(not p.eliminated for p in block.phis):
        lines.append("(empty)")
    for inst in block.ssa_instructions:
        if isinstance(inst, SSAAssignment):
            lines.append(f"{inst.result} = {builder._render_ssa_expr(inst.expr)}")
        elif isinstance(inst, SSAConditionalBranch):
            lines.append(
                f"if {builder._render_ssa_expr(inst.condition)}"
                f" -> B{inst.true_dest} else -> B{inst.false_dest}"
            )
        elif isinstance(inst, SSAUnconditionalJump):
            lines.append(f"jump -> B{inst.dest}")
        else:
            lines.append(str(inst))

    return "\n".join(lines)


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
    """Render the graph to a PNG file using Matplotlib."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Scale figure size based on graph complexity.
    n_blocks = len(cfg.blocks)
    max_layer = max(len(layer) for layer in _group_by_depth(pos)) if pos else 1
    fig_width = max(10, min(24, max_layer * 4))
    fig_height = max(8, min(20, n_blocks * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_list = [node_colors.get(bid, "#ADD8E6") for bid in sorted(cfg.blocks)]

    # Draw nodes.
    nx.draw_networkx_nodes(
        graph, pos, node_color=color_list, node_size=3000,
        node_shape="s", linewidths=1.5, ax=ax,
    )

    # Draw normal edges.
    normal_edges = [(u, v) for u, v in graph.edges() if (u, v) not in backedges]
    if normal_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=normal_edges,
            edge_color="#333333", arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.1", ax=ax,
        )

    # Draw backedges with distinct style.
    if backedges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=list(backedges),
            edge_color="#CC0000", arrows=True, arrowsize=20,
            style="dashed", connectionstyle="arc3,rad=0.3", ax=ax,
        )

    # Draw node labels.
    nx.draw_networkx_labels(
        graph, pos, labels=node_labels,
        font_size=7, font_family="monospace", ax=ax,
    )

    # Draw edge labels (True/False).
    if edge_labels:
        # Offset edge labels slightly for readability.
        edge_pos = {}
        for (u, v), label in edge_labels.items():
            if (u, v) in pos and (u, v) in graph.edges():
                edge_pos[(u, v)] = label
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_pos,
            font_size=8, font_color="#0066CC", ax=ax,
        )

    # Legend for backedges.
    if backedges:
        ax.plot([], [], color="#CC0000", linestyle="dashed",
                label="Backedge (loop)")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _group_by_depth(
    pos: dict[int, tuple[float, float]]
) -> list[list[int]]:
    """Group node IDs by their y-coordinate (depth)."""
    if not pos:
        return []
    by_y: dict[float, list[int]] = {}
    for bid, (_, y) in pos.items():
        rounded = round(y, 2)
        by_y.setdefault(rounded, []).append(bid)
    return [by_y[k] for k in sorted(by_y, reverse=True)]
