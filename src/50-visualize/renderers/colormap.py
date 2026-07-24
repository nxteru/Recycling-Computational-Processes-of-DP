"""Renders a value_accuracy_by_cell matrix (shape (D, U*D), pulled from one
*.result.jsonl via data.source: raw_result) as a 2D per-task accuracy
heatmap indexed by (i, j) -- the nested prefix position i and the target
partial-sum value j. Cells outside row i's valid range (j > (i+1)*U) are
masked to NaN before plotting: the underlying scorer never writes those
cells, so leaving them at their zero-initialized value would show as a
misleading 0% accuracy band instead of "not applicable".

With data.source: raw_result_multi, `values` is instead a list of
(label, values) tuples -- one colormap per panel, side by side in one
figure (e.g. INPUT/ELM/NGRC/DECSSP reservoirs for the same target).

Optional plot_spec keys:
- cmap: matplotlib colormap name (default "viridis")
- tick_fontsize: font size for the tick labels on both axes
- label_fontsize: font size for the x/y axis titles
- panel_title_fontsize / panel_title_weight: font size/weight for each
  panel's own title (the method label)
- square: truthy forces each panel's drawing area to be square
- share_y: truthy (multi-panel only) links the y-axis scale across
  panels and only draws y tick labels / the y title on the first column
- colorbar_label: override the (single, shared) colorbar's label
  (default "accuracy")
- colorbar_label_fontsize / colorbar_tick_fontsize: font sizes for the
  colorbar's label / tick labels
- colorbar_shrink: fraction of the axes height the colorbar spans
  (default 0.8; higher = taller)
- colorbar_aspect: length-to-width ratio of the colorbar (default 20;
  lower = thicker/wider)
- title: omitted (or empty) suppresses the figure-level title entirely"""
import matplotlib.pyplot as plt
import numpy as np


def _masked(values) -> np.ndarray:
    table = np.asarray(values, dtype=float)
    D, width = table.shape
    U = width // D
    masked = table.copy()
    for i in range(D):
        masked[i, (i + 1) * U :] = np.nan
    return masked


def _draw(ax, values, plot_spec: dict, label: str | None = None, show_ylabel: bool = True):
    masked = _masked(values)
    vmin, vmax = plot_spec.get("vlim", (0, 1))
    cmap = plot_spec.get("cmap", "viridis")
    tick_fontsize = plot_spec.get("tick_fontsize")
    label_fontsize = plot_spec.get("label_fontsize")

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    title = label if label is not None else plot_spec.get("title", "")
    if title:
        ax.set_title(title, fontsize=plot_spec.get("panel_title_fontsize"), fontweight=plot_spec.get("panel_title_weight"))
    ax.set_xlabel("j", fontsize=label_fontsize)
    if show_ylabel:
        ax.set_ylabel("i", fontsize=label_fontsize)

    # i is 1-indexed on display (row 0 is prefix position 1); j's ticks are
    # spaced across the full 0..width range (11 marks) rather than
    # matplotlib's default locator, so every panel reads the same scale.
    D, width = masked.shape
    ax.set_yticks(range(D))
    ax.set_yticklabels(range(1, D + 1))
    xticks = range(0, width + 1, max(width // 10, 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    if tick_fontsize:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if plot_spec.get("square"):
        ax.set_box_aspect(1)
    return im


def _add_colorbar(fig, im, axes, plot_spec: dict):
    cbar = fig.colorbar(
        im, ax=axes,
        shrink=plot_spec.get("colorbar_shrink", 0.8),
        aspect=plot_spec.get("colorbar_aspect", 20),
        label=plot_spec.get("colorbar_label", "accuracy"),
    )
    if "colorbar_label_fontsize" in plot_spec:
        cbar.set_label(plot_spec.get("colorbar_label", "accuracy"), fontsize=plot_spec["colorbar_label_fontsize"])
    if "colorbar_tick_fontsize" in plot_spec:
        cbar.ax.tick_params(labelsize=plot_spec["colorbar_tick_fontsize"])


def render(values, plot_spec: dict, output_path):
    if isinstance(values, list) and values and isinstance(values[0], tuple):
        panels = values
        share_y = plot_spec.get("share_y")
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False, sharey=share_y)
        im = None
        for j, (ax, (label, panel_values)) in enumerate(zip(axes[0], panels)):
            im = _draw(ax, panel_values, plot_spec, label, show_ylabel=not share_y or j == 0)
        _add_colorbar(fig, im, axes.ravel().tolist(), plot_spec)
        if plot_spec.get("title"):
            fig.suptitle(plot_spec["title"])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    im = _draw(ax, values, plot_spec)
    _add_colorbar(fig, im, ax, plot_spec)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
