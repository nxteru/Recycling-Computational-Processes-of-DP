"""Renders a reconstruction-accuracy-by-n array (a
reconstruction_accuracy_by_cell value pulled from one *.result.jsonl via
data.source: raw_result) as a bar chart: x is the target partial sum n, y
is the greedy-tour-reconstruction accuracy at that n.

With data.source: raw_result_multi, `values` is instead a list of
(label, values) tuples -- one bar chart per panel, side by side in one
figure (e.g. INPUT/ELM/NGRC/DECSSP reservoirs for the same target).

Optional plot_spec keys:
- tick_fontsize: font size for the tick labels on both axes
- label_fontsize: font size for the x/y axis titles
- panel_title_fontsize / panel_title_weight: font size/weight for each
  panel's own title (the method label)
- square: truthy forces each panel's drawing area to be square
- share_y: truthy (multi-panel only) links the y-axis scale across
  panels and only draws y tick labels / the y title on the first column
- title: omitted (or empty) suppresses the figure-level title entirely"""
import matplotlib.pyplot as plt
import numpy as np


def _draw(ax, values, plot_spec: dict, label: str | None = None, show_ylabel: bool = True):
    values = np.asarray(values, dtype=float)
    n = np.arange(1, len(values) + 1)
    tick_fontsize = plot_spec.get("tick_fontsize")
    label_fontsize = plot_spec.get("label_fontsize")

    # rasterized=True: at width=1.0 adjacent bars share an edge, and vector
    # renderers (SVG) anti-alias that shared edge into a visible hairline
    # gap that a raster PNG doesn't show. Rasterizing just the bar patches
    # avoids the seam while keeping axes/text vector.
    ax.bar(n, values, width=1.0, rasterized=True)
    ax.set_ylim(*plot_spec.get("ylim", (0, 1)))
    ax.set_xlabel(plot_spec.get("xlabel", "n"), fontsize=label_fontsize)
    if show_ylabel:
        ax.set_ylabel(plot_spec.get("ylabel", "accuracy"), fontsize=label_fontsize)
    title = label if label is not None else plot_spec.get("title", "")
    if title:
        ax.set_title(title, fontsize=plot_spec.get("panel_title_fontsize"), fontweight=plot_spec.get("panel_title_weight"))
    if tick_fontsize:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if plot_spec.get("square"):
        ax.set_box_aspect(1)


def render(values, plot_spec: dict, output_path):
    if isinstance(values, list) and values and isinstance(values[0], tuple):
        panels = values
        share_y = plot_spec.get("share_y")
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False, sharey=share_y)
        for j, (ax, (label, panel_values)) in enumerate(zip(axes[0], panels)):
            _draw(ax, panel_values, plot_spec, label, show_ylabel=not share_y or j == 0)
        if plot_spec.get("title"):
            fig.suptitle(plot_spec["title"])
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw(ax, values, plot_spec)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
