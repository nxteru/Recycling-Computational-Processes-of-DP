"""Renders a tidy DataFrame + plot spec as a grid of (x, y) -> value
heatmaps, one subplot per (facet_row, facet_col) combination.

Optional plot_spec keys:
- cmap: matplotlib colormap name (default "viridis")
- colors: [color, ...] builds a linear segmented colormap through these
  colors instead of using a named cmap (e.g. ["red", "blue"])
- tick_fontsize: font size for the tick labels on both axes
- label_fontsize: font size for the x/y axis titles
- square: truthy forces each panel's drawing area to be square
- share_y: truthy links the y-axis scale across panels and only draws
  y tick labels / the y title on the first column
- xtick_rotation: rotation (degrees) for x tick labels, to avoid overlap
  on dense axes
- colorbar_label: override the colorbar's label (default: the value column)
- colorbar_label_fontsize: font size for the colorbar's label
- colorbar_tick_fontsize: font size for the colorbar's tick labels
- panel_titles: false suppresses each panel's own facet-value title
  (default true)
- panel_title_fontsize / panel_title_weight: font size/weight for each
  panel's title
- title: omitted (or empty) suppresses the figure title entirely"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def render(df, plot_spec: dict, output_path):
    x = plot_spec["x"]
    y = plot_spec["y"]
    value = plot_spec.get("value", "metric_value")
    facet_row = plot_spec.get("facet_row")
    facet_col = plot_spec.get("facet_col")
    tick_fontsize = plot_spec.get("tick_fontsize")
    label_fontsize = plot_spec.get("label_fontsize")
    share_y = plot_spec.get("share_y")

    if "colors" in plot_spec:
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", plot_spec["colors"])
    else:
        cmap = plot_spec.get("cmap", "viridis")

    # facet_row_order/facet_col_order: explicit panel order instead of the
    # default alphabetical sort.
    rows = plot_spec.get("facet_row_order") or (sorted(df[facet_row].unique()) if facet_row else [None])
    cols = plot_spec.get("facet_col_order") or (sorted(df[facet_col].unique()) if facet_col else [None])

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4 * len(cols), 4 * len(rows)), squeeze=False, sharey=share_y)

    vmin, vmax = plot_spec.get("vlim", (df[value].min(), df[value].max()))
    im = None
    for i, row_value in enumerate(rows):
        for j, col_value in enumerate(cols):
            ax = axes[i][j]
            sub = df
            if facet_row:
                sub = sub[sub[facet_row] == row_value]
            if facet_col:
                sub = sub[sub[facet_col] == col_value]
            pivot = sub.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
            im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=plot_spec.get("xtick_rotation", 0))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel(x, fontsize=label_fontsize)
            if not share_y or j == 0:
                ax.set_ylabel(y, fontsize=label_fontsize)
            if plot_spec.get("panel_titles", True):
                title_bits = [b for b in (str(col_value) if facet_col else None, str(row_value) if facet_row else None) if b]
                if title_bits:
                    ax.set_title(
                        " / ".join(title_bits),
                        fontsize=plot_spec.get("panel_title_fontsize"),
                        fontweight=plot_spec.get("panel_title_weight"),
                    )
            if tick_fontsize:
                ax.tick_params(axis="both", labelsize=tick_fontsize)
            if plot_spec.get("square"):
                ax.set_box_aspect(1)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=plot_spec.get("colorbar_label", value))
        if "colorbar_label_fontsize" in plot_spec:
            cbar.set_label(plot_spec.get("colorbar_label", value), fontsize=plot_spec["colorbar_label_fontsize"])
        if "colorbar_tick_fontsize" in plot_spec:
            cbar.ax.tick_params(labelsize=plot_spec["colorbar_tick_fontsize"])
    if plot_spec.get("title"):
        fig.suptitle(plot_spec["title"])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
