"""Renders a tidy DataFrame + plot spec as a line chart (e.g. accuracy vs
use_size).

x may be a plain column name, or {ratio: [numerator_col, denominator_col]}
for a computed x axis (e.g. use_size/reservoir_size). hue picks the base
color per group; an optional shade column draws every hue's sub-groups in
one color family, from light to dark by increasing shade value (e.g. one
line per reservoir_size within each reservoir_method's own color).

Optional plot_spec keys:
- colors: {hue_value: color} overrides for the default color cycle
- markers: {hue_value: matplotlib marker} overrides for the default "o"
- marker_sizes: {hue_value: markersize} overrides for the default markersize
- x_label / y_label: override the axis labels derived from the data
- tick_fontsize: font size for the tick labels on both axes
- label_fontsize: font size for the x/y axis titles
- legend_fontsize: font size for the legend text
- legend_loc: matplotlib legend location string (e.g. "lower right"), or
  "outside right" to place it outside the axes on the right
- grid: truthy draws light gridlines on both axes
- figsize: [width, height] in inches, overriding the default 6-per-facet x 5
- title: omitted (or empty) suppresses the figure title entirely"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def _resolve_x(df, x_spec):
    if isinstance(x_spec, dict):
        num, denom = x_spec["ratio"]
        return df[num] / df[denom], f"{num}/{denom}"
    return df[x_spec], x_spec


def _shade(base_color: str, level: int, n_levels: int) -> tuple:
    """Blend base_color with white: level 0 (of n_levels) is lightest,
    the last level is the full base color, so every shade in a hue group
    reads as one color family at increasing intensity."""
    rgb = mcolors.to_rgb(base_color)
    frac = 1.0 if n_levels <= 1 else 0.3 + 0.7 * (level / (n_levels - 1))
    return tuple(1 - frac * (1 - c) for c in rgb)


def render(df, plot_spec: dict, output_path):
    x_spec = plot_spec["x"]
    y = plot_spec.get("y", "metric_value")
    hue = plot_spec.get("hue")
    shade = plot_spec.get("shade")
    facet = plot_spec.get("facet")
    colors = plot_spec.get("colors", {})
    markers = plot_spec.get("markers", {})
    marker_sizes = plot_spec.get("marker_sizes", {})
    tick_fontsize = plot_spec.get("tick_fontsize")
    label_fontsize = plot_spec.get("label_fontsize")
    legend_fontsize = plot_spec.get("legend_fontsize", 7)
    legend_loc = plot_spec.get("legend_loc", "best")

    x_values, x_label = _resolve_x(df, x_spec)
    df = df.assign(_x=x_values)

    facets = sorted(df[facet].unique()) if facet else [None]
    default_figsize = (6 * len(facets), 5)
    figsize = plot_spec.get("figsize", default_figsize)
    fig, axes = plt.subplots(1, len(facets), figsize=figsize, squeeze=False)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, facet_value in zip(axes[0], facets):
        sub = df[df[facet] == facet_value] if facet else df
        hue_values = sorted(sub[hue].unique()) if hue else [None]
        for hue_idx, hue_value in enumerate(hue_values):
            group = sub[sub[hue] == hue_value] if hue else sub
            base_color = colors.get(hue_value, color_cycle[hue_idx % len(color_cycle)])
            marker = markers.get(hue_value, "o")
            markersize = marker_sizes.get(hue_value)
            if shade:
                shade_values = sorted(group[shade].unique())
                for level, shade_value in enumerate(shade_values):
                    series = group[group[shade] == shade_value].sort_values("_x")
                    label = f"{hue_value} ({shade_value})" if hue else str(shade_value)
                    ax.plot(series["_x"], series[y], marker=marker, markersize=markersize, color=_shade(base_color, level, len(shade_values)), label=label)
            else:
                series = group.sort_values("_x")
                ax.plot(series["_x"], series[y], marker=marker, markersize=markersize, color=base_color, label=str(hue_value) if hue else None)
        if plot_spec.get("x_scale") == "log":
            ax.set_xscale("log")
        if "ylim" in plot_spec:
            ax.set_ylim(*plot_spec["ylim"])
        ax.set_xlabel(plot_spec.get("x_label", x_label), fontsize=label_fontsize)
        ax.set_ylabel(plot_spec.get("y_label", y), fontsize=label_fontsize)
        if facet_value is not None:
            ax.set_title(f"{facet}={facet_value}")
        if hue or shade:
            if legend_loc == "outside right":
                ax.legend(fontsize=legend_fontsize, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
            else:
                ax.legend(fontsize=legend_fontsize, ncol=2 if shade else 1, loc=legend_loc)
        if tick_fontsize:
            ax.tick_params(axis="both", labelsize=tick_fontsize)
        if plot_spec.get("grid"):
            ax.grid(True, alpha=0.3)

    if plot_spec.get("title"):
        fig.suptitle(plot_spec["title"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
