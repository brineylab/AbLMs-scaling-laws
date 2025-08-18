import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def pointplot(
    data: pl.DataFrame,
    x: str,
    y: str,
    order: str | None = None,
    hue: str | None = None,
    hue_order: list | None = None,
    palette: str = "viridis",
    show_legend: bool = True,
    legend_kwargs: dict | None = None,
    linestyle: str = "none",
    dodge: bool | float = 0.5,
    estimator: str = "median",
    errorbar: tuple = ("ci", 95),
    yscale: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = 14,
    ylabel: str | None = None,
    ylabel_fontsize: int = 14,
    xtick_labels: list | None = None,
    xtick_labelsize: int | None = 12,
    ytick_labels: list | None = None,
    ytick_labelsize: int | None = 12,
    figsize: list = [5, 4],
    figfile: str | None = None,
) -> None:

    plt.figure(figsize=figsize)

    ax = sns.pointplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        legend=show_legend,
        linestyle=linestyle,
        dodge=dodge,
        estimator=estimator,
        errorbar=errorbar,
    )

    if show_legend:
        default_legend_kwargs = dict(
            loc="upper left",
            bbox_to_anchor=[0.05, 0.95],
            title="model",
            title_fontsize=12,
            fontsize=11,
        )
        legend_kwargs = legend_kwargs or {}
        default_legend_kwargs.update(legend_kwargs)
        ax.legend(**default_legend_kwargs)

    if yscale is not None:
        ax.set_yscale(yscale)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    ax.tick_params(axis="x", labelsize=xtick_labelsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)
    ax.tick_params(axis="y", labelsize=ytick_labelsize)

    plt.tight_layout()
    if figfile is not None:
        plt.savefig(figfile, bbox_inches="tight", dpi=300)
    else:
        plt.show()
