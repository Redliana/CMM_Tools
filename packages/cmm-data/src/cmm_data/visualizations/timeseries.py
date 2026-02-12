"""Time series visualizations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..exceptions import ConfigurationError


def _get_matplotlib():
    """Get matplotlib.pyplot, raising helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ConfigurationError(
            "matplotlib required for visualizations. Install with: pip install cmm-data[viz]"
        )


def plot_commodity_timeseries(
    commodity_code: str,
    metrics: list[str] | None = None,
    figsize: tuple = (12, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot time series for a commodity's salient statistics.

    Args:
        commodity_code: Commodity code (e.g., 'lithi')
        metrics: list of metrics to plot (default: production, imports, exports)
        figsize: Figure size tuple
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    from ..loaders.usgs_commodity import USGSCommodityLoader

    loader = USGSCommodityLoader()
    df = loader.load_salient_statistics(commodity_code)

    if metrics is None:
        metrics = ["USprod_t", "Imports_t", "Exports_t"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Sort by year
    if "Year" in df.columns:
        df = df.sort_values("Year")
        x = df["Year"]
    else:
        x = range(len(df))

    # Plot each metric
    for metric in metrics:
        col = f"{metric}_clean" if f"{metric}_clean" in df.columns else metric
        if col in df.columns:
            label = metric.replace("_t", " (t)").replace("_", " ")
            ax.plot(x, df[col], marker="o", label=label, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Quantity (metric tons)")
    ax.set_title(f"{loader.get_commodity_name(commodity_code)} - Time Series")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_price_trends(
    commodity_code: str,
    figsize: tuple = (10, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot price trends for a commodity.

    Args:
        commodity_code: Commodity code
        figsize: Figure size tuple
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    from ..loaders.usgs_commodity import USGSCommodityLoader

    loader = USGSCommodityLoader()
    df = loader.load_salient_statistics(commodity_code)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    price_col = "Price_dt_clean" if "Price_dt_clean" in df.columns else "Price_dt"

    if price_col not in df.columns:
        ax.text(0.5, 0.5, "No price data available", ha="center", va="center")
        return fig

    # Sort by year
    if "Year" in df.columns:
        df = df.sort_values("Year")
        x = df["Year"]
    else:
        x = range(len(df))

    ax.plot(x, df[price_col], marker="o", color="green", linewidth=2)
    ax.fill_between(x, 0, df[price_col], alpha=0.3, color="green")

    ax.set_xlabel("Year")
    ax.set_ylabel("Price ($/metric ton)")
    ax.set_title(f"{loader.get_commodity_name(commodity_code)} - Price Trend")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_critical_minerals_comparison(
    year: int | None = None,
    metric: str = "USprod_t",
    top_n: int = 15,
    figsize: tuple = (14, 8),
) -> Any:
    """
    Compare multiple critical minerals.

    Args:
        year: Year to compare (default: latest available)
        metric: Metric to compare
        top_n: Number of minerals to show
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    from ..loaders.usgs_commodity import CRITICAL_MINERALS, USGSCommodityLoader

    loader = USGSCommodityLoader()

    data = []
    for code in CRITICAL_MINERALS:
        try:
            df = loader.load_salient_statistics(code)
            if year and "Year" in df.columns:
                row = df[df["Year"] == year]
                if row.empty:
                    row = df.iloc[-1:]
            else:
                row = df.iloc[-1:]

            col = f"{metric}_clean" if f"{metric}_clean" in row.columns else metric
            if col in row.columns:
                value = row[col].iloc[0]
                if pd.notna(value) and value > 0:
                    data.append(
                        {"commodity": loader.get_commodity_name(code), "code": code, "value": value}
                    )
        except (OSError, ValueError):
            continue

    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    plot_df = pd.DataFrame(data).nlargest(top_n, "value")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["commodity"], plot_df["value"], color="steelblue")
    ax.set_xlabel(metric.replace("_t", " (metric tons)").replace("_", " "))
    ax.set_title(f"Critical Minerals - {metric}")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig
