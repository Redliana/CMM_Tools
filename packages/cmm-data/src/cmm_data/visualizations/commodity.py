"""Commodity data visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    import pandas as pd


def _get_matplotlib():
    """Get matplotlib.pyplot, raising helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ConfigurationError(
            "matplotlib required for visualizations. Install with: pip install cmm-data[viz]"
        )


def plot_world_production(
    df: pd.DataFrame,
    commodity_name: str,
    top_n: int = 10,
    year_col: str = "Prod_t_est_2022",
    country_col: str = "Country",
    figsize: tuple = (10, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot bar chart of world production by country.

    Args:
        df: DataFrame with production data (from USGSCommodityLoader)
        commodity_name: Name for chart title
        top_n: Number of top producers to show
        year_col: Column name for production values
        country_col: Column name for country names
        figsize: Figure size tuple
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    # Use cleaned column if available
    if f"{year_col}_clean" in df.columns:
        year_col = f"{year_col}_clean"

    # Filter out totals and sort
    plot_df = df[~df[country_col].str.contains("World|total", case=False, na=False)]
    plot_df = plot_df.dropna(subset=[year_col])
    plot_df = plot_df.nlargest(top_n, year_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.barh(plot_df[country_col], plot_df[year_col])
    ax.set_xlabel("Production (metric tons)")
    ax.set_ylabel("Country")
    ax.set_title(f"Top {top_n} {commodity_name} Producers")
    ax.invert_yaxis()  # Largest at top

    # Add value labels
    for bar, val in zip(bars, plot_df[year_col]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {val:,.0f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_production_timeseries(
    df: pd.DataFrame,
    commodity_name: str,
    country: str | None = None,
    figsize: tuple = (10, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot production over time from salient statistics.

    Args:
        df: DataFrame with salient statistics (from USGSCommodityLoader)
        commodity_name: Name for chart title
        country: Optional country to highlight (default: shows US)
        figsize: Figure size tuple
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use cleaned production column
    prod_col = "USprod_t_clean" if "USprod_t_clean" in df.columns else "USprod_t"

    # Sort by year
    if "Year" in df.columns:
        df = df.sort_values("Year")
        x = df["Year"]
    else:
        x = range(len(df))

    # Plot production
    ax.plot(x, df[prod_col], marker="o", linewidth=2, label="US Production")

    # Add imports/exports if available
    if "Imports_t_clean" in df.columns:
        ax.plot(
            x,
            df["Imports_t_clean"],
            marker="s",
            linewidth=1,
            linestyle="--",
            label="Imports",
            alpha=0.7,
        )
    if "Exports_t_clean" in df.columns:
        ax.plot(
            x,
            df["Exports_t_clean"],
            marker="^",
            linewidth=1,
            linestyle="--",
            label="Exports",
            alpha=0.7,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Quantity (metric tons)")
    ax.set_title(f"{commodity_name} - U.S. Statistics Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_import_reliance(
    df: pd.DataFrame,
    commodity_name: str,
    figsize: tuple = (10, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot net import reliance over time.

    Args:
        df: DataFrame with salient statistics
        commodity_name: Name for chart title
        figsize: Figure size tuple
        ax: Optional matplotlib axes to plot on

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use cleaned NIR column
    nir_col = "NIR_pct_clean" if "NIR_pct_clean" in df.columns else "NIR_pct"

    if nir_col not in df.columns:
        raise ValueError("NIR_pct column not found in DataFrame")

    # Sort by year
    if "Year" in df.columns:
        df = df.sort_values("Year")
        x = df["Year"]
    else:
        x = range(len(df))

    # Plot NIR
    ax.bar(x, df[nir_col], color="steelblue", alpha=0.7)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")

    ax.set_xlabel("Year")
    ax.set_ylabel("Net Import Reliance (%)")
    ax.set_title(f"{commodity_name} - U.S. Net Import Reliance")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_multiple_commodities(
    commodities: list,
    data_type: str = "world",
    metric: str = "production",
    figsize: tuple = (12, 8),
) -> Any:
    """
    Compare multiple commodities in a single chart.

    Args:
        commodities: list of commodity codes
        data_type: 'world' or 'salient'
        metric: Metric to compare
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    from ..loaders.usgs_commodity import USGSCommodityLoader

    loader = USGSCommodityLoader()

    fig, ax = plt.subplots(figsize=figsize)

    for commodity in commodities:
        try:
            if data_type == "world":
                df = loader.load_world_production(commodity)
                # Get world total
                world_row = df[df["Country"].str.contains("World", case=False, na=False)]
                if not world_row.empty:
                    value = world_row.iloc[0].get("Prod_t_est_2022_clean", 0)
                    ax.bar(loader.get_commodity_name(commodity), value)
            else:
                df = loader.load_salient_statistics(commodity)
                # Get latest year
                latest = df.iloc[-1]
                value = latest.get("USprod_t_clean", 0)
                ax.bar(loader.get_commodity_name(commodity), value)
        except (OSError, ValueError):
            continue

    ax.set_xlabel("Commodity")
    ax.set_ylabel("Production (metric tons)")
    ax.set_title("Commodity Production Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig
