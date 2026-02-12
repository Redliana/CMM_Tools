"""Geospatial visualizations."""

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


def plot_deposit_locations(
    df: pd.DataFrame,
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    color_by: str | None = None,
    title: str = "Ore Deposit Locations",
    figsize: tuple = (12, 8),
    ax: Any | None = None,
) -> Any:
    """
    Plot ore deposit locations on a map.

    Args:
        df: DataFrame with deposit data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        color_by: Optional column to color points by
        title: Chart title
        figsize: Figure size tuple
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    # Find coordinate columns (case-insensitive)
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "long" in c.lower()]

    if lat_cols and lat_col not in df.columns:
        lat_col = lat_cols[0]
    if lon_cols and lon_col not in df.columns:
        lon_col = lon_cols[0]

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Coordinate columns not found. Available: {list(df.columns)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Drop rows with missing coordinates
    plot_df = df.dropna(subset=[lat_col, lon_col])

    if color_by and color_by in plot_df.columns:
        # Color by category
        categories = plot_df[color_by].unique()
        colors = plt.cm.tab10(range(len(categories)))

        for cat, color in zip(categories, colors):
            mask = plot_df[color_by] == cat
            ax.scatter(
                plot_df.loc[mask, lon_col],
                plot_df.loc[mask, lat_col],
                c=[color],
                label=str(cat)[:30],
                alpha=0.6,
                s=20,
            )
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.scatter(plot_df[lon_col], plot_df[lat_col], c="steelblue", alpha=0.6, s=20)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_surface_depth(
    df: pd.DataFrame,
    title: str = "Surface Depth Profile",
    figsize: tuple = (12, 8),
    ax: Any | None = None,
) -> Any:
    """
    Plot depth surface from GA chronostratigraphic data.

    Args:
        df: DataFrame with x, y, z columns (from GAChronostratigraphicLoader)
        title: Chart title
        figsize: Figure size tuple
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create scatter plot colored by depth
    scatter = ax.scatter(
        df["x"],
        df["y"],
        c=df["z"],
        cmap="viridis_r",  # Reversed so deeper is darker
        s=1,
        alpha=0.7,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Depth (m)")

    ax.set_xlabel("X (Albers)")
    ax.set_ylabel("Y (Albers)")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_ree_distribution(
    df: pd.DataFrame,
    elements: list[str] | None = None,
    title: str = "REE Distribution",
    figsize: tuple = (12, 6),
    ax: Any | None = None,
) -> Any:
    """
    Plot REE concentration distribution.

    Args:
        df: DataFrame with REE concentration data
        elements: list of element symbols to include
        title: Chart title
        figsize: Figure size tuple
        ax: Optional matplotlib axes

    Returns:
        matplotlib Figure object
    """
    plt = _get_matplotlib()

    if elements is None:
        elements = [
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Find concentration columns for each element
    means = []
    stds = []
    found_elements = []

    for elem in elements:
        # Look for ppm column
        elem_cols = [c for c in df.columns if elem in c and ("ppm" in c.lower())]
        if elem_cols:
            col = elem_cols[0]
            values = pd.to_numeric(df[col], errors="coerce")
            valid = values[values >= 0]  # Filter out below-detection
            if len(valid) > 0:
                means.append(valid.mean())
                stds.append(valid.std())
                found_elements.append(elem)

    if not found_elements:
        ax.text(0.5, 0.5, "No REE data found", ha="center", va="center")
        return fig

    x = range(len(found_elements))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(found_elements)
    ax.set_xlabel("Element")
    ax.set_ylabel("Concentration (ppm)")
    ax.set_title(title)
    ax.set_yscale("log")  # Log scale for REE

    plt.tight_layout()
    return fig
