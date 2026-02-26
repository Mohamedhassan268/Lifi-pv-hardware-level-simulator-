# systems/schematic_style.py
"""
Proteus-Style Schematic Rendering
==================================

Professional schematic styling with grid background, clean colors,
and consistent rendering across all papers.

Matches Proteus 8 Professional appearance:
- Light warm-gray canvas with visible dot grid
- Dark blue wires and component outlines
- Clean black labels with good spacing
- Thin border around drawing area

Usage:
    from systems.schematic_style import create_styled_drawing, save_styled

    d = create_styled_drawing(unit=3, fontsize=10)
    d += elm.Resistor().label('R1')
    save_styled(d, 'output.png', dpi=150)
"""

import numpy as np

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =============================================================================
# Proteus-Inspired Color Palette
# =============================================================================

SCHEMATIC_COLORS = {
    # Canvas
    'background':   '#EAEAE2',   # Warm light gray (Proteus canvas)
    'grid_dot':     '#C0C0B8',   # Visible gray-green grid dots
    'border':       '#808078',   # Medium gray border

    # Wires & components
    'wire':         '#000080',   # Navy blue (classic Proteus wire color)
    'component':    '#000080',   # Component outlines match wires
    'label':        '#000000',   # Black text labels
    'value':        '#000060',   # Dark navy for component values
    'title':        '#1A1A1A',   # Near-black for titles
    'annotation':   '#006400',   # Dark green for notes/annotations
}

# Grid settings
GRID_SPACING = 0.5      # Spacing between grid dots (in drawing units)
GRID_DOT_SIZE = 1.2     # Size of each grid dot marker (visible like Proteus)


# =============================================================================
# Drawing Factory
# =============================================================================

def create_styled_drawing(unit=3, fontsize=10, **kwargs):
    """
    Create a schemdraw Drawing with Proteus-style configuration.

    Args:
        unit: Element length in drawing units (default 3)
        fontsize: Default font size (default 10)
        **kwargs: Extra kwargs passed to schemdraw.Drawing()

    Returns:
        schemdraw.Drawing with professional colors applied.
    """
    if not SCHEMDRAW_AVAILABLE:
        raise ImportError("schemdraw not installed. Run: pip install schemdraw")

    d = schemdraw.Drawing(**kwargs)
    d.config(
        unit=unit,
        fontsize=fontsize,
        bgcolor=SCHEMATIC_COLORS['background'],
        color=SCHEMATIC_COLORS['wire'],
        lw=2.0,
    )
    return d


# =============================================================================
# Styled Save (Grid Overlay + Background + Border)
# =============================================================================

def save_styled(drawing, path, dpi=150):
    """
    Render a schemdraw Drawing with Proteus-style grid and save.

    Draws the schematic, overlays a dot grid on the matplotlib figure,
    adds a clean border, then saves to the specified path.

    Args:
        drawing: schemdraw.Drawing (must not have been drawn yet)
        path: Output file path (.png, .svg, .pdf)
        dpi: Resolution for raster output (default 150)
    """
    # Render the drawing to get matplotlib fig/ax
    result = drawing.draw(show=False)
    fig = result.fig
    ax = result.ax

    # Apply background
    fig.patch.set_facecolor(SCHEMATIC_COLORS['background'])
    ax.set_facecolor(SCHEMATIC_COLORS['background'])

    # Ensure equal aspect ratio so vertical spacing is preserved
    ax.set_aspect('equal')

    # Add dot grid (behind everything)
    _add_dot_grid(ax)

    # Add border around drawing area
    _add_border(ax)

    # Clean up axes (no ticks, no spines)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save
    fig.savefig(
        path,
        dpi=dpi,
        facecolor=SCHEMATIC_COLORS['background'],
        bbox_inches='tight',
        pad_inches=0.4,
    )
    plt.close(fig)


def _add_dot_grid(ax):
    """Overlay a regular dot grid behind all schematic elements."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Extend grid slightly beyond visible area
    margin = GRID_SPACING * 2
    x_start = np.floor((xlim[0] - margin) / GRID_SPACING) * GRID_SPACING
    x_end = np.ceil((xlim[1] + margin) / GRID_SPACING) * GRID_SPACING
    y_start = np.floor((ylim[0] - margin) / GRID_SPACING) * GRID_SPACING
    y_end = np.ceil((ylim[1] + margin) / GRID_SPACING) * GRID_SPACING

    x_pts = np.arange(x_start, x_end + GRID_SPACING, GRID_SPACING)
    y_pts = np.arange(y_start, y_end + GRID_SPACING, GRID_SPACING)
    xx, yy = np.meshgrid(x_pts, y_pts)

    ax.plot(
        xx.flatten(), yy.flatten(), '.',
        color=SCHEMATIC_COLORS['grid_dot'],
        markersize=GRID_DOT_SIZE,
        zorder=0,       # Behind all elements
        clip_on=True,
    )

    # Restore original limits (grid points may have expanded them)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _add_border(ax):
    """Add a thin professional border around the drawing area."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Draw border rectangle
    border_rect = plt.Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=1.5,
        edgecolor=SCHEMATIC_COLORS['border'],
        facecolor='none',
        zorder=50,
        clip_on=False,
    )
    ax.add_patch(border_rect)
