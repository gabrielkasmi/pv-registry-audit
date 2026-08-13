"""Shared figure style for every notebook in the repository.

Lives at the repository root so that a single discovery mechanism reaches it
whatever the depth of the calling notebook.

Usage, at the top of any notebook::

    import sys
    from pathlib import Path

    def _find_root(start=None, marker='paths.py'):
        p = Path(start or Path.cwd()).resolve()
        for parent in [p, *p.parents]:
            if (parent / marker).exists():
                return parent
        raise FileNotFoundError(f"{marker} not found above {p}")

    sys.path.insert(0, str(_find_root()))
    import style
    style.apply()

The settings follow the journal's figure specifications:

- standard sans-serif face (Arial or Helvetica; where unavailable, the
  metrically compatible substitutes Liberation Sans and Nimbus Sans)
- 5 to 7 pt body text, 8 pt bold panel labels
- no background gridlines, no coloured text
- a colour-blind safe palette: no red next to green, no rainbow or jet scale.
  Okabe-Ito for categorical variables (Wong, B. "Points of view: Colour
  blindness." Nature Methods 8, 441, 2011), viridis for continuous ones
- RGB, at least 300 dpi, and editable PDF text (``pdf.fonttype=42``). Without
  that last setting the text is converted to vector outlines and cannot be
  edited in the production stage, which is a common cause of rejection.
- vector (PDF) alongside raster (PNG) on every export; see ``save_fig`` in
  ``code/0_pipeline/evaluation/src/helpers.py``, which writes both.
"""
import sys as _sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Font fallback chain: matplotlib takes the first name available on the system.
# ---------------------------------------------------------------------------
FONT_FAMILY = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans']

FONT_SIZE_BODY = 7          # pt, the maximum the journal allows for body text
FONT_SIZE_SMALL = 6         # pt, ticks and legends; stays inside the 5-7 pt band
FONT_SIZE_PANEL_LABEL = 8   # pt, bold. Reserved for a/b/c panel labels; set by hand
                             # on multi-panel figures rather than applied automatically.

# Reference width for the font scaling factor. The sizes above are those of a
# single-column figure (89 mm). The paper's multi-panel figures are included at
# roughly 0.95\textwidth, about 6.5 inches, and REF_WIDTH_IN is set so that a
# wide canvas renders text around 9-10 pt once reduced to that width.
#
# These sizes are only correct if the figure is produced at that width. A figure
# built on a 16-inch canvas and then included at 0.95\textwidth is shrunk by a
# factor of about four, and its 7 pt text prints below 2 pt. Hence
# `apply(fig_width_in=...)` below, which scales the fonts to the canvas so that
# the rendered size stays constant.
REF_WIDTH_IN = 4.8

# Immutable reference values. `apply()` rebinds FONT_SIZE_* to their scaled
# version; these three stay the starting point of the computation.
_BASE_BODY, _BASE_SMALL, _BASE_PANEL = FONT_SIZE_BODY, FONT_SIZE_SMALL, FONT_SIZE_PANEL_LABEL

# ---------------------------------------------------------------------------
# Resolution / export
# ---------------------------------------------------------------------------
SCREEN_DPI = 100     # inline display in the notebook, not the export resolution
EXPORT_DPI = 450      # 300 dpi is the floor; 450 is the most that helps at proof stage

# ---------------------------------------------------------------------------
# Okabe-Ito categorical palette, colour-blind safe (Wong 2011, see module docstring)
# ---------------------------------------------------------------------------
OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
}

# Semantics reused throughout the pipeline (precision and recall points, recall
# sources). One place to change to re-colour the whole paper.
COLOR_TP = OKABE_ITO['bluish_green']         # true positive, both metrics
COLOR_FP = OKABE_ITO['vermillion']           # false positive (precision)
COLOR_FN = OKABE_ITO['orange']               # false negative (recall), kept distinct from FP
COLOR_SOURCE_OSM = OKABE_ITO['blue']
COLOR_SOURCE_MANUAL = OKABE_ITO['reddish_purple']
COLOR_SCATTER_DEFAULT = OKABE_ITO['blue']
COLOR_MEAN = OKABE_ITO['vermillion']         # central estimate on a bootstrap histogram
COLOR_RAW = OKABE_ITO['black']               # raw value, before correction
COLOR_REGISTRY = OKABE_ITO['orange']         # a registry value overlaid on a distribution

# ---------------------------------------------------------------------------
# Colormaps. Never red beside green, never rainbow or jet.
# ---------------------------------------------------------------------------
CMAP_SEQUENTIAL = 'viridis'      # rate bounded on 0-1, higher is better
CMAP_SEQUENTIAL_BAD = 'viridis_r'  # same family, higher is worse (relative uncertainty)
CMAP_SEQUENTIAL_BAD_ALT = 'magma_r'  # also higher-is-worse, but for a quantity of a
                                     # different nature. Use it when two incomparable
                                     # uncertainties share a figure, for instance the
                                     # uncertainty on a detection rate beside the
                                     # uncertainty on a corrected capacity: two scales
                                     # in the same colormap invite a false comparison.
CMAP_MAGNITUDE = 'Purples'       # positive magnitude with no good/bad reading
CMAP_DIVERGING = 'RdBu_r'        # signed gap around zero; red positive, blue negative

# ---------------------------------------------------------------------------
# Basemap colours (Natural Earth 1:110m, see paths.WORLD)
# ---------------------------------------------------------------------------
BASEMAP_LAND = '#eeebe3'
BASEMAP_SEA = '#dceefb'
BASEMAP_EDGE = '#b8b0a0'


def scale_for(fig_width_in):
    """Font scaling factor for a canvas ``fig_width_in`` inches wide.

    The style's sizes are set for a single-column figure (89 mm). A wider figure
    is shrunk by the same ratio when included in the document, so its fonts must
    grow by that ratio for the printed size to come out unchanged.
    """
    return max(1.0, float(fig_width_in) / REF_WIDTH_IN)


def apply(fig_width_in=None):
    """Call once at the top of a notebook, after importing matplotlib.

    Sets the global rcParams, so every figure created afterwards inherits the
    style without repeating the same arguments at each ``plt.subplots()``.

    ``fig_width_in`` is the width of the notebook's figure canvas, in inches;
    fonts are scaled accordingly (see ``scale_for``). Leave it at None for
    figures produced directly at print width.
    """
    import matplotlib.pyplot as plt

    k = 1.0 if fig_width_in is None else scale_for(fig_width_in)
    # Always start from the reference values, never the current ones: two
    # successive calls to apply() would otherwise compose their scale factors.
    body, small, panel = (_BASE_BODY * k, _BASE_SMALL * k, _BASE_PANEL * k)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': FONT_FAMILY,
        'font.size': body,
        'axes.titlesize': panel,
        'axes.labelsize': body,
        'xtick.labelsize': small,
        'ytick.labelsize': small,
        'legend.fontsize': small,
        'axes.linewidth': 0.8 * k,
        'xtick.major.width': 0.8 * k,
        'ytick.major.width': 0.8 * k,
        'legend.frameon': False,
        'figure.dpi': SCREEN_DPI,
        'savefig.dpi': EXPORT_DPI,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,   # keeps text editable in exported PDFs, not outlined
        'ps.fonttype': 42,
        'axes.grid': False,   # no background gridlines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'text.color': '#000000',       # no coloured text: colour carries meaning on
        'axes.labelcolor': '#000000',  # markers and areas, never on the text itself
    })

    # Figures often pass `fontsize=style.FONT_SIZE_SMALL` directly, in annotations,
    # colorbars and hand-placed legends, which rcParams cannot reach. The module
    # constants are therefore rebound to their scaled values: any code reading them
    # after `apply()` gets the right size without being modified.
    global FONT_SIZE_BODY_SCALED, FONT_SIZE_SMALL_SCALED, FONT_SIZE_PANEL_LABEL_SCALED
    FONT_SIZE_BODY_SCALED, FONT_SIZE_SMALL_SCALED, FONT_SIZE_PANEL_LABEL_SCALED = body, small, panel
    _mod = _sys.modules[__name__]
    _mod.FONT_SIZE_BODY = body
    _mod.FONT_SIZE_SMALL = small
    _mod.FONT_SIZE_PANEL_LABEL = panel
    return k


def find_style(start=None, marker='paths.py'):
    """Walk up from ``start`` until ``marker`` is found.

    Notebooks and modules use this to locate the repository root without
    depending on a fixed number of ``../`` levels, which breaks the moment the
    tree is reorganised. Returns the folder containing the marker, ready to be
    inserted into ``sys.path``, not the file itself.
    """
    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"{marker} introuvable en remontant depuis {p}")
