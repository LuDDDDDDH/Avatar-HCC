from __future__ import annotations
from avatar_paths import get_value, get_column
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
MM_PER_INCH = 25.4
FIGURE_WIDTH_MM: Mapping[str, float] = {'single': 89.0, 'double': 183.0}
MAX_FIGURE_HEIGHT_MM = 170.0
PALETTE_ID = get_value('figures.config.item_342')
PALETTE_SELECTED_ON = '2026-09-01'
ENDPOINT_COLORS: Mapping[str, str] = {'Response': '#DE646C', 'PFS': '#2F6FB0', 'OS': '#AB4AA0'}
MODALITY_COLORS: Mapping[str, str] = {'Metagenomics': '#2A8F78', 'Radiomics': '#2F91C3', 'Blood': '#7D70D1', 'Urine': '#C34D9B'}
MBP_COLORS: Mapping[str, str] = {'MBP-I': '#285F9E', 'MBP-II': '#7163C6', 'MBP-III': '#B74496'}
MASTER_SPECTRUM: tuple[str, ...] = ('#2A8F78', '#2F91C3', '#2F6FB0', '#617FC4', '#7D70D1', '#8F59C0', '#AB4AA0', '#C34D9B', '#DE646C')
SEQUENTIAL_COLORS: tuple[str, ...] = ('#F7FBFF', '#E1F1F8', '#C4E2EF', '#98C8E1', '#6CA8D1', '#617FC4', '#745EB6', '#98499F', '#A72C79')
DIVERGING_COLORS: tuple[str, ...] = ('#2F6FB0', '#79AAD1', '#C7DEEC', '#F8F8F8', '#F2D2CF', '#E49592', '#C44A58')
NEUTRAL_COLORS: Mapping[str, str] = {'text': '#26313D', 'muted_text': '#65727E', 'axis': '#26313D', 'reference': '#7A8590', 'missing': '#C4CBD2', 'light_background': '#F3F7FA', 'background': '#FFFFFF'}
ENDPOINT_MARKERS: Mapping[str, str] = {'Response': 'o', 'PFS': 's', 'OS': '^'}
ENDPOINT_LINESTYLES: Mapping[str, str] = {'Response': '-', 'PFS': '--', 'OS': '-.'}
MODALITY_HATCHES: Mapping[str, str] = {'Metagenomics': '///', 'Radiomics': '\\\\', 'Blood': '...', 'Urine': 'xxx'}
MODALITY_MARKERS: Mapping[str, str] = {'Metagenomics': 'o', 'Radiomics': 's', 'Blood': 'D', 'Urine': '^'}
MODALITY_LINESTYLES: Mapping[str, str] = {'Metagenomics': '-', 'Radiomics': '--', 'Blood': ':', 'Urine': '-.'}
MBP_MARKERS: Mapping[str, str] = {'MBP-I': 'o', 'MBP-II': 's', 'MBP-III': '^'}
MBP_LINESTYLES: Mapping[str, str] = {'MBP-I': '-', 'MBP-II': '--', 'MBP-III': ':'}
MBP_HATCHES: Mapping[str, str] = {'MBP-I': '', 'MBP-II': '///', 'MBP-III': 'xxx'}

def mm_to_inch(value_mm: float) -> float:
    return float(value_mm) / MM_PER_INCH

def figure_size(width: str='double', height_mm: float=120.0) -> tuple[float, float]:
    if width not in FIGURE_WIDTH_MM:
        allowed = ', '.join(sorted(FIGURE_WIDTH_MM))
        raise ValueError(f'Unknown width preset {width!r}; choose one of: {allowed}')
    if not 0 < float(height_mm) <= MAX_FIGURE_HEIGHT_MM:
        raise ValueError(f'height_mm must be > 0 and <= {MAX_FIGURE_HEIGHT_MM:g}; got {height_mm!r}')
    return (mm_to_inch(FIGURE_WIDTH_MM[width]), mm_to_inch(height_mm))

def rc_params() -> dict[str, object]:
    return {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 6.5, 'axes.labelsize': 7.0, 'axes.titlesize': 7.0, 'axes.titleweight': 'normal', 'axes.linewidth': 0.6, 'axes.edgecolor': NEUTRAL_COLORS['axis'], 'axes.labelcolor': NEUTRAL_COLORS['text'], 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'xtick.color': NEUTRAL_COLORS['axis'], 'ytick.color': NEUTRAL_COLORS['axis'], 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.fontsize': 6.0, 'legend.title_fontsize': 6.5, 'legend.frameon': False, 'lines.linewidth': 1.0, 'lines.markersize': 4.0, 'lines.markeredgewidth': 0.4, 'patch.linewidth': 0.5, 'text.color': NEUTRAL_COLORS['text'], 'figure.facecolor': NEUTRAL_COLORS['background'], 'axes.facecolor': NEUTRAL_COLORS['background'], 'savefig.facecolor': NEUTRAL_COLORS['background'], 'savefig.edgecolor': NEUTRAL_COLORS['background'], 'savefig.transparent': False, 'savefig.dpi': 450, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False}

def apply_theme() -> None:
    mpl.rcParams.update(rc_params())

@contextmanager
def theme_context():
    with mpl.rc_context(rc=rc_params()):
        yield

def subplots(*, width: str='double', height_mm: float=120.0, layout: str | None='constrained', **kwargs):
    return plt.subplots(figsize=figure_size(width=width, height_mm=height_mm), layout=layout, **kwargs)

def style_axis(ax: mpl.axes.Axes, *, grid: bool=False) -> mpl.axes.Axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(direction='out', width=0.6, length=2.5, pad=2)
    ax.grid(bool(grid), which='major', color='#D9E0E6', linewidth=0.35, alpha=0.65, zorder=0)
    return ax

def add_panel_label(ax: mpl.axes.Axes, label: str, *, x_pt: float=-8.0, y_pt: float=4.0) -> mpl.text.Text:
    normalized = str(label).strip().lower()
    if not normalized:
        raise ValueError('Panel label cannot be empty')
    transform = ax.transAxes + mpl.transforms.ScaledTranslation(x_pt / 72.0, y_pt / 72.0, ax.figure.dpi_scale_trans)
    return ax.text(0.0, 1.0, normalized, transform=transform, fontsize=8.0, fontweight='bold', fontstyle='normal', ha='left', va='bottom', color=NEUTRAL_COLORS['text'], clip_on=False)

def save_figure(fig: mpl.figure.Figure, output_stem: str | Path, *, formats: Sequence[str]=('pdf', 'svg', 'png'), dpi: int=450, close: bool=False, tight: bool=False, metadata: Mapping[str, str] | None=None) -> list[Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    requested = tuple((str(fmt).lower().lstrip('.') for fmt in formats))
    allowed = {'pdf', 'svg', 'png'}
    unsupported = sorted(set(requested) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported format(s): {', '.join(unsupported)}")
    width_in, height_in = fig.get_size_inches()
    height_mm = float(height_in) * MM_PER_INCH
    if height_mm > MAX_FIGURE_HEIGHT_MM + 0.01:
        raise ValueError(f'Figure height is {height_mm:.1f} mm; maximum is {MAX_FIGURE_HEIGHT_MM:.0f} mm')
    saved: list[Path] = []
    for fmt in requested:
        path = stem.with_suffix(f'.{fmt}')
        kwargs: dict[str, object] = {'format': fmt, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': 'tight' if tight else None, 'pad_inches': 0.02 if tight else 0.0}
        if fmt == 'png':
            kwargs['dpi'] = int(dpi)
        if metadata and fmt in {'pdf', 'svg'}:
            kwargs['metadata'] = dict(metadata)
        fig.savefig(path, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved

def semantic_style(labels: Iterable[str], *, colors: Mapping[str, str], markers: Mapping[str, str] | None=None, linestyles: Mapping[str, str] | None=None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label in labels:
        if label not in colors:
            raise KeyError(f'No semantic colour registered for {label!r}')
        result[label] = {'color': colors[label]}
        if markers and label in markers:
            result[label]['marker'] = markers[label]
        if linestyles and label in linestyles:
            result[label]['linestyle'] = linestyles[label]
    return result
