from avatar_paths import get_value, get_column
from pathlib import Path
import hashlib
import json
import math
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from lifelines import KaplanMeierFitter
PILOT_ROOT = Path(__file__).resolve().parent
STYLE_ROOT = PILOT_ROOT.parents[1]
SNAPSHOT_DIR = PILOT_ROOT / 'source_snapshot'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_352')
PANEL_DIR = OUTPUT_DIR / 'panels'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)
if str(STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLE_ROOT))
from figure_style import ENDPOINT_COLORS, MODALITY_COLORS, NEUTRAL_COLORS, PALETTE_ID, add_panel_label, apply_theme, figure_size, save_figure, style_axis
try:
    from figure_library.public_reporting import public_path_display
except ImportError:
    from public_reporting import public_path_display
FONT_SIZES = {'base': 6.4, 'panel_label': 8.5, 'panel_title': 7.2, 'axis_label': 6.8, 'tick': 6.0, 'legend': 5.9, 'legend_title': 6.1, 'annotation': 5.7, 'bar_value': 5.7, 'cohort_direct': 5.8, 'result_value': 6.0, 'risk_value': 5.7, 'risk_tick': 5.5, 'risk_header': 5.7, 'risk_axis_label': 6.2}
LAYOUT = {'outer_width_ratios': [0.4, 0.6], 'outer_height_ratios': [0.34, 0.33, 0.33], 'outer_left': 0.074, 'outer_right': 0.992, 'outer_top': 0.952, 'outer_bottom': 0.06, 'outer_wspace': 0.14, 'outer_hspace': 0.28, 'top_right_height_ratios': [0.57, 0.43], 'top_right_hspace': 0.42}
PANEL_SIZES_MM = {'a': (89.0, 55.0), 'b': (120.0, 52.0), 'c': (89.0, 150.0), 'd': (120.0, 45.0), 'e': (89.0, 82.0), 'f': (89.0, 82.0), 'g': (89.0, 75.0), 'h': (89.0, 75.0)}
EXPORT_STANDALONE_PANELS = True
apply_theme()
mpl.rcParams.update({'font.size': FONT_SIZES['base'], 'axes.labelsize': FONT_SIZES['axis_label'], 'axes.titlesize': FONT_SIZES['panel_title'], 'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'], 'legend.fontsize': FONT_SIZES['legend'], 'legend.title_fontsize': FONT_SIZES['legend_title'], 'axes.linewidth': 0.55, 'lines.linewidth': 1.0})
VERSION = 'v005'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_265')}{VERSION}"
COHORT_ORDER = ['Internal', '200', '33', '72', '133']
COHORT_LABELS = {'Internal': 'Internal', '200': 'Radiomics', '33': 'Urine', '72': 'Blood', '133': 'Microbiota'}
COHORT_COLORS = {'Internal': ENDPOINT_COLORS['PFS'], '200': MODALITY_COLORS['Radiomics'], '33': MODALITY_COLORS['Urine'], '72': MODALITY_COLORS['Blood'], '133': MODALITY_COLORS['Metagenomics']}
COHORT_MARKERS = {'Internal': 'o', '200': 's', '33': '^', '72': 'D', '133': 'P'}
COHORT_LINESTYLES = {'Internal': '-', '200': '--', '33': '-.', '72': (0, (1.2, 1.1)), '133': (0, (3.0, 1.0, 1.0, 1.0))}
RESPONSE_COLORS = {'Responder': ENDPOINT_COLORS['PFS'], 'Non-Responder': ENDPOINT_COLORS['Response']}
RESPONSE_LINESTYLES = {'Responder': '-', 'Non-Responder': '--'}
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
LIGHT = NEUTRAL_COLORS['light_background']
CENSOR = '#D7DEE4'
EXPECTED_HASHES = {get_value('figures.config.item_102'): 'CBF36C3DB976A80458F43EE5598A12F927CBA0993DBC72DC7F1B0DACF675C551', get_value('figures.config.item_101'): '29722DAB7A9B619C9AE7624271C7734ACB54516945B2D3DC6D73375D6DDB428E', get_value('figures.config.item_085'): '10AF3D6A0243590C5D9D59A09153D311A1E1EAED5A81586FFC26A5554C7DE116', get_value('figures.config.item_222'): '9F0804087DB445DB73394ADBE0A5BB3419B5B50B13099CB6641896E666752CA6', get_value('figures.config.item_172'): '3F1ACA3D7D4B34CB103CE5DF56A81010257A29E2C160FA8566FFA405E18D2CF8'}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()
for name, expected in EXPECTED_HASHES.items():
    path = SNAPSHOT_DIR / name
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f'Hash mismatch for {name}: {actual}')
all_df = pd.read_csv(SNAPSHOT_DIR / get_value('figures.config.item_102'))
summary_df = pd.read_csv(SNAPSHOT_DIR / get_value('figures.config.item_101'))
comp = pd.read_csv(SNAPSHOT_DIR / get_value('figures.config.item_085'))
fp = pd.read_csv(SNAPSHOT_DIR / get_value('figures.config.item_222'))
timeline_df = pd.read_csv(SNAPSHOT_DIR / get_value('figures.config.item_172'))
numeric_columns = ['Response', 'PFS_months', 'PFS_status', 'OS_months', 'OS_status']
for frame in [all_df, timeline_df]:
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
if len(all_df) != 547 or len(timeline_df) != 547:
    raise RuntimeError('Expected exactly 547 patients in the frozen snapshot')
expected_counts = {'Internal': 109, '200': 200, '33': 33, '72': 72, '133': 133}
actual_counts = all_df.groupby('Cohort').size().to_dict()
if actual_counts != expected_counts:
    raise RuntimeError(f'Unexpected cohort counts: {actual_counts}')
P_VALUES = {'response_pearson_chi2': 0.8317695369232401, 'cohort_pfs_logrank': 0.2925842832752559, 'cohort_os_logrank': 0.9843375977984964, 'response_pfs_descriptive_logrank': 2.1112076896811645e-23, 'response_os_descriptive_logrank': 1.0932435868666199e-24}
summary_by_cohort = summary_df.set_index('Cohort').loc[COHORT_ORDER]
fp_by_cohort = fp.set_index('Cohort').loc[COHORT_ORDER]
for cohort in COHORT_ORDER:
    for endpoint in ['PFS', 'OS']:
        d = all_df.loc[all_df['Cohort'].eq(cohort), [f'{endpoint}_months', f'{endpoint}_status']].dropna()
        kmf = KaplanMeierFitter().fit(d.iloc[:, 0], d.iloc[:, 1])
        observed = float(kmf.median_survival_time_)
        expected = float(summary_by_cohort.loc[cohort, f'median_{endpoint}_months'])
        if not np.isclose(observed, expected, rtol=0, atol=1e-10):
            raise RuntimeError(f'Median mismatch for {cohort} {endpoint}: {observed} vs {expected}')

def cohort_label(value):
    return COHORT_LABELS.get(value, str(value))

def fmt_p(value):
    return '$P$ < 0.001' if value < 0.001 else f'$P$ = {value:.3f}'
QA_ITEMS = []

def qa_tag(artist, *, name, role, panel, container='figure', check_data=False, allow_data_overlap=False):
    if hasattr(artist, 'set_gid'):
        artist.set_gid(f'qa:{name}')
    artist._qa_meta = {'name': name, 'role': role, 'panel': panel, 'container': container, 'check_data': bool(check_data), 'allow_data_overlap': bool(allow_data_overlap)}
    QA_ITEMS.append(artist)
    return artist

def panel_header(ax, title, panel, *, y=0.7, title_x=0.075):
    ax.axis('off')
    qa_tag(ax, name=f'{panel}-header', role='header_axes', panel=panel)
    panel_text = ax.text(0.0, y, panel, transform=ax.transAxes, ha='left', va='center', fontsize=FONT_SIZES['panel_label'], fontweight='bold', color=TEXT)
    title_text = ax.text(title_x, y, title, transform=ax.transAxes, ha='left', va='center', fontsize=FONT_SIZES['panel_title'], fontweight='bold', color=TEXT)
    qa_tag(panel_text, name=f'{panel}-label', role='panel_label', panel=panel)
    qa_tag(title_text, name=f'{panel}-title', role='panel_title', panel=panel)
    return (panel_text, title_text)

def decorate_axis(ax, title, panel):
    style_axis(ax)
    qa_tag(ax, name=f'{panel}-main', role='main_axes', panel=panel)
    title_text = ax.set_title(title, loc='left', fontsize=FONT_SIZES['panel_title'], fontweight='bold', pad=1.5)
    panel_text = add_panel_label(ax, panel, x_pt=-11.0, y_pt=1.5)
    panel_text.set_fontsize(FONT_SIZES['panel_label'])
    qa_tag(title_text, name=f'{panel}-title', role='panel_title', panel=panel)
    qa_tag(panel_text, name=f'{panel}-label', role='panel_label', panel=panel)
    return ax

def draw_panel_a(fig, spec, *, panel_label='a'):
    grid = spec.subgridspec(2, 1, height_ratios=[0.16, 0.84], hspace=0.02)
    header_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    panel_header(header_ax, 'Survival and follow-up', panel_label, y=0.76)
    d = fp_by_cohort.reset_index()
    y = np.arange(len(d))
    endpoint_specs = [('Median_PFS_months', 's', ENDPOINT_COLORS['PFS'], 'Median PFS'), ('Median_OS_months', '^', ENDPOINT_COLORS['OS'], 'Median OS'), ('Median_followup_reverseKM_months', 'D', MUTED, 'Median follow-up')]
    for i, row in d.iterrows():
        values = [float(row[item[0]]) for item in endpoint_specs if pd.notna(row[item[0]])]
        ax.plot([min(values), max(values)], [i, i], color=CENSOR, linewidth=0.8, zorder=1)
        for column, marker, color, _ in endpoint_specs:
            value = row[column]
            if pd.notna(value):
                ax.scatter(value, i, s=20, marker=marker, facecolor=color, edgecolor='white', linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([cohort_label(c) for c in d['Cohort']], color=TEXT)
    ax.invert_yaxis()
    xmax = max(36, int(math.ceil(np.nanmax(d.iloc[:, 1:].to_numpy(float)) / 6) * 6))
    ax.set_xlim(-1.0, xmax + 1.0)
    ax.set_xticks(np.arange(0, xmax, 5))
    ax.set_xlabel('Time (months)')
    style_axis(ax)
    qa_tag(ax, name=f'{panel_label}-main', role='main_axes', panel=panel_label)
    handles = [Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.35, markersize=4.1, label=label) for _, marker, color, label in endpoint_specs]
    legend = header_ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.075, -0.04, 0.925, 0.55), mode='expand', ncol=3, columnspacing=0.7, handletextpad=0.3, borderaxespad=0)
    qa_tag(legend, name=f'{panel_label}-legend', role='legend', panel=panel_label)
    return {'header': header_ax, 'main': ax, 'legend': legend}

def draw_panel_b(fig, spec, *, panel_label='b'):
    grid = spec.subgridspec(2, 2, height_ratios=[0.3, 0.7], hspace=0.02, wspace=0.2)
    header_ax = fig.add_subplot(grid[0, :])
    panel_header(header_ax, 'Event and censoring', panel_label, y=0.72)
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    endpoint_event_colors = {'PFS': ENDPOINT_COLORS['PFS'], 'OS': ENDPOINT_COLORS['OS']}
    for j, (ax, endpoint) in enumerate(zip(axes, ['PFS', 'OS'])):
        d = comp.loc[comp['Endpoint'].eq(endpoint)].set_index('Cohort').loc[COHORT_ORDER[::-1]]
        y = np.arange(len(d))
        event = d['Event_fraction'].to_numpy(float) * 100
        censored = d['Censored_fraction'].to_numpy(float) * 100
        color = endpoint_event_colors[endpoint]
        ax.barh(y, event, color=color, height=0.62, linewidth=0)
        ax.barh(y, censored, left=event, color=CENSOR, height=0.62, linewidth=0)
        for yi, ev, ce in zip(y, event, censored):
            if ev >= 24:
                text_artist = ax.text(ev / 2, yi, f'{ev:.0f}', ha='center', va='center', color='white', fontsize=FONT_SIZES['bar_value'])
                qa_tag(text_artist, name=f'{panel_label}-{endpoint}-event-{yi}', role='bar_value', panel=panel_label, allow_data_overlap=True)
            if ce >= 24:
                text_artist = ax.text(ev + ce / 2, yi, f'{ce:.0f}', ha='center', va='center', color=TEXT, fontsize=FONT_SIZES['bar_value'])
                qa_tag(text_artist, name=f'{panel_label}-{endpoint}-censor-{yi}', role='bar_value', panel=panel_label, allow_data_overlap=True)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Patients (%)')
        endpoint_title = ax.set_title(endpoint, fontsize=FONT_SIZES['axis_label'], fontweight='bold', pad=1)
        qa_tag(endpoint_title, name=f'{panel_label}-{endpoint}-subtitle', role='subplot_title', panel=panel_label)
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels([cohort_label(c) for c in d.index])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        style_axis(ax)
        qa_tag(ax, name=f'{panel_label}-{endpoint.lower()}-main', role='main_axes', panel=panel_label)
    handles = [Line2D([0], [0], color=TEXT, linewidth=4.0, label='Event / death'), Line2D([0], [0], color=CENSOR, linewidth=4.0, label='Censored')]
    legend = header_ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.72), ncol=2, columnspacing=0.9, handlelength=1.4, borderaxespad=0)
    qa_tag(legend, name=f'{panel_label}-legend', role='legend', panel=panel_label)
    return {'header': header_ax, 'main': axes, 'legend': legend}

def draw_panel_c(fig, spec, *, panel_label='c'):
    grid = spec.subgridspec(3, 1, height_ratios=[0.046, 0.052, 0.902], hspace=0.0)
    header_ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[1, 0])
    legend_ax.axis('off')
    qa_tag(legend_ax, name=f'{panel_label}-legend-axes', role='legend_axes', panel=panel_label)
    panel_header(header_ax, 'Patient-level outcomes', panel_label, y=0.62)
    body = grid[2, 0].subgridspec(1, 2, width_ratios=[0.79, 0.21], wspace=0.02)
    ax = fig.add_subplot(body[0, 0])
    label_ax = fig.add_subplot(body[0, 1], sharey=ax)
    t = timeline_df.sort_values('_y').copy()
    xmax = float(t['OS_months'].max()) * 1.015
    boundaries = []
    for index, cohort in enumerate(COHORT_ORDER):
        d = t.loc[t['Cohort'].eq(cohort)]
        start, end = (int(d['_y'].min()), int(d['_y'].max()))
        boundaries.append((cohort, start, end, len(d)))
        if index % 2 == 0:
            ax.axhspan(start - 0.5, end + 0.5, color=LIGHT, alpha=0.6, zorder=0)
            label_ax.axhspan(start - 0.5, end + 0.5, color=LIGHT, alpha=0.6, zorder=0)
        color = COHORT_COLORS[cohort]
        for _, row in d.iterrows():
            y = float(row['_y'])
            linestyle = '-' if row['Response'] == 1 else (0, (2.0, 1.4))
            os_time = float(row['OS_months'])
            ax.plot([0, os_time], [y, y], color=color, linewidth=0.38, linestyle=linestyle, alpha=0.78, solid_capstyle='butt', zorder=1)
            if pd.notna(row['PFS_months']) and pd.notna(row['PFS_status']):
                if int(row['PFS_status']) == 1:
                    ax.scatter(row['PFS_months'], y, s=1.8, marker='o', color=color, linewidth=0, zorder=3)
                else:
                    ax.scatter(row['PFS_months'], y, s=6.0, marker='|', color=color, linewidth=0.45, zorder=3)
            if pd.notna(row['OS_status']):
                if int(row['OS_status']) == 1:
                    ax.scatter(os_time, y, s=3.2, marker='x', color=TEXT, linewidth=0.45, zorder=4)
                else:
                    ax.scatter(os_time, y, s=6.0, marker='|', color=TEXT, linewidth=0.45, zorder=4)
    for index, (cohort, start, end, n) in enumerate(boundaries):
        if index > 0:
            ax.axhline(start - 0.5, color='#BFC8D0', linewidth=0.45, zorder=2)
            label_ax.axhline(start - 0.5, color='#BFC8D0', linewidth=0.45, zorder=2)
        label_text = label_ax.text(0.96, (start + end) / 2, f'{cohort_label(cohort)}\nn={n}', transform=label_ax.get_yaxis_transform(), ha='right', va='center', fontsize=FONT_SIZES['cohort_direct'], fontweight='bold', color=COHORT_COLORS[cohort], zorder=5)
        qa_tag(label_text, name=f'{panel_label}-cohort-{cohort}', role='cohort_label', panel=panel_label, container='axes')
    ax.set_xlim(0, xmax)
    ax.set_ylim(-1, len(t))
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel('Time from treatment initiation (months)')
    style_axis(ax)
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(ax.get_ylim())
    label_ax.axis('off')
    qa_tag(ax, name=f'{panel_label}-main', role='main_axes', panel=panel_label)
    qa_tag(label_ax, name=f'{panel_label}-label-band', role='label_axes', panel=panel_label)
    handles = [Line2D([0], [0], color=TEXT, linestyle='-', linewidth=0.9, label='Responder'), Line2D([0], [0], color=TEXT, linestyle='--', linewidth=0.9, label='Non-responder'), Line2D([0], [0], marker='o', color='none', markerfacecolor=TEXT, markeredgewidth=0, markersize=3.0, label='PFS event'), Line2D([0], [0], marker='x', color=TEXT, linestyle='none', markersize=3.2, label='Death'), Line2D([0], [0], marker='|', color=TEXT, linestyle='none', markersize=4.0, label='Censoring')]
    legend = legend_ax.legend(handles=handles, loc='center', ncol=3, columnspacing=0.7, handletextpad=0.28, borderaxespad=0)
    qa_tag(legend, name=f'{panel_label}-legend', role='legend', panel=panel_label)
    return {'header': header_ax, 'legend_axes': legend_ax, 'main': ax, 'labels': label_ax, 'legend': legend}

def draw_panel_d(fig, spec, *, panel_label='d'):
    grid = spec.subgridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.02)
    header_ax = fig.add_subplot(grid[0, 0])
    panel_header(header_ax, 'Objective response', panel_label, y=0.78)
    p_text = header_ax.text(1.0, 0.78, 'Pearson $\\chi^2$ ' + fmt_p(P_VALUES['response_pearson_chi2']), transform=header_ax.transAxes, ha='right', va='center', fontsize=FONT_SIZES['annotation'], color=TEXT)
    qa_tag(p_text, name=f'{panel_label}-pvalue', role='pvalue', panel=panel_label)
    orr_header = header_ax.text(0.755, 0.02, 'ORR (n/N)', transform=header_ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['annotation'], fontweight='bold', color=MUTED)
    qa_tag(orr_header, name=f'{panel_label}-result-header', role='column_header', panel=panel_label)
    body = grid[1, 0].subgridspec(1, 2, width_ratios=[0.74, 0.26], wspace=0.02)
    ax = fig.add_subplot(body[0, 0])
    value_ax = fig.add_subplot(body[0, 1], sharey=ax)
    d = summary_by_cohort.reset_index().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    for yi, row in d.iterrows():
        cohort = row['Cohort']
        rate = float(row['ORR']) * 100
        low = float(row['ORR_CI_lower']) * 100
        high = float(row['ORR_CI_upper']) * 100
        ax.errorbar(rate, yi, xerr=[[rate - low], [high - rate]], fmt=COHORT_MARKERS[cohort], color=COHORT_COLORS[cohort], ecolor=COHORT_COLORS[cohort], markersize=3.4, markeredgecolor='white', markeredgewidth=0.3, elinewidth=0.9, capsize=1.8, capthick=0.7, zorder=3)
        value_text = value_ax.text(0.02, yi, f"{rate:.1f}%  ({int(row['Responders'])}/{int(row['N'])})", transform=value_ax.get_yaxis_transform(), ha='left', va='center', fontsize=FONT_SIZES['result_value'], color=TEXT)
        qa_tag(value_text, name=f'{panel_label}-value-{cohort}', role='result_value', panel=panel_label, container='axes')
    ax.set_yticks(y)
    ax.set_yticklabels([cohort_label(c) for c in d['Cohort']])
    ax.set_xlim(25, 75)
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.set_xlabel('Objective response rate (%)')
    style_axis(ax)
    value_ax.set_xlim(0, 1)
    value_ax.set_ylim(ax.get_ylim())
    value_ax.axis('off')
    qa_tag(ax, name=f'{panel_label}-main', role='main_axes', panel=panel_label)
    qa_tag(value_ax, name=f'{panel_label}-value-band', role='value_axes', panel=panel_label)
    return {'header': header_ax, 'main': ax, 'values': value_ax}

def risk_count(durations, time):
    return int(np.sum(np.asarray(durations, dtype=float) >= float(time)))

def endpoint_ticks(endpoint):
    return (48, np.arange(0, 49, 12)) if endpoint == 'PFS' else (60, np.arange(0, 61, 12))

def km_settings(endpoint, mode):
    if mode == 'cohort':
        p_value = P_VALUES['cohort_pfs_logrank'] if endpoint == 'PFS' else P_VALUES['cohort_os_logrank']
        labels = COHORT_ORDER
    else:
        p_value = P_VALUES['response_pfs_descriptive_logrank'] if endpoint == 'PFS' else P_VALUES['response_os_descriptive_logrank']
        labels = ['Responder', 'Non-Responder']
    return (p_value, labels)

def plot_km_main(ax, endpoint, mode, p_value, *, panel_label, show_y):
    if mode == 'cohort':
        labels = COHORT_ORDER
        selectors = {label: all_df['Cohort'].eq(label) for label in labels}
        colors = COHORT_COLORS
        linestyles = COHORT_LINESTYLES
    else:
        labels = ['Responder', 'Non-Responder']
        selectors = {'Responder': all_df['Response'].eq(1), 'Non-Responder': all_df['Response'].eq(0)}
        colors = RESPONSE_COLORS
        linestyles = RESPONSE_LINESTYLES
    max_time, ticks = endpoint_ticks(endpoint)
    risk_groups = []
    medians = {}
    handles = []
    for label in labels:
        d = all_df.loc[selectors[label], [f'{endpoint}_months', f'{endpoint}_status']].dropna()
        kmf = KaplanMeierFitter().fit(d.iloc[:, 0], d.iloc[:, 1], label=str(label))
        kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=colors[label], linestyle=linestyles[label], linewidth=1.05, censor_styles={'marker': '|', 'ms': 3.5, 'mew': 0.58})
        medians[label] = float(kmf.median_survival_time_)
        risk_groups.append((label, d.iloc[:, 0].to_numpy(float)))
        handles.append(Line2D([0], [0], color=colors[label], linestyle=linestyles[label], linewidth=1.15, label=cohort_label(label)))
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', labelbottom=False, length=0)
    ax.set_xlabel('')
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    if show_y:
        ax.set_ylabel('Progression-free survival' if endpoint == 'PFS' else 'Overall survival')
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    style_axis(ax)
    prefix = 'Descriptive ' if mode == 'response' else ''
    p_text = ax.text(0.985, 0.965, prefix + fmt_p(p_value), transform=ax.transAxes, ha='right', va='top', fontsize=FONT_SIZES['annotation'], fontweight='bold', color=TEXT, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.86, 'pad': 0.6})
    qa_tag(p_text, name=f'{panel_label}-pvalue', role='pvalue', panel=panel_label, container='axes', allow_data_overlap=True)
    return (handles, risk_groups, ticks, medians)

def draw_risk_table(label_ax, number_ax, groups, ticks, *, panel_label, show_labels):
    count = len(groups)
    tick_start = float(ticks[0])
    tick_end = float(ticks[-1])
    tick_span = max(tick_end - tick_start, 1.0)
    number_ax.set_xlim(tick_start - 0.07 * tick_span, tick_end + 0.03 * tick_span)
    number_ax.set_ylim(-0.95, count + 0.35)
    number_ax.axis('off')
    qa_tag(number_ax, name=f'{panel_label}-risk-values', role='risk_axes', panel=panel_label)
    if label_ax is not None:
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(-0.95, count + 0.35)
        label_ax.axis('off')
        qa_tag(label_ax, name=f'{panel_label}-risk-labels', role='risk_label_axes', panel=panel_label)
    for x in ticks:
        tick_text = number_ax.text(x, count + 0.02, f'{int(x)}', ha='center', va='bottom', fontsize=FONT_SIZES['risk_tick'], color=MUTED)
        qa_tag(tick_text, name=f'{panel_label}-risk-tick-{int(x)}', role='risk_tick', panel=panel_label)
    for index, (label, durations) in enumerate(groups):
        y = count - 1 - index
        if show_labels and label_ax is not None:
            label_text = label_ax.text(0.98, y, cohort_label(label), transform=label_ax.get_yaxis_transform(), ha='right', va='center', fontsize=FONT_SIZES['risk_value'], color=TEXT)
            qa_tag(label_text, name=f'{panel_label}-risk-label-{label}', role='risk_label', panel=panel_label)
        for x in ticks:
            value_text = number_ax.text(x, y, str(risk_count(durations, x)), ha='center', va='center', fontsize=FONT_SIZES['risk_value'], color=TEXT)
            qa_tag(value_text, name=f'{panel_label}-risk-{label}-{int(x)}', role='risk_value', panel=panel_label)
    if show_labels and label_ax is not None:
        header_text = label_ax.text(0.98, count + 0.02, 'No. at risk', transform=label_ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=FONT_SIZES['risk_header'], fontweight='bold', color=TEXT)
        qa_tag(header_text, name=f'{panel_label}-risk-header', role='risk_header', panel=panel_label)
    axis_text = number_ax.text(0.5, -0.08, 'Time (months)', transform=number_ax.transAxes, ha='center', va='top', fontsize=FONT_SIZES['risk_axis_label'], color=TEXT)
    qa_tag(axis_text, name=f'{panel_label}-risk-axis-label', role='axis_label', panel=panel_label)

def draw_km_content(main_ax, risk_label_ax, risk_number_ax, *, endpoint, mode, panel_label, show_y, show_risk_labels):
    p_value, _ = km_settings(endpoint, mode)
    handles, groups, ticks, medians = plot_km_main(main_ax, endpoint, mode, p_value, panel_label=panel_label, show_y=show_y)
    title_group = 'cohort' if mode == 'cohort' else 'response'
    decorate_axis(main_ax, f'{endpoint} by {title_group}', panel_label)
    draw_risk_table(risk_label_ax, risk_number_ax, groups, ticks, panel_label=panel_label, show_labels=show_risk_labels)
    return {'main': main_ax, 'risk_labels': risk_label_ax, 'risk_values': risk_number_ax, 'handles': handles, 'medians': medians}

def draw_km_pair(fig, spec, mode, panels):
    row_ratios = [3.0, 0.52, 1.85] if mode == 'cohort' else [3.25, 0.52, 1.45]
    grid = spec.subgridspec(3, 4, height_ratios=row_ratios, width_ratios=[0.28, 0.72, 0.08, 0.92], hspace=0.05, wspace=0.0)
    main_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    legend_ax = fig.add_subplot(grid[1, 1:])
    risk_label_ax = fig.add_subplot(grid[2, 0])
    risk_axes = [fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[2, 3])]
    legend_ax.axis('off')
    qa_tag(legend_ax, name=f'{panels[0]}-{panels[1]}-legend-axes', role='legend_axes', panel=f'{panels[0]}{panels[1]}')
    contents = []
    for index, endpoint in enumerate(['PFS', 'OS']):
        contents.append(draw_km_content(main_axes[index], risk_label_ax if index == 0 else None, risk_axes[index], endpoint=endpoint, mode=mode, panel_label=panels[index], show_y=index == 0, show_risk_labels=index == 0))
    _, legend_labels = km_settings('PFS', mode)
    legend = legend_ax.legend(handles=contents[0]['handles'], labels=[cohort_label(label) for label in legend_labels], loc='center', ncol=len(legend_labels), columnspacing=0.9, handlelength=1.7, handletextpad=0.35, borderaxespad=0)
    qa_tag(legend, name=f'{panels[0]}-{panels[1]}-legend', role='legend', panel=f'{panels[0]}{panels[1]}')
    return {'main': main_axes, 'risk_labels': risk_label_ax, 'risk_values': risk_axes, 'legend': legend, 'contents': contents}

def draw_km_panel(fig, spec, *, endpoint, mode, panel_label):
    row_ratios = [3.0, 0.62, 1.92] if mode == 'cohort' else [3.25, 0.62, 1.55]
    grid = spec.subgridspec(3, 2, height_ratios=row_ratios, width_ratios=[0.24, 0.76], hspace=0.05, wspace=0.0)
    main_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[1, :])
    risk_label_ax = fig.add_subplot(grid[2, 0])
    risk_number_ax = fig.add_subplot(grid[2, 1])
    legend_ax.axis('off')
    content = draw_km_content(main_ax, risk_label_ax, risk_number_ax, endpoint=endpoint, mode=mode, panel_label=panel_label, show_y=True, show_risk_labels=True)
    _, legend_labels = km_settings(endpoint, mode)
    ncol = 3 if mode == 'cohort' else 2
    legend = legend_ax.legend(handles=content['handles'], labels=[cohort_label(label) for label in legend_labels], loc='center', ncol=ncol, columnspacing=0.9, handlelength=1.7, handletextpad=0.35, borderaxespad=0)
    qa_tag(legend, name=f'{panel_label}-legend', role='legend', panel=panel_label)
    return {'main': main_ax, 'legend_axes': legend_ax, 'legend': legend, 'risk_labels': risk_label_ax, 'risk_values': risk_number_ax}

def bbox_metrics(a, b):
    ix0, iy0 = (max(a.x0, b.x0), max(a.y0, b.y0))
    ix1, iy1 = (min(a.x1, b.x1), min(a.y1, b.y1))
    iw, ih = (max(0.0, ix1 - ix0), max(0.0, iy1 - iy0))
    intersection = iw * ih
    area_a = max(a.width * a.height, 1e-09)
    area_b = max(b.width * b.height, 1e-09)
    union = area_a + area_b - intersection
    return {'intersection': intersection, 'overlap_min': intersection / min(area_a, area_b), 'iou': intersection / union if union else 0.0}

def collision_threshold(role_a, role_b, cross_axes=False):
    dense_roles = {'tick', 'risk_value', 'risk_tick', 'risk_label'}
    dense_count = sum((role in dense_roles for role in (role_a, role_b)))
    if dense_count == 2:
        warning, critical = (0.35, 0.75)
    elif dense_count == 1:
        warning, critical = (0.18, 0.5)
    else:
        warning, critical = (0.06, 0.28)
    if cross_axes and dense_count < 2:
        warning = min(warning, 0.03)
    return (warning, critical)

def audit_figure_layout(fig, figure_name, tagged_items):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    one_pt_px = fig.dpi / 72.0
    safe_pad_px = 2.0 * one_pt_px
    items = []
    seen = set()

    def append_item(artist, role, panel, name):
        if id(artist) in seen or not artist.get_visible():
            return
        if isinstance(artist, Text) and (not artist.get_text().strip()):
            return
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            return
        if bbox.width <= 0 or bbox.height <= 0:
            return
        seen.add(id(artist))
        axes = getattr(artist, 'axes', None)
        items.append({'artist': artist, 'name': str(name), 'role': str(role), 'panel': str(panel), 'axes_id': id(axes) if axes is not None else None, 'bbox': bbox})
    for artist in tagged_items:
        if getattr(artist, 'get_figure', lambda: None)() is not fig:
            continue
        meta = getattr(artist, '_qa_meta', {})
        if isinstance(artist, (Text, Legend)):
            append_item(artist, meta.get('role', 'tagged'), meta.get('panel', 'unknown'), meta.get('name', f'artist-{id(artist)}'))
    for axis_index, ax in enumerate(fig.axes):
        if not ax.get_visible() or not ax.axison:
            continue
        axis_meta = getattr(ax, '_qa_meta', {})
        panel = axis_meta.get('panel', 'unknown')
        for label_index, text_artist in enumerate(list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
            append_item(text_artist, 'tick', panel, f'axis-{axis_index}-tick-{label_index}')
        for text_artist in [ax.xaxis.label, ax.yaxis.label]:
            append_item(text_artist, 'axis_label', panel, f'axis-{axis_index}-label-{id(text_artist)}')
    issues = []
    safe_bbox = Bbox.from_extents(figure_bbox.x0 + safe_pad_px, figure_bbox.y0 + safe_pad_px, figure_bbox.x1 - safe_pad_px, figure_bbox.y1 - safe_pad_px)
    for item in items:
        bbox = item['bbox']
        outside = bbox.x0 < figure_bbox.x0 or bbox.y0 < figure_bbox.y0 or bbox.x1 > figure_bbox.x1 or (bbox.y1 > figure_bbox.y1)
        low_clearance = bbox.x0 < safe_bbox.x0 or bbox.y0 < safe_bbox.y0 or bbox.x1 > safe_bbox.x1 or (bbox.y1 > safe_bbox.y1)
        if outside:
            issues.append({'type': 'figure_outside', 'severity': 'critical', 'item': item['name'], 'panel': item['panel']})
        elif low_clearance:
            issues.append({'type': 'low_figure_clearance', 'severity': 'low_clearance', 'item': item['name'], 'panel': item['panel']})
    for i, item_a in enumerate(items):
        for item_b in items[i + 1:]:
            metrics = bbox_metrics(item_a['bbox'], item_b['bbox'])
            if metrics['intersection'] <= 0:
                continue
            cross_axes = item_a['axes_id'] != item_b['axes_id']
            warning_threshold, critical_threshold = collision_threshold(item_a['role'], item_b['role'], cross_axes=cross_axes)
            ratio = metrics['overlap_min']
            if ratio >= critical_threshold:
                severity = 'critical'
            elif ratio >= warning_threshold:
                severity = 'warning'
            else:
                continue
            issues.append({'type': 'text_or_legend_overlap', 'severity': severity, 'item_a': item_a['name'], 'item_b': item_b['name'], 'panel_a': item_a['panel'], 'panel_b': item_b['panel'], 'overlap_min': round(float(ratio), 4), 'cross_axes': bool(cross_axes)})
    counts = {level: sum((issue['severity'] == level for issue in issues)) for level in ['critical', 'warning', 'low_clearance']}
    serialized_items = []
    for item in items:
        bbox = item['bbox']
        serialized_items.append({'name': item['name'], 'role': item['role'], 'panel': item['panel'], 'bbox_pt': [round(float(value) / one_pt_px, 3) for value in [bbox.x0, bbox.y0, bbox.x1, bbox.y1]]})
    return {'figure': {'name': figure_name, 'width_mm': round(float(fig.get_figwidth() * 25.4), 3), 'height_mm': round(float(fig.get_figheight() * 25.4), 3)}, 'summary': {**counts, 'pass': counts['critical'] == 0 and counts['warning'] == 0}, 'items': serialized_items, 'issues': issues}

def write_audit(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public_path_display(report), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = report['summary']

def make_panel_figure(panel):
    width_mm, height_mm = PANEL_SIZES_MM[panel]
    panel_fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    margins = {'a': (0.18, 0.99, 0.98, 0.15), 'b': (0.13, 0.99, 0.98, 0.17), 'c': (0.1, 0.99, 0.99, 0.075), 'd': (0.15, 0.99, 0.96, 0.18), 'e': (0.04, 0.99, 0.92, 0.06), 'f': (0.04, 0.99, 0.92, 0.06), 'g': (0.04, 0.99, 0.92, 0.065), 'h': (0.04, 0.99, 0.92, 0.065)}
    left, right, top, bottom = margins[panel]
    spec = panel_fig.add_gridspec(1, 1, left=left, right=right, top=top, bottom=bottom)[0, 0]
    if panel == 'a':
        artists = draw_panel_a(panel_fig, spec, panel_label='a')
    elif panel == 'b':
        artists = draw_panel_b(panel_fig, spec, panel_label='b')
    elif panel == 'c':
        artists = draw_panel_c(panel_fig, spec, panel_label='c')
    elif panel == 'd':
        artists = draw_panel_d(panel_fig, spec, panel_label='d')
    else:
        endpoint = 'PFS' if panel in {'e', 'g'} else 'OS'
        mode = 'cohort' if panel in {'e', 'f'} else 'response'
        artists = draw_km_panel(panel_fig, spec, endpoint=endpoint, mode=mode, panel_label=panel)
    return (panel_fig, artists)
QA_ITEMS.clear()
composite_fig = plt.figure(figsize=figure_size('double', height_mm=170), facecolor='white')
outer = composite_fig.add_gridspec(3, 2, width_ratios=LAYOUT['outer_width_ratios'], height_ratios=LAYOUT['outer_height_ratios'], left=LAYOUT['outer_left'], right=LAYOUT['outer_right'], top=LAYOUT['outer_top'], bottom=LAYOUT['outer_bottom'], wspace=LAYOUT['outer_wspace'], hspace=LAYOUT['outer_hspace'])
panel_a = draw_panel_a(composite_fig, outer[0, 0], panel_label='a')
top_right = outer[0, 1].subgridspec(2, 1, height_ratios=LAYOUT['top_right_height_ratios'], hspace=LAYOUT['top_right_hspace'])
panel_b = draw_panel_b(composite_fig, top_right[0], panel_label='b')
panel_d = draw_panel_d(composite_fig, top_right[1], panel_label='d')
panel_c = draw_panel_c(composite_fig, outer[1:, 0], panel_label='c')
panels_ef = draw_km_pair(composite_fig, outer[1, 1], 'cohort', ['e', 'f'])
panels_gh = draw_km_pair(composite_fig, outer[2, 1], 'response', ['g', 'h'])
composite_audit = audit_figure_layout(composite_fig, get_value('figures.config.item_266'), QA_ITEMS)
write_audit(composite_audit, OUTPUT_DIR / get_value('figures.config.item_118'))
if any((composite_audit['summary'][key] for key in ['critical', 'warning', 'low_clearance'])):
    raise RuntimeError('Composite layout audit found unresolved layout issues')
saved = save_figure(composite_fig, FIGURE_STEM, formats=('pdf', 'svg', 'png'), dpi=600, tight=False, metadata={'Title': get_value('figures.config.item_345'), 'Creator': 'Avatar-HCC Figure Harmonization'})
panel_audits = {}
panel_saved = []
if EXPORT_STANDALONE_PANELS:
    for panel in 'abcdefgh':
        QA_ITEMS.clear()
        panel_fig, _ = make_panel_figure(panel)
        panel_audit = audit_figure_layout(panel_fig, f"{get_value('figures.config.item_267')}{panel}{get_value('figures.config.item_365')}", QA_ITEMS)
        panel_audits[panel] = panel_audit
        panel_paths = save_figure(panel_fig, PANEL_DIR / f"{get_value('figures.config.item_267')}{panel}{get_value('figures.config.item_365')}", formats=('pdf', 'svg', 'png'), dpi=600, tight=False, metadata={'Title': f"Avatar-HCC Figure 2 panel {panel}{get_value('figures.config.item_343')}", 'Creator': 'Avatar-HCC Figure Harmonization'})
        panel_saved.extend(panel_paths)
        plt.close(panel_fig)
    combined_panel_audit = {'figure': {'name': get_value('figures.config.item_268')}, 'summary': {'critical': sum((report['summary']['critical'] for report in panel_audits.values())), 'warning': sum((report['summary']['warning'] for report in panel_audits.values())), 'low_clearance': sum((report['summary']['low_clearance'] for report in panel_audits.values())), 'pass': all((report['summary']['pass'] for report in panel_audits.values()))}, 'panels': panel_audits}
    write_audit(combined_panel_audit, OUTPUT_DIR / get_value('figures.config.item_167'))
    if any((combined_panel_audit['summary'][key] for key in ['critical', 'warning', 'low_clearance'])):
        raise RuntimeError('Standalone panel layout audit found unresolved layout issues')
plt.show()
expected_suffixes = ['.pdf', '.svg', '.png']
for suffix in expected_suffixes:
    path = FIGURE_STEM.with_suffix(suffix)
    if not path.exists() or path.stat().st_size < 1000:
        raise RuntimeError(f'Missing or unexpectedly small export: {path}')
    file_sha256(path)
panel_expected = [PANEL_DIR / f"{get_value('figures.config.item_267')}{panel}{get_value('figures.config.item_365')}{suffix}" for panel in 'abcdefgh' for suffix in ('.pdf', '.svg', '.png')]
for path in panel_expected:
    if not path.exists() or path.stat().st_size < 1000:
        raise RuntimeError(f'Missing or unexpectedly small panel export: {path}')
audit_paths = [OUTPUT_DIR / get_value('figures.config.item_118'), OUTPUT_DIR / get_value('figures.config.item_167')]
for path in audit_paths:
    if not path.exists() or path.stat().st_size < 100:
        raise RuntimeError(f'Missing layout audit: {path}')
width_in, height_in = composite_fig.get_size_inches()
if not np.isclose(width_in * 25.4, 183.0, atol=0.05):
    raise RuntimeError('Unexpected figure width')
if not np.isclose(height_in * 25.4, 170.0, atol=0.05):
    raise RuntimeError('Unexpected figure height')
