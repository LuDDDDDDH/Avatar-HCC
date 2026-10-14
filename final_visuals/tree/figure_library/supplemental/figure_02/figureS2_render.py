from __future__ import annotations
from avatar_paths import get_value, get_column
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure_library.figure_style import DIVERGING_COLORS, ENDPOINT_COLORS, ENDPOINT_LINESTYLES, ENDPOINT_MARKERS, MODALITY_COLORS, MODALITY_LINESTYLES, MODALITY_MARKERS, NEUTRAL_COLORS, PALETTE_ID, SEQUENTIAL_COLORS, apply_theme, style_axis
from figure_library.figure_qa import ArtistRegistry, audit_figure_layout, sha256_file, write_json
VERSION = 'v001'
PILOT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PILOT_ROOT / 'inputs'
OUTPUT_DIR = PILOT_ROOT / 'outputs' / get_value('figures.config.item_354')
PANEL_DIR = OUTPUT_DIR / 'panels'
FIGURE_STEM = OUTPUT_DIR / f"{get_value('figures.config.item_308')}{VERSION}"
FULL_WIDTH_MM = 180.0
FULL_HEIGHT_MM = 270.0
MM_PER_INCH = 25.4
DPI = 600
MIN_FONT_PT = 5.4
ORIGINAL_CIRCOS_DIR = Path(__import__('avatar_final_paths').resolve(get_value('figures.config.item_376')))
ORIGINAL_BOOTSTRAP_DIR = Path(__import__('avatar_final_paths').resolve(get_value('figures.config.item_330')))
ORIGINAL_PERFORMANCE_DIR = Path(__import__('avatar_final_paths').resolve(get_value('figures.config.item_378')))
TEXT = NEUTRAL_COLORS['text']
MUTED = NEUTRAL_COLORS['muted_text']
GRID = '#DCE3E9'
LIGHT = '#F2F6F9'
TEAL = '#2A8F78'
FULL_GREY = '#7A8590'
ROLE_COLORS = {'Best single': '#98C8E1', 'Best pair': '#7D70D1', 'MRU': TEAL, 'Full': FULL_GREY}
ROLE_MARKERS = {'Best single': 'o', 'Best pair': 's', 'MRU': 'D', 'Full': 's'}
ROLE_LINES = {'Best single': ':', 'Best pair': '--', 'MRU': '-', 'Full': '-.'}
DIV_CMAP = LinearSegmentedColormap.from_list(get_value('figures.config.item_362'), DIVERGING_COLORS)
SEQ_CMAP = LinearSegmentedColormap.from_list(get_value('figures.config.item_375'), SEQUENTIAL_COLORS)
PANEL_RECTS: Mapping[str, tuple[float, float, float, float]] = {'a': (0.045, 0.818, 0.925, 0.157), 'b': (0.045, 0.642, 0.925, 0.145), 'c': (0.045, 0.435, 0.925, 0.175), 'd': (0.045, 0.295, 0.925, 0.105), 'e': (0.045, 0.145, 0.925, 0.115), 'f': (0.045, 0.022, 0.925, 0.09)}
PANEL_SIZES_MM: Mapping[str, tuple[float, float]] = {'a': (180.0, 60.0), 'b': (180.0, 75.0), 'c': (180.0, 65.0), 'd': (180.0, 50.0), 'e': (180.0, 55.0), 'f': (180.0, 50.0)}
EXPECTED_FILES = (get_value('figures.config.item_212'), get_value('figures.config.item_226'), get_value('figures.config.item_186'), get_value('figures.config.item_169'), get_value('figures.config.item_066'), get_value('figures.config.item_065'), get_value('figures.config.item_187'), get_value('figures.config.item_170'), get_value('figures.config.item_254'), get_value('figures.config.item_253'), get_value('figures.config.item_093'), get_value('figures.config.item_091'), get_value('figures.config.item_092'), get_value('figures.config.item_106'), get_value('figures.config.item_094'), get_value('figures.config.item_090'), get_value('figures.config.item_088'), get_value('figures.config.item_087')) + tuple((f"{modality}_{endpoint}{get_value('figures.config.item_067')}" for modality in ('Metagenomics', 'Radiomics', 'Blood', 'Urine') for endpoint in ('Response', 'PFS', 'OS')))

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def _fmt_p(value: float) -> str:
    if value < 0.001:
        return 'P<0.001'
    if value < 0.01:
        return f'P={value:.3f}'
    return f'P={value:.2f}'

def _short(value: str, limit: int=23) -> str:
    text = str(value).replace('_', ' ')
    if len(text) <= limit:
        return text
    return text[:max(4, limit - 1)].rstrip() + '…'

def _add_axis(fig: mpl.figure.Figure, outer: Sequence[float], inner: Sequence[float], **kwargs):
    x, y, w, h = outer
    ix, iy, iw, ih = inner
    return fig.add_axes([x + ix * w, y + iy * h, iw * w, ih * h], **kwargs)

def _panel_letter(fig: mpl.figure.Figure, rect: Sequence[float], label: str, registry: ArtistRegistry):
    full_master = fig.get_figheight() * MM_PER_INCH > 100.0
    y_position = rect[1] + rect[3] + 0.002 if full_master else rect[1] + rect[3] - 0.032
    artist = fig.text(rect[0] - 0.024, y_position, label, fontsize=8.5, fontweight='bold', ha='left', va='bottom', color=TEXT)
    registry.tag(artist, name=f'panel-{label}-letter', role='panel_label', panel=label)

def _tag_axis(ax: mpl.axes.Axes, panel: str) -> mpl.axes.Axes:
    ax._qa_meta = {'panel': panel}
    return ax

def load_bundle() -> dict[str, Any]:
    missing = [name for name in EXPECTED_FILES if not (INPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing S2 frozen inputs: {missing}')

    def original_path(name: str) -> Path:
        if name == get_value('figures.config.item_226'):
            return ORIGINAL_BOOTSTRAP_DIR / name
        if 'circos_patient_order_and_values' in name or name.startswith(get_value('figures.config.item_310')):
            return ORIGINAL_CIRCOS_DIR / name
        return ORIGINAL_PERFORMANCE_DIR / name
    inputs = {name: {'path': str(INPUT_DIR / name), 'original_path': str(original_path(name)), 'sha256': _sha256(INPUT_DIR / name), 'bytes': (INPUT_DIR / name).stat().st_size} for name in EXPECTED_FILES}
    tables = {name: pd.read_csv(INPUT_DIR / name) for name in EXPECTED_FILES}
    expected_columns = {get_value('figures.config.item_212'): {'modal', 'feature', 'assigned_task', 'rank', 'mean_abs_shap'}, get_value('figures.config.item_226'): {'modal_clean', 'feature_clean', 'outcome_clean', 'bootstrap_topk_frequency', 'bootstrap_median_rank', 'median_rank_quality'}, get_value('figures.config.item_186'): {'short', 'n_modalities', 'Response_AUPRC_mean', 'Response_AUPRC_sd', 'AUPRC_rank'}, get_value('figures.config.item_066'): {'role', 'short', 'mean_predicted_probability', 'observed_response_rate', 'observed_CI_lower', 'observed_CI_upper'}, get_value('figures.config.item_065'): {'role', 'short', 'Response_Brier'}, get_value('figures.config.item_187'): {'role', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Balanced accuracy', 'F1'}, get_value('figures.config.item_254'): {'endpoint', 'role', 'time', 'AUC', 'AUC_CI_lower', 'AUC_CI_upper'}, get_value('figures.config.item_092'): {'model', 'Response_AUROC', 'Response_AUPRC', 'n_external'}, get_value('figures.config.item_090'): {'endpoint', 'model', 'HR_higher_vs_lower', 'HR_CI_lower', 'HR_CI_upper', 'logrank_P', 'n_lower_risk', 'n_higher_risk'}, get_value('figures.config.item_088'): {'model', 'mean_predicted_probability', 'observed_response_rate', 'observed_CI_lower', 'observed_CI_upper'}}
    for name, columns in expected_columns.items():
        absent = columns - set(tables[name].columns)
        if absent:
            raise ValueError(f'{name} missing columns: {sorted(absent)}')
    for modality in MODALITY_COLORS:
        for endpoint in ENDPOINT_COLORS:
            name = f"{modality}_{endpoint}{get_value('figures.config.item_067')}"
            required = {'sample', 'True_Score', 'Pred_Score', 'shared_shap_sum', 'specific_shap_sum'}
            absent = required - set(tables[name].columns)
            if absent:
                raise ValueError(f'{name} missing columns: {sorted(absent)}')
            if len(tables[name]) != 109:
                raise ValueError(f'Unexpected patient count in {name}: {len(tables[name])}')
    numeric_qa = {'figure': 'Supplementary Figure S2', 'version': VERSION, 'mode': 'frozen-table structural and arithmetic identities only; no refitting', 'files': [{'file': name, 'shape': list(tables[name].shape), **inputs[name], 'pass': True} for name in EXPECTED_FILES], 'checks': {'circos_tables': 12, 'patients_per_circos_table': 109, 'bootstrap_rows': int(len(tables[get_value('figures.config.item_226')])), 'response_models': int(len(tables[get_value('figures.config.item_186')])), 'time_dependent_auc_rows': int(len(tables[get_value('figures.config.item_254')])), 'external_survival_rows': int(len(tables[get_value('figures.config.item_090')])), 'all_hashes_recorded': True}, 'pass': True}
    return {'tables': tables, 'input_manifest': inputs, 'numeric_qa': numeric_qa}

def draw_panel_a(fig, rect, bundle, registry):
    tables = bundle['tables']
    selection = tables[get_value('figures.config.item_212')]
    modalities = list(MODALITY_COLORS)
    endpoints = list(ENDPOINT_COLORS)
    axes = []
    gap = 0.032
    width = (1.0 - gap * 3) / 4.0
    for index, modality in enumerate(modalities):
        ax = _add_axis(fig, rect, (index * (width + gap), 0.09, width, 0.86), projection='polar')
        _tag_axis(ax, 'a')
        axes.append(ax)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_ylim(0.05, 1.57)
        ax.set_axis_off()
        rows = selection.loc[selection['modal'] == modality].copy()
        rows = rows.sort_values(['assigned_task', 'rank'], kind='stable')
        features = [str(v) for v in rows['feature'].tolist()[:5]]
        sector_gap = 0.14
        sector_width = (2 * np.pi - 3 * sector_gap) / 3
        for endpoint_index, endpoint in enumerate(endpoints):
            data = tables[f"{modality}_{endpoint}{get_value('figures.config.item_067')}"]
            theta0 = endpoint_index * (sector_width + sector_gap)
            theta_edges = np.linspace(theta0, theta0 + sector_width, len(data) + 1)
            theta = (theta_edges[:-1] + theta_edges[1:]) / 2
            present_features = [feature for feature in features if feature in data.columns]
            matrix = data[present_features].to_numpy(float).T
            if matrix.shape[0] < 5:
                pad = np.zeros((5 - matrix.shape[0], matrix.shape[1]))
                matrix = np.vstack([matrix, pad])
            matrix = np.clip(matrix[:5], -2.5, 2.5)
            r_edges = np.linspace(1.02, 1.5, 6)
            ax.pcolormesh(theta_edges, r_edges, matrix, cmap=DIV_CMAP, vmin=-2.5, vmax=2.5, shading='flat')
            for radius in r_edges:
                ax.plot([theta0, theta0 + sector_width], [radius, radius], color='#D9E0E6', lw=0.22)
            pred = data['Pred_Score'].to_numpy(float)
            true = data['True_Score'].to_numpy(float)
            ax.plot(theta, 0.38 + 0.24 * pred, color=ENDPOINT_COLORS[endpoint], lw=0.75)
            stride = 5
            ax.scatter(theta[::stride], 0.38 + 0.24 * true[::stride], s=3.0, facecolor='white', edgecolor=ENDPOINT_COLORS[endpoint], linewidth=0.35, zorder=4)
            shap = data['shared_shap_sum'].to_numpy(float) + data['specific_shap_sum'].to_numpy(float)
            scale = max(float(np.nanmax(np.abs(shap))), 1e-12)
            ax.plot(theta, 0.76 + 0.14 * shap / scale, color=MODALITY_COLORS[modality], lw=0.55)
            mid = theta0 + sector_width / 2
            ax.text(mid, 1.56, endpoint, color=ENDPOINT_COLORS[endpoint], fontsize=5.4, fontweight='bold', ha='center', va='center', rotation=np.degrees(-mid + np.pi / 2), rotation_mode='anchor', clip_on=False)
        ax.text(0.5, 0.5, modality, transform=ax.transAxes, ha='center', va='center', fontsize=6.0, fontweight='bold', color=MODALITY_COLORS[modality])
    handles = [Line2D([0], [0], color=TEXT, lw=0.8, label='Predicted'), Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=TEXT, markersize=3.6, label='Observed'), Line2D([0], [0], color=TEAL, lw=0.8, label='Shared + specific SHAP')]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(rect[0] + rect[2] * 0.55, rect[1] - 0.02), ncol=3, fontsize=5.4, frameon=False, handlelength=1.4, columnspacing=1.1)
    registry.tag(legend, name='a-legend', role='legend', panel='a')
    return axes

def draw_panel_b(fig, rect, bundle, registry):
    data = bundle['tables'][get_value('figures.config.item_226')].copy()
    modalities = list(MODALITY_COLORS)
    endpoints = list(ENDPOINT_COLORS)
    axes = []
    for index, modality in enumerate(modalities):
        row, col = divmod(index, 2)
        inner_y = 0.62 if row == 0 else 0.08
        ax = _add_axis(fig, rect, (0.06 + col * 0.5, inner_y, 0.4, 0.28))
        _tag_axis(ax, 'b')
        axes.append(ax)
        subset = data.loc[data['modal_clean'] == modality].copy()
        order = subset.groupby('feature_clean', sort=False)['bootstrap_topk_frequency'].max().sort_values(ascending=False).head(5).index.tolist()
        order = list(reversed(order))
        y_map = {feature: i for i, feature in enumerate(order)}
        x_map = {endpoint: i for i, endpoint in enumerate(endpoints)}
        for _, row in subset.iterrows():
            feature = str(row['feature_clean'])
            endpoint = str(row['outcome_clean'])
            if feature not in y_map or endpoint not in x_map:
                continue
            frequency = float(row['bootstrap_topk_frequency'])
            quality = float(row['median_rank_quality'])
            ax.scatter(x_map[endpoint], y_map[feature], s=8 + 42 * frequency, c=[MODALITY_COLORS[modality]], edgecolor=TEXT, linewidth=0.25)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xticks(range(3), endpoints, rotation=25, ha='right', fontsize=5.4)
        ax.set_yticks(range(len(order)), [_short(v, 14) for v in order], fontsize=5.4)
        ax.tick_params(length=0, pad=1.0)
        ax.set_title(modality, color=MODALITY_COLORS[modality], fontsize=6.2, fontweight='bold', pad=2)
        for x in range(3):
            ax.axvline(x, color=GRID, lw=0.35, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    size_handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#98C8E1', markeredgecolor=TEXT, markeredgewidth=0.25, markersize=math.sqrt(8 + 42 * value) * 0.58, label=f'{int(value * 100)}%') for value in (0.25, 0.6, 1.0)]
    full_master = fig.get_figheight() * MM_PER_INCH > 100.0
    legend_y = rect[1] - 0.002 if full_master else rect[1] + 0.01
    legend = fig.legend(handles=size_handles, loc='upper center', bbox_to_anchor=(rect[0] + rect[2] * 0.53, legend_y), ncol=3, title=None, fontsize=5.4, frameon=False, handletextpad=0.2, columnspacing=0.8)
    registry.tag(legend, name='b-size-legend', role='legend', panel='b')
    return axes

def draw_panel_c(fig, rect, bundle, registry):
    tables = bundle['tables']
    axes = []
    ax1 = _add_axis(fig, rect, (0.045, 0.1, 0.31, 0.84))
    _tag_axis(ax1, 'c')
    axes.append(ax1)
    ax2 = _add_axis(fig, rect, (0.405, 0.12, 0.255, 0.8))
    _tag_axis(ax2, 'c')
    axes.append(ax2)
    ax3 = _add_axis(fig, rect, (0.72, 0.17, 0.28, 0.74))
    _tag_axis(ax3, 'c')
    axes.append(ax3)
    perf = tables[get_value('figures.config.item_186')].sort_values('Response_AUPRC_mean')
    y = np.arange(len(perf))
    for i, (_, row) in enumerate(perf.iterrows()):
        role = 'MRU' if str(row['short']) == 'M+R+U' else 'Full' if str(row['short']) == 'M+R+B+U' else 'Other'
        color = ROLE_COLORS.get(role, '#E49592')
        marker = 'D' if role == 'MRU' else 's' if role == 'Full' else 'o'
        mean, sd = (float(row['Response_AUPRC_mean']), float(row['Response_AUPRC_sd']))
        ax1.errorbar(mean, i, xerr=sd, fmt=marker, ms=3.2, color=color, ecolor=color, elinewidth=0.65, capsize=1.5, markeredgecolor=TEXT, markeredgewidth=0.25)
        ax1.text(min(mean + sd + 0.012, 0.985), i, f'{mean:.3f}', fontsize=5.4, va='center', ha='left', color=TEXT, fontweight='bold' if role in {'MRU', 'Full'} else 'normal')
    ax1.set_yticks(y, perf['short'].astype(str), fontsize=5.4)
    ax1.set_xlim(0.5, 1.01)
    ax1.set_xlabel('Internal Response AUPRC')
    style_axis(ax1, grid=True)
    cal = tables[get_value('figures.config.item_066')]
    brier = tables[get_value('figures.config.item_065')].set_index('role')
    ax2.plot([0, 1], [0, 1], color=FULL_GREY, lw=0.7, ls='--', zorder=0)
    for role in ('Best single', 'Best pair', 'MRU', 'Full'):
        sub = cal.loc[cal['role'] == role].sort_values('mean_predicted_probability')
        if sub.empty:
            continue
        x = sub['mean_predicted_probability'].to_numpy(float)
        yy = sub['observed_response_rate'].to_numpy(float)
        lo = sub['observed_CI_lower'].to_numpy(float)
        hi = sub['observed_CI_upper'].to_numpy(float)
        bv = float(brier.loc[role, 'Response_Brier'])
        ax2.errorbar(x, yy, yerr=[yy - lo, hi - yy], color=ROLE_COLORS[role], marker=ROLE_MARKERS[role], ms=3.0, lw=0.8, elinewidth=0.55, capsize=1.3, ls=ROLE_LINES[role], label=f'{role} ({bv:.3f})')
    ax2.set_xlim(0.15, 0.95)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel('Predicted response probability')
    ax2.set_ylabel('Observed response rate')
    style_axis(ax2, grid=False)
    ax2.grid(False)
    leg = ax2.legend(title='Brier score', loc='upper left', fontsize=5.4, title_fontsize=5.5, handlelength=1.5, borderaxespad=0.2)
    registry.tag(leg, name='c-calibration-legend', role='legend', panel='c')
    threshold = tables[get_value('figures.config.item_187')].copy()
    roles = [r for r in ('Best single', 'Best pair', 'MRU', 'Full') if r in set(threshold['role'])]
    metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Balanced accuracy', 'F1']
    matrix = threshold.set_index('role').loc[roles, metrics].to_numpy(float)
    im = ax3.imshow(matrix, cmap=SEQ_CMAP, vmin=0.55, vmax=1.0, aspect='auto')
    ax3.set_xticks(range(len(metrics)), ['Sens.', 'Spec.', 'PPV', 'NPV', 'Bal. acc.', 'F1'], rotation=45, ha='right', fontsize=5.4)
    ax3.set_yticks(range(len(roles)), roles, fontsize=5.4)
    ax3.tick_params(length=0, pad=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if matrix[i, j] > 0.81 else TEXT
            ax3.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=5.4, color=color, fontweight='bold' if roles[i] == 'MRU' else 'normal')
    for spine in ax3.spines.values():
        spine.set_visible(False)
    return axes

def draw_panel_d(fig, rect, bundle, registry):
    source = bundle['tables'][get_value('figures.config.item_254')]
    summary = bundle['tables'][get_value('figures.config.item_253')]
    axes = []
    for index, endpoint in enumerate(('PFS', 'OS')):
        ax = _add_axis(fig, rect, (0.02 + index * 0.5, 0.16, 0.45, 0.75))
        _tag_axis(ax, 'd')
        axes.append(ax)
        for role in ('Best single', 'Best pair', 'MRU', 'Full'):
            sub = source.loc[(source['endpoint'] == endpoint) & (source['role'] == role)].sort_values('time')
            if sub.empty:
                continue
            x = sub['time'].to_numpy(float)
            yy = sub['AUC'].to_numpy(float)
            lo = sub['AUC_CI_lower'].to_numpy(float)
            hi = sub['AUC_CI_upper'].to_numpy(float)
            mean_row = summary.loc[(summary['endpoint'] == endpoint) & (summary['role'] == role)]
            mean_auc = float(mean_row.iloc[0]['mean_AUC']) if not mean_row.empty else float(np.mean(yy))
            ax.fill_between(x, lo, hi, color=ROLE_COLORS[role], alpha=0.1, linewidth=0)
            ax.plot(x, yy, color=ROLE_COLORS[role], marker=ROLE_MARKERS[role], ms=2.8, ls=ROLE_LINES[role], lw=0.85, label=f'{role} {mean_auc:.3f}')
        ax.set_ylim(0.55, 1.0)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Time-dependent AUC' if index == 0 else '')
        ax.set_title(endpoint, color=ENDPOINT_COLORS[endpoint], fontweight='bold', fontsize=6.5, pad=2)
        style_axis(ax, grid=True)
        if index == 1:
            ax.tick_params(labelleft=False)
        leg = ax.legend(loc='lower left', ncol=2, fontsize=5.4, handlelength=1.2, columnspacing=0.7, borderaxespad=0.2)
        registry.tag(leg, name=f'd-{endpoint}-legend', role='legend', panel='d')
    return axes

def draw_panel_e(fig, rect, bundle, registry):
    tables = bundle['tables']
    roc = tables[get_value('figures.config.item_093')]
    pr = tables[get_value('figures.config.item_091')]
    metrics = tables[get_value('figures.config.item_092')].set_index('model')
    cal = tables[get_value('figures.config.item_088')]
    brier = tables[get_value('figures.config.item_087')].set_index('model')
    axes = []
    ax1 = _add_axis(fig, rect, (0.03, 0.18, 0.25, 0.73))
    _tag_axis(ax1, 'e')
    axes.append(ax1)
    ax2 = _add_axis(fig, rect, (0.365, 0.18, 0.27, 0.73))
    _tag_axis(ax2, 'e')
    axes.append(ax2)
    ax3 = _add_axis(fig, rect, (0.73, 0.18, 0.26, 0.73))
    _tag_axis(ax3, 'e')
    axes.append(ax3)
    for model in MODALITY_COLORS:
        if model not in metrics.index:
            continue
        color = MODALITY_COLORS[model]
        marker = MODALITY_MARKERS[model]
        rr = roc.loc[roc['model'] == model]
        pp = pr.loc[pr['model'] == model]
        auroc = float(metrics.loc[model, 'Response_AUROC'])
        auprc = float(metrics.loc[model, 'Response_AUPRC'])
        ax1.plot(rr['FPR'], rr['TPR'], color=color, ls=MODALITY_LINESTYLES[model], lw=0.9, label=f'{model} {auroc:.3f}')
        ax2.plot(pp['Recall'], pp['Precision'], color=color, ls=MODALITY_LINESTYLES[model], lw=0.9, label=f'{model} {auprc:.3f}')
        cc = cal.loc[cal['model'] == model].sort_values('mean_predicted_probability')
        x = cc['mean_predicted_probability'].to_numpy(float)
        yy = cc['observed_response_rate'].to_numpy(float)
        lo = cc['observed_CI_lower'].to_numpy(float)
        hi = cc['observed_CI_upper'].to_numpy(float)
        bv = float(brier.loc[model, 'Response_Brier'])
        ax3.errorbar(x, yy, yerr=[yy - lo, hi - yy], color=color, marker=marker, ms=2.7, ls=MODALITY_LINESTYLES[model], lw=0.75, elinewidth=0.5, capsize=1.1, label=f'{model} {bv:.3f}')
    ax1.plot([0, 1], [0, 1], color=FULL_GREY, ls='--', lw=0.6)
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel='False-positive rate', ylabel='True-positive rate')
    ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision')
    ax3.plot([0, 1], [0, 1], color=FULL_GREY, ls='--', lw=0.6)
    ax3.set(xlim=(0.15, 0.9), ylim=(0, 1), xlabel='Predicted response probability', ylabel='Observed response rate')
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax3.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    legend_locations = {'AUROC': 'lower right', 'AUPRC': 'lower left', 'Brier': 'upper left'}
    for ax, title, legend_title in ((ax1, 'External ROC', 'AUROC'), (ax2, 'External PR', 'AUPRC'), (ax3, 'External calibration', 'Brier')):
        ax.set_title(title, fontsize=6.2, fontweight='bold', pad=2)
        style_axis(ax, grid=False)
        ax.grid(False)
        leg = ax.legend(title=legend_title, fontsize=5.4, title_fontsize=5.5, loc=legend_locations[legend_title], handlelength=1.3, borderaxespad=0.2)
        registry.tag(leg, name=f'e-{legend_title}-legend', role='legend', panel='e')
    return axes

def draw_panel_f(fig, rect, bundle, registry):
    data = bundle['tables'][get_value('figures.config.item_090')].copy()
    axes = []
    for index, endpoint in enumerate(('PFS', 'OS')):
        base_x = index * 0.5
        forest = _add_axis(fig, rect, (base_x + 0.045, 0.18, 0.215, 0.7))
        _tag_axis(forest, 'f')
        axes.append(forest)
        table_ax = _add_axis(fig, rect, (base_x + 0.275, 0.18, 0.215, 0.7))
        _tag_axis(table_ax, 'f')
        axes.append(table_ax)
        sub = data.loc[data['endpoint'] == endpoint].copy()
        sub['_order'] = sub['model'].map({m: i for i, m in enumerate(MODALITY_COLORS)})
        sub = sub.sort_values('_order')
        y = np.arange(len(sub))[::-1]
        for yi, (_, row) in zip(y, sub.iterrows()):
            model = str(row['model'])
            hr, lo, hi = (float(row[k]) for k in ('HR_higher_vs_lower', 'HR_CI_lower', 'HR_CI_upper'))
            forest.errorbar(hr, yi, xerr=[[hr - lo], [hi - hr]], fmt=MODALITY_MARKERS[model], color=MODALITY_COLORS[model], ecolor=MODALITY_COLORS[model], ms=3.5, elinewidth=0.7, capsize=1.4, markeredgecolor=TEXT, markeredgewidth=0.25)
            table_ax.text(0.0, yi, f"{hr:.2f} ({lo:.2f}–{hi:.2f}); {_fmt_p(float(row['logrank_P']))}; n={int(row['n_lower_risk'])}/{int(row['n_higher_risk'])}", fontsize=5.4, va='center', ha='left', color=TEXT)
        forest.axvline(1.0, color=FULL_GREY, ls='--', lw=0.7)
        forest.set_xscale('log')
        forest.set_xlim(0.4, 10.5)
        forest.set_ylim(-0.6, len(sub) - 0.4)
        forest.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0, 5.0, 10.0]))
        forest.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        short_models = {'Metagenomics': 'Meta', 'Radiomics': 'Radio', 'Blood': 'Blood', 'Urine': 'Urine'}
        forest.set_yticks(y, [short_models[str(v)] for v in sub['model']], fontsize=5.4)
        forest.set_xlabel('HR, higher vs lower frozen risk')
        forest.set_title(endpoint, color=ENDPOINT_COLORS[endpoint], fontweight='bold', fontsize=6.5, pad=2)
        style_axis(forest, grid=True)
        table_ax.set_xlim(0, 1)
        table_ax.set_ylim(forest.get_ylim())
        table_ax.axis('off')
        table_ax.text(0.0, len(sub) - 0.08, 'HR (95% CI); log-rank P; n lower/higher', fontsize=5.4, fontweight='bold', va='bottom', ha='left', color=TEXT)
    return axes
DRAWERS = {'a': draw_panel_a, 'b': draw_panel_b, 'c': draw_panel_c, 'd': draw_panel_d, 'e': draw_panel_e, 'f': draw_panel_f}

def build_full(bundle):
    fig = plt.figure(figsize=(FULL_WIDTH_MM / MM_PER_INCH, FULL_HEIGHT_MM / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    all_axes = []
    for panel, rect in PANEL_RECTS.items():
        all_axes.extend(DRAWERS[panel](fig, rect, bundle, registry))
        _panel_letter(fig, rect, panel, registry)
    return (fig, registry, all_axes)

def build_panel(panel: str, bundle):
    width_mm, height_mm = PANEL_SIZES_MM[panel]
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor='white')
    registry = ArtistRegistry()
    rect = (0.055, 0.08, 0.91, 0.86)
    axes = DRAWERS[panel](fig, rect, bundle, registry)
    _panel_letter(fig, rect, panel, registry)
    return (fig, registry, axes)

def _save(fig: mpl.figure.Figure, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'Title': f'Avatar-HCC Supplementary Figure S2 {VERSION}', 'Subject': 'Render-only from frozen results', 'Keywords': f'{PALETTE_ID}; render-only; frozen tables'}
    paths = []
    for suffix in ('pdf', 'svg', 'png'):
        path = stem.with_suffix(f'.{suffix}')
        kwargs: dict[str, Any] = {'format': suffix, 'facecolor': 'white', 'edgecolor': 'white', 'transparent': False, 'bbox_inches': None, 'pad_inches': 0}
        if suffix == 'png':
            kwargs['dpi'] = DPI
        if suffix == 'pdf':
            kwargs['metadata'] = metadata
        elif suffix == 'svg':
            kwargs['metadata'] = {'Title': metadata['Title'], 'Description': metadata['Subject']}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths

def render_all():
    apply_theme()
    mpl.rcParams.update({'savefig.dpi': DPI, 'figure.dpi': 150})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    write_json({'figure': 'Supplementary Figure S2', 'version': VERSION, 'mode': 'byte-for-byte frozen input snapshots', 'files': bundle['input_manifest']}, INPUT_DIR / get_value('figures.config.item_116'))
    write_json(bundle['numeric_qa'], OUTPUT_DIR / f"{get_value('figures.config.item_311')}{VERSION}.json")
    fig, registry, _ = build_full(bundle)
    full_audit = audit_figure_layout(fig, f"{get_value('figures.config.item_308')}{VERSION}", registry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.5, check_collisions=True)
    write_json(full_audit, OUTPUT_DIR / f"{get_value('figures.config.item_312')}{VERSION}.json")
    _save(fig, FIGURE_STEM)
    plt.close(fig)
    panel_audits = {}
    for panel in DRAWERS:
        pfig, pregistry, _ = build_panel(panel, bundle)
        report = audit_figure_layout(pfig, f"{get_value('figures.config.item_313')}{panel}_Nature_{VERSION}", pregistry, minimum_font_pt=MIN_FONT_PT, safe_pad_pt=1.5, check_collisions=True)
        panel_audits[panel] = report
        _save(pfig, PANEL_DIR / f"{get_value('figures.config.item_313')}{panel}_Nature_{VERSION}")
        plt.close(pfig)
    write_json(panel_audits, OUTPUT_DIR / f"{get_value('figures.config.item_314')}{VERSION}.json")
    manifest_path = OUTPUT_DIR / f"{get_value('figures.config.item_316')}{VERSION}.json"
    outputs = sorted([p for p in OUTPUT_DIR.rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.svg', '.png', '.json'} and (p != manifest_path)], key=lambda p: str(p.relative_to(OUTPUT_DIR)))
    manifest = {'figure': 'Supplementary Figure S2', 'version': VERSION, 'palette': PALETTE_ID, 'mode': 'render-only from frozen inputs', 'working_master_size_mm': [FULL_WIDTH_MM, FULL_HEIGHT_MM], 'dpi': DPI, 'inputs': bundle['input_manifest'], 'outputs': [{'path': str(p.relative_to(OUTPUT_DIR)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in outputs], 'strict_layout_pass': bool(full_audit['summary']['strict_pass']) and all((panel_audits[p]['summary']['strict_pass'] for p in panel_audits))}
    write_json(manifest, manifest_path)
    return {'full_stem': str(FIGURE_STEM), 'manifest': str(manifest_path), 'full_layout': full_audit['summary'], 'panel_layout': {p: panel_audits[p]['summary'] for p in panel_audits}}
if __name__ == '__main__':
    render_all()
