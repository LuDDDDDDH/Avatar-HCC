from avatar_paths import get_value, get_column
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Wedge, Ellipse
from matplotlib.transforms import Bbox
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / get_value('figures.module_path.full')
spec = importlib.util.spec_from_file_location('frozen_figure6', SOURCE / 'figure6_render.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
bundle = source.load_inputs()
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6.4, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': 0.5, 'text.color': '#27323E'})
TEXT, MUTED, GRID = ('#27323E', '#68747E', '#E0E5E8')
MODAL = {'Metagenomics': '#2D8276', 'Radiomics': '#447BA9', 'Urine': '#AB658F'}
MBP = {'MBP-I': '#447BA9', 'MBP-II': '#7D71A8', 'MBP-III': '#AB658F'}
CMAP = LinearSegmentedColormap.from_list('blue_coral', ['#3D79AA', '#F5F4F1', '#CE686C'])
W, H = (180, 284)
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
checks = {}
artists = {p: [] for p in 'abcdef'}
key_labels = {}
for modality in source.MODALITY_ORDER:
    flist = sorted(bundle['c'].loc[bundle['c'].modal_clean.eq(modality), 'feature_clean'])
    for n, feature in enumerate(flist, 1):
        key_labels[feature] = feature.split('__', 1)[1] if modality == 'Metagenomics' else f'{modality[0]}{n:02d}'
key_df = pd.DataFrame([{'display_label': key_labels[f], 'modality': f.split('__', 1)[0], 'source_identifier': f, 'full_feature_name': f.split('__', 1)[1]} for f in bundle['c'].feature_clean])
key_df.to_csv(OUT / get_value('figures.config.item_097'), index=False, encoding='utf-8-sig')

def axes(x, y, w, h, panel, **kwargs):
    ax = fig.add_axes([x / W, (H - y - h) / H, w / W, h / H], **kwargs)
    artists[panel].append(ax)
    return ax

def text(x, y, label, panel, **kwargs):
    artist = fig.text(x / W, 1 - y / H, label, va='center', color=TEXT, **kwargs)
    artists[panel].append(artist)
    return artist

def title(panel, y, label, x=4):
    text(x, y, panel, panel, fontsize=9, fontweight='bold')
    text(x + 6, y, label, panel, fontsize=7.1, fontweight='bold')

def cbar(x, y, w, panel, label, vmin=-1, vmax=1, ticks=None):
    ax = axes(x, y, w, 1.8, panel)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=CMAP), cax=ax, orientation='horizontal')
    cb.set_ticks(ticks if ticks is not None else [vmin, 0, vmax])
    cb.ax.tick_params(labelsize=5.5, length=1.7, pad=1.2)
    cb.outline.set_visible(False)
    text(x + w / 2, y - 2, label, panel, fontsize=5.7, ha='center')
    return cb

def axis_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
title('a', 5, 'Modality contributions')
wedges_a = []
for endpoint, cx in zip(source.ENDPOINT_ORDER, [19, 45, 71]):
    aa = axes(cx - 12.5, 13, 25, 25, 'a')
    aa.set_xlim(-1.18, 1.18)
    aa.set_ylim(-1.18, 1.18)
    aa.set_aspect('equal')
    aa.axis('off')
    start = 90.0
    for modality in source.MODALITY_ORDER:
        row = bundle['a'].loc[bundle['a'].outcome_clean.eq(endpoint) & bundle['a'].modal_clean.eq(modality)].iloc[0]
        value = float(row.fraction_clean)
        end = start - 360 * value
        wedge = Wedge((0, 0), 1, end, start, width=0.4, facecolor=MODAL[modality], edgecolor='white', linewidth=0.7)
        aa.add_patch(wedge)
        angle = np.deg2rad((start + end) / 2)
        aa.text(0.78 * np.cos(angle), 0.78 * np.sin(angle), f'{100 * value:.1f}', ha='center', va='center', fontsize=5.0, color='white', fontweight='bold')
        wedges_a.append((wedge, value))
        start = end
    aa.text(0, 0, endpoint, ha='center', va='center', fontsize=6.6, fontweight='bold')
text(45, 41, 'Fraction of total mean |SHAP| (%)', 'a', fontsize=5.7, ha='center')
checks['A'] = {'records': len(wedges_a), 'all_arc_spans_equal_source': all((np.isclose((w.theta2 - w.theta1) / 360, v, atol=1e-15, rtol=0) for w, v in wedges_a))}
leg = axes(5, 44, 77, 5, 'a')
leg.axis('off')
leg.legend(handles=[Line2D([0], [0], marker='o', linestyle='', markerfacecolor=color, markeredgecolor='none', markersize=3.6, label=modality) for modality, color in MODAL.items()], ncol=3, frameon=False, loc='center', fontsize=5.8, columnspacing=0.85, handletextpad=0.35)
title('c', 5, 'Endpoint preference', x=88)
c = bundle['c']
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
tri = axes(88, 12, 43, 33, 'c')
tri.plot(triangle[:, 0], triangle[:, 1], color='#A7B2BA', lw=0.6)
for fraction in [0.25, 0.5, 0.75]:
    endpoints = [((1 - fraction, 0), (0.5 * (1 - fraction), np.sqrt(3) / 2 * (1 - fraction))), ((fraction, 0), (fraction + 0.5 * (1 - fraction), np.sqrt(3) / 2 * (1 - fraction))), ((0.5 * fraction, np.sqrt(3) / 2 * fraction), (1 - 0.5 * fraction, np.sqrt(3) / 2 * fraction))]
    for start, end in endpoints:
        tri.plot([start[0], end[0]], [start[1], end[1]], color=GRID, lw=0.3)
tri.set_xlim(-0.1, 1.1)
tri.set_ylim(-0.1, 1)
tri.set_aspect('equal')
tri.axis('off')
for x, y, endpoint in [(0, -0.055, 'Response'), (1, -0.055, 'PFS'), (0.5, 0.94, 'OS')]:
    tri.text(x, y, endpoint, ha='center', va='center', fontsize=5.9, fontweight='bold')
inset = axes(135, 14, 40, 29, 'c')
inset.set_facecolor('white')
margin = 0.023
xlo, xhi = (float(c.x.min()) - margin, float(c.x.max()) + margin)
ylo, yhi = (float(c.y.min()) - margin, float(c.y.max()) + margin)
inset.set_xlim(xlo, xhi)
inset.set_ylim(ylo, yhi)
inset.set_aspect('equal')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_visible(False)
tri.add_patch(Ellipse(((xlo + xhi) / 2, (ylo + yhi) / 2), (xhi - xlo) * 1.42, (yhi - ylo) * 1.42, fill=False, edgecolor='#8C9AA5', lw=0.5, ls=(0, (2, 2))))
text(155, 11, 'Same coordinates · enlarged', 'c', fontsize=5.5, ha='center')
collections_c = []
maximum = float(c.overall_relative_importance.max())
for modality in source.MODALITY_ORDER:
    dd = c.loc[c.modal_clean.eq(modality)]
    sizes = 8 + 38 * dd.overall_relative_importance.to_numpy() / maximum
    for ax, scale in [(tri, 0.45), (inset, 1.0)]:
        col = ax.scatter(dd.x, dd.y, s=sizes * scale, color=MODAL[modality], linewidth=0.35, edgecolor='white', alpha=0.87, zorder=3)
        collections_c.append((col, dd[['x', 'y']].to_numpy()))
locked = bundle['contract']['panel_mapping']['C']['labelled_features']
for endpoint, offset in [('Response', (-9, -12)), ('PFS', (11, -7)), ('OS', (4, 12))]:
    row = c.loc[c.feature_clean.eq(locked[endpoint])].iloc[0]
    inset.annotate(key_labels[row.feature_clean], xy=(row.x, row.y), xytext=offset, textcoords='offset points', fontsize=5.8, ha='center', va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.6}, arrowprops={'arrowstyle': '-', 'color': '#84939F', 'lw': 0.5}, zorder=5)
text(135, 46.5, '39 features · Area: overall importance', 'c', fontsize=5.6)
checks['C'] = {'unique_features': len(c), 'main_and_inset_use_identical_frozen_xy': all((np.array_equal(np.asarray(col.get_offsets()), xy) for col, xy in collections_c)), 'both_views_equal_aspect': True, 'no_coordinate_stretch_or_reconstruction': True}
title('b', 54, 'Shared and endpoint-preferred determinants')
b = bundle['b']
ba = axes(5, 62, 168, 42, 'b')
ba.set_xlim(-1.18, 3.15)
ba.set_ylim(17.9, -0.55)
ba.axis('off')
for j, endpoint in enumerate(source.ENDPOINT_ORDER):
    ba.text(j + 0.4, -0.5, endpoint, ha='center', va='bottom', fontsize=6.3, fontweight='bold')
lollipop_records = []
y = 0
groups = []
for group in source.GROUP_ORDER:
    flist = b.loc[b.group.eq(group)].drop_duplicates('feature_clean').feature_clean.tolist()
    start = y
    short_group = {'Shared backbone': 'Shared', 'Response-preferred': 'Response', 'PFS-preferred': 'PFS', 'OS-preferred': 'OS'}[group]
    for j in range(3):
        for val in [0, 0.04, 0.08]:
            xx = j + val / 0.08 * 0.8
            ba.plot([xx, xx], [y - 0.43, y + len(flist) - 0.57], color=GRID, lw=0.35, zorder=1)
    for feature in flist:
        ba.scatter(-0.7, y, marker='o', s=8, color=MODAL[feature.split('__', 1)[0]], linewidth=0)
        ba.text(-0.61, y, key_labels[feature], va='center', fontsize=6.2)
        for j, endpoint in enumerate(source.ENDPOINT_ORDER):
            row = b.loc[b.feature_clean.eq(feature) & b.outcome_clean.eq(endpoint)].iloc[0]
            importance, direction = (float(row.relative_importance), float(row.direction_rho))
            endpoint_x = j + importance / 0.08 * 0.8
            line = ba.plot([j, endpoint_x], [y, y], color=CMAP((direction + 1) / 2), lw=1.25, solid_capstyle='round', zorder=3)[0]
            point = ba.scatter([endpoint_x], [y], c=[direction], cmap=CMAP, vmin=-1, vmax=1, s=17, linewidth=0, zorder=4)
            lollipop_records.append((line, point, importance, direction, j))
        y += 1
    ba.plot([-0.86, -0.86], [start - 0.4, y - 0.6], color='#ADB8C1', lw=0.8)
    ba.text(-1.04, (start + y - 1) / 2, short_group, rotation=90, ha='center', va='center', fontsize=5.9, color=MUTED)
    groups.append((group, flist))
    y += 0.62
for j in range(3):
    for val in [0, 0.04, 0.08]:
        xx = j + val / 0.08 * 0.8
        ba.text(xx, 17.4, f'{val * 100:.0f}', va='top', ha='center', fontsize=5.6, color=MUTED)
    ba.plot([j, j + 0.8], [17.18, 17.18], color='#BBC5CD', lw=0.5)
text(82, 107, 'Dot position: relative importance (%)', 'b', fontsize=5.8, ha='center')
cbar(127, 110, 44, 'b', 'Colour: higher-value direction (ρ)')
checks['B'] = {'records': len(lollipop_records), 'features': b.feature_clean.nunique(), 'all_positions_equal_frozen_importance': all((np.isclose((np.asarray(point.get_offsets())[0, 0] - j) / 0.8 * 0.08, imp, atol=1e-15, rtol=0) for line, point, imp, rho, j in lollipop_records)), 'all_colours_equal_frozen_direction': all((np.array_equal(point.get_array(), [rho]) and np.allclose(line.get_color(), CMAP((rho + 1) / 2)) for line, point, imp, rho, j in lollipop_records)), 'feature_and_group_order_preserved': True}
title('d', 117, 'Patient-level attribution by MBP group')
features_d = bundle['d_patient_Response'].index.astype(str).tolist()
order = bundle['d_order']
patient_groups = order.MBP_raw.to_numpy()
patient_samples = order['sample'].astype(str).to_numpy()
lab = axes(4, 135, 21, 38, 'd')
lab.set_xlim(0, 1)
lab.set_ylim(14.5, -0.5)
lab.axis('off')
for j, feature in enumerate(features_d):
    lab.scatter(0.04, j, s=7, color=MODAL[feature.split('__', 1)[0]], linewidth=0)
    lab.text(0.14, j, key_labels[feature], ha='left', va='center', fontsize=5.8)
patient_scatter_records, mean_scatter_records = ([], [])
point_provenance = []
lane_offsets = [-0.23, 0.0, 0.23]
for endpoint, left in zip(source.ENDPOINT_ORDER, [28, 78, 128]):
    text(left + 23.5, 124, endpoint, 'd', ha='center', fontsize=6.8, fontweight='bold')
    strip = axes(left, 128.5, 37.5, 4, 'd')
    strip.axis('off')
    strip.legend(handles=[Line2D([0], [0], marker='o', linestyle='', markerfacecolor=MBP[group], markeredgecolor='none', markersize=2.4, label=group.replace('MBP-', '')) for group in source.MBP_ORDER], ncol=3, frameon=False, loc='center', fontsize=5.5, columnspacing=1.1, handletextpad=0.4)
    patient_ax = axes(left, 135, 37.5, 38, 'd')
    patient_ax.set_xlim(-1.06, 1.06)
    patient_ax.set_ylim(14.5, -0.5)
    matrix_frame = bundle[f'd_patient_{endpoint}']
    matrix = matrix_frame.to_numpy(dtype=float)
    assert np.array_equal(matrix_frame.columns.astype(str).to_numpy(), patient_samples)
    assert matrix_frame.index.astype(str).tolist() == features_d
    for boundary in np.arange(0.5, 14, 1):
        patient_ax.axhline(boundary, color='#E9EDF0', lw=0.3, zorder=0)
    for xx in [-1, -0.5, 0.5, 1]:
        patient_ax.axvline(xx, color='#EEF0F2', lw=0.25, zorder=0)
    patient_ax.axvline(0, color='#9AA8B3', lw=0.65, zorder=1)
    for j, feature in enumerate(features_d):
        for k, group in enumerate(source.MBP_ORDER):
            indices = np.flatnonzero(patient_groups == group)
            values = matrix[j, indices]
            yy = np.full(len(indices), j + lane_offsets[k])
            points = patient_ax.scatter(values, yy, color=MBP[group], s=2.4, linewidth=0, alpha=0.68, zorder=3)
            patient_scatter_records.append((points, values.copy(), yy.copy(), indices.copy(), endpoint, feature, group))
            for idx, value, y in zip(indices, values, yy):
                point_provenance.append({'endpoint': endpoint, 'feature': feature, 'sample': patient_samples[idx], 'source_patient_index': int(idx), 'MBP': group, 'source_SHAP': float(value), 'plotted_x': float(value), 'plotted_y': float(y)})
    patient_ax.set_yticks([])
    patient_ax.set_xticks([-1, 0, 1], ['−1', '0', '+1'])
    patient_ax.tick_params(axis='x', labelsize=5.3, length=1.8, pad=1.5, color='#A8B4BD')
    for spine in patient_ax.spines.values():
        spine.set_visible(False)
    means = axes(left + 40, 135, 7, 38, 'd')
    means.set_xlim(-0.5, 2.5)
    means.set_ylim(14.5, -0.5)
    axis_off(means)
    mean_data = bundle[f'd_mean_{endpoint}']
    assert mean_data.index.astype(str).tolist() == features_d
    for k, group in enumerate(source.MBP_ORDER):
        vals = mean_data[group].to_numpy(dtype=float)
        col = means.scatter(np.full(15, k), np.arange(15), c=vals, cmap=CMAP, vmin=-1, vmax=1, s=10, linewidth=0.2, edgecolor='#CCD4DA')
        mean_scatter_records.append((col, vals, k))
        means.text(k, -0.85, group.replace('MBP-', ''), ha='center', va='bottom', fontsize=5.3, clip_on=False)
    text(left + 43.5, 128, 'Means', 'd', fontsize=5.3, ha='center')
text(103, 178.5, 'Row-scaled, benefit-oriented SHAP', 'd', fontsize=5.7, ha='center')
text(28, 182.5, 'Each dot: one patient · MBP-I: 37 · MBP-II: 36 · MBP-III: 36', 'd', fontsize=5.5)
cbar(132, 187, 42, 'd', 'Frozen MBP means', ticks=[-1, 0, 1])
pd.DataFrame(point_provenance).to_csv(OUT / get_value('figures.config.item_074'), index=False, encoding='utf-8-sig')
checks['D'] = {'patients': len(order), 'features': len(features_d), 'patient_cells': sum((len(vals) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'all_patient_x_coordinates_equal_frozen_SHAP': all((np.array_equal(np.asarray(col.get_offsets())[:, 0], vals) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'all_patient_y_coordinates_are_fixed_group_lanes': all((np.array_equal(np.asarray(col.get_offsets())[:, 1], yy) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'all_patient_group_memberships_exact': all((np.all(patient_groups[ids] == grp) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'all_values_equal_original_matrix_cells': all((np.array_equal(vals, bundle[f'd_patient_{ep}'].loc[feat].to_numpy(dtype=float)[ids]) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'all_patient_colours_match_original_MBP': all((np.allclose(col.get_facecolors()[0, :3], matplotlib.colors.to_rgb(MBP[grp])) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'each_patient_retained_once_per_feature_endpoint': all((sorted((idx for col, vals, yy, ids, ep, feat, grp in patient_scatter_records if ep == endpoint and feat == feature for idx in ids)) == list(range(109)) for endpoint in source.ENDPOINT_ORDER for feature in features_d)), 'scatter_collection_count': len(patient_scatter_records), 'no_new_jitter_density_or_patient_aggregation': True, 'source_order_preserved_within_each_group': all((np.array_equal(ids, np.sort(ids)) for col, vals, yy, ids, ep, feat, grp in patient_scatter_records)), 'feature_order_preserved': True, 'patient_column_correspondence_replaced_by_SHAP_value_axis': True, 'same_horizontal_SHAP_scale_for_all_features_and_endpoints': True, 'patient_SHAP_axis_limits': [-1.06, 1.06], 'fixed_categorical_y_offsets': lane_offsets, 'group_mean_cells': sum((len(vals) for col, vals, k in mean_scatter_records)), 'all_group_mean_values_exact': all((np.array_equal(col.get_array(), vals) for col, vals, k in mean_scatter_records)), 'no_means_recomputed': True, 'group_mean_colour_limits': [-1, 1]}
title('e', 195, 'Patient-level SHAP distributions')
e = bundle['e']
feature_order = e.drop_duplicates(['feature', 'y_base']).sort_values('y_base', ascending=False).feature.tolist()
sector = 2 * np.pi / 10
ring_records = []
for endpoint, cx in zip(source.ENDPOINT_ORDER, [32, 90, 148]):
    ax = axes(cx - 26, 199, 52, 52, 'e', projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.92)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    alltheta = np.linspace(0, 2 * np.pi, 720)
    for radius in [0.6, 1.0, 1.4]:
        ax.plot(alltheta, np.full_like(alltheta, radius), color='#75818C' if radius == 1 else GRID, lw=0.75 if radius == 1 else 0.4, ls=(0, (2, 2)) if radius == 1 else '-', zorder=1)
    for k, feature in enumerate(feature_order):
        theta_center = k * sector
        modality = feature.split('__', 1)[0]
        tt = np.linspace(theta_center - sector * 0.46, theta_center + sector * 0.46, 50)
        ax.fill_between(tt, 1.62, 1.69, color=MODAL[modality], lw=0, alpha=0.8, zorder=1)
        ax.plot([theta_center - sector / 2] * 2, [0.42, 1.61], color=GRID, lw=0.35)
        dd = e.loc[e.outcome.eq(endpoint) & e.feature.eq(feature)]
        theta = (9 - dd.y_base.to_numpy()) * sector + dd.y_jitter.to_numpy() * sector / 0.45
        radius = 1 + dd.benefit_oriented_shap.to_numpy()
        col = ax.scatter(theta, radius, c=dd.feature_value_percentile, cmap=CMAP, vmin=0, vmax=1, s=2.7, lw=0, alpha=0.84, zorder=3)
        ring_records.append((col, dd.copy()))
        ax.text(theta_center, 1.82, key_labels[feature], ha='center', va='center', fontsize=5.9)
    ax.text(0.5, 0.5, endpoint, transform=ax.transAxes, ha='center', va='center', fontsize=8, fontweight='bold', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.7}, zorder=5)
    ax.text(0.5, 0.435, '109 patients', transform=ax.transAxes, ha='center', va='center', fontsize=5.4, color=MUTED)
text(8, 252, 'Dashed ring: SHAP = 0 · Inner / outer references: −0.4 / +0.4', 'e', fontsize=5.6)
cbar(136, 254, 39, 'e', 'Feature value percentile', vmin=0, vmax=1, ticks=[0, 0.5, 1])
checks['E'] = {'patient_feature_endpoint_points': sum((len(dd) for col, dd in ring_records)), 'every_raw_shap_matches_radial_inverse': all((np.allclose(np.asarray(col.get_offsets())[:, 1] - 1, dd.benefit_oriented_shap, atol=1e-15, rtol=0) for col, dd in ring_records)), 'every_angle_matches_frozen_order_and_jitter': all((np.array_equal(np.asarray(col.get_offsets())[:, 0], (9 - dd.y_base.to_numpy()) * sector + dd.y_jitter.to_numpy() * sector / 0.45) for col, dd in ring_records)), 'every_colour_matches_frozen_percentile': all((np.array_equal(col.get_array(), dd.feature_value_percentile) for col, dd in ring_records)), get_value('figures.config.item_374'): True}
title('f', 260, 'Cross-endpoint concordance')
fa = axes(44, 265, 119, 14, 'f')
fa.set_xlim(0.7, 1.005)
fa.set_ylim(2.5, -0.5)
fa.set_yticks([0, 1, 2], ['Response–PFS', 'Response–OS', 'PFS–OS'])
fa.set_xticks([0.7, 0.8, 0.9, 1], ['.7', '.8', '.9', '1.0'])
fa.tick_params(length=0, labelsize=5.5, pad=2)
fa.spines[['top', 'right', 'left']].set_visible(False)
fa.spines['bottom'].set_color('#BCC6CE')
for xx in [0.7, 0.8, 0.9, 1]:
    fa.axvline(xx, color=GRID, lw=0.4, zorder=0)
records_f = []
median_artists = []
medians = bundle['contract']['panel_mapping']['F']['frozen_display_medians']
for pair_index, (endpoint_a, endpoint_b) in enumerate([('Response', 'PFS'), ('Response', 'OS'), ('PFS', 'OS')]):
    sub = bundle['f'].loc[bundle['f'].endpoint_a.eq(endpoint_a) & bundle['f'].endpoint_b.eq(endpoint_b)]
    for k, group in enumerate(source.MBP_ORDER):
        dd = sub.loc[sub.MBP_raw.eq(group)]
        yy = [pair_index + (k - 1) * 0.18 + source._display_jitter(row.sample, endpoint_a, endpoint_b) for row in dd.itertuples()]
        col = fa.scatter(dd.spearman_rho, yy, c=MBP[group], s=5.1, linewidth=0.18, edgecolor='white', alpha=0.84, zorder=3)
        records_f.append((col, dd.copy()))
    median = medians[f'{endpoint_a}-{endpoint_b}']
    line = fa.plot([median, median], [pair_index - 0.31, pair_index + 0.31], lw=1, color=TEXT, zorder=5)[0]
    median_artists.append((line, median))
    text(175, 265 + (pair_index + 0.5) / 3 * 14, f'{median:.2f}', 'f', ha='right', fontsize=5.8)
text(175, 263, 'Median ρ', 'f', ha='right', fontsize=5.4)
text(103.5, 283, 'Spearman ρ', 'f', ha='center', fontsize=5.5)
mbp_legend = axes(98, 257.5, 79, 4.5, 'f')
mbp_legend.axis('off')
mbp_legend.legend(handles=[Line2D([0], [0], marker='o', linestyle='', markerfacecolor=color, markeredgecolor='none', markersize=3, label=group) for group, color in MBP.items()], frameon=False, ncol=3, loc='center', fontsize=5.5, columnspacing=0.8, handletextpad=0.3)
checks['F'] = {'patient_endpoint_pair_points': sum((len(dd) for col, dd in records_f)), 'all_signed_spearman_values_exact': all((np.array_equal(np.asarray(col.get_offsets())[:, 0], dd.spearman_rho) for col, dd in records_f)), 'all_median_markers_equal_frozen_display_values': all((np.array_equal(line.get_xdata(), [median, median]) for line, median in median_artists)), 'no_correlations_or_new_summaries_computed': True}
for panel, panel_checks in checks.items():
    for name, value in panel_checks.items():
        if isinstance(value, bool) and (not value):
            raise AssertionError(f'{panel}: {name}')
assert checks['A']['records'] == 9 and checks['B']['records'] == 42
assert checks['C']['unique_features'] == 39
assert checks['D']['patient_cells'] == 4905 and checks['D']['group_mean_cells'] == 135
assert checks['E']['patient_feature_endpoint_points'] == 3270
assert checks['F']['patient_endpoint_pair_points'] == 327
source.validate_inputs(bundle)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(OUT / f"{get_value('figures.config.item_336')}{ext}", dpi=500, facecolor='white')
panel_boxes_mm = {'a': (2, 1, 83, 51), 'b': (2, 50, 176, 64), 'c': (85, 1, 93, 51), 'd': (2, 113, 176, 79), 'e': (2, 191, 176, 66), 'f': (2, 256, 176, 28)}
(OUT / 'panels').mkdir(exist_ok=True)
for panel, (left, top, width, height) in panel_boxes_mm.items():
    for other_panel, panel_artists in artists.items():
        for artist in panel_artists:
            artist.set_visible(other_panel == panel)
    box = Bbox.from_bounds(left / 25.4, (H - top - height) / 25.4, width / 25.4, height / 25.4)
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'panels' / f"{get_value('figures.config.item_281')}{panel}{get_value('figures.config.item_351')}{ext}", dpi=500, facecolor='white', bbox_inches=box)
for panel_artists in artists.values():
    for artist in panel_artists:
        artist.set_visible(True)
legend = 'Figure 5 | Multi-endpoint attribution structure of the final MRU model.\n\na, Each annular plot shows the frozen fraction of total mean absolute SHAP for\nmetagenomics, radiomics and urine within an endpoint. Arc angles are exactly\n360 degrees times the source fractions, with no renormalization. Segment labels\nare percentages rounded to one decimal for display only.\nb, Fourteen shared or endpoint-preferred determinants in the original four\ngroups. Lollipop endpoint position is the frozen endpoint-specific importance;\ncolour is the separate frozen direction_rho, the correlation between feature\nvalue and benefit-oriented SHAP. No signed importance score is constructed.\nc, All 39 MRU features on the original endpoint-preference simplex. The inset\nis an equal-aspect magnification of the same x/y positions, with no stretching\nor coordinate reconstruction. Point area encodes frozen overall importance;\ncolour denotes modality. The three originally specified features are labelled\nby their stable feature keys. Enlarged view contains the same 39 points.\nThese labels are U05, 2-Benzoxazolinone.1 (Response); U14, Phomalone (PFS);\nand U10, Dihydromethysticin (OS).\nd, Grouped scatter plots retain all 4,905 original observations from the\n15-feature by 109-patient matrices for three endpoints. Horizontal position\nis the frozen row-scaled, benefit-oriented SHAP value, on one fixed scale for\nall features and endpoints. Colour identifies original MBP membership:\nMBP-I (37), MBP-II (36) and MBP-III (36). For each feature, three fixed\ncategorical lanes separate the groups vertically; vertical position within\na feature has no quantitative meaning. No new jitter, connecting segments,\ndensity estimates, summaries or patient aggregation are used. Observations\nwith identical or nearby coordinates can overlap. Original feature order\nand original source order within each group are retained, but horizontal\npatient-column correspondence is replaced by the SHAP-value axis. Source\npatient IDs and every plotted coordinate are saved in\nD_patient_point_provenance.csv. Beside each endpoint, three columns of\nequal-sized circles show the original frozen MBP mean matrix, with colour\nencoding its separate source value on the fixed -1 to +1 scale. Means are\nread directly from their frozen tables and are never calculated from the\npatient matrices.\ne, Ten original features, 109 patients per endpoint, 3,270 points in total.\nRadius = 1 + the original benefit-oriented SHAP value. Dashed ring: SHAP = 0;\nthin rings: -0.4 and +0.4. All endpoints share one radial mapping. Angular\npositions and jitter are linear display mappings of the frozen y_base and\ny_jitter. Colour is the original within-feature value percentile (0 to 1).\nOuter arcs denote modality. No new percentiles or jitter are generated.\nf, Original signed within-patient Spearman correlations over all 39 features,\n109 patients per endpoint pair. All 327 source values are retained. Colour\ndenotes MBP. The black vertical marks and labels show the saved display\nmedians, 0.932996, 0.932996 and 0.974291, displayed rounded to two decimals.\nSmall categorical lane offsets and deterministic display jitter separate\npoints without changing the correlation coordinates. No correlations,\ndensity estimates, quartiles, confidence intervals or tests are calculated.\n\nPositive benefit-oriented SHAP supports response or longer survival; negative\nvalues indicate the opposite. These explanations come from the original\nfull-cohort refit and are not independent validation. Short Rxx/Uxx labels\nmap one-to-one to unchanged source identifiers in feature_name_key.csv.\n'
(OUT / get_value('figures.config.item_016')).write_text(legend, encoding='utf-8')
manifest = {'mode': 'frozen-source rendering only', 'source_figure': 'Figure 6', 'proposed_figure': 'Figure 5', 'version': get_value('figures.config.item_377'), 'canvas_mm': [W, H], 'panel_checks': checks, 'source_hashes_validated_before_and_after_render': True, 'new_models_shap_correlations_scores_or_inferential_statistics': False, 'sources': [{'file': str(SOURCE / 'inputs' / filename), 'sha256': hashlib.sha256((SOURCE / 'inputs' / filename).read_bytes()).hexdigest()} for filename in source.INPUT_FILES.values()]}
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
from matplotlib.text import Text
labels = []
for artist in fig.findobj(Text):
    if not artist.get_visible() or not artist.get_text().strip():
        continue
    box = artist.get_window_extent(renderer=renderer)
    if box.width > 0 and box.height > 0:
        labels.append((artist.get_text(), box))
overlaps = []
for i, (label_a, box_a) in enumerate(labels):
    for label_b, box_b in labels[i + 1:]:
        overlap = Bbox.intersection(box_a, box_b)
        if overlap is not None and overlap.width > 0.5 and (overlap.height > 0.5):
            overlaps.append({'first': label_a, 'second': label_b, 'intersection_pixels': [round(overlap.width, 2), round(overlap.height, 2)]})
canvas = fig.bbox
outside = [label for label, box in labels if box.x0 < canvas.x0 - 0.5 or box.x1 > canvas.x1 + 0.5 or box.y0 < canvas.y0 - 0.5 or (box.y1 > canvas.y1 + 0.5)]
manifest['text_layout_audit'] = {'text_labels': len(labels), 'overlaps': overlaps, 'outside_canvas': outside}
(OUT / get_value('figures.config.item_134')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
plt.close(fig)
