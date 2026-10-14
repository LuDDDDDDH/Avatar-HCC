from avatar_paths import get_value, get_column
from pathlib import Path
import hashlib
import importlib.util
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CODE = ROOT / get_value('figures.module_path.render')
spec = importlib.util.spec_from_file_location('original_figure5', CODE)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from figure_qa import ArtistRegistry, audit_figure_layout
W, H = (180.0, 252.0)
TEXT, MUTED = (old.TEXT, old.MUTED)
GROUPS = old.MBP_ORDER
COLORS = old.MBP_COLORS
ENDPOINTS = old.ENDPOINT_COLORS
REG = ArtistRegistry()
CHECKS = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def pformat(value):
    value = float(value)
    if value < 0.001:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return f'{mantissa}e{int(exponent):+d}'.replace('-', '−')
    return f'{value:.3g}'

def verify(panel, key, fields, expected, actual):
    expected, actual = (np.asarray(expected, float), np.asarray(actual, float))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=f'{panel}: {key}')
    CHECKS.append({'panel': panel, 'key': key, 'fields': fields, 'source': expected.tolist(), 'artist': actual.tolist()})

def provenance(bundle):
    rows = []
    for e in bundle['manifest']['files']:
        source = Path(e['original'])
        snapshot = old.INPUT_DIR / e['snapshot']
        if not source.is_file():
            raise FileNotFoundError(source)
        pd.testing.assert_frame_equal(pd.read_csv(source), pd.read_csv(snapshot), check_exact=True)
        rows.append({**e, 'original_sha256_now': sha(source), 'snapshot_sha256_now': sha(snapshot), 'exact_source_table_match': True})
    nb = ROOT / get_value('figures.module_path.original') / bundle['manifest']['source_notebook_in_archive']
    original = json.loads(nb.read_text(encoding='utf-8'))
    export_cells = []
    for i, c in enumerate(original['cells']):
        source = ''.join(c.get('source', []))
        lines = [s.strip() for s in source.splitlines() if any((name.replace('.csv', '') in s for name in old.INPUT_FILES.values()))]
        if lines:
            export_cells.append({'cell': i, 'export_lines': lines})
    return {'original_renderer': str(CODE), 'original_renderer_sha256': sha(CODE), 'source_notebook': str(nb), 'source_notebook_sha256': sha(nb), 'source_notebook_export_cells': export_cells, 'original_tables': rows}

def axis(fig, rect, panel):
    x, t, w, h = rect
    a = fig.add_axes([x / W, 1 - (t + h) / H, w / W, h / H])
    a._qa_meta = {'panel': panel}
    return a

def ft(fig, x, t, text, *, size=6.0, color=TEXT, ha='left', weight='normal', panel='figure'):
    v = fig.text(x / W, 1 - t / H, text, fontsize=size, ha=ha, va='top', color=color, weight=weight)
    REG.tag(v, name=f'{panel}-{text}-{len(REG.artists)}', role='figure_text', panel=panel)
    return v

def txt(a, x, y, text, *, size=5.7, color=TEXT, ha='left', va='center', weight='normal', trans=None):
    v = a.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, weight=weight, transform=trans or a.transData, clip_on=False)
    REG.tag(v, name=f"{a._qa_meta['panel']}-{text}-{len(REG.artists)}", role='value', panel=a._qa_meta['panel'])
    return v

def title(fig, letter, x, t, name):
    ft(fig, x, t - 0.2, letter, size=9, weight='bold', panel=letter)
    ft(fig, x + 7, t, name, size=7.2, weight='bold', panel=letter)

def clean(a, *, left=True):
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    a.spines['left'].set_visible(left)
    for side in ('bottom', 'left'):
        a.spines[side].set_color('#81909E')
        a.spines[side].set_linewidth(0.45)
    a.tick_params(length=2, width=0.45, pad=2, colors=MUTED)
    a.tick_params(axis='y', length=2 if left else 0)
    a.grid(False)

def draw_a(fig, b):
    title(fig, 'a', 2, 4, 'Observed response & MBP')
    cells = b['association_cells']
    summary = b['association_summary'].iloc[0]
    flow = axis(fig, (15, 12, 43, 21), 'a')
    flow.axis('off')
    response_counts = cells.groupby('mRECIST').Observed_n.sum()
    mbp_counts = b['group_boundaries'].set_index('group')['n']
    gap = 6.0
    nr_count = float(response_counts.loc['NR'])
    r_count = float(response_counts.loc['R'])
    source_boxes = {'R': (nr_count + gap, nr_count + gap + r_count), 'NR': (0.0, nr_count)}
    dest_boxes = {}
    node_bottom = 0.0
    for group in reversed(GROUPS):
        n = float(mbp_counts.loc[group])
        dest_boxes[group] = (node_bottom, node_bottom + n)
        node_bottom += n + gap
    flow.set(xlim=(0, 1), ylim=(0, node_bottom - gap))
    source_cursor = {k: v[1] for k, v in source_boxes.items()}
    dest_cursor = {k: v[1] for k, v in dest_boxes.items()}
    for stratum in ['R', 'NR']:
        for group in GROUPS:
            r = cells.loc[cells.mRECIST.eq(stratum) & cells.MBP.eq(group)].iloc[0]
            n = float(r.Observed_n)
            shi = source_cursor[stratum]
            slo = shi - n
            dhi = dest_cursor[group]
            dlo = dhi - n
            vertices = [(0.04, slo), (0.38, slo), (0.62, dlo), (0.96, dlo), (0.96, dhi), (0.62, dhi), (0.38, shi), (0.04, shi), (0.04, slo)]
            path = MPath(vertices, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY])
            patch = PathPatch(path, facecolor=COLORS[group], alpha=0.42 if stratum == 'R' else 0.27, edgecolor='white', lw=0.35)
            flow.add_patch(patch)
            verify('a', stratum + ' ' + group + ' ribbon', ['Observed_n'], [n], [patch.get_path().vertices[7, 1] - patch.get_path().vertices[0, 1]])
            source_cursor[stratum] = slo
            dest_cursor[group] = dlo
    for stratum, (lo, hi) in source_boxes.items():
        node = Rectangle((0, lo), 0.035, hi - lo, facecolor='#A8B3C0', edgecolor='none')
        flow.add_patch(node)
        verify('a', stratum + ' node', ['row_total_n'], [response_counts.loc[stratum]], [node.get_height()])
        txt(flow, -0.05, (lo + hi) / 2, f'{stratum}\nn = {int(hi - lo)}', ha='right', size=5.7)
    for group, (lo, hi) in dest_boxes.items():
        node = Rectangle((0.965, lo), 0.035, hi - lo, facecolor=COLORS[group], edgecolor='none')
        flow.add_patch(node)
        verify('a', group + ' node', ['group_n'], [mbp_counts.loc[group]], [node.get_height()])
        txt(flow, 1.04, (lo + hi) / 2, f'{group[4:]}\nn = {int(hi - lo)}', color=COLORS[group], size=5.7, weight='bold')
    for j, group in enumerate(GROUPS):
        x = 23 + 17 * j
        ft(fig, x, 35, group, size=5.7, color=COLORS[group], weight='bold', ha='center', panel='a')
        for stratum, t in [('R', 40), ('NR', 48)]:
            r = cells.loc[cells.mRECIST.eq(stratum) & cells.MBP.eq(group)].iloc[0]
            count = ft(fig, x, t, f'{int(r.Observed_n)} · {r.Row_percent:.1f}%', size=5.5, ha='center', panel='a')
            asr = ft(fig, x, t + 3.2, f'ASR {r.Adjusted_standardized_residual:+.2f}', size=5.5, ha='center', color=MUTED, panel='a')
            verify('a', stratum + ' ' + group + ' annotations', ['Observed_n', 'Row_percent', 'ASR'], [r.Observed_n, round(r.Row_percent, 1), round(r.Adjusted_standardized_residual, 2)], [int(count.get_text().split(' · ')[0]), float(count.get_text().split(' · ')[1][:-1]), float(asr.get_text()[4:])])
    ft(fig, 10, 40, 'R', size=5.7, weight='bold', panel='a')
    ft(fig, 10, 48, 'NR', size=5.7, weight='bold', panel='a')
    ft(fig, 10, 55, f"n = {int(summary['n'])} · V = {summary.Cramers_V:.2f} · P = {pformat(summary.chi_square_P)}", size=5.6, color=MUTED, panel='a')

def draw_d(fig, b):
    title(fig, 'd', 76, 4, 'Effect per worsening MBP level')
    plot = axis(fig, (98, 17, 32, 30), 'd')
    table = axis(fig, (134, 17, 43, 30), 'd')
    table.set(xlim=(0, 1), ylim=(-0.55, 5.55))
    table.axis('off')
    endpoints = ['Nonresponse', 'PFS', 'OS']
    for i, endpoint in enumerate(endpoints):
        for j, kind in enumerate(['Unadjusted', 'Adjusted']):
            y = 5 - 2 * i - j
            r = b['adjusted_association'].loc[b['adjusted_association'].Endpoint.eq(endpoint) & b['adjusted_association'].Effect_type.eq(kind)].iloc[0]
            color = ENDPOINTS['Response' if endpoint == 'Nonresponse' else endpoint] if kind == 'Adjusted' else '#7D8B9A'
            line, = plot.plot([r.CI_lower, r.CI_upper], [y, y], color=color, lw=0.85 if j else 0.65)
            point, = plot.plot(r.Estimate, y, ls='none', marker='o', ms=3.1, mfc=color if j else 'white', mec=color, mew=0.8)
            verify('d', endpoint + ' ' + kind, ['CI_lower', 'Estimate', 'CI_upper'], [r.CI_lower, r.Estimate, r.CI_upper], [line.get_xdata()[0], point.get_xdata()[0], line.get_xdata()[1]])
            txt(table, 0, y, f'{r.Estimate:.2f} ({r.CI_lower:.2f}–{r.CI_upper:.2f})', size=5.6, color=color)
            txt(table, 1, y, pformat(r.P), size=5.5, color=color, ha='right')
    plot.set_xscale('log')
    plot.set(xlim=(0.7, 32), ylim=(-0.55, 5.55), xticks=[1, 2, 5, 10, 20], yticks=[4.5, 2.5, 0.5], yticklabels=['Nonresponse\nOR', 'PFS\nHR', 'OS\nHR'])
    plot.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    plot.xaxis.set_minor_locator(NullLocator())
    plot.axvline(1, color='#A9B4C0', lw=0.5, ls=(0, (2, 3)), zorder=0)
    clean(plot, left=False)
    plot.set_xlabel('OR / HR (log scale)', fontsize=5.8, labelpad=2)
    ft(fig, 134, 12, 'Estimate (95% CI)', size=5.5, color=MUTED, panel='d')
    ft(fig, 177, 12, 'P', size=5.5, color=MUTED, ha='right', panel='d')
    legend = plot.legend([Line2D([0], [0], marker='o', mfc='white', mec='#9DA8B4', color='#9DA8B4', ms=3), Line2D([0], [0], marker='o', mfc=TEXT, mec=TEXT, color=TEXT, ms=3)], ['Unadjusted', 'Adjusted'], ncol=2, loc='lower left', bbox_to_anchor=(-0.24, 1.08), fontsize=5.5, handlelength=0.8, handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    REG.tag(legend, name='d-adjustment-key', role='legend', panel='d')
    ft(fig, 91, 55, 'Adjusted for ALBI, TBS, AFP, NLR and PVTT', size=5.6, color=MUTED, panel='d')

def survival(fig, b, endpoint, panel, x):
    title(fig, panel, x - 14, 63, 'Progression-free survival' if endpoint == 'PFS' else 'Overall survival')
    curve = b[endpoint.lower() + '_curve']
    patients = b[endpoint.lower() + '_patients']
    summary = b[endpoint.lower() + '_summary']
    plot = axis(fig, (x, 81, 69, 32), panel)
    risk = axis(fig, (x, 118, 69, 10), panel)
    ticks = np.array([0.0, 12.0, 24.0, 36.0, 48.0])
    for i, group in enumerate(GROUPS):
        r = summary.loc[summary.MBP.eq(group)].iloc[0]
        ft(fig, x + 23 * i, 70, f"{group}\nMedian {r[f'median_{endpoint}_months']:.1f} mo", size=5.7, color=COLORS[group], panel=panel)
        vals = curve.loc[curve.MBP.eq(group)].sort_values('time_months')
        t = vals.time_months.to_numpy(float)
        s = vals.survival_probability.to_numpy(float)
        lower = vals.CI_lower.to_numpy(float)
        upper = vals.CI_upper.to_numpy(float)
        band = plot.fill_between(t, lower, upper, step='post', color=COLORS[group], alpha=0.07, lw=0, zorder=1)
        line, = plot.step(t, s, where='post', lw=1.15, color=COLORS[group], zorder=3)
        vertices = band.get_paths()[0].vertices
        for j, source in vals.reset_index(drop=True).iterrows():
            assert np.any(np.all(np.isclose(vertices, [source.time_months, source.CI_lower], rtol=0, atol=1e-12), axis=1))
            assert np.any(np.all(np.isclose(vertices, [source.time_months, source.CI_upper], rtol=0, atol=1e-12), axis=1))
            verify(panel, group + ' curve ' + str(j), ['time_months', 'survival_probability', 'CI_lower', 'CI_upper'], [source.time_months, source.survival_probability, source.CI_lower, source.CI_upper], [line.get_xdata()[j], line.get_ydata()[j], lower[j], upper[j]])
        censored = patients.loc[patients.MBP_raw.eq(group) & patients[endpoint + '_status'].eq(0), 'time_plot'].to_numpy(float)
        if len(censored):
            ypos = old._step_value_at(vals, censored)
            marks, = plot.plot(censored, ypos, ls='none', marker='|', ms=2.7, mew=0.6, color=COLORS[group], zorder=4)
            verify(panel, group + ' censor times', ['time_plot'], censored, marks.get_xdata())
            verify(panel, group + ' censor heights', ['stored_step_probability'], ypos, marks.get_ydata())
        for time in ticks:
            n = int((patients.loc[patients.MBP_raw.eq(group), 'time_plot'] >= time).sum())
            label = txt(risk, time, 2 - i, str(n), ha='center', size=5.7)
            verify(panel, group + ' risk at ' + str(time), ['n_at_risk'], [n], [float(label.get_text())])
        txt(risk, -2.3, 2 - i, group, size=5.6, color=COLORS[group], ha='right', weight='bold')
    plot.set(xlim=(0, 48), ylim=(0, 1.025), xticks=ticks, yticks=[0, 0.5, 1])
    plot.tick_params(axis='x', labelbottom=False)
    plot.set_ylabel('Survival probability', fontsize=6.0, labelpad=3)
    clean(plot)
    ft(fig, x + 69, 76.2, f'Log-rank P = {pformat(summary.global_logrank_P.iloc[0])}', ha='right', size=5.6, color=MUTED, panel=panel)
    risk.set(xlim=(0, 48), ylim=(-0.5, 2.5), xticks=ticks, yticks=[])
    for spine in risk.spines.values():
        spine.set_visible(False)
    risk.tick_params(axis='x', length=0, pad=2, labelsize=5.7)
    risk.set_xlabel('Time (months)', fontsize=6.0, labelpad=2)
    txt(risk, 0, 1.17, 'No. at risk', size=5.5, va='bottom', trans=risk.transAxes, color=MUTED)

def draw_e(fig, b):
    title(fig, 'e', 2, 141, 'Effect within observed mRECIST strata')
    plot = axis(fig, (28, 154, 22, 25), 'e')
    table = axis(fig, (53, 154, 24, 25), 'e')
    table.set(xlim=(0, 1), ylim=(-0.7, 3.7))
    table.axis('off')
    labels = []
    for i, endpoint in enumerate(['PFS', 'OS']):
        for j, stratum in enumerate(['Responder', 'Nonresponder']):
            y = 3 - 2 * i - j
            r = b['within_mrecist'].loc[b['within_mrecist'].Endpoint.eq(endpoint) & b['within_mrecist'].mRECIST.eq(stratum)].iloc[0]
            label = f'{endpoint} · ' + ('R' if stratum == 'Responder' else 'NR')
            labels.append(label + f'\nn={int(r.n)}, events={int(r.events)}')
            line, = plot.plot([r.CI_lower, r.CI_upper], [y, y], color=ENDPOINTS[endpoint], lw=0.9)
            point, = plot.plot(r.HR_per_MBP_level_worsening, y, ls='none', marker='o', ms=3.2, color=ENDPOINTS[endpoint])
            verify('e', endpoint + ' ' + stratum, ['CI_lower', 'HR_per_MBP_level_worsening', 'CI_upper'], [r.CI_lower, r.HR_per_MBP_level_worsening, r.CI_upper], [line.get_xdata()[0], point.get_xdata()[0], line.get_xdata()[1]])
            txt(table, 0, y + 0.13, f'{r.HR_per_MBP_level_worsening:.2f} ({r.CI_lower:.2f}–{r.CI_upper:.2f})', size=5.6, color=ENDPOINTS[endpoint])
            txt(table, 0, y - 0.19, f'P = {pformat(r.P)}', size=5.5, color=MUTED)
    plot.set_xscale('log')
    plot.set(xlim=(0.8, 9), ylim=(-0.7, 3.7), xticks=[1, 2, 4, 8], yticks=[3, 2, 1, 0], yticklabels=labels)
    plot.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    plot.xaxis.set_minor_locator(NullLocator())
    plot.axvline(1, ls=(0, (2, 3)), lw=0.5, color='#A9B4C0', zorder=0)
    plot.tick_params(axis='y', labelsize=5.5, pad=3)
    clean(plot, left=False)
    plot.set_xlabel('HR per MBP level', fontsize=5.8, labelpad=2)
    ft(fig, 53, 149, 'HR (95% CI)', size=5.5, color=MUTED, panel='e')

def draw_f(fig, b):
    title(fig, 'f', 82, 141, 'Incremental prognostic information')
    sequence = [('PFS', 'Pretreatment'), ('PFS', 'Post-mRECIST'), ('OS', 'Pretreatment'), ('OS', 'Post-mRECIST')]
    for i, (endpoint, context) in enumerate(sequence):
        x = 91 + 22 * i
        plot = axis(fig, (x, 155, 17, 21), 'f')
        row = b['incremental_summary'].loc[b['incremental_summary'].Endpoint.eq(endpoint) & b['incremental_summary'].Context.eq(context)].iloc[0]
        base = float(row.Cindex_without_MBP_mean)
        plus = float(row.Cindex_with_MBP_mean)
        line, = plot.plot([0, 1], [base, plus], color=ENDPOINTS[endpoint], lw=1.3, solid_capstyle='round', zorder=2)
        plot.plot([0, 1], [base, plus], ls='none', marker='_', ms=4, mew=1.1, color=ENDPOINTS[endpoint])
        verify('f', endpoint + ' ' + context, ['Cindex_without_MBP_mean', 'Cindex_with_MBP_mean'], [base, plus], line.get_ydata())
        txt(plot, 0, base - 0.019, f'{base:.3f}', ha='center', va='top', size=5.6, color=MUTED)
        txt(plot, 1, plus + 0.013, f'{plus:.3f}', ha='center', va='bottom', size=5.7, color=ENDPOINTS[endpoint], weight='bold')
        plot.set(xlim=(-0.22, 1.25), ylim=(0.49, 0.81), xticks=[0, 1], xticklabels=['Base', '+MBP'], yticks=[0.5, 0.6, 0.7, 0.8])
        if i != 0:
            plot.set_yticks([])
        clean(plot, left=i == 0)
        plot.tick_params(axis='x', labelsize=5.5, length=0, pad=2)
        if i == 0:
            plot.set_ylabel('C-index', fontsize=5.8, labelpad=3)
        ft(fig, x + 8.5, 147, endpoint, size=6.3, color=ENDPOINTS[endpoint], weight='bold', ha='center', panel='f')
        ft(fig, x + 8.5, 151, 'Pretreatment' if context == 'Pretreatment' else 'Post-mRECIST', size=5.5, ha='center', color=MUTED, panel='f')
        ft(fig, x + 8.5, 182, f'ΔC {row.Delta_Cindex_mean:+.3f}', size=5.7, ha='center', color=ENDPOINTS[endpoint], weight='bold', panel='f')
        ft(fig, x + 8.5, 186, f'LRT P = {pformat(row.LRT_P)}', size=5.5, ha='center', color=MUTED, panel='f')

def draw_g(fig, b):
    title(fig, 'g', 2, 196, 'Continuous benefit score')
    for i, endpoint in enumerate(['PFS', 'OS']):
        x = 16 + 40 * i
        plot = axis(fig, (x, 209, 31, 28), 'g')
        row = b['spline_stats'].set_index('Endpoint').loc[endpoint]
        values = b['spline_curve'].loc[b['spline_curve'].Endpoint.eq(endpoint)].sort_values('score')
        xx = values.score.to_numpy(float)
        mean = values.HR.to_numpy(float)
        lower = values.HR_lower.to_numpy(float)
        upper = values.HR_upper.to_numpy(float)
        band = plot.fill_between(xx, lower, upper, color='#BFC7D0', alpha=0.24, lw=0)
        line, = plot.plot(xx, mean, color=ENDPOINTS[endpoint], lw=1.15)
        vertices = band.get_paths()[0].vertices
        for j, source in values.reset_index(drop=True).iterrows():
            assert np.any(np.all(np.isclose(vertices, [source.score, source.HR_lower], rtol=0, atol=1e-12), axis=1))
            assert np.any(np.all(np.isclose(vertices, [source.score, source.HR_upper], rtol=0, atol=1e-12), axis=1))
            verify('g', endpoint + ' curve ' + str(j), ['score', 'HR', 'HR_lower', 'HR_upper'], [source.score, source.HR, source.HR_lower, source.HR_upper], [line.get_xdata()[j], line.get_ydata()[j], lower[j], upper[j]])
        plot.set_yscale('log')
        plot.set(xlim=(0, 1), ylim=(0.025, 30), xticks=[0, 0.5, 1], yticks=[0.03, 0.3, 1, 3, 30])
        plot.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        plot.yaxis.set_minor_locator(NullLocator())
        plot.axhline(1, ls=(0, (2, 3)), color='#9FABB8', lw=0.5)
        median = plot.axvline(row.score_reference, color='#C0C8D0', lw=0.5, ls=(0, (1, 3)))
        verify('g', endpoint + ' median reference', ['score_reference'], [row.score_reference], [median.get_xdata()[0]])
        clean(plot)
        plot.set_xlabel('MRU benefit score', fontsize=5.8, labelpad=2)
        if i == 0:
            plot.set_ylabel('HR vs median', fontsize=5.8, labelpad=3)
        else:
            plot.tick_params(axis='y', labelleft=False)
        ft(fig, x + 15.5, 203, endpoint, size=6.3, color=ENDPOINTS[endpoint], weight='bold', ha='center', panel='g')
        txt(plot, 0.99, 0.97, f'Overall P = {pformat(row.P_overall)}\nNonlinear P = {pformat(row.P_nonlinear)}', size=5.5, ha='right', va='top', trans=plot.transAxes, color=MUTED)

def draw_h(fig, b):
    title(fig, 'h', 96, 196, 'Clinical phenotype profiles')
    variables = list(dict.fromkeys(b['phenotype'].Variable))
    for i, group in enumerate(GROUPS):
        x = 115 + 21 * i
        plot = axis(fig, (x, 209, 17.5, 28), 'h')
        for j, variable in enumerate(variables):
            row = b['phenotype'].loc[b['phenotype'].Variable.eq(variable) & b['phenotype'].MBP_group.eq(group)].iloc[0]
            line, = plot.plot([row.Lower, row.Upper], [6 - j, 6 - j], color='#98A6B5', lw=0.65)
            point, = plot.plot(row.Effect, 6 - j, ls='none', marker='o', ms=3.2, color=COLORS[group], mec='white', mew=0.35)
            verify('h', variable + ' ' + group, ['Lower', 'Effect', 'Upper'], [row.Lower, row.Effect, row.Upper], [line.get_xdata()[0], point.get_xdata()[0], line.get_xdata()[1]])
        plot.axvline(0, color='#B2BDCA', lw=0.5, ls=(0, (2, 3)), zorder=0)
        plot.set(xlim=(-0.72, 0.78), ylim=(-0.6, 6.6), xticks=[-0.5, 0, 0.5], yticks=[])
        if i == 0:
            plot.set_yticks(list(range(6, -1, -1)), labels=variables)
            plot.tick_params(axis='y', labelsize=5.5, pad=3)
        clean(plot, left=False)
        plot.tick_params(axis='x', labelsize=5.5)
        ft(fig, x + 8.75, 203, group, size=6.1, color=COLORS[group], weight='bold', ha='center', panel='h')
    ft(fig, 145, 243, 'Standardized deviation (95% interval)', size=5.7, color=MUTED, ha='center', panel='h')

def main():
    bundle = old.load_inputs()
    sources = provenance(bundle)
    old.configure_style()
    plt.rcParams.update({'font.size': 6.0, 'xtick.labelsize': 5.7, 'ytick.labelsize': 5.7, 'axes.labelsize': 6.0, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white', dpi=120)
    draw_a(fig, bundle)
    draw_d(fig, bundle)
    survival(fig, bundle, 'PFS', 'b', 16)
    survival(fig, bundle, 'OS', 'c', 103)
    draw_e(fig, bundle)
    draw_f(fig, bundle)
    draw_g(fig, bundle)
    draw_h(fig, bundle)
    layout = audit_figure_layout(fig, get_value('figures.config.item_332'), REG, minimum_font_pt=5.5, safe_pad_pt=1.2, check_collisions=True)
    intact = all((sha(e['original']) == e['original_sha256_now'] and sha(old.INPUT_DIR / e['snapshot']) == e['snapshot_sha256_now'] for e in sources['original_tables']))
    assert intact and sha(CODE) == sources['original_renderer_sha256']
    report = {'figure': 'Current Figure4 / original Figure5, complete a–h', 'size_mm': [W, H], 'source_provenance': sources, 'original_input_validation': bundle['numeric_qa'], 'quantitative_artist_checks': len(CHECKS), 'all_artist_values_match_original_sources': True, 'original_files_unchanged': intact, 'original_renderer_unchanged': True, 'original_notebook_unchanged': sha(sources['source_notebook']) == sources['source_notebook_sha256'], 'new_model_km_cox_spline_cv_bootstrap_runs': 0, 'scientific_contract': bundle['contract'], 'display_changes': ['count-preserving R/NR–MBP alluvial with all original n, row%, ASR', 'solid KM step curves, lighter original confidence bands, all original censor/risk data', 'separated unadjusted/adjusted forest rows, original effects/CI/P', 'within-mRECIST forest replaces effect heatmap', 'four distinct nested-model slope facets replace horizontal dumbbells', 'unchanged spline coordinates with light neutral original CI bands', 'three aligned clinical-profile facets instead of overlaid group intervals']}
    for name, data in [(get_value('figures.config.item_210'), report), (get_value('figures.config.item_057'), CHECKS), (get_value('figures.config.item_119'), layout)]:
        (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    stem = OUT / get_value('figures.config.item_333')
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(stem.with_suffix('.' + fmt), facecolor='white', bbox_inches=None, **{'dpi': 600} if fmt == 'png' else {})
    fig.savefig(OUT / get_value('figures.config.item_012'), dpi=190, facecolor='white', bbox_inches=None)
    plt.close(fig)
if __name__ == '__main__':
    main()
