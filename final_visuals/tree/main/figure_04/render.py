from avatar_paths import get_value, get_column
from pathlib import Path
import importlib.util
import json
import numpy as np
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRIOR = ROOT / get_value('figures.module_path.render_2')
spec = importlib.util.spec_from_file_location('prior_figure4', PRIOR)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def draw_e(fig, b):
    base.title(fig, 'e', 2, 141, 'Effect within observed mRECIST strata')
    plot = base.axis(fig, (28, 154, 22, 25), 'e')
    table = base.axis(fig, (53, 154, 24, 25), 'e')
    table.set(xlim=(0, 1), ylim=(-0.7, 3.7))
    table.axis('off')
    labels = []
    for i, endpoint in enumerate(['PFS', 'OS']):
        for j, stratum in enumerate(['Responder', 'Nonresponder']):
            y = 3 - 2 * i - j
            r = b['within_mrecist'].loc[b['within_mrecist'].Endpoint.eq(endpoint) & b['within_mrecist'].mRECIST.eq(stratum)].iloc[0]
            labels.append(f'{endpoint} · ' + ('R' if stratum == 'Responder' else 'NR') + f'\nn={int(r.n)}, events={int(r.events)}')
            color = base.ENDPOINTS[endpoint]
            band = plot.fill_betweenx([y - 0.16, y + 0.16], r.CI_lower, r.CI_upper, color=color, alpha=0.23, linewidth=0)
            tick, = plot.plot([r.HR_per_MBP_level_worsening] * 2, [y - 0.18, y + 0.18], color=color, lw=1.65, solid_capstyle='butt')
            v = band.get_paths()[0].vertices[:, 0]
            base.verify('e', endpoint + ' ' + stratum, ['CI_lower', 'HR_per_MBP_level_worsening', 'CI_upper'], [r.CI_lower, r.HR_per_MBP_level_worsening, r.CI_upper], [v.min(), tick.get_xdata()[0], v.max()])
            base.txt(table, 0, y + 0.13, f'{r.HR_per_MBP_level_worsening:.2f} ({r.CI_lower:.2f}–{r.CI_upper:.2f})', size=5.6, color=color)
            base.txt(table, 0, y - 0.19, f'P = {base.pformat(r.P)}', size=5.5, color=base.MUTED)
    plot.set_xscale('log')
    plot.set(xlim=(0.8, 9), ylim=(-0.7, 3.7), xticks=[1, 2, 4, 8], yticks=[3, 2, 1, 0], yticklabels=labels)
    plot.xaxis.set_major_formatter(base.FuncFormatter(lambda x, _: f'{x:g}'))
    plot.xaxis.set_minor_locator(base.NullLocator())
    plot.axvline(1, ls=(0, (2, 3)), lw=0.5, color='#A9B4C0', zorder=0)
    plot.tick_params(axis='y', labelsize=5.5, pad=3)
    base.clean(plot, left=False)
    plot.set_xlabel('HR per MBP level', fontsize=5.8, labelpad=2)
    base.ft(fig, 53, 149, 'HR (95% CI)', size=5.5, color=base.MUTED, panel='e')

def draw_f(fig, b):
    base.title(fig, 'f', 82, 141, 'Incremental prognostic information')
    plot = base.axis(fig, (91, 155, 84, 25), 'f')
    sequence = [('PFS', 'Pretreatment'), ('PFS', 'Post-mRECIST'), ('OS', 'Pretreatment'), ('OS', 'Post-mRECIST')]
    for i, (endpoint, context) in enumerate(sequence):
        folds = b['incremental_folds'].loc[b['incremental_folds'].Endpoint.eq(endpoint) & b['incremental_folds'].Context.eq(context)].sort_values('fold_index')
        r = b['incremental_summary'].loc[b['incremental_summary'].Endpoint.eq(endpoint) & b['incremental_summary'].Context.eq(context)].iloc[0]
        v = folds.Delta_Cindex.to_numpy(float)
        assert len(v) == 25
        offsets = np.tile(np.array([-0.12, -0.06, 0, 0.06, 0.12]), 5)
        color = base.ENDPOINTS[endpoint]
        dots = plot.scatter(i + offsets, v, s=7, color=color, alpha=0.58, edgecolors='none', zorder=3)
        base.verify('f', endpoint + ' ' + context + ' saved fold values', ['Delta_Cindex (25 saved folds)'], v, dots.get_offsets()[:, 1])
        mean, = plot.plot([i - 0.2, i + 0.2], [r.Delta_Cindex_mean] * 2, color=color, lw=1.55, solid_capstyle='butt', zorder=4)
        base.verify('f', endpoint + ' ' + context + ' saved mean', ['Delta_Cindex_mean'], [r.Delta_Cindex_mean], [mean.get_ydata()[0]])
        base.verify('f', endpoint + ' ' + context + ' fold mean consistency', ['Delta_Cindex_mean'], [r.Delta_Cindex_mean], [np.mean(v)])
        x = 101.5 + 21 * i
        base.ft(fig, x, 147, endpoint, size=6.3, color=color, weight='bold', ha='center', panel='f')
        base.ft(fig, x, 151, context, size=5.5, ha='center', color=base.MUTED, panel='f')
        base.ft(fig, x, 187, f'ΔC {r.Delta_Cindex_mean:+.3f}', size=5.7, ha='center', color=color, weight='bold', panel='f')
        base.ft(fig, x, 191, f'LRT P = {base.pformat(r.LRT_P)}', size=5.5, ha='center', color=base.MUTED, panel='f')
    plot.axhline(0, color='#A9B4C0', lw=0.5, ls=(0, (2, 3)), zorder=0)
    plot.set(xlim=(-0.5, 3.5), ylim=(-0.13, 0.41), xticks=[], yticks=[-0.1, 0, 0.1, 0.2, 0.3])
    base.clean(plot)
    plot.set_ylabel('ΔC-index', fontsize=5.8, labelpad=3)
    base.ft(fig, 133, 182, 'Dots: saved CV folds · line: mean', size=5.5, color=base.MUTED, ha='center', panel='f')

def main():
    b = base.old.load_inputs()
    sources = base.provenance(b)
    previous_code_hash = base.sha(PRIOR)
    base.old.configure_style()
    base.plt.rcParams.update({'font.size': 6.0, 'xtick.labelsize': 5.7, 'ytick.labelsize': 5.7, 'axes.labelsize': 6.0, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = base.plt.figure(figsize=(base.W / 25.4, base.H / 25.4), facecolor='white', dpi=120)
    base.draw_a(fig, b)
    base.draw_d(fig, b)
    base.survival(fig, b, 'PFS', 'b', 16)
    base.survival(fig, b, 'OS', 'c', 103)
    draw_e(fig, b)
    draw_f(fig, b)
    base.draw_g(fig, b)
    base.draw_h(fig, b)
    layout = base.audit_figure_layout(fig, get_value('figures.config.item_334'), base.REG, minimum_font_pt=5.5, safe_pad_pt=1.2, check_collisions=True)
    intact = all((base.sha(e['original']) == e['original_sha256_now'] and base.sha(base.old.INPUT_DIR / e['snapshot']) == e['snapshot_sha256_now'] for e in sources['original_tables']))
    assert intact and base.sha(base.CODE) == sources['original_renderer_sha256'] and (base.sha(PRIOR) == previous_code_hash)
    report = {'figure': 'Current Figure4 / original Figure5, complete a–h', 'size_mm': [base.W, base.H], 'source_provenance': sources, 'original_input_validation': b['numeric_qa'], 'quantitative_artist_checks': len(base.CHECKS), 'all_artist_values_match_original_sources': True, 'original_files_unchanged': intact, 'original_renderer_unchanged': True, 'original_notebook_unchanged': base.sha(sources['source_notebook']) == sources['source_notebook_sha256'], 'new_model_km_cox_spline_cv_bootstrap_runs': 0, 'unchanged_panels': ['a', 'b', 'c', 'd', 'g', 'h'], 'display_changes': ['e: uniform original 95% CI bands and HR ticks; original numeric CI/P/n/events retained', 'f: 100 original CV fold increments and four stored means; original full-data LRT P retained'], 'fold_interpretation': '25 repeated-CV folds per context share patients; no new confidence intervals or tests', 'scientific_contract': b['contract']}
    for name, data in [(get_value('figures.config.item_211'), report), (get_value('figures.config.item_058'), base.CHECKS), (get_value('figures.config.item_120'), layout)]:
        (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    legend = (PRIOR.parent / get_value('figures.config.item_022')).read_text(encoding='utf-8')
    lines = legend.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('e, '):
            lines[i] = 'e, Original within-mRECIST HRs for PFS and OS per one-level worsening of MBP. Uniformly shaded horizontal bands span the original 95% CI; vertical ticks mark the original HR. The shading does not encode a probability density. R/NR, patient counts, event counts, numeric HR/CI and P are retained. No subgroup model or test was refitted.'
        elif line.startswith('f, '):
            lines[i] = 'f, Incremental prognostic information in four endpoint/context comparisons. Each dot is one of the 25 original saved repeated-CV fold differences in Harrell C-index between the clinical base model and that model plus MBP. Horizontal displacement prevents visual occlusion and has no quantitative meaning. Coloured horizontal bars and displayed mean increments (Delta C) are the original paired mean differences. Original full-data nested-Cox likelihood-ratio-test P values are retained below. Base in the pretreatment context uses ALBI, TBS, AFP, NLR and PVTT; Base in the post-mRECIST context additionally includes observed mRECIST. Repeated-CV folds share patients and are not independent patient observations. No confidence interval, new test, CV run or model fitting is added. The post-mRECIST context is early-response prognostic refinement rather than a pretreatment decision model.'
    legend = '\n'.join(lines).replace(get_value('figures.config.item_358'), get_value('figures.config.item_359'))
    legend += get_value('figures.annotation.figure_04')
    (OUT / get_value('figures.config.item_023')).write_text(legend, encoding='utf-8')
    stem = OUT / get_value('figures.config.item_335')
    for fmt in ['png', 'svg', 'pdf']:
        fig.savefig(stem.with_suffix('.' + fmt), facecolor='white', bbox_inches=None, **{'dpi': 600} if fmt == 'png' else {})
    fig.savefig(OUT / get_value('figures.config.item_013'), dpi=190, facecolor='white', bbox_inches=None)
    base.plt.close(fig)
if __name__ == '__main__':
    main()
