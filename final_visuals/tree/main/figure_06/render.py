from avatar_paths import get_value, get_column
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT = OUT / 'inputs'
INPUT.mkdir(exist_ok=True)
SOURCE = ROOT / get_value('figures.module_path.render_3')
TABLE_ROOT = Path(__import__('avatar_final_paths').resolve(get_value('figures.config.item_341')))
manifest = []
for name in [get_value('figures.config.item_036'), get_value('figures.config.item_039'), get_value('figures.config.item_040')]:
    src, dst = (TABLE_ROOT / name, INPUT / name)
    if not dst.exists():
        shutil.copyfile(src, dst)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
    manifest.append({'source': str(src), 'snapshot': str(dst), 'sha256': digest})
table = pd.read_csv(INPUT / get_value('figures.config.item_040'))
internal = pd.read_csv(INPUT / get_value('figures.config.item_036'))
external = pd.read_csv(INPUT / get_value('figures.config.item_039'))
assert table.module.tolist() == internal.module.tolist() == external.module.tolist()
for cohort, source_table in [('internal', internal), ('external', external)]:
    np.testing.assert_array_equal(table[f'{cohort}_effect'], source_table.effect_z_R_minus_NR)
    np.testing.assert_array_equal(table[f'{cohort}_q'], source_table.q_BH_across_modules)
    assert (source_table.n_R + source_table.n_NR).nunique() == 1
n_internal = int(internal.n_R.iloc[0] + internal.n_NR.iloc[0])
n_external = int(external.n_R.iloc[0] + external.n_NR.iloc[0])
spec = importlib.util.spec_from_file_location('figure6_external_validation_base', SOURCE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
bundle = base.load_inputs()
reference, _ = base.build_full_figure(bundle)
fig, registry = base.build_full_figure(bundle)
original_axes = len(fig.axes)
for ax in fig.axes:
    lo, hi = ax.get_ylim()
    ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])

def text_mm(x, top, value, **kwargs):
    artist = fig.text(x / 180, 1 - top / 200, value, color=base.TEXT, fontsize=5.4, **kwargs)
    registry.tag(artist, name='validation-' + value, role='direct_label', panel='a')
    return artist
text_mm(145, 5.2, 'Internal–external validation', ha='center', va='top', fontweight='bold')
col_internal, col_external = ('#4279B1', '#278D78')
for x, color, marker, label in [(115.8, col_internal, 'o', f'Internal (n={n_internal})'), (147.4, col_external, 'D', f'External (n={n_external})')]:
    fig.add_artist(base.mpl.lines.Line2D([x / 180], [1 - 11.2 / 200], transform=fig.transFigure, marker=marker, markersize=3.1, color=color, linestyle='none'))
    text_mm(x + 2.4, 11.2, label, ha='left', va='center')
ax = fig.add_axes([144 / 180, 1 - 29.4 / 200, 32 / 180, 14 / 200])
ax._qa_meta = {'panel': 'a'}
ys = np.array([2.0, 1.0, 0.0])
for index, row in table.iterrows():
    ax.plot([row.internal_effect, row.external_effect], [ys[index] + 0.09, ys[index] - 0.09], color='#B9C3CC', linewidth=0.8, zorder=1)
ax.scatter(table.internal_effect, ys + 0.09, s=17, color=col_internal, edgecolors='white', linewidths=0.35, zorder=3)
ax.scatter(table.external_effect, ys - 0.09, s=17, marker='D', color=col_external, edgecolors='white', linewidths=0.35, zorder=3)
ax.axvline(0, color='#CCD2D8', linewidth=0.65, zorder=0)
ax.set(xlim=(-1.12, 1.02), ylim=(-0.42, 2.42), xticks=[-1, 0, 1], yticks=[])
ax.tick_params(axis='x', labelsize=5.4, length=2, pad=1.8)
for side in ['left', 'right', 'top']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#7B8792')
ax.spines['bottom'].set_linewidth(0.6)
labels = ['Low-gray-weighted', 'Coarse organization', 'Fine heterogeneity']
for y, label in zip(ys, labels):
    top = 29.4 - (y + 0.42) / 2.84 * 14
    text_mm(142, top, label, ha='right', va='center')
text_mm(160, 35.6, 'Response effect (R − NR, z)', ha='center', va='center')
counts = {'axes': original_axes, 'lines': 0, 'scatter_arrays': 0, 'patches': 0}
for a, b in zip(reference.axes, fig.axes[:original_axes]):
    np.testing.assert_allclose(a.get_position().bounds, b.get_position().bounds, rtol=0, atol=1e-14)
    assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()
    assert len(a.lines) == len(b.lines) and len(a.collections) == len(b.collections)
    assert len(a.patches) == len(b.patches)
    for c, d in zip(a.lines, b.lines):
        np.testing.assert_allclose(c.get_xydata(), d.get_xydata(), rtol=0, atol=1e-14)
        assert c.get_color() == d.get_color()
        counts['lines'] += 1
    for c, d in zip(a.collections, b.collections):
        np.testing.assert_allclose(c.get_offsets(), d.get_offsets(), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(c.get_facecolors(), d.get_facecolors())
        for p, q in zip(c.get_paths(), d.get_paths()):
            np.testing.assert_allclose(p.vertices, q.vertices, rtol=0, atol=1e-14)
        counts['scatter_arrays'] += 1
    for c, d in zip(a.patches, b.patches):
        np.testing.assert_allclose(c.get_path().vertices, d.get_path().vertices, rtol=0, atol=1e-14)
        assert c.get_facecolor() == d.get_facecolor()
        counts['patches'] += 1
    assert [(t.get_text(), t.get_position()) for t in a.texts] == [(t.get_text(), t.get_position()) for t in b.texts]
assert [(t.get_text(), t.get_position()) for t in reference.texts] == [(t.get_text(), t.get_position()) for t in fig.texts[:len(reference.texts)]]
layout = base.audit_figure_layout(fig, get_value('figures.config.item_338'), registry, minimum_font_pt=5.4, safe_pad_pt=1.2, check_collisions=True)
stars = base.audit_star_visibility(fig, get_value('figures.config.item_338'))
(OUT / get_value('figures.config.item_121')).write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding='utf-8')
assert layout['summary']['critical'] == 0 and layout['summary']['warning'] == 0, layout['summary']
assert stars['summary']['pass'], stars['summary']
base.VERSION = get_value('figures.config.item_338')
base._save_exact(fig, OUT / get_value('figures.config.item_339'), 180, 200)
fig.savefig(OUT / get_value('figures.config.item_014'), dpi=240, facecolor='white')
fig.savefig(OUT / get_value('figures.config.item_015'), dpi=400, facecolor='white', bbox_inches=base.mpl.transforms.Bbox.from_bounds(113 / 25.4, 163 / 25.4, 66 / 25.4, 35 / 25.4))
audit = {'inputs': manifest, 'source_renderer': str(SOURCE), 'source_renderer_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'original_artists_and_geometry_unchanged': True, 'original_artist_checks': counts, 'added_effect_records': table.to_dict(orient='records'), 'internal_n': n_internal, 'external_n': n_external, 'new_analysis': False, 'layout_summary': layout['summary'], 'star_summary': stars['summary']}
(OUT / get_value('figures.config.item_060')).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
