exec(open('analysis/figs.py').read().split('# ---------- Figure: protocol schematic')[0])
E = json.load(open('results/extra_analysis.json'))
PI, PP = '#b8b6b0', '#52514e'          # image level (light), patient level (dark)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.15, 0.9, 1.2], wspace=0.32))
x = np.arange(4); w = 0.38
for k, (p, col) in enumerate([('image', PI), ('patient', PP)]):
    v = [100 * E[f'{p}_{m}']['top3'] / E[f'{p}_{m}']['n_wrong'] for m in M]
    bars = a.bar(x + (k - 0.5) * w, v, w - 0.03, color=col, label=f'{p}-level split')
    for xi, vi, m in zip(x + (k - 0.5) * w, v, M):
        a.text(xi, vi + 1.5, f"{E[f'{p}_{m}']['top3']}/{E[f'{p}_{m}']['n_wrong']}", ha='center', fontsize=9, color=INK)
a.set_xticks(x); a.set_xticklabels([NAME[m] for m in M], fontsize=10); a.set_ylim(0, 100)
a.set_ylabel('Share of slice errors in the 3 worst patients (%)'); a.grid(axis='y', color=GRID); a.set_axisbelow(True)
a.legend(frameon=False, loc='upper left', fontsize=10); tag(a, '(a)')
k = np.arange(1, 5)
for j, (p, col) in enumerate([('image', PI), ('patient', PP)]):
    v = [E[f'{p}_overlap']['by_k'][str(i)] for i in k]
    b.bar(k + (j - 0.5) * w, v, w - 0.03, color=col, label=f"{p}-level ({E[f'{p}_overlap']['union']} slices)")
    for xi, vi in zip(k + (j - 0.5) * w, v): b.text(xi, vi + 0.4, str(vi), ha='center', fontsize=9, color=INK)
b.set_xticks(k); b.set_xlabel('Number of models that misclassify the slice'); b.set_ylabel('Misclassified slices')
b.grid(axis='y', color=GRID); b.set_axisbelow(True); b.legend(frameon=False, fontsize=10); tag(b, '(b)')
rng = np.random.default_rng(0)
for i, m in enumerate(M):
    for j, p in enumerate(['image', 'patient']):
        d = load(p, m); P = d[[f'prob_{q}' for q in range(3)]].values; wrong = d.pred != d.label
        cf = P.max(1)[wrong.values]; xx = i * 2.6 + j + rng.normal(0, 0.07, len(cf))
        c.scatter(xx, 100 * cf, s=26, marker=MK[m], facecolor=COL[m] if p == 'patient' else 'white', edgecolor=COL[m], linewidth=1.2, zorder=3)
        c.text(i * 2.6 + j, 32, p[:3] + '.', ha='center', fontsize=9, color=INK2)
c.set_xticks([i * 2.6 + 0.5 for i in range(4)]); c.set_xticklabels([NAME[m] for m in M], fontsize=10)
c.axhline(90, color=INK2, lw=0.8, ls='--'); c.text(-0.45, 90.8, '0.9', fontsize=9, color=INK2, ha='left', va='bottom')
c.set_ylim(28, 101); c.set_ylabel('Confidence of misclassified slices (%)'); c.grid(axis='y', color=GRID); c.set_axisbelow(True)
c.text(0.0, -0.2, 'open markers: image-level split; filled markers: patient-level split', transform=c.transAxes, fontsize=9.5, color=INK2)
tag(c, '(c)')
fig.savefig('figures/fig_p2_errors.png', dpi=250, bbox_inches='tight', facecolor='white'); print('ok')
