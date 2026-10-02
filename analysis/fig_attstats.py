exec(open('analysis/figs.py').read().split('# ---------- Figure: protocol schematic')[0])
A = pd.read_csv('results/attention_in_mask.csv')
CI, CP = '#2a78d6', '#eb6834'
G = [('threeclass_image', True), ('threeclass_image', False), ('threeclass_patient', True), ('threeclass_patient', False)]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(wspace=0.22))
for ax, (k, title) in zip(axs, [('bifam_in', 'BiFAM map, stride 4 (mask-supervised)'), ('ag_in', 'Attention gate α, stride 4'), ('cam_in', 'Grad-CAM, final decoder block (unsupervised)')]):
    data = [A[(A.task == t) & (A.correct == c)][k].values for t, c in G]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
    for b, (t, c) in zip(bp['boxes'], G):
        col = CI if t == 'threeclass_image' else CP
        b.set_facecolor(col if c else 'white'); b.set_edgecolor(col); b.set_linewidth(1.5)
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.07, len(d)), d, s=5, color=INK2, alpha=0.25, zorder=3, linewidths=0)
        ax.text(i + 1, 1.03, f"{np.median(d):.2f}", ha='center', fontsize=10, color=INK)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"{'Image' if t == 'threeclass_image' else 'Patient'}\n{'correct' if c else 'wrong'}\n(n={len(d)})" for (t, c), d in zip(G, data)], fontsize=9.5)
    ax.set_ylim(0, 1.09); ax.set_title(title, fontsize=11.5, pad=18); ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel('Share of map activation inside the tumor mask')
for ax, s in zip(axs, '(a) (b) (c)'.split()): tag(ax, s, -0.1)
fig.text(0.5, -0.06, 'Filled boxes: correctly classified slices; open boxes: misclassified slices. Blue: image-level partition; orange: patient-level partition. Numbers: medians. Median tumor area: 1.3% (image level) and 1.5% (patient level) of the slice.', ha='center', fontsize=9.5, color=INK2)
fig.savefig('figures/fig_p2_attention_stats.png', dpi=250, bbox_inches='tight', facecolor='white'); print('ok')
