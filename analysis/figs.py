import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.metrics import confusion_matrix, f1_score
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
M = ['resnet50', 'efficientnet_b0', 'densenet201', 'bifam_next']
NAME = {'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0', 'densenet201': 'DenseNet201', 'bifam_next': 'BiFAM-Next'}
COL = dict(zip(M, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'])); MK = dict(zip(M, ['o', 's', '^', 'D']))
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
R = json.load(open('results/protocol_results.json')); A = json.load(open('results/analysis.json'))
def load(prot, m):
    d = pd.read_csv(f'results/preds/{prot}_{m}.csv', keep_default_na=False).sort_values('path').reset_index(drop=True)
    d['pred'] = d[[c for c in d if c.startswith('prob_')]].values.argmax(1); return d
def tag(ax, s, x=-0.12): ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

# ---------- Figure: protocol schematic ----------
fig, axs = plt.subplots(1, 2, figsize=(14, 3.9))
rng = np.random.default_rng(3)
pcol = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#4a3aa7', '#008300']
for k, (ax, title, mode) in enumerate(zip(axs, ['(a) Image-level split: slices are shuffled', '(b) Patient-level split: patients are separated'], ['image', 'patient'])):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis('off'); ax.set_title(title, fontsize=12.5, fontweight='bold', loc='left')
    for x0, lab in [(0.3, 'Training set'), (6.3, 'Test set')]:
        ax.add_patch(FancyBboxPatch((x0, 0.3), 3.4 if x0 > 1 else 5.4, 3.4, boxstyle='round,pad=0.05,rounding_size=0.15', fc='#f4f3ef', ec=INK2, lw=1))
        ax.text(x0 + 0.1, 3.85, lab, fontsize=11, color=INK, fontweight='bold')
    slices = [(p, s) for p in range(6) for s in range(5)]
    if mode == 'image':
        idx = rng.permutation(len(slices)); test = set(idx[:8])
    else:
        test = {i for i, (p, s) in enumerate(slices) if p in (4, 5)}
    tr = [s for i, s in enumerate(slices) if i not in test]; te = [s for i, s in enumerate(slices) if i in test]
    for j, (p, s) in enumerate(tr):
        ax.add_patch(Rectangle((0.6 + (j % 8) * 0.62, 0.7 + (j // 8) * 1.05), 0.5, 0.8, fc=pcol[p], ec='white', lw=1.5))
    for j, (p, s) in enumerate(te):
        ax.add_patch(Rectangle((6.6 + (j % 5) * 0.62, 0.7 + (j // 5) * 1.05), 0.5, 0.8, fc=pcol[p], ec='white', lw=1.5))
    note = ('Slices of the same patient (same colour) appear on both sides:\nthe test set rewards recognising a lesion already seen in training.'
            if mode == 'image' else 'Every patient is entirely in training or entirely in test:\nthe test set measures generalisation to new patients.')
    ax.text(0.3, -0.35, note, fontsize=10.5, color=INK2, va='top')
fig.text(0.5, 1.02, 'Each coloured tile is one MRI slice; colour identifies the patient', ha='center', fontsize=10.5, color=INK2)
fig.savefig('figures/fig_p2_protocols.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Figure: protocol gap (slope) + patient-level CIs + paired differences ----------
fig = plt.figure(figsize=(16, 5.2))
a0 = fig.add_axes([0.05, 0.12, 0.27, 0.78]); a1 = fig.add_axes([0.42, 0.12, 0.25, 0.78]); a2 = fig.add_axes([0.77, 0.12, 0.22, 0.78])
for m in M:
    yv = [R[m]['image']['f1'], R[m]['patient']['f1']]
    a0.plot([0, 1], yv, '-', color=COL[m], lw=2, marker=MK[m], ms=9, mec='white', mew=1.5, zorder=3)
    off_r = {'densenet201': 0.12, 'bifam_next': -0.12}.get(m, 0); off_l = {'bifam_next': 0.1, 'densenet201': -0.1}.get(m, 0)
    a0.text(1.05, yv[1] + off_r, f"{NAME[m]}  {yv[1]:.2f}", va='center', fontsize=10, color=INK)
    a0.text(-0.05, yv[0] + off_l, f"{yv[0]:.2f}", va='center', ha='right', fontsize=10, color=INK)
a0.set_xticks([0, 1]); a0.set_xticklabels(['Image level\n(613 slices)', 'Patient level\n(46 new patients)']); a0.set_xlim(-0.35, 1.75)
a0.set_ylabel('Macro F1-score (%)'); a0.grid(axis='y', color=GRID); a0.set_ylim(92.5, 99)
a0.text(0.5, 92.75, 'drop: 2.2–2.4 points for every model', ha='center', fontsize=10, color=INK2)
tag(a0, '(a)', -0.2)
for i, m in enumerate(M):
    lo, hi = A[m]['patient_f1_ci']; v = R[m]['patient']['f1']
    a1.plot([lo, hi], [i, i], color=COL[m], lw=2.5, solid_capstyle='round'); a1.plot(v, i, MK[m], color=COL[m], ms=10, mec='white', mew=1.5)
    a1.text(hi + 0.3, i, f"{v:.2f} ({lo:.1f}–{hi:.1f})", va='center', fontsize=10, color=INK)
a1.set_yticks(range(4)); a1.set_yticklabels([NAME[m] for m in M]); a1.set_xlim(84, 104.5)
a1.set_xlabel('Patient-level macro F1 (%), patient-bootstrap 95% CI'); a1.grid(axis='x', color=GRID); tag(a1, '(b)', -0.33)
D = A['diff_vs_bifam']
for i, m in enumerate([x for x in M if x != 'bifam_next']):
    mu, lo, hi = D[m]
    a2.plot([lo, hi], [i, i], color=COL[m], lw=2.5, solid_capstyle='round'); a2.plot(mu, i, MK[m], color=COL[m], ms=10, mec='white', mew=1.5)
    a2.text(hi + 0.25, i, f"{mu:+.2f}", va='center', fontsize=10, color=INK)
a2.axvline(0, color=INK2, lw=1, ls='--')
a2.set_yticks(range(3)); a2.set_yticklabels([NAME[m] for m in M if m != 'bifam_next']); a2.set_xlim(-3, 7.5)
a2.set_xlabel('BiFAM-Next minus model (macro F1 points)'); a2.grid(axis='x', color=GRID); tag(a2, '(c)', -0.45)
fig.savefig('figures/fig_p2_gap.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Figure: confusion matrices, 2 protocols x 4 models ----------
CLS = ['Glioma', 'Mening.', 'Pituitary']
fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), gridspec_kw=dict(wspace=0.35, hspace=0.45))
for r, prot in enumerate(['image', 'patient']):
    for c, m in enumerate(M):
        d = load(prot, m); cm = confusion_matrix(d.label, d.pred, labels=range(3)); rp = cm / cm.sum(1, keepdims=True) * 100
        ax = axs[r, c]; ax.imshow(rp, cmap='Blues', vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', fontsize=12, fontweight='bold', color='white' if rp[i, j] > 60 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CLS, fontsize=9.5); ax.set_yticklabels(CLS, fontsize=9.5)
        for s in ax.spines.values(): s.set_visible(False)
        acc = 100 * (d.pred == d.label).mean()
        ax.set_title(f"{NAME[m]}\n{'Image' if prot == 'image' else 'Patient'} level · acc {acc:.2f}%", fontsize=10.5)
        if c == 0: ax.set_ylabel('True class')
        if r == 1: ax.set_xlabel('Predicted class')
fig.savefig('figures/fig_p2_confusion.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Figure: per-patient slice error rate, 4 models x 46 patients ----------
P = {m: load('patient', m) for m in M}
g0 = P[M[0]].groupby('group'); cls = g0.label.first(); n = g0.size()
err = pd.DataFrame({m: P[m].assign(e=P[m].pred != P[m].label).groupby('group').e.mean() for m in M})
order = sorted(err.index, key=lambda g: (cls[g], -err.loc[g].sum(), g))
E = err.loc[order, M].T.values * 100
fig, ax = plt.subplots(figsize=(17, 4.4))
im = ax.imshow(E, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i, m in enumerate(M):
    wrong = set(A[m]['wrong_patients'])
    for j, g in enumerate(order):
        if g in wrong: ax.text(j, i, '×', ha='center', va='center', fontsize=13, fontweight='bold', color='white' if E[i, j] > 60 else INK)
ax.set_yticks(range(4)); ax.set_yticklabels([NAME[m] for m in M])
ax.set_xticks(range(len(order))); ax.set_xticklabels([str(n[g]) for g in order], fontsize=8)
ax.set_xlabel('Test patients, grouped by tumor type and sorted by error rate (numbers: slices per patient; ×: patient misclassified)')
b = 0
for k, lab in enumerate(['Glioma', 'Meningioma', 'Pituitary']):
    cnt = int((cls[order] == k).sum()); ax.axvline(b + cnt - 0.5, color=INK, lw=1.5) if k < 2 else None
    ax.text(b + cnt / 2 - 0.5, -0.8, f"{lab} ({cnt} patients)", ha='center', fontsize=11, fontweight='bold'); b += cnt
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label('Slices misclassified (%)')
fig.savefig('figures/fig_p2_patients.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- Figure: error concentration ----------
H = ['fs_MR037458B', 'fs_105538', 'fs_112074']
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for i, m in enumerate(M):
    e = A[m]['err_by_patient']; h = sum(e.get(g, 0) for g in H); o = A[m]['slice_errors'] - h
    ax.barh(i, h, color=COL[m], height=0.6); ax.barh(i, o, left=h + 0.4, color=COL[m], alpha=0.35, height=0.6)
    ax.text(h / 2, i, f"{h}", ha='center', va='center', color=INK, fontsize=11, fontweight='bold')
    ax.text(h + o + 1.2, i, f"{h} of {A[m]['slice_errors']} errors ({100 * h / A[m]['slice_errors']:.0f}%) in 3 patients", va='center', fontsize=10, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([NAME[m] for m in M]); ax.set_xlim(0, 50); ax.set_axisbelow(True)
ax.set_xlabel('Misclassified test slices (dark: the same three patients; light: all other patients)')
ax.grid(axis='x', color=GRID)
fig.savefig('figures/fig_p2_concentration.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('ok')
