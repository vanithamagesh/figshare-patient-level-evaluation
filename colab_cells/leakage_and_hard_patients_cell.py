# ===== Paper 2: image figures (leakage evidence + hard patients). Runs on a CPU runtime in ~5-10 min. =====
import os, json, subprocess, numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
if 'PROJECT' not in globals():
    from google.colab import drive; drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/BiFAM'
D = globals().get('DATA_DIR', '/content/data_v2_s20')
if not os.path.exists(f'{D}/splits/threeclass_image.json'):
    print('restoring prepared data from Drive ...')
    subprocess.run(['unzip', '-q', '-o', f'{PROJECT}/prepared_data_v2_s20.zip', '-d', '/content'], check=True)
OUT = f'{PROJECT}/paper2'; os.makedirs(OUT, exist_ok=True)
idx = pd.read_csv(f'{D}/figshare.csv', keep_default_na=False).set_index('path')
CLS = ['glioma', 'meningioma', 'pituitary']
def img(p, s=None):
    a = np.asarray(Image.open(f'{D}/{p}').convert('L'), dtype=np.float32)
    return a if s is None else np.asarray(Image.fromarray(a).resize((s, s), Image.BILINEAR), dtype=np.float32)
cache = {}
def vec(p):
    if p not in cache:
        v = img(p, 64).ravel(); v = (v - v.mean()) / (v.std() + 1e-6); cache[p] = v / np.sqrt(v.size)
    return cache[p]
stats, nn = {}, {}
for task in ['threeclass_image', 'threeclass_patient']:
    sp = json.load(open(f'{D}/splits/{task}.json'))
    tr = [p for p in sp['train'] + sp['val'] if p in idx.index]; te = [p for p in sp['test'] if p in idx.index]
    gtr = set(idx.loc[tr, 'group']); gte = idx.loc[te, 'group']
    Xtr = np.stack([vec(p) for p in tr]); Xte = np.stack([vec(p) for p in te])
    S = Xte @ Xtr.T; j = S.argmax(1); r = S.max(1)
    same = idx.loc[[tr[k] for k in j], 'group'].values == gte.values
    nn[task] = pd.DataFrame(dict(test=te, train=[tr[k] for k in j], r=r, same_patient=same, group=gte.values))
    stats[task] = dict(test_slices=len(te), test_patients=int(gte.nunique()),
                       test_slices_with_patient_in_training=int(gte.isin(gtr).sum()),
                       test_patients_in_training=int(len(set(gte) & gtr)),
                       nn_r_median=round(float(np.median(r)), 3), nn_r_q90=round(float(np.quantile(r, 0.9)), 3),
                       share_r_above_0_9=round(float((r > 0.9).mean()), 3), share_nn_same_patient=round(float(same.mean()), 3))
print(json.dumps(stats, indent=1)); json.dump(stats, open(f'{OUT}/leak_stats.json', 'w'), indent=1)
for t, d in nn.items(): d.to_csv(f'{OUT}/nn_{t}.csv', index=False)

# --- Figure A: nearest training slice for image-level test slices + similarity histograms ---
d = nn['threeclass_image'].sort_values('r', ascending=False).reset_index(drop=True)
picks = [d.iloc[int(q * (len(d) - 1))] for q in (0.05, 0.25, 0.5, 0.75)]
fig = plt.figure(figsize=(16, 7.6))
for k, row in enumerate(picks):
    for m, (p, lab) in enumerate([(row.test, 'test slice'), (row.train, 'nearest training slice')]):
        ax = fig.add_axes([0.02 + k * 0.155, 0.53 - m * 0.47, 0.145, 0.43]); ax.imshow(img(p), cmap='gray'); ax.axis('off')
        ax.set_title(f"{lab}\n{'same patient' if row.same_patient else 'other patient'}, r = {row.r:.3f}" if m else f"{lab} ({CLS[int(idx.loc[p, 'label'])]})", fontsize=10)
ax = fig.add_axes([0.67, 0.12, 0.31, 0.78])
bins = np.linspace(min(x.r.min() for x in nn.values()), 1, 40)
for t, c, lab in [('threeclass_image', '#2a78d6', 'image-level test slices'), ('threeclass_patient', '#eb6834', 'patient-level test slices')]:
    ax.hist(nn[t].r, bins=bins, color=c, alpha=0.6, label=f"{lab} (median r = {stats[t]['nn_r_median']:.2f})")
ax.set_xlabel('Correlation with the most similar training slice'); ax.set_ylabel('Test slices'); ax.legend(fontsize=9, loc='upper left')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.savefig(f'{OUT}/fig_p2_leakage.png', dpi=250, bbox_inches='tight', facecolor='white'); plt.close(fig)

# --- Figure B: slices of the three hard test patients with the predictions of the four models ---
H = globals().get('HARD', ['fs_MR037458B', 'fs_105538', 'fs_112074'])
RUNS = dict(BiFAM_Next='bifamlite', ResNet50='base_resnet50', EfficientNet_B0='base_efficientnet_b0', DenseNet201='base_densenet201')
pred = {}
for k, r in RUNS.items():
    f = f'{PROJECT}/runs/next/threeclass_patient/{r}/predictions_seed11.csv'
    if os.path.exists(f):
        q = pd.read_csv(f, keep_default_na=False); pred[k] = q.set_index('path')[[c for c in q if c.startswith('prob_')]].values.argmax(1)
        pred[k] = dict(zip(q.path, pred[k]))
H = [g for g in H if g in set(idx.group)]
fig, axs = plt.subplots(len(H), 6, figsize=(15, 2.9 * len(H) + 0.3), squeeze=False)
for i, g in enumerate(H):
    ps = sorted(idx.index[idx.group == g]); ps = [ps[int(t)] for t in np.linspace(0, len(ps) - 1, min(6, len(ps)))]
    for jj in range(6):
        ax = axs[i, jj]; ax.axis('off')
        if jj >= len(ps): continue
        p = ps[jj]; ax.imshow(img(p), cmap='gray')
        mp = idx.loc[p, 'mask'] if 'mask' in idx.columns else ''
        if mp and os.path.exists(f'{D}/{mp}'):
            mk = np.asarray(Image.open(f'{D}/{mp}').convert('L').resize(img(p).shape[::-1])) > 0
            if mk.any(): ax.contour(mk, levels=[0.5], colors='#1baf7a', linewidths=1)
        lab = '  '.join(f"{k[0]}:{CLS[pred[k][p]][0].upper()}" for k in ['BiFAM_Next', 'ResNet50', 'EfficientNet_B0', 'DenseNet201'] if k in pred and p in pred[k])
        ax.set_title(lab, fontsize=8)
    axs[i, 0].text(-0.08, 0.5, f"patient {i + 1}\ntrue: {CLS[int(idx.loc[ps[0], 'label'])]}\n{int((idx.group == g).sum())} slices",
                   transform=axs[i, 0].transAxes, ha='right', va='center', fontsize=10)
fig.savefig(f'{OUT}/fig_p2_hardpatients.png', dpi=250, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('saved to', OUT, os.listdir(OUT))
