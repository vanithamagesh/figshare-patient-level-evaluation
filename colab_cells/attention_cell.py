# ===== Paper 2: attention maps from the saved runs. GPU runtime recommended (~10 min); CPU works but is slow. =====
# Uses only saved checkpoints; nothing is trained. Outputs go to MyDrive/BiFAM/paper2/.
import os, sys, json, subprocess, numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
if 'PROJECT' not in globals():
    from google.colab import drive; drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/BiFAM'
sys.path.insert(0, PROJECT); import bifamnet
bifamnet.TIMM_NAMES.setdefault('efficientnet_b0', 'efficientnet_b0')
from torch import nn
if not hasattr(bifamnet, 'PyramidPool'):           # BiFAM-Next = BiFAMLite + pyramid pooling (same code as the jobs notebook)
    class PyramidPool(nn.Module):
        def __init__(self, c, bins=(1, 2, 3, 6)):
            super().__init__()
            self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(c, c // 4, 1, bias=False),
                                                       nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True)) for b in bins])
            self.fuse = nn.Sequential(nn.Conv2d(c + len(bins) * (c // 4), c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        def forward(self, x):
            h, w = x.shape[-2:]
            return self.fuse(torch.cat([x] + [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False) for s in self.stages], 1))
    _Lite = bifamnet.BiFAMLite
    class BiFAMNext(_Lite):
        def __init__(self, **cfg):
            super().__init__(**cfg)
            if self.cfg.get('ppm', 'convnext' in str(self.cfg.get('backbone', ''))):
                self.bottleneck = nn.Sequential(self.bottleneck, PyramidPool(self.cfg['bottleneck_channels']))
    bifamnet.PyramidPool, bifamnet.BiFAMLite = PyramidPool, BiFAMNext
D = globals().get('DATA_DIR', '/content/data_v2_s20')
if not os.path.exists(f'{D}/splits/threeclass_image.json'):
    print('restoring prepared data from Drive ...')
    subprocess.run(['unzip', '-q', '-o', f'{PROJECT}/prepared_data_v2_s20.zip', '-d', '/content'], check=True)
OUT = f'{PROJECT}/paper2'; os.makedirs(OUT, exist_ok=True)
RUNS = f'{PROJECT}/runs/next'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device:', dev)
CLS = ['glioma', 'meningioma', 'pituitary']
idx = bifamnet.load_index(D, 'threeclass').set_index('path', drop=False)
SPL = {t: json.load(open(f'{D}/splits/{t}.json')) for t in ['threeclass_image', 'threeclass_patient']}
img_train = set(SPL['threeclass_image']['train']) | set(SPL['threeclass_image']['val'])

def ckpt(task, run):
    p = f'{RUNS}/{task}/{run}/seed11.pt'
    return p if os.path.exists(p) else None
MAIN = {t: ckpt(t, 'bifamlite') for t in SPL}
print('BiFAM-Next checkpoints:', MAIN)
assert all(MAIN.values()), 'a BiFAM-Next checkpoint (seed11.pt) is missing - tell Claude which one'
BASE = {m: ckpt('threeclass_patient', f'base_{m}_ck') for m in ['resnet50', 'efficientnet_b0', 'densenet201']}
print('baseline checkpoints (optional, from the retrain cell):', BASE)

def inputs(paths, size):
    return torch.stack([bifamnet._row_input(D, idx.loc[p], size, 1) for p in paths])
def mask(p, size):
    m = idx.loc[p, 'mask']
    return (bifamnet._png(D, m, size) > 127) if m else np.zeros((size, size), bool)
def share_in(h, m):                                    # share of the map's total activation that lies inside the tumor mask
    return float((h * m).sum() / (h.sum() + 1e-8))

def next_maps(model, paths, bs=8):
    """BiFAM-Next: Grad-CAM, BiFAM 1/4 and attention gate 1/4 (all in [0,1]) + single-view probabilities."""
    size = model.cfg['img_size']; res = []
    for s in range(0, len(paths), bs):
        x = inputs(paths[s:s + bs], size).to(dev)
        mp = bifamnet.compute_maps(model, x, level=2)
        with torch.no_grad():
            pr = bifamnet.to_prob(model(x)['logits'].float(), 3).cpu().numpy()
        for i in range(len(x)):
            res.append(dict(cam=mp['gradcam'][i].cpu().numpy(), bifam=mp['bifam'][i].cpu().numpy(),
                            ag=mp['ag'][i].cpu().numpy() if 'ag' in mp else None, prob=pr[i]))
    return res

def timm_cam(model, paths, bs=8):
    """Grad-CAM on the last feature map of a timm classifier (ResNet50, EfficientNet-B0, DenseNet201)."""
    size = model.cfg['img_size']; net = model.net; res = []
    for s in range(0, len(paths), bs):
        x = inputs(paths[s:s + bs], size).to(dev)
        with torch.enable_grad():
            f = net.forward_features(x); f.retain_grad(); lg = net.forward_head(f)
            lg.gather(1, lg.argmax(1, keepdim=True)).sum().backward()
        cam = F.relu((f.grad.mean((2, 3), keepdim=True) * f).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode='bilinear', align_corners=False)[:, 0]
        cam = bifamnet._norm(cam.detach()).cpu().numpy(); pr = torch.softmax(lg.detach().float(), 1).cpu().numpy()
        res += [dict(cam=cam[i], prob=pr[i]) for i in range(len(x))]
    return res

# ---------- 1) maps for every test slice of both partitions: how much attention falls on the tumor ----------
rows = []
models = {}
for task, ck in MAIN.items():
    model, _ = bifamnet.load_checkpoint(ck, dev); models[task] = model; size = model.cfg['img_size']
    test = [p for p in SPL[task]['test'] if p in idx.index]
    pred = pd.read_csv(f'{RUNS}/{task}/bifamlite/predictions_seed11.csv', keep_default_na=False).set_index('path')
    pc = [c for c in pred.columns if c.startswith('prob_')]
    for s in range(0, len(test), 64):
        chunk = test[s:s + 64]
        for p, r in zip(chunk, next_maps(model, chunk)):
            m = mask(p, size); y = int(idx.loc[p, 'label']); yh = int(pred.loc[p, pc].values.argmax())
            rows.append(dict(task=task, model='bifam_next', path=p, group=idx.loc[p, 'group'], label=y, pred=yh, correct=yh == y,
                             conf=float(pred.loc[p, pc].values.max()), mask_frac=float(m.mean()),
                             cam_in=share_in(r['cam'], m), bifam_in=share_in(r['bifam'], m),
                             ag_in=share_in(r['ag'], m) if r['ag'] is not None else np.nan))
    print(task, 'done:', sum(r['task'] == task for r in rows), 'slices')
for mname, ck in BASE.items():
    if not ck: continue
    model, _ = bifamnet.load_checkpoint(ck, dev); size = model.cfg['img_size']
    test = [p for p in SPL['threeclass_patient']['test'] if p in idx.index]
    for s in range(0, len(test), 64):
        chunk = test[s:s + 64]
        for p, r in zip(chunk, timm_cam(model, chunk)):
            m = mask(p, size); y = int(idx.loc[p, 'label']); yh = int(r['prob'].argmax())
            rows.append(dict(task='threeclass_patient', model=mname, path=p, group=idx.loc[p, 'group'], label=y, pred=yh,
                             correct=yh == y, conf=float(r['prob'].max()), mask_frac=float(m.mean()),
                             cam_in=share_in(r['cam'], m), bifam_in=np.nan, ag_in=np.nan))
    print(mname, 'done')
A = pd.DataFrame(rows); A.to_csv(f'{OUT}/attention_in_mask.csv', index=False)
summ = (A.groupby(['model', 'task', 'correct'])[['cam_in', 'bifam_in', 'ag_in', 'mask_frac']]
          .median().round(3).reset_index())
summ['n'] = A.groupby(['model', 'task', 'correct']).size().values
print(summ.to_string()); summ.to_csv(f'{OUT}/attention_summary.csv', index=False)

# ---------- 2) figure: the same patient-level test slices seen by the patient-level and the image-level model ----------
P = A[(A.model == 'bifam_next') & (A.task == 'threeclass_patient')]
H = globals().get('HARD', ['fs_MR037458B', 'fs_105538', 'fs_112074'])
pick = []
for g in H:                                         # the most confident wrong slice of each hard patient
    q = P[(P.group == g)].sort_values(['correct', 'conf'], ascending=[True, False])
    if len(q): pick.append(q.iloc[0].path)
clean = P.groupby('group').correct.all(); clean = set(clean[clean].index)
for c in range(3):                                   # one correct slice per class: the largest tumor of an error-free patient
    q = P[(P.label == c) & P.group.isin(clean)].sort_values('mask_frac', ascending=False)
    if len(q): pick.append(q.iloc[0].path)
size = models['threeclass_patient'].cfg['img_size']
mp_pat = next_maps(models['threeclass_patient'], pick); mp_img = next_maps(models['threeclass_image'], pick)
base_maps = {m: timm_cam(bifamnet.load_checkpoint(ck, dev)[0], pick) for m, ck in BASE.items() if ck}
def ov(ax, im, h, m, title, color='black'):
    ax.imshow(im, cmap='gray'); ax.imshow(h, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    if m.any(): ax.contour(m, levels=[0.5], colors='#1baf7a', linewidths=1)
    ax.set_title(title, fontsize=9, color=color); ax.axis('off')
cols = ['Input + mask', 'Patient-level model\nGrad-CAM', 'Patient-level model\nBiFAM 1/4', 'Image-level model\nGrad-CAM', 'Image-level model\nBiFAM 1/4']
fig, axs = plt.subplots(len(pick), 5, figsize=(13, 2.75 * len(pick)), squeeze=False)
for i, p in enumerate(pick):
    im = inputs([p], size)[0, 0].numpy(); m = mask(p, size); y = int(idx.loc[p, 'label'])
    axs[i, 0].imshow(im, cmap='gray'); axs[i, 0].contour(m, levels=[0.5], colors='#1baf7a', linewidths=1) if m.any() else None
    axs[i, 0].axis('off'); axs[i, 0].set_title(f"true: {CLS[y]}" + ("\n(hard patient)" if idx.loc[p, 'group'] in H else ''), fontsize=9)
    for j, (r, tag) in enumerate([(mp_pat[i], 'new patient'), (mp_img[i], 'seen in training' if p in img_train else 'image-level test')]):
        yh = int(r['prob'].argmax()); col = 'black' if yh == y else '#c00000'
        t = f"pred {CLS[yh]} ({r['prob'].max():.2f})\n{tag}"
        ov(axs[i, 1 + 2 * j], im, r['cam'], m, t, col); ov(axs[i, 2 + 2 * j], im, r['bifam'], m, '', col)
for j, c in enumerate(cols): axs[0, j].set_title(c + ('\n' + axs[0, j].get_title() if axs[0, j].get_title() else ''), fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_p2_attention_protocols.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- 3) figure (only if baseline checkpoints exist): Grad-CAM of the four models on the same slices ----------
if base_maps:
    names = [('BiFAM-Next', mp_pat)] + [(n, base_maps[k]) for k, n in [('resnet50', 'ResNet50'), ('efficientnet_b0', 'EfficientNet-B0'), ('densenet201', 'DenseNet201')] if k in base_maps]
    fig, axs = plt.subplots(len(pick), len(names) + 1, figsize=(2.6 * (len(names) + 1), 2.75 * len(pick)), squeeze=False)
    for i, p in enumerate(pick):
        im = inputs([p], size)[0, 0].numpy(); m = mask(p, size); y = int(idx.loc[p, 'label'])
        axs[i, 0].imshow(im, cmap='gray'); axs[i, 0].axis('off'); axs[i, 0].set_title(f'true: {CLS[y]}', fontsize=9)
        if m.any(): axs[i, 0].contour(m, levels=[0.5], colors='#1baf7a', linewidths=1)
        for j, (n, mm) in enumerate(names):
            yh = int(mm[i]['prob'].argmax())
            ov(axs[i, j + 1], im, mm[i]['cam'], m, (n + '\n' if i == 0 else '') + f"pred {CLS[yh]} ({mm[i]['prob'].max():.2f})",
               'black' if yh == y else '#c00000')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_p2_attention_models.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---------- 4) figure: share of Grad-CAM activation inside the tumor mask ----------
groups = [(m, t, c) for m in ['bifam_next', 'resnet50', 'efficientnet_b0', 'densenet201'] for t in ['threeclass_image', 'threeclass_patient'] for c in [True, False]]
groups = [g for g in groups if ((A.model == g[0]) & (A.task == g[1]) & (A.correct == g[2])).any()]
fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups)), 4.2))
data = [A[(A.model == m) & (A.task == t) & (A.correct == c)].cam_in.values for m, t, c in groups]
bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color='black'))
for b, (m, t, c) in zip(bp['boxes'], groups):
    b.set_facecolor('#2a78d6' if t == 'threeclass_image' else '#eb6834'); b.set_alpha(0.85 if c else 0.4)
for k, d in enumerate(data):
    ax.scatter(np.random.default_rng(k).normal(k + 1, 0.06, len(d)), d, s=6, color='#333333', alpha=0.35, zorder=3)
NM = dict(bifam_next='BiFAM-Next', resnet50='ResNet50', efficientnet_b0='EffNet-B0', densenet201='DenseNet201')
ax.set_xticks(range(1, len(groups) + 1))
ax.set_xticklabels([f"{NM[m]}\n{'image' if t == 'threeclass_image' else 'patient'}\n{'correct' if c else 'wrong'} (n={len(d)})" for (m, t, c), d in zip(groups, data)], fontsize=8)
ax.set_ylabel('Share of Grad-CAM inside tumor mask'); ax.set_ylim(0, 1); ax.grid(axis='y', color='#e5e5e5'); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_p2_attention_stats.png', dpi=250, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('saved to', OUT, sorted(os.listdir(OUT)))
