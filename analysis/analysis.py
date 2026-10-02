"""Patient-level analysis of the four models (Paper 2): per-patient decisions, error concentration,
patient-bootstrap intervals, paired differences to BiFAM-Next and the exploratory four-model average.
Usage (from the repository root): python analysis/analysis.py  ->  results/analysis_recomputed.json"""
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
M = ['resnet50', 'efficientnet_b0', 'densenet201', 'bifam_next']
R = 'results'
def load(prot, m):
    d = pd.read_csv(f'{R}/preds/{prot}_{m}.csv', keep_default_na=False).sort_values('path').reset_index(drop=True)
    d['pred'] = d[['prob_0', 'prob_1', 'prob_2']].values.argmax(1); return d
f1 = lambda y, p: 100 * f1_score(y, p, average='macro')
out = {}
P = {m: load('patient', m) for m in M}; I = {m: load('image', m) for m in M}
groups = P[M[0]].group.values; ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
rng = np.random.default_rng(0); boot = [np.concatenate([gi[g] for g in rng.choice(ug, len(ug))]) for _ in range(2000)]
y = P[M[0]].label.values
for m in M:
    d = P[m]; wrong = d.pred != d.label
    mean_p = d.groupby('group')[['prob_0', 'prob_1', 'prob_2']].mean(); yl = d.groupby('group').label.first()
    pp = mean_p.values.argmax(1); maj = d.groupby('group').pred.agg(lambda s: s.value_counts().idxmax())
    ebp = wrong.groupby(d.group).sum(); ebp = ebp[ebp > 0]
    out[m] = dict(per_patient_correct=int((pp == yl.values).sum()), majority_correct=int((maj.values == yl.values).sum()),
                  per_patient_f1=round(f1(yl.values, pp), 2), slice_errors=int(wrong.sum()),
                  top3_share=int(ebp.sort_values(ascending=False).head(3).sum()), patients_with_error=int(len(ebp)),
                  wrong_patients=sorted(yl.index[pp != yl.values]), err_by_patient={k: int(v) for k, v in ebp.items()},
                  patient_f1_ci=[round(float(x), 1) for x in np.percentile([f1(y[b], d.pred.values[b]) for b in boot], [2.5, 97.5])],
                  image_errors=int((I[m].pred != I[m].label).sum()))
bf = P['bifam_next'].pred.values
def paired(m):                                           # bootstrap mean and 95% interval of the paired macro-F1 difference
    d = [f1(y[b], bf[b]) - f1(y[b], P[m].pred.values[b]) for b in boot]
    return [round(float(np.mean(d)), 2)] + [round(float(v), 2) for v in np.percentile(d, [2.5, 97.5])]
out['diff_vs_bifam'] = {m: paired(m) for m in M[:3]}
out['image_test_patients'] = int(I[M[0]].group.nunique()); out['image_test_slices'] = int(len(I[M[0]]))
for prot, D in [('patient', P), ('image', I)]:
    avg = np.mean([D[m][['prob_0', 'prob_1', 'prob_2']].values for m in M], 0).argmax(1); yy = D[M[0]].label.values
    out[f'ens4_{prot}'] = dict(acc=round(100 * (avg == yy).mean(), 2), f1=round(f1(yy, avg), 2))
json.dump(out, open(f'{R}/analysis_recomputed.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['diff_vs_bifam', 'ens4_patient', 'ens4_image']}))
print({m: (out[m]['per_patient_correct'], out[m]['majority_correct'], out[m]['slice_errors'], out[m]['top3_share'], out[m]['patient_f1_ci']) for m in M})
