import pandas as pd, numpy as np, json, itertools
from sklearn.metrics import precision_recall_fscore_support
M=['resnet50','efficientnet_b0','densenet201','bifam_next']; out={}
def ece(conf,correct,nb=15):
    b=np.linspace(0,1,nb+1); e=0
    for lo,hi in zip(b[:-1],b[1:]):
        s=(conf>lo)&(conf<=hi)
        if s.any(): e+=s.mean()*abs(correct[s].mean()-conf[s].mean())
    return e*100
for p in ['image','patient']:
    W={}
    for m in M:
        d=pd.read_csv(f'results/preds/{p}_{m}.csv'); P=d[['prob_0','prob_1','prob_2']].values; y=d.label.values; yh=P.argmax(1)
        pr,rc,f1,n=precision_recall_fscore_support(y,yh,zero_division=0)
        conf=P.max(1); c=(yh==y)
        Y=np.eye(3)[y]; brier=((P-Y)**2).sum(1).mean()
        g=d.group.values; eg=pd.Series(~c).groupby(g).sum().sort_values(ascending=False)
        out[f'{p}_{m}']=dict(prec=(pr*100).round(2).tolist(),rec=(rc*100).round(2).tolist(),f1=(f1*100).round(2).tolist(),n=n.tolist(),
          ece=round(ece(conf,c),2),brier=round(brier,4),conf_correct=round(conf[c].mean()*100,2),conf_wrong=round(conf[~c].mean()*100,2),
          wrong_conf_gt90=int((conf[~c]>0.9).sum()),n_wrong=int((~c).sum()),
          err_patients=int((eg>0).sum()),top3=int(eg.head(3).sum()))
        W[m]=set(d.path[~c])
        o=out[f'{p}_{m}']; print(p,m,o)
    allw=set().union(*W.values()); cnt={k:0 for k in range(1,5)}
    for s in allw: cnt[sum(s in W[m] for m in M)]+=1
    jac={f'{a}|{b}':round(len(W[a]&W[b])/len(W[a]|W[b]),3) for a,b in itertools.combinations(M,2)}
    out[f'{p}_overlap']=dict(union=len(allw),by_k=cnt,jaccard=jac); print(p,'overlap',out[f'{p}_overlap'])
json.dump(out,open('results/extra_analysis.json','w'),indent=1)
