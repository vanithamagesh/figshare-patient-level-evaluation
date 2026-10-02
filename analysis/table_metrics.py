import pandas as pd, numpy as np, json
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,cohen_kappa_score,confusion_matrix
M=['resnet50','efficientnet_b0','densenet201','bifam_next']; out={}
rng=np.random.default_rng(0)
for p in ['image','patient']:
  for m in M:
    d=pd.read_csv(f'results/preds/{p}_{m}.csv'); P=d[['prob_0','prob_1','prob_2']].values; y=d.label.values; yh=P.argmax(1)
    pr,rc,f1,_=precision_recall_fscore_support(y,yh,average='macro',zero_division=0)
    # per-class recall/specificity
    cm=confusion_matrix(y,yh)
    spec=np.mean([ (cm.sum()-cm[k].sum()-cm[:,k].sum()+cm[k,k])/(cm.sum()-cm[k].sum()) for k in range(3)])
    r=dict(acc=accuracy_score(y,yh)*100,prec=pr*100,rec=rc*100,f1=f1*100,spec=spec*100,auc=roc_auc_score(y,P,multi_class='ovr',average='macro')*100,kappa=cohen_kappa_score(y,yh),cm=cm.tolist())
    # bootstrap acc ci
    g=d.group.values if p=='patient' else np.arange(len(d)); ug=np.unique(g); idx={u:np.where(g==u)[0] for u in ug}
    rr=np.random.default_rng(0); accs=[];f1s=[]
    for b in range(2000):
      s=rr.choice(ug,len(ug)); ii=np.concatenate([idx[u] for u in s])
      accs.append((y[ii]==yh[ii]).mean()*100); f1s.append(precision_recall_fscore_support(y[ii],yh[ii],average='macro',zero_division=0)[2]*100)
    r['acc_ci']=list(np.percentile(accs,[2.5,97.5])); r['f1_ci']=list(np.percentile(f1s,[2.5,97.5]))
    out[f'{p}_{m}']=r
    print(p,m,' '.join(f'{k}={v:.2f}' for k,v in r.items() if isinstance(v,float)), 'accCI %.1f-%.1f f1CI %.1f-%.1f'%(*r['acc_ci'],*r['f1_ci']), cm.tolist())
json.dump(out,open('results/table_metrics.json','w'),indent=1,default=float)
