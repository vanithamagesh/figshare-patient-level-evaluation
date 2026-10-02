# ===== Paper 2 (OPTIONAL): retrain the three standard models on the patient-level split and KEEP the checkpoints,
# so that the attention cell can draw their Grad-CAM maps. Paste below the Step 6 cell of the jobs notebook (GPU).
# Jobs 20-22, one model each (~same time as jobs 17-19, patient-level half only). The reported results are NOT changed:
# the new runs go to base_<model>_ck, and the cell prints how often they agree with the reported predictions.
import sys, json, pandas as pd
from pathlib import Path
bifamnet.TIMM_NAMES['efficientnet_b0'] = 'efficientnet_b0'
if 'efficientnet_b0' not in bifamnet.BASELINES: bifamnet.BASELINES.append('efficientnet_b0')
CK_JOBS = {20: 'resnet50', 21: 'densenet201', 22: 'efficientnet_b0'}

def run_ck_job(job, settings):
    model, task = CK_JOBS[job], 'threeclass_patient'
    P = run_local.PRESETS['next']
    data = settings['data'] + (f"_s{P['slices_per_case']}" if P['slices_per_case'] != bifamnet.SLICES_PER_CASE else '')
    runs = Path(settings['runs']) / 'next'; out = runs / task / f'base_{model}_ck'
    if not (out / 'seed11.pt').exists():
        tov = dict(epochs=P['epochs'], batch=P['batch'], workers=settings.get('workers', 2), tta=1, patience=P['patience'])
        bifamnet.loss_fn = smooth_loss
        bifamnet.run_training(data, task, out, model=model, seeds=P['seeds'], train_over=tov, img_size=P['img_size'],
                              device=settings.get('device'), keep_checkpoints=True, channels='t1ce')
    a = pd.read_csv(out / 'predictions_seed11.csv'); b = pd.read_csv(runs / task / f'base_{model}' / 'predictions_seed11.csv')
    pa = a.set_index('path').filter(like='prob_').values.argmax(1); pb = b.set_index('path').loc[a.path].filter(like='prob_').values.argmax(1)
    print(f'[job {job}] {model}: checkpoint kept; slice predictions agree with the reported run on {100 * (pa == pb).mean():.1f}% of slices')

run_ck_job(JOB, dict(raw='/content/raw', data='/content/data_v2', runs=f'{PROJECT}/runs', device='cuda:0', workers=2))
