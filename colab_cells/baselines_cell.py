# ===== Paper 2: standard models under BOTH protocols (jobs 17-19). Paste below the Step 6 cell and run. =====
import sys, json
from pathlib import Path
bifamnet.TIMM_NAMES['efficientnet_b0'] = 'efficientnet_b0'
if 'efficientnet_b0' not in bifamnet.BASELINES: bifamnet.BASELINES.append('efficientnet_b0')
BASE_JOBS = {17: 'resnet50', 18: 'densenet201', 19: 'efficientnet_b0'}      # each job: one model x two protocols
BASE_TASKS = ['threeclass_image', 'threeclass_patient']

def run_base_job(job, settings):
    model = BASE_JOBS[job]
    P = run_local.PRESETS['next']
    data = settings['data'] + (f"_s{P['slices_per_case']}" if P['slices_per_case'] != bifamnet.SLICES_PER_CASE else '')
    runs = Path(settings['runs']) / 'next'
    missing = [t for t in BASE_TASKS if not (Path(data) / 'splits' / f'{t}.json').exists()]
    if missing:                                    # prepare the data once (no training here)
        run_local.SETTINGS.update(settings, backbone=CNX, run_main=False, run_figures=False, run_baselines=False,
                                  run_ablation=False, run_fusion=False, run_analysis=False)
        sys.argv = ['run_local.py', '--tasks', *missing, '--preset', 'next', '--skip-doctor']
        run_local.main()
    tov = dict(epochs=P['epochs'], batch=P['batch'], workers=settings.get('workers', 2), tta=1, patience=P['patience'])
    for task in BASE_TASKS:
        out = runs / task / f'base_{model}'
        if (out / 'summary.json').exists():
            print(f'[job {job}] {task}/base_{model} already finished - skipping'); continue
        print(f'\n[job {job}] ===== {task} / {model} (standard classifier, label smoothing on) =====')
        bifamnet.loss_fn = smooth_loss
        bifamnet.run_training(data, task, out, model=model, seeds=P['seeds'], train_over=tov, img_size=P['img_size'],
                              device=settings.get('device'), keep_checkpoints=False, channels='t1ce')
    for task in BASE_TASKS:
        s = runs / task / f'base_{model}' / 'summary.json'
        if s.exists():
            m = json.load(open(s))['mean']
            print(f"{task:20s} {model:16s} accuracy {100 * m['accuracy']:.2f}   macro F1 {100 * m['macro_f1']:.2f}")
    print(f'\n[job {job}] finished.')

run_base_job(JOB, dict(raw='/content/raw', data='/content/data_v2', runs=f'{PROJECT}/runs', device='cuda:0', workers=2))
