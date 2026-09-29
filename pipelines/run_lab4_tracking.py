import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
print('[INFO] =========================================')
print('[INFO] Starting Lab 4: MLflow Experiment Tracking')
print('[INFO] =========================================')
for script in ('src/preprocess.py','src/train_mlflow.py'):
    print(f'[INFO] ---> Executing {script}...')
    r=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT)
    if r.returncode: raise SystemExit(f'[ERROR] Pipeline halted. {script} failed.')
print('[SUCCESS] Lab 4 Pipeline fully executed!')
