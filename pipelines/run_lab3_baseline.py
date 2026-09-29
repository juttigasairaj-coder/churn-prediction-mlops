import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for script in ('src/preprocess.py','src/train.py','src/evaluate.py'):
    print(f'[INFO] ---> Executing {script}...')
    r=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT)
    if r.returncode: raise SystemExit(r.returncode)
print('[SUCCESS] Lab 3 Baseline Pipeline completed!')
