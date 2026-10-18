# Извлечение кэша на Kaggle GPU: копируем код из датасета forte-code и гоняем extract.py по forte-videos.
import os, pathlib, shutil, subprocess, sys
inp = pathlib.Path("/kaggle/input")
code = next(p.parent for p in inp.rglob("extract.py"))
vids = next(p.parent for p in inp.rglob("v26.mov"))
work = pathlib.Path("/kaggle/working/code"); shutil.copytree(code, work, dirs_exist_ok=True)
env = {**os.environ, "CACHE_DIR": "/kaggle/working/cache", "PYTHONUNBUFFERED": "1"}
r = subprocess.run([sys.executable, "extract.py", "--videos", str(vids)], cwd=work, env=env)
shutil.rmtree(work, ignore_errors=True)
print("exit", r.returncode)
