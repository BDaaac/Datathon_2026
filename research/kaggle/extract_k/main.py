# Извлечение кэша на Kaggle: код и эталоны из forte-code (файлы лежат плоско), видео из forte-videos.
import os, pathlib, shutil, subprocess, sys
inp = pathlib.Path("/kaggle/input")
code = next(p.parent for p in inp.rglob("extract.py"))
vids = next(p.parent for p in inp.rglob("v26.mov"))
work = pathlib.Path("/kaggle/working/code"); (work / "assets").mkdir(parents=True, exist_ok=True)
for f in code.iterdir():
    dst = work / "assets" / f.name if (f.suffix == ".jpg" or f.name == "views.json") else work / f.name
    shutil.copy(f, dst)
print("assets:", sorted(p.name for p in (work / "assets").iterdir()), flush=True)
env = {**os.environ, "CACHE_DIR": "/kaggle/working/cache", "PYTHONUNBUFFERED": "1"}
r = subprocess.run([sys.executable, "extract.py", "--videos", str(vids)], cwd=work, env=env)
shutil.rmtree(work, ignore_errors=True)
print("exit", r.returncode)
