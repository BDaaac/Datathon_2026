"""Замер: 5 видео одновременно, отдельные процессы как в сервисе (torch 2 потока, декодер 1 поток). Время и пик RSS на процесс."""
import json, os, subprocess, sys, time, glob
import psutil
V = r"C:/Users/Димаш/Downloads/open (1)/open/videos"
IDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["14", "39", "4", "5", "35"]
env = dict(os.environ, DET_BACKEND="onnx", OPENCV_FFMPEG_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
code = ("import sys,time,json,torch;torch.set_num_threads(2);sys.path.insert(0,'.');from inference import Analyzer;"
        "t=time.time();a=Analyzer();lt=time.time()-t;r=a.run(sys.argv[1]);"
        "print(json.dumps({'load':round(lt,1),'proc':r.get('processing_sec'),'verdict':r.get('verdict')}))")
procs = []
t0 = time.time()
for i in IDS:
    p = next(x for x in glob.glob(V + "/*.mov") if x.endswith(f"-{i}.mov"))
    procs.append((i, subprocess.Popen([sys.executable, "-c", code, p], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True), time.time()))
peak = {i: 0 for i, _, _ in procs}; done = {}; sys_peak = 0
while len(done) < len(procs):
    for i, pr, ts in procs:
        if i in done: continue
        try:
            rss = psutil.Process(pr.pid).memory_info().rss
            for c in psutil.Process(pr.pid).children(recursive=True): rss += c.memory_info().rss
            peak[i] = max(peak[i], rss)
        except psutil.Error: pass
        if pr.poll() is not None: done[i] = (time.time() - ts, pr.stdout.read().strip())
    sys_peak = max(sys_peak, psutil.virtual_memory().used)
    time.sleep(0.5)
for i in IDS:
    wall, out = done[i]; print(f"видео {i}: стена {wall:.0f} с, пик RAM {peak[i] / 2**30:.2f} ГБ, {out[-160:]}")
print(f"ВСЕГО {time.time() - t0:.0f} с; RAM системы пик {sys_peak / 2**30:.1f} из {psutil.virtual_memory().total / 2**30:.1f} ГБ; CPU {psutil.cpu_count()}")
