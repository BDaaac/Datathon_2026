import glob, os, re, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import Analyzer
V = r"C:/Users/Димаш/Downloads/open (1)/open/videos"
LOCK = {"08", "10", "11", "12", "13", "14", "15"}
a = Analyzer()
for p in sorted(glob.glob(V + "/*.mov")):
    fid = "%02d" % int(re.search(r"-(\d+)\.mov$", p).group(1))
    if fid in ("47", "48"): continue
    r = a.run(p)
    for v in r["violations"]: v.pop("evidence_frames", None)
    json.dump(r, open(f"runs/service_reg_ov/{fid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(fid, r["verdict"], r["processing_sec"], flush=True)
