"""Faz 3 hacim çıkarımı + H2 kuru denemesi için tarayıcı paneli — yalnızca okur.

  python scripts/m2_izle_web.py            # http://127.0.0.1:8765 (yalnız bu makine)
  python scripts/m2_izle_web.py --faz 3d   # 3D: kırpıntılar (62) + inceltici eğitimi (63)

Kaynaklar: logs/m2_features.log (60_m2_features.py), logs/h2.log (61_m2_score.py),
results/faz3/m2/h2_calib.json, nvidia-smi. Günlük her 2.000 dosyada bir satır yazar; aradaki ilerleme
son hızdan tahmin edilir (panelde "tahmini" olarak işaretli).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOG, HLOG = ROOT / "logs/m2_features.log", ROOT / "logs/h2.log"
H2J = ROOT / "results/faz3/m2/h2_calib.json"
PROG = re.compile(r"(\w+): (\d+)/(\d+) \((\d+) ms/dosya")
DONE = re.compile(r"(\w+): (\d+) satır, (\d+) s; pozitif (?:\(tüm dönmeler\) )?tepe hatası medyan ([\d.]+) m, ≤ 25 m %([\d.]+)")
EPOCH = re.compile(r"tohum (\d+) epoch (\d+)/(\d+): kayıp ([\d.]+), val AUROC S25 ([\d.]+), Snull ([\d.]+), "
                   r"val başarı tavanı ([\d.]+) \[(\d+) s\]")
MODE = {"faz": "3bc", "mark": "M2F_BITTI"}
RLOG, RJ = ROOT / "logs/m2_ref.log", ROOT / "results/faz3/m2/refiner_calib.json"
TOTALS = {"calib": 12392, "train": 41400}
HIST: list[tuple[float, float]] = []


def gpu() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=3).stdout
        u, m, mt, t, p = (float(x) for x in q.strip().split(", "))
        return {"util": u, "mem": m / 1024, "mem_total": mt / 1024, "temp": t, "power": p}
    except Exception:  # noqa: BLE001
        return {}


def status() -> dict:
    txt = FLOG.read_text(encoding="utf-8", errors="replace") if FLOG.exists() else ""
    mtime = FLOG.stat().st_mtime if FLOG.exists() else time.time()
    stages = {s: {"done": 0, "total": TOTALS[s], "ms": None, "finished": False, "summary": None} for s in ("calib", "train")}
    for m in PROG.finditer(txt):
        st = stages[m[1]]
        st.update(done=int(m[2]), total=int(m[3]), ms=int(m[4]))
    for m in DONE.finditer(txt):
        st = stages[m[1]]
        st.update(done=int(m[2]), total=int(m[2]), finished=True,
                  summary={"seconds": int(m[3]), "median_err_m": float(m[4]), "le25_pct": float(m[5])})
    feat_done = MODE["mark"] in txt
    now = time.time()
    active = next((s for s in ("calib", "train") if not stages[s]["finished"]), None)
    if active and not feat_done:
        st = stages[active]
        ms = st["ms"] or 47
        # son günlük satırından bu yana geçen süreyle tahmin (bir önceki aşama bittiyse oradan)
        est = st["done"] + int((now - mtime) * 1000 / ms)
        st["est"] = min(est, st["total"] - 1)
        st["eta_min"] = (st["total"] - st["est"]) * ms / 60000
    total_files = sum(s["total"] for s in stages.values())
    done_files = sum(s.get("est", s["done"]) for s in stages.values())
    HIST.append((now, done_files))
    del HIST[:-300]
    htxt = HLOG.read_text(encoding="utf-8", errors="replace") if HLOG.exists() else ""
    h2 = json.loads(H2J.read_text(encoding="utf-8")) if H2J.exists() and "H2_BITTI" in htxt else None
    remaining = sum(((s["total"] - s.get("est", s["done"])) * (s["ms"] or 47)) / 60000
                    for s in stages.values() if not s["finished"])
    ref = None
    if MODE["faz"] == "3d":
        rtxt = RLOG.read_text(encoding="utf-8", errors="replace") if RLOG.exists() else ""
        ep = [{"seed": int(m[1]), "epoch": int(m[2]), "epochs": int(m[3]), "loss": float(m[4]), "s25": float(m[5]),
               "snull": float(m[6]), "ceil": float(m[7]), "t": int(m[8])} for m in EPOCH.finditer(rtxt)]
        done_r = "M2R_BITTI" in rtxt
        ref = {"started": bool(rtxt), "epochs": ep, "done": done_r,
               "result": json.loads(RJ.read_text(encoding="utf-8")) if done_r and RJ.exists() else None}
        htxt = rtxt
        h2 = None
    return {"mode": MODE["faz"], "ref": ref, "stages": stages, "active": active, "feat_done": feat_done, "pct": 100 * done_files / total_files,
            "remaining_min": remaining, "h2_running": feat_done and h2 is None, "h2": h2,
            "h2_md": (ROOT / "results/faz3/m2/h2_calib.md").read_text(encoding="utf-8") if h2 else None,
            "errors": [ln for ln in (txt + htxt).splitlines() if "Error" in ln or "Traceback" in ln][-3:],
            "gpu": gpu(), "hist": HIST[-120:], "now": now}


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GüvenLoc · Faz 3 izleme</title><style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3de;--acc:#2f6fd6;--ok:#2e8a4f;--bad:#c0392b;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#9a9992;--line:#33332f;--acc:#6ea0f0;--ok:#5cbf7d;--bad:#e5685a;--warn:#e0a84a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 20px 60px}h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mute);margin:0 0 22px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.k{color:var(--mute);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}
.stage{display:grid;grid-template-columns:150px 1fr 150px;gap:14px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}
.stage:first-of-type{border-top:0}.bar{height:10px;background:var(--line);border-radius:99px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--acc);border-radius:99px;transition:width .8s}
.done .bar>i{background:var(--ok)}.wait{color:var(--mute)}.r{text-align:right;font-variant-numeric:tabular-nums;color:var(--mute);font-size:13.5px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:var(--mute)}.run .dot{background:var(--acc);animation:p 1.4s infinite}.done .dot{background:var(--ok)}
@keyframes p{50%{opacity:.25}}h2{font-size:16px;margin:26px 0 10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
th{color:var(--mute);font-weight:550;font-size:12.5px}.verdict{font-size:18px;font-weight:650;margin:6px 0 10px}.ok{color:var(--ok)}.bad{color:var(--bad)}
svg{display:block;width:100%;height:70px}.err{color:var(--bad);font-family:ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap}.note{color:var(--mute);font-size:13px}
</style></head><body><main>
<h1>Faz 3 · oylama hacmi ve H2</h1><p class="sub" id="sub">bağlanıyor…</p>
<div class="grid"><div class="card"><div class="k">Toplam ilerleme</div><div class="v" id="pct">–</div><div class="bar" style="margin-top:8px"><i id="pbar" style="width:0"></i></div></div>
<div class="card"><div class="k">Kalan süre (çıkarım)</div><div class="v" id="eta">–</div><div class="note" id="rate"></div></div>
<div class="card"><div class="k">GPU</div><div class="v" id="gu">–</div><div class="note" id="gm"></div></div></div>
<div class="card" style="margin-top:14px" id="stages"></div>
<div class="card" style="margin-top:14px"><div class="k">İşlenen dosya (son dakikalar)</div><svg id="spark" viewBox="0 0 600 70" preserveAspectRatio="none"></svg></div>
<h2 id="h2t">H2 kuru denemesi (calib)</h2><div class="card" id="h2"><span class="wait">Çıkarım bitince otomatik çalışacak.</span></div>
<div class="err" id="errs"></div>
<p class="note">Günlük 2.000 dosyada bir yazılıyor; aradaki ilerleme son hızdan tahmin ediliyor. Sayfa 2 sn'de bir yenilenir.</p>
</main><script>
const f=(x,d=1)=>x==null?"–":Number(x).toLocaleString("tr-TR",{minimumFractionDigits:d,maximumFractionDigits:d});
const mins=m=>m==null?"–":m<1?"< 1 dk":m<60?`${Math.round(m)} dk`:`${Math.floor(m/60)} sa ${Math.round(m%60)} dk`;
function stageRow(name,label,s,active){const d=s.est??s.done,p=100*d/s.total,cls=s.finished?"done":active?"run":"wait";
 const right=s.finished?`bitti · ${mins(s.summary.seconds/60)}`:active?`${f(d,0)} / ${f(s.total,0)}${s.est!=null?" (tahmini)":""}`:"sırada";
 const extra=s.finished?`<div class="note">pozitif tepe hatası medyan ${f(s.summary.median_err_m)} m · ≤ 25 m %${f(s.summary.le25_pct)}</div>`:"";
 return `<div class="stage ${cls}"><div><span class="dot"></span>${label}${extra}</div><div class="bar"><i style="width:${p}%"></i></div><div class="r">${right}</div></div>`}
function h2(st){if(!st.h2)return st.h2_running?'<span class="wait">H2 skoru hesaplanıyor…</span>':'<span class="wait">Çıkarım bitince otomatik çalışacak.</span>';
 const h=st.h2,r=h.rows,ok=h.h2.H2,names={"4a":"M1-C 1 çağrı · RANSAC + inlier","5a":"M1-C 1 çağrı · hacim + S","2":"4 yön · RANSAC + inlier","2v":"4 yön · hacim + S"};
 let rows=Object.keys(r).map(k=>`<tr><td>${k}</td><td>${names[k]??"hacim, tek metrik "+k.slice(3)}</td><td>${f(r[k]["success@25"],3)}</td><td>${f(r[k].false_accept,3)}</td><td>${f(r[k].far_neg,3)}</td><td>${f(r[k].far_loc,3)}</td><td>${f(r[k].auroc,3)} [${f(r[k].auroc_ci[0],3)}, ${f(r[k].auroc_ci[1],3)}]</td></tr>`).join("");
 const m=h.mcnemar_5a_4a,d=h.auroc_diff;
 return `<div class="verdict ${ok?"ok":"bad"}">H2 ${ok?"sağlandı":"sağlanmadı"} (calib)</div>
 <div class="note">(i) yanlış kabul 5a &lt; 4a, McNemar p = ${m.p.toExponential(2)} (yalnız 4a: ${m.a_only}, yalnız 5a: ${m.b_only}) → <b class="${h.h2.i?"ok":"bad"}">${h.h2.i?"✓":"✗"}</b> ·
 (ii) AUROC farkı ${f(d[0],3)} [${f(d[1],3)}, ${f(d[2],3)}] → <b class="${h.h2.ii?"ok":"bad"}">${h.h2.ii?"✓":"✗"}</b></div>
 <table style="margin-top:12px"><tr><th>satır</th><th>yöntem</th><th>başarı@25</th><th>yanlış kabul</th><th>FAR-neg</th><th>FAR-loc</th><th>AUROC [GA]</th></tr>${rows}</table>`}
function curve(ep){if(!ep.length)return"";const W=600,H=150,n=ep.at(-1).epochs,ys=ep.flatMap(e=>[e.s25,e.snull]),lo=Math.min(...ys)-0.005,hi=Math.max(...ys)+0.005;
 const X=e=>W*(e.epoch-0.5)/n,Y=v=>H-6-(H-12)*(v-lo)/(hi-lo),line=(k,c)=>`<polyline points="${ep.map(e=>`${X(e)},${Y(e[k])}`).join(" ")}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
 return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:150px">${line("s25","var(--acc)")}${line("snull","var(--warn)")}</svg>
 <div class="note"><span style="color:var(--acc)">■</span> val AUROC S25 · <span style="color:var(--warn)">■</span> val AUROC Snull · aralık ${f(lo,3)}–${f(hi,3)}</div>`}
function refiner(st){const r=st.ref;if(!r||!r.started)return'<span class="wait">Kırpıntılar bitince eğitim otomatik başlayacak.</span>';
 const ep=r.epochs,last=ep.at(-1);let h="";
 if(last){const p=100*last.epoch/last.epochs;h+=`<div class="stage ${r.done?"done":"run"}" style="grid-template-columns:150px 1fr 230px"><div><span class="dot"></span>Eğitim (tohum ${last.seed})</div><div class="bar"><i style="width:${p}%"></i></div><div class="r">epoch ${last.epoch}/${last.epochs} · kayıp ${f(last.loss,4)} · ${mins(last.t/60)}</div></div>`;
  h+=`<div class="note" style="margin:4px 0 8px">son val 04: AUROC S25 ${f(last.s25,4)}, Snull ${f(last.snull,4)}, başarı tavanı ${f(last.ceil,3)}</div>`+curve(ep)}
 else h+='<span class="wait">Veri yükleniyor…</span>';
 if(r.result){for(const [seed,res] of Object.entries(r.result)){const s=res.summary,e=s.equal_accept_rate,d=s.auroc_diff_5r_4a,ds=s.auroc_diff_same_label;
  const rows=["4a","5a_v0","5r"].map(k=>`<tr><td>${k}</td><td>${{"4a":"RANSAC + inlier","5a_v0":"v0 hacim (destek)","5r":"inceltici ("+res.main_score+")"}[k]}</td><td>${f(s[k].auroc,4)}</td><td>${f(s[k].auroc_same_label_ransac,4)}</td><td>${f(s[k]["ceiling@25"],3)}</td></tr>`).join("");
  const better=d[1]>0;h+=`<h2 style="margin-top:18px">Calib sonucu (tohum ${seed})</h2><div class="verdict ${better?"ok":"bad"}">AUROC farkı (inceltici − RANSAC) ${f(d[0],4)} [${f(d[1],4)}, ${f(d[2],4)}]</div>
  <div class="note">aynı etiketle ${f(ds[0],4)} [${f(ds[1],4)}, ${f(ds[2],4)}] · eşit kabul oranında (%${f(100*e.accept_rate)}) yanlış kabul 4a ${f(e.fa_4a,3)} / inceltici ${f(e.fa_5r,3)}, başarı@25 ${f(e.succ_4a,3)} / ${f(e.succ_5r,3)}, McNemar p = ${e.mcnemar.p.toExponential(2)}</div>
  <table style="margin-top:10px"><tr><th>satır</th><th>yöntem</th><th>AUROC (kendi etiketi)</th><th>AUROC (RANSAC etiketi)</th><th>başarı tavanı</th></tr>${rows}</table>`}}
 return h}
function spark(hist){const s=document.getElementById("spark");if(hist.length<2){s.innerHTML="";return}
 const t0=hist[0][0],t1=hist.at(-1)[0],y0=hist[0][1],y1=Math.max(hist.at(-1)[1],y0+1);
 const pts=hist.map(([t,y])=>`${600*(t-t0)/Math.max(t1-t0,1)},${66-60*(y-y0)/(y1-y0)}`).join(" ");
 s.innerHTML=`<polyline points="${pts}" fill="none" stroke="var(--acc)" stroke-width="2" vector-effect="non-scaling-stroke"/>`}
async function tick(){try{const st=await (await fetch("/status")).json();
 document.getElementById("sub").textContent=st.feat_done?(st.h2?"Tamamlandı.":"Çıkarım bitti, H2 hesaplanıyor."):`Çalışıyor: ${st.active==="calib"?"calib":"train"} ${st.mode==="3d"?"kırpıntıları":"çıkarımı"} · son güncelleme ${new Date(st.now*1000).toLocaleTimeString("tr-TR")}`;
 document.getElementById("pct").textContent=`%${f(st.pct)}`;document.getElementById("pbar").style.width=`${st.pct}%`;
 document.getElementById("eta").textContent=st.feat_done?"bitti":mins(st.remaining_min);
 const a=st.active&&st.stages[st.active];document.getElementById("rate").textContent=a&&a.ms?`${a.ms} ms/dosya · ${f(1000/a.ms,0)} dosya/sn`:"";
 const g=st.gpu;document.getElementById("gu").textContent=g.util!=null?`%${f(g.util,0)}`:"–";
 document.getElementById("gm").textContent=g.util!=null?`${f(g.mem)} / ${f(g.mem_total,0)} GB · ${f(g.temp,0)}°C · ${f(g.power,0)} W`:"";
 document.getElementById("stages").innerHTML=stageRow("calib","Calib çıkarımı (05, 11)",st.stages.calib,st.active==="calib")+stageRow("train","Train çıkarımı (01, 04, 08, 09, 10)",st.stages.train,st.active==="train")+
  (st.mode==="3d"?"":`<div class="stage ${st.h2?"done":st.h2_running?"run":"wait"}"><div><span class="dot"></span>H2 skoru ve kuru deneme</div><div class="bar"><i style="width:${st.h2?100:0}%"></i></div><div class="r">${st.h2?"bitti":st.h2_running?"çalışıyor":"sırada"}</div></div>`);
 document.getElementById("h2").innerHTML=st.mode==="3d"?refiner(st):h2(st);
 if(st.mode==="3d"){document.getElementById("h2t").textContent="Öğrenilmiş inceltici (3D)";document.querySelector("h1").textContent="Faz 3D · öğrenilmiş inceltici"}spark(st.hist);document.getElementById("errs").textContent=st.errors.join("\n");
}catch(e){document.getElementById("sub").textContent="Panel sunucusuna ulaşılamıyor."}}
tick();setInterval(tick,2000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body, ctype = ((json.dumps(status()).encode(), "application/json") if self.path.startswith("/status")
                       else (PAGE.encode(), "text/html; charset=utf-8"))
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", choices=["3bc", "3d"], default="3bc")
    if ap.parse_args().faz == "3d":
        MODE.update(faz="3d", mark="M2C_BITTI")
        FLOG = ROOT / "logs/m2_crops.log"
    print("Panel: http://127.0.0.1:8765  (Ctrl+C ile kapat)")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
