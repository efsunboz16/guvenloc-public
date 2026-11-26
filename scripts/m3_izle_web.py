"""Faz 4 tuzak hattı (F4-5) için tarayıcı paneli — yalnızca okur.

  python scripts/m3_izle_web.py            # http://127.0.0.1:8765 (yalnız bu makine)

Kaynaklar: logs/decoy.log (30_build_cache → 40_baselines feats → 60_m2_features), logs/decoy_sel.log
(65_m3_decoy.py --real), results/faz4/m3/decoy_sel{,_real}.json, nvidia-smi.
Önbellek günlüğü 25 karede bir yazar; aradaki ilerleme son hızdan tahmin edilir ("tahmini").
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG, SLOG = ROOT / "logs/decoy.log", ROOT / "logs/decoy_sel.log"
SIM, REAL = ROOT / "results/faz4/m3/decoy_sel.json", ROOT / "results/faz4/m3/decoy_sel_real.json"
CACHE = re.compile(r"kare (\d+)/(\d+) \| (\d+) dosya \| ([\d.]+) s/dosya")
FHEAD = re.compile(r"(train|calib): (\d+) önbellek dosyası")
FPROG = re.compile(r"^  (\d+)/(\d+)$", re.MULTILINE)
VPROG = re.compile(r"(train|calib): (\d+)/(\d+) \((\d+) ms/dosya")
VDONE = re.compile(r"(train|calib): (\d+) satır, (\d+) s")
FRAMES = 4561
HIST: list[tuple[float, float]] = []


def gpu() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=3).stdout
        u, m, mt, t, p = (float(x) for x in q.strip().split(", "))
        return {"util": u, "mem": m / 1024, "mem_total": mt / 1024, "temp": t, "power": p}
    except Exception:  # noqa: BLE001
        return {}


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def status() -> dict:
    txt, now = read(LOG), time.time()
    mtime = LOG.stat().st_mtime if LOG.exists() else now
    cache_txt, _, rest = txt.partition("bitti:")
    cache_done = bool(rest)
    c = {"done": 0, "total": FRAMES, "s_file": None, "finished": cache_done}
    for m in CACHE.finditer(cache_txt):
        c.update(done=int(m[1]), total=int(m[2]), s_file=float(m[4]))
    if cache_done:
        c["done"] = c["total"]
    elif c["s_file"]:
        c["est"] = min(c["total"] - 1, c["done"] + int((now - mtime) / (4 * c["s_file"])))
        c["eta_min"] = (c["total"] - c["est"]) * 4 * c["s_file"] / 60
    # RANSAC öznitelikleri (train, calib)
    feats = {s: {"done": 0, "total": None, "finished": False} for s in ("train", "calib")}
    for seg in re.split(r"(?=(?:train|calib): \d+ önbellek dosyası)", rest)[1:]:
        h = FHEAD.match(seg)
        st = feats[h[1]]
        st["total"] = int(h[2])
        for m in FPROG.finditer(seg):
            st["done"] = int(m[1])
        if f"feats_decoy_v1_{h[1]}_px5.csv" in seg:
            st.update(done=st["total"], finished=True)
    vol = {s: {"done": 0, "total": None, "ms": None, "finished": False} for s in ("train", "calib")}
    for m in VPROG.finditer(rest):
        vol[m[1]].update(done=int(m[2]), total=int(m[3]), ms=int(m[4]))
    for m in VDONE.finditer(rest):
        vol[m[1]].update(done=int(m[2]), total=int(m[2]), finished=True)
    chain_done = "DECOY_BITTI" in txt
    stxt = read(SLOG)
    real = json.loads(REAL.read_text(encoding="utf-8")) if REAL.exists() and "seçilen" in stxt else None
    sim = json.loads(SIM.read_text(encoding="utf-8")) if SIM.exists() else None
    done_units = c.get("est", c["done"])
    HIST.append((now, done_units))
    del HIST[:-300]
    errs = [ln for ln in (txt + stxt).splitlines() if "Traceback" in ln or re.search(r"\w+Error\b", ln)
            and "apszKnownConfigOptions" not in ln][-3:]
    return {"cache": c, "feats": feats, "vol": vol, "chain_done": chain_done, "sel_running": chain_done and real is None,
            "real": real, "sim": sim, "errors": errs, "gpu": gpu(), "hist": HIST[-150:], "now": now}


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GüvenLoc · Faz 4 tuzak hattı</title><style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3de;--acc:#2f6fd6;--ok:#2e8a4f;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#9a9992;--line:#33332f;--acc:#6ea0f0;--ok:#5cbf7d;--bad:#e5685a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 20px 60px}h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mute);margin:0 0 22px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.k{color:var(--mute);font-size:12.5px;text-transform:uppercase;letter-spacing:.04em}.v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}
.stage{display:grid;grid-template-columns:260px 1fr 170px;gap:14px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}
.stage:first-of-type{border-top:0}.bar{height:10px;background:var(--line);border-radius:99px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--acc);border-radius:99px;transition:width .8s}
.done .bar>i{background:var(--ok)}.wait{color:var(--mute)}.r{text-align:right;font-variant-numeric:tabular-nums;color:var(--mute);font-size:13.5px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;background:var(--mute)}.run .dot{background:var(--acc);animation:p 1.4s infinite}.done .dot{background:var(--ok)}
@keyframes p{50%{opacity:.25}}h2{font-size:16px;margin:26px 0 10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--mute);font-weight:550;font-size:12.5px}.verdict{font-size:18px;font-weight:650;margin:6px 0 10px}.ok{color:var(--ok)}.bad{color:var(--bad)}
.best{font-weight:650}svg{display:block;width:100%}.err{color:var(--bad);font-family:ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap}.note{color:var(--mute);font-size:13px}
</style></head><body><main>
<h1>Faz 4 · gerçek tuzak karo deneyi (F4-5)</h1><p class="sub" id="sub">bağlanıyor…</p>
<div class="grid"><div class="card"><div class="k">RoMa önbelleği</div><div class="v" id="pct">–</div><div class="bar" style="margin-top:8px"><i id="pbar" style="width:0"></i></div></div>
<div class="card"><div class="k">Kalan süre (önbellek)</div><div class="v" id="eta">–</div><div class="note" id="rate"></div></div>
<div class="card"><div class="k">GPU</div><div class="v" id="gu">–</div><div class="note" id="gm"></div></div></div>
<div class="card" style="margin-top:14px" id="stages"></div>
<div class="card" style="margin-top:14px"><div class="k">İşlenen kare (son dakikalar)</div><svg id="spark" viewBox="0 0 600 70" preserveAspectRatio="none" style="height:70px"></svg></div>
<h2>Seçim: tuzak normalizasyonu kuralı geçiyor mu?</h2>
<div class="card" id="sel"></div>
<h2>Bölge başına FAR-neg (eğitim bölgeleri, bölge-dışarıda; hedef α = 0,05)</h2>
<div class="card"><svg id="far" viewBox="0 0 640 230"></svg><div class="note" id="farnote"></div></div>
<div class="err" id="errs"></div>
<p class="note">Kural (sonuçtan önce): normalize skor, ham skora göre ort. |FAR − α| ve en büyük FAR'ı birlikte düşürmeli; AUROC kaybı ≤ 0,005. Sayfa 2 sn'de bir yenilenir.</p>
</main><script>
const f=(x,d=1)=>x==null?"–":Number(x).toLocaleString("tr-TR",{minimumFractionDigits:d,maximumFractionDigits:d});
const mins=m=>m==null?"–":m<1?"< 1 dk":m<60?`${Math.round(m)} dk`:`${Math.floor(m/60)} sa ${Math.round(m%60)} dk`;
const REG=["01","04","08","09","10"],NAMES={inlier:"inlier",inlier_norm:"inlier (normalize)",S:"S (ham)",S_norm:"S (normalize)"};
function row(label,d,total,fin,active,extra=""){const p=total?100*d/total:0,cls=fin?"done":active?"run":"wait";
 const r=fin?"bitti":active?`${f(d,0)} / ${total?f(total,0):"?"}${extra}`:"sırada";
 return `<div class="stage ${cls}"><div><span class="dot"></span>${label}</div><div class="bar"><i style="width:${p}%"></i></div><div class="r">${r}</div></div>`}
function table(res,sel){return `<table><tr><th>skor</th>${REG.map(r=>`<th>FAR ${r}</th>`).join("")}<th>ort. |FAR−α|</th><th>en büyük FAR</th><th>AUROC</th><th>doğru poz. kabul</th></tr>`+
 Object.entries(res).map(([k,v])=>`<tr class="${k===sel?"best":""}"><td>${NAMES[k]}</td>${REG.map(r=>`<td>${f(v.far[r],3)}</td>`).join("")}<td>${f(v.mean_abs_dev,3)}</td><td>${f(v.max_far,3)}</td><td>${f(v.auroc,4)}</td><td>${f(Object.values(v.accept_correct_pos).reduce((a,b)=>a+b)/5,3)}</td></tr>`).join("")+"</table>"}
function farChart(sim,real){const svg=document.getElementById("far"),W=640,H=230,L=40,B=200,T=14,series=[["S","S (ham)","#8a8a84"],["S_norm","S normalize · benzetim","#e0a84a"]];
 if(real)series.push(["S_norm","S normalize · gerçek tuzak","#2f6fd6"]);const src=[sim,sim,real];
 const max=Math.max(0.1,...series.flatMap((s,i)=>src[i]?REG.map(r=>src[i].results[s[0]].far[r]):[]))*1.1,y=v=>B-(B-T)*v/max,gw=(W-L-10)/REG.length,bw=gw/(series.length+1);
 let g=`<line x1="${L}" x2="${W-6}" y1="${y(0.05)}" y2="${y(0.05)}" stroke="var(--bad)" stroke-dasharray="4 4"/><text x="${W-8}" y="${y(0.05)-5}" text-anchor="end" font-size="11" fill="var(--bad)">α = 0,05</text>`;
 for(const t of [0,0.1,0.2,0.3].filter(t=>t<=max))g+=`<text x="${L-6}" y="${y(t)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${f(t,1)}</text><line x1="${L}" x2="${W-6}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/>`;
 REG.forEach((r,j)=>{series.forEach((s,i)=>{if(!src[i])return;const v=src[i].results[s[0]].far[r],x=L+j*gw+bw*(i+0.5);
  g+=`<rect x="${x}" y="${y(v)}" width="${bw*0.9}" height="${B-y(v)}" fill="${s[2]}" rx="2"><title>${s[1]} · ${r}: ${f(v,3)}</title></rect>`});
  g+=`<text x="${L+j*gw+gw/2}" y="${B+16}" text-anchor="middle" font-size="12" fill="var(--mute)">bölge ${r}</text>`});
 g+=series.map((s,i)=>src[i]?`<rect x="${L+i*190}" y="${H-12}" width="10" height="10" fill="${s[2]}"/><text x="${L+i*190+14}" y="${H-3}" font-size="12" fill="var(--ink)">${s[1]}</text>`:"").join("");
 svg.innerHTML=g;document.getElementById("farnote").textContent=real?"":"Gerçek tuzak sonucu gelince mavi çubuklar eklenecek."}
function spark(h){const svg=document.getElementById("spark");if(h.length<2){svg.innerHTML="";return}const t0=h[0][0],t1=h.at(-1)[0]||t0+1,v0=h[0][1],v1=Math.max(h.at(-1)[1],v0+1);
 const pts=h.map(([t,v])=>`${600*(t-t0)/Math.max(1,t1-t0)},${66-60*(v-v0)/(v1-v0)}`).join(" ");svg.innerHTML=`<polyline points="${pts}" fill="none" stroke="var(--acc)" stroke-width="2"/>`}
async function tick(){let st;try{st=await (await fetch("status")).json()}catch(e){document.getElementById("sub").textContent="bağlantı yok";return}
 const c=st.cache,d=c.est??c.done;document.getElementById("pct").textContent=`%${f(100*d/c.total)}`;document.getElementById("pbar").style.width=`${100*d/c.total}%`;
 document.getElementById("eta").textContent=c.finished?"bitti":mins(c.eta_min);document.getElementById("rate").textContent=c.s_file?`${f(c.s_file,2)} s/dosya · kare başına 4 dönme`:"";
 const g=st.gpu;document.getElementById("gu").textContent=g.util!=null?`%${f(g.util,0)}`:"–";document.getElementById("gm").textContent=g.util!=null?`${f(g.mem)} / ${f(g.mem_total)} GB · ${f(g.temp,0)} °C · ${f(g.power,0)} W`:"";
 const fe=st.feats,vo=st.vol,cfin=c.finished;
 const stage=!cfin?"RoMa önbelleği":!(fe.train.finished&&fe.calib.finished)?"RANSAC öznitelikleri":!(vo.train.finished&&vo.calib.finished)?"hacim öznitelikleri":st.real?"tamamlandı":"seçim hesaplanıyor";
 document.getElementById("sub").textContent=`Aşama: ${stage} · son güncelleme ${new Date(st.now*1000).toLocaleTimeString("tr-TR")}`;
 document.getElementById("stages").innerHTML=row("RoMa: kare × tuzak karo (4 dönme)",d,c.total,cfin,!cfin,c.est!=null?" (tahmini)":"")+
  row("RANSAC öznitelikleri · train",fe.train.done,fe.train.total,fe.train.finished,cfin&&!fe.train.finished)+
  row("RANSAC öznitelikleri · calib",fe.calib.done,fe.calib.total,fe.calib.finished,fe.train.finished&&!fe.calib.finished)+
  row("Hacim öznitelikleri · train",vo.train.done,vo.train.total,vo.train.finished,fe.calib.finished&&!vo.train.finished)+
  row("Hacim öznitelikleri · calib",vo.calib.done,vo.calib.total,vo.calib.finished,vo.train.finished&&!vo.calib.finished)+
  `<div class="stage ${st.real?"done":st.sel_running?"run":"wait"}"><div><span class="dot"></span>Seçim (65_m3_decoy.py --real)</div><div class="bar"><i style="width:${st.real?100:0}%"></i></div><div class="r">${st.real?"bitti":st.sel_running?"çalışıyor":"sırada"}</div></div>`;
 let s="";if(st.real){const ok=st.real.selected!=="S";s+=`<div class="verdict ${ok?"ok":"bad"}">${ok?"Kural geçildi → seçilen: "+NAMES[st.real.selected]:"Kural geçilmedi → ham S kalır"}</div><div class="note">Gerçek tuzak (aynı kare)</div>${table(st.real.results,st.real.selected)}`}
 else s+=`<div class="wait">Gerçek tuzak sonucu, hat bitince burada görünecek.</div>`;
 if(st.sim)s+=`<h2 style="font-size:14px">Karşılaştırma: benzetim (komşu kare, F4-4) — kural geçilmedi</h2>${table(st.sim.results,null)}`;
 document.getElementById("sel").innerHTML=s;if(st.sim)farChart(st.sim,st.real);spark(st.hist);document.getElementById("errs").textContent=st.errors.join("\n")}
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
    print("http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
