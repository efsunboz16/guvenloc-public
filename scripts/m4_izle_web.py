"""Faz 5 (Modül 4, zamansal füzyon) için tarayıcı paneli — yalnızca okur.

  python scripts/m4_izle_web.py            # http://127.0.0.1:8765 (yalnız bu makine)

Kaynaklar: results/faz5/m4/ (seq_summary.json, meas_*.csv, tune.json, h4_dry.json, traj.csv, gamma.json, *.png),
docs/faz5_m4.md (kararlar), configs/h4.yaml (ölçütler), logs/m4_*.log (süren işler), nvidia-smi, tmux.
Sayfa 5 saniyede bir yenilenir; dosya değişince yörünge verisi yeniden okunur.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results/faz5/m4"
DOC = ROOT / "docs/faz5_m4.md"
H4 = ROOT / "configs/h4.yaml"
_TRAJ: dict = {"mtime": None, "rows": {}}


def gpu() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True,
                           timeout=3).stdout
        u, m, mt, t = (float(x) for x in q.strip().split(", "))
        return {"util": u, "mem": m / 1024, "mem_total": mt / 1024, "temp": t}
    except Exception:  # noqa: BLE001
        return {}


def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def steps() -> list[dict]:
    meas = sorted(R5.glob("meas_*.npz"))
    tune = jload(R5 / "tune.json")
    dry = jload(R5 / "h4_dry.json")
    return [
        {"id": "5A", "name": "Uçuş dizileri, drone↔drone hareket", "done": (R5 / "seq_summary.json").exists(),
         "detail": "70_m4_sequences.py"},
        {"id": "5B·1", "name": "Kare ölçümleri P(x, y)", "done": len(meas) >= 7,
         "detail": f"{len(meas)}/7 bölge (71_m4_measure.py)"},
        {"id": "5B·2", "name": "Filtre ayarı (eğitim bölgeleri)", "done": tune is not None,
         "detail": f"{len(tune['grid'])} ayar denendi" if tune else "72 --tune"},
        {"id": "5B·3", "name": "H4 kuru denemesi (calib)", "done": dry is not None, "detail": "72 --dry"},
        {"id": "5C", "name": "γ eşikli uydu eşleştirmesi", "done": (R5 / "gamma.json").exists(),
         "detail": "sırada"},
        {"id": "5D", "name": "Görseller (yörünge, bulut)", "done": any(R5.glob("*.png")), "detail": "sırada"},
        {"id": "F6", "name": "Test (02, 03, 06), tek sefer", "done": False, "detail": "Faz 6"},
    ]


def decisions() -> list[dict]:
    try:
        t = DOC.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for m in re.finditer(r"^\| (F5-\d+) \| (.+?) \| (.+?) \|$", t, re.MULTILINE):
        title = re.sub(r"\*\*(.+?)\*\*.*", r"\1", m.group(2)).strip()
        out.append({"id": m.group(1), "title": title[:160]})
    return out


def criteria() -> dict:
    c = {"median_reduction_min": 0.2, "wilcoxon_p_max": 0.05, "coverage_min": 0.88}
    try:
        t = H4.read_text(encoding="utf-8")
        for k in c:
            m = re.search(rf"{k}:\s*([\d.]+)", t)
            if m:
                c[k] = float(m.group(1))
    except OSError:
        pass
    return c


def jobs() -> list[dict]:
    out = []
    try:
        ls = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], check=False, capture_output=True, text=True,
                            timeout=3).stdout.split()
    except Exception:  # noqa: BLE001
        ls = []
    for p in sorted((ROOT / "logs").glob("m4_*.log")):
        lines = [x for x in p.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines() if x.strip()]
        out.append({"log": p.name, "tail": lines[-4:], "err": any("Traceback" in x for x in lines)})
    return [{"sessions": [s for s in ls if s.startswith("m4")]}] + out


def traj_rows() -> dict:
    p = R5 / "traj.csv"
    if not p.exists():
        return {}
    mt = p.stat().st_mtime
    if _TRAJ["mtime"] != mt:
        rows: dict = {}
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["region"], []).append(
                    [int(r["num"]), r["accept"] == "True", r["started"] == "True", r["reset"] == "True",
                     *(round(float(r[k]), 1) if r[k] not in ("", "nan") else None
                       for k in ("err6", "err7", "e6", "n6", "e7", "n7", "ge", "gn"))])
        _TRAJ.update(mtime=mt, rows=rows)
    return _TRAJ["rows"]


def status() -> dict:
    tune = jload(R5 / "tune.json")
    top = sorted(tune["grid"], key=lambda x: x["med7"])[:8] if tune else []
    seq = jload(R5 / "seq_summary.json")
    split = {}
    for f in R5.glob("meas_*.csv"):
        with f.open(encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None)
            if r:
                split[f.stem[5:]] = "calib" if r["region"] in ("05", "11") else "eğitim"
    regions = {}
    for r, rows in traj_rows().items():
        e6 = sorted(x[4] for x in rows if x[4] is not None)
        e7 = sorted(x[5] for x in rows if x[5] is not None)
        regions[r] = {"n": len(rows), "med6": e6[len(e6) // 2] if e6 else None, "med7": e7[len(e7) // 2] if e7 else None,
                      "resets": sum(x[3] for x in rows), "accept": sum(x[1] for x in rows) / max(len(rows), 1),
                      "split": split.get(r, "")}
    return {"steps": steps(), "criteria": criteria(), "dry": jload(R5 / "h4_dry.json"), "tune_top": top,
            "tune_sel": tune["selected"] if tune else None, "seq": seq, "regions": regions,
            "decisions": decisions(), "jobs": jobs(), "gpu": gpu()}


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GüvenLoc · Faz 5 izleme</title><style>
:root{color-scheme:light;--bg:#f6f5f2;--card:#fff;--ink:#191919;--mute:#666;--line:#e4e2dc;--grid:#ecebe6;
--s1:#257de0;--s2:#f47230;--gt:#8a8a84;--good:#008a17;--bad:#d93a3b;--warn:#b07800;--acc:#257de0}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#121212;--card:#191919;--ink:#efefef;--mute:#9c9c9c;
--line:#2c2c2c;--grid:#262626;--s1:#338ced;--s2:#e76624;--gt:#8f8f8a;--good:#0aac23;--bad:#eb6263;--warn:#d28e05;--acc:#338ced}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:22px 20px 60px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:26px 0 10px}
.sub{color:var(--mute);font-size:13px}.grid{display:grid;gap:12px}.g4{grid-template-columns:repeat(4,1fr)}.g2{grid-template-columns:1.6fr 1fr}
@media(max-width:900px){.g4,.g2{grid-template-columns:1fr 1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.k{color:var(--mute);font-size:12px;text-transform:none;letter-spacing:.01em}.v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.ok{color:var(--good)}.no{color:var(--bad)}.pend{color:var(--mute)}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.steps{display:flex;gap:8px;flex-wrap:wrap}.step{flex:1 1 140px;border:1px solid var(--line);border-radius:9px;padding:9px 11px;background:var(--card)}
.step b{font-size:12px;color:var(--mute);font-weight:600}.step div{font-size:13px;margin-top:2px}.step .d{color:var(--mute);font-size:12px}
.step.done{border-color:color-mix(in srgb,var(--good) 45%,var(--line))}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--mute);font-weight:600;font-size:12px}td.n,th.n{text-align:right}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.seg button{border:0;background:transparent;color:var(--ink);padding:6px 11px;font:inherit;font-size:13px;cursor:pointer}
.seg button.on{background:var(--acc);color:#fff}.seg button:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
.ctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.ctrl label{font-size:13px;color:var(--mute);display:flex;gap:8px;align-items:center}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--mute);margin:6px 0 0}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
.legend i.dot{width:8px;height:8px;border-radius:50%}
svg{display:block;width:100%}.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:12.5px;box-shadow:0 4px 14px rgb(0 0 0/.12);display:none;z-index:9;font-variant-numeric:tabular-nums}
.note{color:var(--mute);font-size:12.5px;margin-top:6px}pre{margin:6px 0 0;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;color:var(--mute)}
.bar{height:6px;border-radius:3px;background:var(--grid);overflow:hidden;margin-top:6px}.bar>span{display:block;height:100%;background:var(--acc)}
</style></head><body><main>
<h1>GüvenLoc · Faz 5 — zamansal füzyon (parçacık filtresi)</h1>
<div class="sub" id="upd">yükleniyor…</div>
<h2>Adımlar</h2><div class="steps" id="steps"></div>
<h2>H4 — calib kuru denemesi (test değil; ölçütler <code>configs/h4.yaml</code>)</h2>
<div class="grid g4" id="h4"></div><div class="note" id="h4note"></div>
<h2>Yörünge</h2>
<div class="card"><div class="ctrl"><div class="seg" id="regsel" role="tablist"></div>
<label>Kareler <input type="range" id="r0" min="0" max="100" value="0"> – <input type="range" id="r1" min="0" max="100" value="100"> <span id="rlab"></span></label></div>
<div class="grid g2"><div><svg id="map" viewBox="0 0 640 520" role="img" aria-label="Yörünge haritası"></svg>
<div class="legend"><span><i style="background:var(--gt)"></i>GT (GPS)</span><span><i class="dot" style="background:var(--s2)"></i>Tek kare (satır 6)</span><span><i style="background:var(--s1)"></i>Filtre (satır 7)</span><span>◆ sıfırlama</span></div></div>
<div id="regstats"></div></div></div>
<h2>Kare başına konum hatası (log ölçek)</h2>
<div class="card"><svg id="err" viewBox="0 0 1100 260" role="img" aria-label="Kare başına hata"></svg>
<div class="legend"><span><i class="dot" style="background:var(--s2)"></i>Tek kare</span><span><i style="background:var(--s1)"></i>Filtre</span><span>gri bant: M3 reddi (güncelleme yok)</span></div></div>
<h2>Bölge başına medyan hata</h2>
<div class="card"><svg id="bars" viewBox="0 0 1100 250" role="img" aria-label="Bölge başına medyan hata"></svg>
<div class="legend"><span><i style="background:var(--s2);height:8px"></i>Tek kare</span><span><i style="background:var(--s1);height:8px"></i>Filtre</span></div>
<table id="regtab" style="margin-top:8px"></table></div>
<h2>Filtre ayarı — en iyi 8 (eğitim bölgeleri, havuzlanmış medyan)</h2><div class="card"><table id="tune"></table><div class="note" id="tunesel"></div></div>
<h2>Kararlar</h2><div class="card"><table id="dec"></table></div>
<h2>Süren işler</h2><div class="grid g2"><div class="card" id="jobs"></div><div class="card" id="gpu"></div></div>
</main><div class="tip" id="tip"></div><script>
const $=id=>document.getElementById(id),tip=$("tip");let S=null,TR={},REG=null;
const f1=x=>x==null?"–":x.toFixed(1).replace(".",","),f0=x=>x==null?"–":Math.round(x).toLocaleString("tr-TR"),f3=x=>x==null?"–":x.toFixed(3).replace(".",",");
function showTip(e,h){tip.innerHTML=h;tip.style.display="block";const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);tip.style.left=x+"px";tip.style.top=y+"px"}
function hideTip(){tip.style.display="none"}
function card(k,v,cls,sub){return `<div class="card"><div class="k">${k}</div><div class="v ${cls||""}">${v}</div>${sub?`<div class="note">${sub}</div>`:""}</div>`}
function renderStatus(){const s=S;$("upd").textContent="Son güncelleme "+new Date().toLocaleTimeString("tr-TR")+" · yalnız okur · 5 sn'de bir";
 $("steps").innerHTML=s.steps.map(x=>`<div class="step ${x.done?"done":""}"><b>${x.id}</b><div>${x.done?"✓":"○"} ${x.name}</div><div class="d">${x.detail}</div></div>`).join("");
 const c=s.criteria,d=s.dry&&s.dry["7"];if(d){const p=d.pooled_in_sample,a=d["kappa 05 → 11"],b=d["kappa 11 → 05"];
  const okR=p.reduction>=c.median_reduction_min,okW=p.wilcoxon_p<c.wilcoxon_p_max,okC=p.coverage7>=c.coverage_min;
  $("h4").innerHTML=card("Medyan hata azalması (ölçüt ≥ "+f3(c.median_reduction_min)+")",(okR?"✓ ":"✗ ")+(100*p.reduction).toFixed(1).replace(".",",")+" %",okR?"ok":"no",`${f1(p.med6)} → ${f1(p.med7)} m · %95 GA [${f3(p.reduction_ci[0])}; ${f3(p.reduction_ci[1])}]`)
   +card("Wilcoxon tek yönlü p (< "+c.wilcoxon_p_max+")",(okW?"✓ ":"✗ ")+p.wilcoxon_p.toExponential(1).replace(".",","),okW?"ok":"no",`n = ${p.n} kare, filtre başlamış ${p.n_started}`)
   +card("Kapsama (ölçüt ≥ "+f3(c.coverage_min)+")",(okC?"✓ ":"✗ ")+f3(p.coverage7),okC?"ok":"no",`κ = ${f3(p.kappa)} (havuz, tanım gereği ≈ 0,90) · yarıçap medyan ${f0(p.radius7_median)} m`)
   +card("κ taşınması (kapsama)",`${f3(a.coverage7)} / ${f3(b.coverage7)}`,(a.coverage7>=c.coverage_min&&b.coverage7>=c.coverage_min)?"ok":"no",`05 → 11 / 11 → 05 · açık risk`);
  const cv=s.dry["7-cv"].pooled_in_sample,nu=s.dry["7-noupd"].pooled_in_sample;
  $("h4note").textContent=`Ablasyon: sabit hız (7-cv) azalma ${f3(cv.reduction)} · M3 kapısız (7-noupd) ${f3(nu.reduction)} · başarı@25 ${f3(p.succ25[0])} → ${f3(p.succ25[1])}, @50 ${f3(p.succ50[0])} → ${f3(p.succ50[1])}`}
 else $("h4").innerHTML=card("Kuru deneme","henüz yok","pend");
 const regs=Object.keys(s.regions).sort();if(!REG||!regs.includes(REG))REG=regs.includes("05")?"05":regs[0];
 $("regsel").innerHTML=regs.map(r=>`<button role="tab" aria-selected="${r===REG}" class="${r===REG?"on":""}" data-r="${r}">${r}<span style="opacity:.7;font-size:11px"> ${s.regions[r].split==="calib"?"calib":""}</span></button>`).join("");
 $("regsel").querySelectorAll("button").forEach(b=>b.onclick=()=>{REG=b.dataset.r;$("r0").value=0;$("r1").value=100;renderStatus();loadTraj()});
 bars();
 $("regtab").innerHTML="<tr><th>bölge</th><th>küme</th><th class=n>kare</th><th class=n>M3 kabul</th><th class=n>med tek kare</th><th class=n>med filtre</th><th class=n>azalma</th><th class=n>sıfırlama</th></tr>"+regs.map(r=>{const x=s.regions[r];return `<tr><td>${r}</td><td>${x.split}</td><td class=n>${x.n}</td><td class=n>${f3(x.accept)}</td><td class=n>${f1(x.med6)} m</td><td class=n>${f1(x.med7)} m</td><td class=n>${f3(1-x.med7/x.med6)}</td><td class=n>${x.resets}</td></tr>`}).join("");
 $("tune").innerHTML="<tr><th>#</th><th class=n>λ</th><th class=n>σ_pos oranı</th><th class=n>ε</th><th class=n>θ σ</th><th class=n>sıfırlama</th><th class=n>med filtre</th><th>bölge başına (m)</th></tr>"+s.tune_top.map((x,i)=>`<tr${i===0?' style="font-weight:650"':""}><td>${i+1}</td><td class=n>${x.lik_power}</td><td class=n>${x.sigma_pos_rate}</td><td class=n>${x.eps}</td><td class=n>${x.theta_meas_sigma_deg>1e6?"kapalı":x.theta_meas_sigma_deg+"°"}</td><td class=n>${x.reset_lik}</td><td class=n>${f1(x.med7)} m</td><td>${Object.entries(x.per_region).map(([r,v])=>r+": "+f0(v)).join(" · ")}</td></tr>`).join("");
 $("tunesel").textContent=s.tune_top.length?`Tek kare (aynı bölgeler) havuzlanmış medyan ${f1(s.tune_top[0].med6)} m. Seçilen ayar Faz 6'da değişmeden kullanılır.`:"";
 $("dec").innerHTML=s.decisions.map(x=>`<tr><td style="white-space:nowrap"><b>${x.id}</b></td><td>${x.title}</td></tr>`).join("");
 const j=s.jobs,ses=j[0].sessions;$("jobs").innerHTML=`<div class="k">tmux (m4*)</div><div>${ses.length?ses.join(", "):"çalışan iş yok"}</div>`+j.slice(1).map(x=>`<div class="k" style="margin-top:8px">${x.log}${x.err?' · <span class="no">hata</span>':""}</div><pre>${x.tail.join("\n")}</pre>`).join("");
 const g=s.gpu;$("gpu").innerHTML=g.util==null?'<div class="k">GPU</div><div>okunamadı</div>':`<div class="k">GPU (RTX 3090)</div><div class="v">${g.util} %</div><div class="note">bellek ${g.mem.toFixed(1)} / ${g.mem_total.toFixed(0)} GB · ${g.temp} °C</div><div class="bar"><span style="width:${100*g.mem/g.mem_total}%"></span></div>`}
function bars(){const s=S,regs=Object.keys(s.regions).sort(),svg=$("bars"),W=1100,H=250,L=56,B=215,T=12,lo=Math.log10(10),hi=Math.log10(1000);
 const y=v=>B-(B-T)*(Math.log10(Math.max(10,Math.min(1000,v)))-lo)/(hi-lo),bw=Math.min(36,(W-L-20)/regs.length/2-6);let g="";
 [10,30,100,300,1000].forEach(t=>{g+=`<line x1="${L}" x2="${W-8}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/><text x="${L-8}" y="${y(t)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${t} m</text>`});
 regs.forEach((r,i)=>{const x=s.regions[r],cx=L+20+(i+.5)*(W-L-30)/regs.length;
  [[x.med6,"var(--s2)",-1,"tek kare"],[x.med7,"var(--s1)",1,"filtre"]].forEach(([v,c,side,nm])=>{if(v==null)return;const x0=side<0?cx-bw-1:cx+1,yy=y(v);
   g+=`<path d="M${x0},${B} V${yy+4} q0,-4 4,-4 h${bw-8} q4,0 4,4 V${B} Z" fill="${c}" data-t="${r} · ${nm}: ${f1(v)} m"/>`;
   g+=`<text x="${x0+bw/2}" y="${yy-5}" text-anchor="middle" font-size="11" fill="var(--ink)">${f0(v)}</text>`});
  g+=`<text x="${cx}" y="${B+16}" text-anchor="middle" font-size="12" fill="var(--ink)">${r}</text><text x="${cx}" y="${B+30}" text-anchor="middle" font-size="10.5" fill="var(--mute)">${x.split}</text>`});
 g+=`<line x1="${L}" x2="${W-8}" y1="${B}" y2="${B}" stroke="var(--mute)"/>`;svg.innerHTML=g;
 svg.querySelectorAll("[data-t]").forEach(el=>{el.onmousemove=e=>showTip(e,el.dataset.t);el.onmouseleave=hideTip})}
function rng(rows){const a=+$("r0").value,b=+$("r1").value,n=rows.length,i0=Math.floor(Math.min(a,b)/100*(n-1)),i1=Math.ceil(Math.max(a,b)/100*(n-1));
 $("rlab").textContent=`${rows[i0][0]}–${rows[i1][0]}`;return rows.slice(i0,i1+1)}
function drawMap(){const all=TR[REG];if(!all)return;const rows=rng(all),svg=$("map"),W=640,H=520,P=18;
 const pts=rows.flatMap(r=>[[r[10],r[11]],[r[8],r[9]]]).filter(p=>p[0]!=null);
 // eşit en-boy; uzak tek kare tepeleri haritayı ezmesin diye GT + filtre kapsamına göre ölçek, taşanlar kenara kırpılır
 const core=rows.flatMap(r=>[[r[10],r[11]],[r[8],r[9]]]);let x0=Math.min(...core.map(p=>p[0])),x1=Math.max(...core.map(p=>p[0])),y0=Math.min(...core.map(p=>p[1])),y1=Math.max(...core.map(p=>p[1]));
 const pad=Math.max(60,0.06*Math.max(x1-x0,y1-y0));x0-=pad;x1+=pad;y0-=pad;y1+=pad;const sc=Math.min((W-2*P)/(x1-x0),(H-2*P)/(y1-y0));
 const ox=(W-sc*(x1-x0))/2,oy=(H-sc*(y1-y0))/2,X=e=>ox+sc*(e-x0),Y=n=>H-oy-sc*(n-y0),cl=(v,a,b)=>Math.max(a,Math.min(b,v));
 let g="";const step=[50,100,200,500,1000,2000,5000].find(s=>s*sc>=70)||10000;
 for(let e=Math.ceil(x0/step)*step;e<=x1;e+=step)g+=`<line x1="${X(e)}" x2="${X(e)}" y1="0" y2="${H}" stroke="var(--grid)"/>`;
 for(let n=Math.ceil(y0/step)*step;n<=y1;n+=step)g+=`<line x1="0" x2="${W}" y1="${Y(n)}" y2="${Y(n)}" stroke="var(--grid)"/>`;
 g+=`<g transform="translate(${W-90},${H-22})"><line x1="0" x2="${step*sc}" y1="0" y2="0" stroke="var(--mute)" stroke-width="2"/><text x="0" y="-5" font-size="11" fill="var(--mute)">${step} m</text></g><text x="${W-14}" y="22" font-size="12" fill="var(--mute)" text-anchor="middle">K↑</text>`;
 const line=(i,j)=>"M"+rows.map(r=>`${X(r[i]).toFixed(1)},${Y(r[j]).toFixed(1)}`).join("L");
 g+=`<path d="${line(10,11)}" fill="none" stroke="var(--gt)" stroke-width="2" stroke-dasharray="5 3"/>`;
 rows.forEach(r=>{if(r[6]==null)return;const cx=cl(X(r[6]),3,W-3),cy=cl(Y(r[7]),3,H-3),out=cx!==X(r[6])||cy!==Y(r[7]);
  g+=`<circle cx="${cx}" cy="${cy}" r="${out?2.5:3.2}" fill="var(--s2)" opacity="${r[1]?0.85:0.35}"/>`});
 g+=`<path d="${line(8,9)}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
 rows.forEach(r=>{if(r[3])g+=`<path d="M${X(r[8])},${Y(r[9])-6} l5,6 l-5,6 l-5,-6 Z" fill="var(--card)" stroke="var(--ink)" stroke-width="1.3"/>`});
 g+=`<rect id="hit" x="0" y="0" width="${W}" height="${H}" fill="transparent"/><circle id="hov" r="6" fill="none" stroke="var(--ink)" stroke-width="2" style="display:none"/>`;svg.innerHTML=g;
 const hov=svg.querySelector("#hov");svg.querySelector("#hit").onmousemove=e=>{const b=svg.getBoundingClientRect(),mx=(e.clientX-b.left)*W/b.width,my=(e.clientY-b.top)*H/b.height;
  let best=null,bd=1e9;rows.forEach(r=>{const d=(X(r[10])-mx)**2+(Y(r[11])-my)**2;if(d<bd){bd=d;best=r}});if(!best||bd>900){hideTip();hov.style.display="none";return}
  hov.setAttribute("cx",X(best[10]));hov.setAttribute("cy",Y(best[11]));hov.style.display="";
  showTip(e,`<b>${REG} · kare ${best[0]}</b><br>tek kare ${f1(best[4])} m${best[1]?"":" (M3 ret)"}<br>filtre ${f1(best[5])} m${best[3]?" · sıfırlama":""}`)};
 svg.querySelector("#hit").onmouseleave=()=>{hideTip();hov.style.display="none"};
 const x=S.regions[REG];$("regstats").innerHTML=card("Medyan hata — tek kare → filtre",`${f0(x.med6)} → ${f0(x.med7)} m`,"",`${REG} (${x.split}) · ${x.n} kare`)+`<div style="height:10px"></div>`
  +card("Azalma",f3(1-x.med7/x.med6),(1-x.med7/x.med6)>=0.2?"ok":"no","bölge düzeyinde; H4 kararı havuzlanmış calib/test ile")+`<div style="height:10px"></div>`
  +card("M3 kabul / sıfırlama",`${f3(x.accept)} / ${x.resets}`,"","sıfırlama: kabul edilmiş ölçüm bulutu açıklamadı → filtre o kareden yeniden başladı");
 drawErr(rows)}
function drawErr(rows){const svg=$("err"),W=1100,H=260,L=56,R=10,T=10,B=228,lo=Math.log10(3),hi=Math.log10(5000);
 const n=rows.length,X=i=>L+(W-L-R)*i/Math.max(1,n-1),Y=v=>B-(B-T)*(Math.log10(Math.max(3,Math.min(5000,v)))-lo)/(hi-lo);let g="";
 rows.forEach((r,i)=>{if(!r[1])g+=`<rect x="${X(i)-(W-L-R)/n/2}" y="${T}" width="${(W-L-R)/n+.5}" height="${B-T}" fill="var(--grid)"/>`});
 [3,10,30,100,300,1000,5000].forEach(t=>{g+=`<line x1="${L}" x2="${W-R}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--line)"/><text x="${L-8}" y="${Y(t)+4}" text-anchor="end" font-size="11" fill="var(--mute)">${t}</text>`});
 g+=`<text x="${L-8}" y="${B+24}" text-anchor="end" font-size="11" fill="var(--mute)">m</text>`;
 const ticks=6;for(let k=0;k<=ticks;k++){const i=Math.round(k*(n-1)/ticks);g+=`<text x="${X(i)}" y="${B+16}" text-anchor="middle" font-size="11" fill="var(--mute)">${rows[i][0]}</text>`}
 rows.forEach((r,i)=>{if(r[4]!=null)g+=`<circle cx="${X(i)}" cy="${Y(r[4])}" r="2.2" fill="var(--s2)" opacity=".75"/>`});
 g+=`<path d="M${rows.map((r,i)=>`${X(i).toFixed(1)},${Y(r[5]).toFixed(1)}`).join("L")}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
 rows.forEach((r,i)=>{if(r[3])g+=`<line x1="${X(i)}" x2="${X(i)}" y1="${B}" y2="${B-8}" stroke="var(--ink)" stroke-width="1.5"/>`});
 g+=`<line id="xh" y1="${T}" y2="${B}" stroke="var(--mute)" style="display:none"/><rect id="hit2" x="${L}" y="${T}" width="${W-L-R}" height="${B-T}" fill="transparent"/>`;svg.innerHTML=g;
 const xh=svg.querySelector("#xh");svg.querySelector("#hit2").onmousemove=e=>{const b=svg.getBoundingClientRect(),mx=(e.clientX-b.left)*W/b.width,i=Math.max(0,Math.min(n-1,Math.round((mx-L)/(W-L-R)*(n-1)))),r=rows[i];
  xh.setAttribute("x1",X(i));xh.setAttribute("x2",X(i));xh.style.display="";showTip(e,`<b>kare ${r[0]}</b><br><span style="color:var(--s2)">●</span> tek kare ${f1(r[4])} m${r[1]?"":" (ret)"}<br><span style="color:var(--s1)">━</span> filtre ${f1(r[5])} m${r[3]?"<br>◆ sıfırlama":""}`)};
 svg.querySelector("#hit2").onmouseleave=()=>{hideTip();xh.style.display="none"}}
async function loadTraj(){if(!REG)return;if(!TR[REG]){const r=await fetch("/api/traj?region="+REG);TR[REG]=await r.json()}drawMap()}
["r0","r1"].forEach(id=>$(id).oninput=drawMap);
async function tick(){try{const r=await fetch("/api/status");S=await r.json();renderStatus();if(!TR[REG])loadTraj()}catch(e){$("upd").textContent="sunucuya ulaşılamıyor"}}
tick();setInterval(tick,5000);setInterval(()=>{TR={};loadTraj()},60000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/status":
            self._send(json.dumps(status(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
        elif u.path == "/api/traj":
            r = parse_qs(u.query).get("region", [""])[0]
            self._send(json.dumps(traj_rows().get(r, [])).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
