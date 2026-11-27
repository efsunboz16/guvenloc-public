"""GüvenLoc Gradio prototipi (MIMARI 6C).

  python app/app.py                 # http://127.0.0.1:7860 (yalnız bu makine; RoMa v2 base, ≈ 3,6 GB GPU)
  python app/app.py --port 7861

Mod 1 — tek kare: veri kümesinden kare seç (ya da kendi görüntünü yükle; karo seçilen karenin uydu karosu) →
uydu karosu üzerinde tahmin, q̂·σ̂ güven dairesi, kabul/ret, enlem/boylam, yön.
Mod 2 — uçuş: bölge seç → oynatıcı; GT ve filtre yörüngesi, parçacık bulutu, güven bölgesi, uydu eşleştirmesinin
yapıldığı/atlandığı kareler (γ, F5-7). Filtre ayarı/κ/γ Faz 5'te dondurulmuş değerler.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

ENGINE: C.Engine | None = None


def engine() -> C.Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = C.Engine()
    return ENGINE


def frame_slider(rid):
    fr = C.frames_of(rid)
    return gr.update(minimum=fr[0], maximum=fr[-1], value=fr[len(fr) // 3], step=1)


def run_single(rid, num, upload):
    fr = C.frames_of(rid)
    num = min(fr, key=lambda x: abs(x - int(num)))
    return engine().single(rid, num, upload)


def load_flight(rid):
    fl = C.flight(rid)
    n = len(fl["nums"])
    img, thumb, md = C.flight_frame(rid, 0)
    return gr.update(minimum=0, maximum=n - 1, value=0, step=1), img, thumb, md


def show(rid, k, view):
    return C.flight_frame(rid, int(k), float(view))


def tick(rid, k, step, view):
    n = len(C.flight(rid)["nums"])
    k2 = min(int(k) + int(step), n - 1)
    img, thumb, md = C.flight_frame(rid, k2, float(view))
    return k2, img, thumb, md, gr.update(active=k2 < n - 1)


def build() -> gr.Blocks:
    regs = C.region_choices()
    default = next((v for _, v in regs if v == "03"), regs[0][1])
    with gr.Blocks(title="GüvenLoc — prototip") as demo:
        gr.Markdown("# GüvenLoc — drone karesinden uydu haritasında güvenilir konum\n"
                    "TÜBİTAK 2209-A prototipi. M1-C (dönme) → RoMa v2 → M2 (oylama hacmi) → M3 (conformal kabul/ret, "
                    "güven dairesi) → M4 (parçacık filtresi). Test bölgeleri: 02, 03, 06.")
        with gr.Tab("Mod 1 — tek kare"):
            with gr.Row():
                with gr.Column(scale=1):
                    r1 = gr.Dropdown(regs, value=default, label="Bölge")
                    fr = C.frames_of(default)
                    n1 = gr.Slider(fr[0], fr[-1], value=fr[len(fr) // 3], step=1, label="Kare numarası")
                    up = gr.Image(label="İsteğe bağlı: kendi drone görüntün (karo: seçili karenin uydu karosu)",
                                  type="numpy", height=200)
                    b1 = gr.Button("Konumlandır", variant="primary")
                    th1 = gr.Image(label="Drone karesi", interactive=False, height=240)
                with gr.Column(scale=2):
                    im1 = gr.Image(label="Uydu karosu: tahmin, güven dairesi, GT", interactive=False, height=640)
                    md1 = gr.Markdown()
            r1.change(frame_slider, r1, n1)
            b1.click(run_single, [r1, n1, up], [im1, th1, md1])
        with gr.Tab("Mod 2 — uçuş"):
            with gr.Row():
                with gr.Column(scale=1):
                    r2 = gr.Dropdown(regs, value=default, label="Uçuş (bölge)")
                    b2 = gr.Button("Uçuşu yükle (filtre koşulur, ≈ 1–3 s)", variant="primary")
                    k2 = gr.Slider(0, 1, value=0, step=1, label="Kare (sıra)")
                    with gr.Row():
                        play = gr.Button("▶ Oynat")
                        pause = gr.Button("⏸ Duraklat")
                    step = gr.Slider(1, 10, value=2, step=1, label="Adım (kare/tık)")
                    view = gr.Slider(600, 4000, value=1500, step=100, label="Harita genişliği (m)")
                    th2 = gr.Image(label="Drone karesi", interactive=False, height=240)
                with gr.Column(scale=2):
                    im2 = gr.Image(label="Uydu haritası: GT, filtre, parçacıklar, güven bölgesi", interactive=False,
                                   height=640)
                    md2 = gr.Markdown()
            timer = gr.Timer(0.6, active=False)
            b2.click(load_flight, r2, [k2, im2, th2, md2])
            k2.release(show, [r2, k2, view], [im2, th2, md2])
            view.release(show, [r2, k2, view], [im2, th2, md2])
            play.click(lambda: gr.update(active=True), None, timer)
            pause.click(lambda: gr.update(active=False), None, timer)
            timer.tick(tick, [r2, k2, step, view], [k2, im2, th2, md2, timer])
    return demo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
