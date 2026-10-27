"""Membangun notebook Colab Bab 8 versi 2 (notebooks/bab08.ipynb): ilmu sama, biaya T4 jauh lebih kecil."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 8 — Deteksi Objek: Menghitung Sel Darah (BCCD) dengan Faster R-CNN MobileNetV3
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi:
1. **BCCD** (Shenggan, 2017; lisensi MIT): 364 citra apusan darah 640×480, anotasi VOC (RBC, WBC, Platelets), pembagian resmi `ImageSets/Main` (train/val/test).
2. Konsep: IoU, NMS, mAP@0,5 dan mAP@0,5:0,95 (COCO), dihitung dengan `torchmetrics` (*backend* pycocotools).
3. **Faster R-CNN MobileNetV3-Large FPN** dari `torchvision` (BSD-3; bobot pra-latih COCO), varian resolusi penuh vs varian **320 px** (seluler).
4. Hitung sel otomatis vs anotasi per citra (MAE hitung, Bland–Altman), latensi GPU vs CPU.

**Versi 2 (28-10-2026)** — perubahan terhadap `bab08.ipynb` (eksperimen, data, model, dan metrik sama):
- **Mode deterministik PyTorch dimatikan** untuk notebook ini. Dalam mode itu, `roi_align` torchvision di GPU beralih ke implementasi Python
  yang dikompilasi `torch.compile` (lambat, kompilasi ulang untuk bentuk tensor yang berubah) — penyebab utama run v1 sangat lama.
  Seed tetap dipasang, tetapi hasil antar-run dapat berbeda sedikit.
- 12 epoch (bukan 24), evaluasi mAP validasi tiap 3 epoch (+ epoch terakhir); citra disimpan sekali di GPU (uint8), augmentasi di GPU.
- Latensi CPU: 1 pemanasan + 5 ulangan (bukan 3 + 10).
- Batas waktu pengaman per run latih (dicatat bila tercapai), cetak progres per epoch dengan ETA, dan **waktu setiap sel** dicetak otomatis.

Jalur YOLO (Ultralytics, AGPL-3.0) **tidak** dipakai di notebook ini; lihat catatan lisensi di buku.
Hasil → `/content/hasil_bab08/` → `/content/hasil_bab08.zip`. Hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 8.0 Lingkungan")
code(sel_lingkungan("08"))
code(r"""
# --- v2: cetak waktu SETIAP sel secara otomatis (hook IPython) ---
WAKTU_SEL = []
try:
    _ip = get_ipython()
    def _sel_mulai(info=None):
        global _T_SEL; _T_SEL = time.perf_counter()
        print(f"[sel mulai {time.strftime('%H:%M:%S')}]", flush=True)
    def _sel_selesai(hasil=None):
        dt = time.perf_counter() - _T_SEL; tot = time.perf_counter() - T_MULAI
        WAKTU_SEL.append({"sel_ke": len(WAKTU_SEL) + 1, "detik": round(dt, 2), "kumulatif_detik": round(tot, 1)})
        print(f"[sel selesai] {dt:.1f} s | kumulatif notebook {tot/60:.1f} menit", flush=True)
    for _ev, _fn in (("pre_run_cell", _sel_mulai), ("post_run_cell", _sel_selesai)):
        for _f in list(_ip.events.callbacks[_ev]):
            if getattr(_f, "__name__", "") == _fn.__name__: _ip.events.unregister(_ev, _f)
        _ip.events.register(_ev, _fn)
    _T_SEL = time.perf_counter()
except NameError:
    pass

with catat_waktu("1_pip_install"):
    import importlib.util
    perlu = [p for p, m in (("torchmetrics>=1.4", "torchmetrics"), ("pycocotools", "pycocotools")) if importlib.util.find_spec(m) is None]
    if perlu:
        print("pip install:", perlu, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *perlu], check=True, timeout=900)
import numpy as np, pandas as pd, torch, torchvision, sklearn, matplotlib, matplotlib.ticker, matplotlib.patches as mpatches
import matplotlib.pyplot as plt, xml.etree.ElementTree as ET, torchmetrics
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSI = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "torchmetrics": torchmetrics.__version__,
         "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
print(json.dumps(VERSI, indent=1))
""" + "\n" + SEED_FN + "\n" + r"""
# --- v2: matikan mode deterministik (lihat catatan di atas). Seed tetap dipasang. ---
def seed_semua(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
seed_semua()
print("algoritme deterministik:", torch.are_deterministic_algorithms_enabled(), "| cudnn.deterministic:", torch.backends.cudnn.deterministic)
""")

md(r"""
## 8.1 Data BCCD (format Pascal VOC)
Setiap berkas XML memuat `<object>` dengan nama kelas dan `<bndbox>` (xmin, ymin, xmax, ymax). Kotak yang rusak (lebar/tinggi ≤ 1 piksel)
dibuang dan dicatat jumlahnya. Indeks kelas: 0 = latar (khas torchvision), 1 = RBC, 2 = WBC, 3 = Platelets.
""")
code(r"""
KELAS = ["latar", "RBC", "WBC", "Platelets"]
with catat_waktu("2_muat_data"):
    pastikan_dataset("bccd", ["BCCD_Dataset-master.zip"])
    akar = ekstrak_lokal("bccd", "BCCD_Dataset-master.zip") / "BCCD_Dataset-master" / "BCCD"
    def baca_voc(nama):
        r = ET.parse(akar / "Annotations" / f"{nama}.xml").getroot(); kotak, label, rusak = [], [], 0
        for o in r.iter("object"):
            b = o.find("bndbox"); v = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
            v = [min(max(v[0], 0), 639), min(max(v[1], 0), 479), min(max(v[2], 0), 640), min(max(v[3], 0), 480)]
            if v[2] - v[0] <= 1 or v[3] - v[1] <= 1: rusak += 1; continue
            kotak.append(v); label.append(KELAS.index(o.find("name").text.strip()))
        return np.array(kotak, np.float32).reshape(-1, 4), np.array(label, np.int64), rusak
    DATA, RUSAK = {}, 0
    for s in ("train", "val", "test"):
        nama = (akar / "ImageSets" / "Main" / f"{s}.txt").read_text().split()
        nama = [n for n in nama if (akar / "JPEGImages" / f"{n}.jpg").exists()]
        if SMOKE: nama = nama[:{"train": 16, "val": 6, "test": 6}[s]]
        item = []
        for n in nama:
            k, l, rs = baca_voc(n); RUSAK += rs
            img = np.array(Image.open(akar / "JPEGImages" / f"{n}.jpg").convert("RGB"))
            item.append({"nama": n, "img": img, "x": torch.as_tensor(img).permute(2, 0, 1).contiguous().to(DEV),   # v2: uint8 di GPU, sekali
                         "boxes": k, "labels": l, "b_t": torch.as_tensor(k).to(DEV), "l_t": torch.as_tensor(l).to(DEV)})
        DATA[s] = item; print(f"{s}: {len(item)} citra dimuat", flush=True)
T81 = pd.DataFrame([{"split": s, "n_citra": len(v), **{f"n_{KELAS[c]}": int(sum((it["labels"] == c).sum() for it in v)) for c in (1, 2, 3)},
                     "rerata_RBC_per_citra": float(np.mean([(it["labels"] == 1).sum() for it in v]))} for s, v in DATA.items()])
T81.to_csv(OUT / "tabel_8_1_data.csv", index=False); print("kotak rusak dibuang:", RUSAK)
WARNA = {1: "red", 2: "blue", 3: "lime"}
def gambar_kotak(a, img, boxes, labels, skor=None, judul=""):
    a.imshow(img); a.axis("off"); a.set_title(judul, fontsize=8)
    for i, (b, l) in enumerate(zip(boxes, labels)):
        a.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=WARNA[int(l)], lw=1))
        if skor is not None: a.text(b[0], b[1], f"{skor[i]:.2f}", color=WARNA[int(l)], fontsize=5)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, it in zip(ax, DATA["train"][:3]): gambar_kotak(a, it["img"], it["boxes"], it["labels"], judul=f"{it['nama']}: {len(it['labels'])} kotak")
fig.legend(handles=[mpatches.Patch(color=WARNA[c], label=KELAS[c]) for c in (1, 2, 3)], loc="lower center", ncol=3)
plt.tight_layout(); plt.savefig(OUT / "gambar_8_1_contoh_anotasi.png", bbox_inches="tight"); plt.show()
uk = np.concatenate([np.c_[it["boxes"][:, 2] - it["boxes"][:, 0], it["boxes"][:, 3] - it["boxes"][:, 1], it["labels"]] for it in DATA["train"]])
fig, a = plt.subplots(figsize=(5, 3.5))
for c in (1, 2, 3): a.scatter(uk[uk[:, 2] == c, 0], uk[uk[:, 2] == c, 1], s=4, alpha=0.5, c=WARNA[c], label=KELAS[c])
a.set_xlabel("lebar kotak (px)"); a.set_ylabel("tinggi kotak (px)"); a.legend(); plt.tight_layout(); plt.savefig(OUT / "gambar_8_2_ukuran_kotak.png", bbox_inches="tight"); plt.show(); T81
""")

md(r"""
## 8.2 IoU dan NMS dalam beberapa baris
""")
code(r"""
a = torch.tensor([[10., 10, 60, 60]]); b = torch.tensor([[30., 30, 80, 80], [10, 10, 60, 60], [100, 100, 120, 120]])
print("IoU:", torchvision.ops.box_iou(a, b).numpy().round(3))
kotak = torch.tensor([[10., 10, 60, 60], [12, 12, 62, 62], [100, 100, 150, 150]]); skor = torch.tensor([0.9, 0.8, 0.7])
print("NMS (IoU 0,5) menyisakan indeks:", torchvision.ops.nms(kotak, skor, 0.5).tolist())
""")

md(r"""
## 8.3 Pelatihan Faster R-CNN MobileNetV3-Large FPN
- Kepala klasifikasi COCO (91 kelas) diganti `FastRCNNPredictor(…, 4)`.
- SGD (lr 0,01, momentum 0,9, wd 1e-4), *warm-up* 1 epoch + kosinus, batch 4, AMP; augmentasi flip horizontal/vertikal (kotak ikut dibalik) dan kecerahan ±15 %.
- Model terbaik dipilih dari **mAP@0,5 validasi** (dievaluasi tiap 2 epoch).
- Dua varian: **resolusi penuh** (`min_size=800`, bawaan) dan **320 px** (`fasterrcnn_mobilenet_v3_large_320_fpn`).
""")
code(r"""
def buat_model(varian):
    fn = fasterrcnn_mobilenet_v3_large_fpn if varian == "penuh" else fasterrcnn_mobilenet_v3_large_320_fpn
    try: m = fn(weights="DEFAULT")
    except Exception as e:
        if not SMOKE: raise
        print("SMOKE: bobot COCO gagal diunduh, pakai acak:", e); m = fn(weights=None, weights_backbone=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(KELAS))
    m.roi_heads.detections_per_img = 100            # sama dengan maxDets COCO; satu citra BCCD memuat hingga puluhan RBC
    if SMOKE: m.transform.min_size, m.transform.max_size = (256,), 341
    return m.to(DEV)
def ke_input(it, aug=False, g=None):
    x = it["x"].float().div(255); b = it["b_t"].clone(); H, W = x.shape[1:]
    if aug:
        if float(torch.rand(1, generator=g)) < 0.5: x = x.flip(2); b[:, [0, 2]] = W - b[:, [2, 0]]
        if float(torch.rand(1, generator=g)) < 0.5: x = x.flip(1); b[:, [1, 3]] = H - b[:, [3, 1]]
        x = (x * (1 + (float(torch.rand(1, generator=g)) * 2 - 1) * 0.15)).clamp(0, 1)   # v2: skalar Python (generator di CPU, x di GPU)
    return x, {"boxes": b, "labels": it["l_t"]}
@torch.no_grad()
def prediksi(model, item, bs=8):
    model.eval(); out = []
    for i in range(0, len(item), bs):
        xs = [ke_input(it)[0] for it in item[i:i + bs]]
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"): o = model(xs)
        out += [{k: v.float().cpu() if v.is_floating_point() else v.cpu() for k, v in d.items()} for d in o]
    return out
def hitung_map(pred, item, kelas=True):
    mt = MeanAveragePrecision(iou_type="bbox", class_metrics=kelas, backend="pycocotools")   # maxDets bawaan [1, 10, 100]
    mt.update(pred, [{"boxes": torch.as_tensor(it["boxes"]), "labels": torch.as_tensor(it["labels"])} for it in item])
    r = mt.compute(); hasil = {"mAP50": float(r["map_50"]), "mAP50_95": float(r["map"])}
    if kelas:
        for c, ap in zip(r["classes"].tolist(), r["map_per_class"].tolist()): hasil[f"AP50_95_{KELAS[c]}"] = ap
    return hasil
EPOCH, EVAL_TIAP = 12, 3                                # v2: 24 -> 12 epoch, evaluasi tiap 3 epoch
BATAS_DETIK = {"penuh": 300, "320": 150}               # v2: pengaman per run (normalnya tidak tercapai)
def latih_det(varian, seed, n_ep=EPOCH, bs=4, lr=0.01):
    seed_semua(seed); m = buat_model(varian); g = torch.Generator().manual_seed(seed)
    if SMOKE: n_ep = 2
    par = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(par, lr=lr, momentum=0.9, weight_decay=1e-4)
    n_it = int(np.ceil(len(DATA["train"]) / bs)); tot = n_ep * n_it
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / n_it) * 0.5 * (1 + np.cos(np.pi * i / tot)))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda"); log = []; best = (-1, None, 0); T0 = time.perf_counter(); henti = False
    for ep in range(1, n_ep + 1):
        m.train(); t0 = time.perf_counter(); perm = torch.randperm(len(DATA["train"]), generator=g).tolist(); tl = 0
        for i in range(0, len(perm), bs):
            xs, ts = zip(*[ke_input(DATA["train"][j], aug=True, g=g) for j in perm[i:i + bs]])
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                ld = m(list(xs), list(ts)); loss = sum(ld.values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        if DEV.type == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0; baris = {"model": f"{varian} | seed {seed}", "epoch": ep, "loss_latih": tl / n_it, "detik_epoch": dt}
        if time.perf_counter() - T0 > BATAS_DETIK[varian] and ep < n_ep and not SMOKE:
            henti = True; print(f"!! batas waktu {BATAS_DETIK[varian]} s tercapai pada epoch {ep}; pelatihan dihentikan", flush=True)
        if ep % EVAL_TIAP == 0 or ep == n_ep or henti:
            mv = hitung_map(prediksi(m, DATA["val"]), DATA["val"], kelas=False); baris.update({"mAP50_val": mv["mAP50"], "mAP50_95_val": mv["mAP50_95"]})
            if mv["mAP50"] > best[0]: best = (mv["mAP50"], {k: v.detach().clone() for k, v in m.state_dict().items()}, ep)
        baris["berhenti_batas_waktu"] = henti; log.append(baris)
        el = time.perf_counter() - T0
        print(f"[{varian} s{seed}] epoch {ep}/{n_ep} loss {baris['loss_latih']:.3f} | {dt:.1f} s/epoch | "
              + (f"mAP50_val {baris['mAP50_val']:.3f} | " if "mAP50_val" in baris else "")
              + f"berjalan {el/60:.1f} mnt, ETA run {el/ep*(n_ep-ep)/60:.1f} mnt", flush=True)
        if henti: break
    m.load_state_dict(best[1]); return m, pd.DataFrame(log), best[2]
VARIAN = {"penuh": [0] if SMOKE else [0, 1], "320": [0]}
HASIL, LOG, MODEL = [], {}, {}
for v, seeds in VARIAN.items():
    for s in seeds:
        print(f"=== latih varian {v}, seed {s} ===", flush=True)
        with catat_waktu(f"latih_{v}_s{s}"): m, lg, ep = latih_det(v, s)
        with catat_waktu(f"evaluasi_{v}_s{s}"): pu = prediksi(m, DATA["test"]); mu = hitung_map(pu, DATA["test"])
        LOG[(v, s)] = lg; MODEL[(v, s)] = (m, pu)
        HASIL.append({"varian": v, "seed": s, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()), "berhenti_batas_waktu": bool(lg.berhenti_batas_waktu.any()), "mAP50_val_terbaik": float(lg.mAP50_val.max()), "detik_per_epoch": float(lg.detik_epoch.mean()),
                      "n_param_juta": sum(p.numel() for p in m.parameters()) / 1e6, **{f"{k}_uji": val for k, val in mu.items()}})
        print(HASIL[-1])
T82 = pd.DataFrame(HASIL); T82.to_csv(OUT / "tabel_8_2_map.csv", index=False); pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for (v, s), lg in LOG.items():
    ax[0].plot(lg.epoch, lg.loss_latih, label=f"{v} s{s}"); d = lg.dropna(subset=["mAP50_val"]); ax[1].plot(d.epoch, d.mAP50_val, "o-", ms=3, label=f"{v} s{s}")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss latih (jumlah 4 komponen)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mAP@0,5 val")
for a in ax: a.legend(fontsize=7); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.tight_layout(); plt.savefig(OUT / "gambar_8_3_kurva_latih.png", bbox_inches="tight"); plt.show(); T82
""")

md(r"""
## 8.4 Visualisasi prediksi dan AP per kelas
Model utama: varian **penuh** dengan mAP@0,5 validasi tertinggi. Kotak prediksi ditampilkan untuk skor ≥ 0,5.
""")
code(r"""
PIL = max([k for k in MODEL if k[0] == "penuh"], key=lambda k: T82[(T82.varian == k[0]) & (T82.seed == k[1])].mAP50_val_terbaik.iloc[0])
model, PU = MODEL[PIL]; torch.save(model.state_dict(), OUT / "fasterrcnn_mnv3_bccd.pt")
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for j, it in enumerate(DATA["test"][:3]):
    gambar_kotak(ax[0, j], it["img"], it["boxes"], it["labels"], judul=f"anotasi: {len(it['labels'])} kotak")
    p = PU[j]; k = p["scores"] >= 0.5
    gambar_kotak(ax[1, j], it["img"], p["boxes"][k].numpy(), p["labels"][k].numpy(), p["scores"][k].numpy(), judul=f"prediksi (skor ≥ 0,5): {int(k.sum())} kotak")
fig.legend(handles=[mpatches.Patch(color=WARNA[c], label=KELAS[c]) for c in (1, 2, 3)], loc="lower center", ncol=3)
plt.tight_layout(); plt.savefig(OUT / "gambar_8_4_prediksi_vs_anotasi.png", bbox_inches="tight"); plt.show()
r = T82[(T82.varian == PIL[0]) & (T82.seed == PIL[1])].iloc[0]
T83 = pd.DataFrame([{"kelas": KELAS[c], "AP50_95_uji": r.get(f"AP50_95_{KELAS[c]}_uji", np.nan)} for c in (1, 2, 3)])
# AP@0,5 per kelas: hitung ulang per kelas
for c in (1, 2, 3):
    pc = [{"boxes": p["boxes"][p["labels"] == c], "scores": p["scores"][p["labels"] == c], "labels": p["labels"][p["labels"] == c]} for p in PU]
    ic = [{"boxes": it["boxes"][it["labels"] == c], "labels": it["labels"][it["labels"] == c]} for it in DATA["test"]]
    T83.loc[T83.kelas == KELAS[c], "AP50_uji"] = hitung_map(pc, ic, kelas=False)["mAP50"]
T83.to_csv(OUT / "tabel_8_3_ap_per_kelas.csv", index=False); T83
""")

md(r"""
## 8.5 Hitung sel otomatis vs anotasi
Ambang skor per kelas dipilih pada set **validasi** untuk meminimalkan MAE hitung, lalu diterapkan pada set uji.
Catatan: anotasi RBC di BCCD diketahui tidak selalu lengkap (sel bertumpuk atau di tepi), sehingga "acuan" hitung juga mengandung galat.
""")
code(r"""
with catat_waktu("3_prediksi_val_hitung"): PV = prediksi(model, DATA["val"])
AMB = {}
for c in (1, 2, 3):
    yv = np.array([(it["labels"] == c).sum() for it in DATA["val"]])
    kand = np.round(np.arange(0.2, 0.95, 0.05), 2)
    mae = [np.abs(np.array([int(((p["labels"] == c) & (p["scores"] >= t)).sum()) for p in PV]) - yv).mean() for t in kand]
    AMB[c] = float(kand[int(np.argmin(mae))])
baris = []; fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c in zip(ax, (1, 2, 3)):
    y = np.array([(it["labels"] == c).sum() for it in DATA["test"]]); yh = np.array([int(((p["labels"] == c) & (p["scores"] >= AMB[c])).sum()) for p in PU])
    d = yh - y; baris.append({"kelas": KELAS[c], "ambang_skor_val": AMB[c], "rerata_acuan": y.mean(), "rerata_prediksi": yh.mean(), "MAE_hitung": np.abs(d).mean(),
                              "bias": d.mean(), "LoA_bawah": d.mean() - 1.96 * d.std(ddof=1), "LoA_atas": d.mean() + 1.96 * d.std(ddof=1),
                              "persen_citra_tepat": float((d == 0).mean() * 100)})
    a.scatter(y + np.random.default_rng(0).uniform(-0.2, 0.2, len(y)), yh, s=10, alpha=0.6); lim = [0, max(y.max(), yh.max()) + 1]; a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel(f"hitung anotasi {KELAS[c]}"); a.set_ylabel("hitung prediksi"); a.set_title(f"{KELAS[c]} (ambang {AMB[c]}): MAE {np.abs(d).mean():.2f}")
plt.tight_layout(); plt.savefig(OUT / "gambar_8_5_hitung_sel.png", bbox_inches="tight"); plt.show()
T84 = pd.DataFrame(baris); T84.to_csv(OUT / "tabel_8_4_hitung_sel.csv", index=False); T84
""")

md("## 8.6 Latensi inferensi (satu citra 640×480): GPU vs CPU")
code(r"""
def latensi(m, dev, n=20, panas=3):
    m = m.to(dev).eval(); x = [DATA["test"][0]["x"].float().div(255).to(dev)]
    with torch.no_grad():
        for _ in range(panas): m(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x)
            if dev.type == "cuda": torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)
baris = []
with catat_waktu("4_latensi"):
    for (v, s), (m, _) in MODEL.items():
        if s != 0: continue
        r = {"varian": v, "ms_GPU": latensi(m, DEV) if DEV.type == "cuda" else np.nan}
        n_thr = torch.get_num_threads(); torch.set_num_threads(2)
        r["ms_CPU_2thread"] = latensi(m, torch.device("cpu"), n=2 if SMOKE else 5, panas=1); torch.set_num_threads(n_thr)   # v2: 1 + 5 ulangan
        baris.append(r); m.to(DEV); print(r, flush=True)
T85 = pd.DataFrame(baris); T85.to_csv(OUT / "tabel_8_5_latensi.csv", index=False); T85
""")

md("## 8.7 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("08", 'dict(data=T81.to_dict("records"), kotak_rusak=RUSAK, map=T82.to_dict("records"), model_terpilih=list(PIL), ap_per_kelas=T83.to_dict("records"), hitung_sel=T84.to_dict("records"), latensi=T85.to_dict("records"), versi_notebook="v2", waktu_per_sel=WAKTU_SEL, epoch=EPOCH, eval_tiap=EVAL_TIAP, deterministik=False)'))
simpan(C, "bab08.ipynb", Path(__file__).parent / "bab08.ipynb")
