"""Membangun notebook Colab Bab 1 (notebooks/bab01.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 1 — Pengantar Visi Komputer untuk Kesehatan
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Notebook ini berdiri sendiri dan dirancang untuk **Google Colab dengan GPU T4**
(*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

Isi:
1. Info lingkungan (GPU, versi paket) dan pencatat waktu tiap sel penting.
2. Unduh enam subset **MedMNIST v2** berlisensi **CC BY 4.0** dari Zenodo (sumber resmi): PneumoniaMNIST (X-ray), OrganAMNIST (CT), BreastMNIST (USG), BloodMNIST (mikroskopi), RetinaMNIST (fundus), OCTMNIST (OCT).
3. Galeri modalitas, perbandingan resolusi 28/64/128/224 piksel, dan distribusi kelas.
4. Baseline pertama pada PneumoniaMNIST: regresi logistik piksel (28 px) vs CNN kecil (64 px) dengan sensitivitas, spesifisitas, PPV, NPV, AUC (+ CI 95 % bootstrap), matriks kebingungan, kurva ROC, dan contoh salah klasifikasi.
5. Data disimpan **permanen di Google Drive** (`MyDrive/Buku Visi Komputer Kesehatan/dataset/medmnist/`); bila sudah ada dan lengkap (ukuran + MD5 Zenodo), tidak diunduh ulang.
6. Semua hasil disimpan ke `/content/hasil_bab01/` lalu dizip menjadi `/content/hasil_bab01.zip`.

> **Perhatian.** Data MedMNIST sudah dianonimkan dan **tidak dimaksudkan untuk penggunaan klinis**. Model di notebook ini hanya untuk pembelajaran.
> Atribusi: Yang dkk. (2023), *Scientific Data*, 10, 41, https://doi.org/10.1038/s41597-022-01721-8 — data: https://doi.org/10.5281/zenodo.10519652.
""")

md(r"""
## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`
Sel ini sama di semua notebook buku. Colab akan meminta izin akses Drive (akun Anda sendiri).
Dataset disimpan di `MyDrive/Buku Visi Komputer Kesehatan/dataset/<nama_dataset>/`; fungsi `pastikan_dataset(nama)`
memeriksa keberadaan, ukuran, dan MD5 berkas, lalu hanya mengunduh dari sumber resmi bila belum ada atau rusak.
""")
code(kode_sel_standar())

md("## 1.0 Lingkungan dan pencatat waktu")
code(r"""
import os, sys, time, json, platform, subprocess, contextlib, shutil
from pathlib import Path

T_MULAI = time.perf_counter()
SMOKE = os.environ.get("BAB01_SMOKE") == "1"          # uji cepat di luar Colab (penulis); di Colab = False
OUT  = Path(os.environ.get("BAB01_OUT",  "/content/hasil_bab01"))
DATA = DATASET_DIR / "medmnist"                      # folder Drive permanen (dari sel standar)
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

WAKTU = {}
@contextlib.contextmanager
def catat_waktu(nama):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    WAKTU[nama] = round(dt, 2)
    print(f"[waktu] {nama}: {dt:.1f} s")

def jalankan(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return f"(gagal: {e})"

with catat_waktu("0_info_lingkungan"):
    smi = jalankan("nvidia-smi")
    print(smi if smi.strip() else "nvidia-smi tidak tersedia (tidak ada GPU?)")
    info_cpu = jalankan("lscpu | grep -E 'Model name|^CPU\\(s\\)'")
    info_mem = jalankan("free -h | head -2")
    print(info_cpu, info_mem)
    (OUT / "info_lingkungan.txt").write_text(
        f"Python: {sys.version}\nPlatform: {platform.platform()}\n\n{smi}\n{info_cpu}\n{info_mem}")
print("SMOKE =", SMOKE, "| OUT =", OUT)
""")

md("## 1.1 Instalasi paket")
code(r"""
with catat_waktu("1_pip_install"):
    try:
        import medmnist
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist==3.0.2"], check=True)
        import medmnist
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import sklearn
from medmnist import INFO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

PERANGKAT = "cuda" if torch.cuda.is_available() else "cpu"
VERSI = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "medmnist": medmnist.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
         "matplotlib": matplotlib.__version__, "perangkat": PERANGKAT}
print(json.dumps(VERSI, indent=1))
plt.rcParams.update({"font.size": 10, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
""")

md(r"""
## 1.2 Pastikan data ada di Drive, lalu muat (sumber resmi: Zenodo)
Semua subset di bawah berlisensi **CC BY 4.0** (dicek dari `INFO[flag]['license']`). Berkas `.npz` dibaca dari folder Drive
(`root=` folder Drive); pengunduhan hanya terjadi bila `.npz` belum ada atau tidak lolos cek ukuran/MD5.
Berkas `.npz` dibaca sekali ke memori, sehingga tidak perlu disalin ke disk lokal.
""")
code(r"""
MODALITAS = [  # (flag, modalitas, ukuran untuk galeri)
    ("pneumoniamnist", "X-ray dada", 128),
    ("organamnist",    "CT abdomen (irisan aksial)", 64),
    ("breastmnist",    "USG payudara", 128),
    ("bloodmnist",     "Mikroskopi sel darah", 64),
    ("retinamnist",    "Fundus retina", 128),
    ("octmnist",       "OCT retina", 64),
]
if SMOKE:
    MODALITAS = [(f, m, 28) for f, m, _ in MODALITAS]
RES_PNEU = [28, 128] if SMOKE else [28, 64, 128, 224]

NAMA_ID = {
 "pneumoniamnist": ["normal", "pneumonia"],
 "breastmnist":    ["ganas", "normal/jinak"],
 "octmnist":       ["CNV", "DME", "drusen", "normal"],
 "retinamnist":    ["DR 0", "DR 1", "DR 2", "DR 3", "DR 4"],
 "bloodmnist":     ["basofil", "eosinofil", "eritroblas", "granulosit imatur", "limfosit", "monosit", "neutrofil", "trombosit"],
 "organamnist":    ["kandung kemih", "femur kiri", "femur kanan", "jantung", "ginjal kiri", "ginjal kanan", "hati", "paru kiri", "paru kanan", "pankreas", "limpa"],
}

_SUDAH_CEK = set()
def muat(flag, size, split):
    fn = berkas_medmnist(flag, size)
    if (flag, size) not in _SUDAH_CEK:
        pastikan_dataset("medmnist", [fn]); _SUDAH_CEK.add((flag, size))
    kelas = getattr(medmnist, INFO[flag]["python_class"])
    ds = kelas(split=split, download=False, size=size, root=str(DATA))
    return ds.imgs, ds.labels.reshape(-1)

DATASET = {}
with catat_waktu("2_pastikan_dan_muat_data"):
    for flag, mod, size in MODALITAS:
        assert INFO[flag]["license"] == "CC BY 4.0", flag
        assert len(NAMA_ID[flag]) == len(INFO[flag]["label"]), flag
        DATASET[(flag, size)] = {s: muat(flag, size, s) for s in ("train", "val", "test")}
    for size in RES_PNEU:
        if ("pneumoniamnist", size) not in DATASET:
            DATASET[("pneumoniamnist", size)] = {s: muat("pneumoniamnist", size, s) for s in ("train", "val", "test")}
    if ("pneumoniamnist", 64) not in DATASET:          # dipakai CNN
        DATASET[("pneumoniamnist", 64)] = {s: muat("pneumoniamnist", 64, s) for s in ("train", "val", "test")}

baris = []
for flag, mod, size in MODALITAS:
    d = DATASET[(flag, size)]
    baris.append({"dataset": INFO[flag]["python_class"], "modalitas": mod, "tugas": INFO[flag]["task"],
                  "kelas": len(INFO[flag]["label"]), "latih": len(d["train"][1]), "validasi": len(d["val"][1]),
                  "uji": len(d["test"][1]), "bentuk_citra": "x".join(map(str, d["train"][0].shape[1:])),
                  "lisensi": INFO[flag]["license"]})
TABEL_DATASET = pd.DataFrame(baris)
TABEL_DATASET.to_csv(OUT / "tabel_1_1_dataset.csv", index=False)
TABEL_DATASET
""")

md("## 1.3 Galeri enam modalitas (Gambar 1.1)")
code(r"""
def ambil_contoh(y, n=6, rng=np.random.default_rng(SEED)):
    # satu contoh per kelas lebih dulu, lalu acak sisanya
    pilih = []
    for k in np.unique(y):
        pilih.append(int(rng.choice(np.where(y == k)[0])))
        if len(pilih) == n: break
    sisa = np.setdiff1d(np.arange(len(y)), pilih)
    pilih += list(rng.choice(sisa, n - len(pilih), replace=False))
    return pilih

with catat_waktu("3_galeri_modalitas"):
    fig, ax = plt.subplots(len(MODALITAS), 6, figsize=(10.5, 1.95 * len(MODALITAS)))
    for i, (flag, mod, size) in enumerate(MODALITAS):
        X, y = DATASET[(flag, size)]["test"]
        for j, idx in enumerate(ambil_contoh(y)):
            a = ax[i, j]; img = X[idx]
            a.imshow(img, cmap="gray" if img.ndim == 2 else None)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(NAMA_ID[flag][y[idx]], fontsize=8)
            if j == 0: a.set_ylabel(mod, fontsize=8.5)
    fig.suptitle("Contoh citra uji MedMNIST v2 (CC BY 4.0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gambar_1_1_galeri_modalitas.png"); plt.show()
""")

md("## 1.4 Perbandingan resolusi PneumoniaMNIST (Gambar 1.2)")
code(r"""
with catat_waktu("4_resolusi"):
    Xs = {s: DATASET[("pneumoniamnist", s)]["test"] for s in RES_PNEU}
    y_t = Xs[RES_PNEU[0]][1]
    idx = [int(np.where(y_t == 0)[0][3]), int(np.where(y_t == 1)[0][3])]
    fig, ax = plt.subplots(2, len(RES_PNEU), figsize=(2.6 * len(RES_PNEU), 5.4))
    for r, i in enumerate(idx):
        for c, s in enumerate(RES_PNEU):
            ax[r, c].imshow(Xs[s][0][i], cmap="gray"); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{s}×{s} px" + (" — " + NAMA_ID['pneumoniamnist'][y_t[i]] if c == 0 else ""), fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_2_resolusi_pneumonia.png"); plt.show()
    pd.DataFrame([{"ukuran_px": s, "MB_npz": round(sum(p.stat().st_size for p in DATA.glob(f"pneumoniamnist{'' if s==28 else '_'+str(s)}.npz"))/1e6, 1)} for s in RES_PNEU]).to_csv(OUT / "tabel_1_2_ukuran_resolusi.csv", index=False)
""")

md("## 1.5 Distribusi kelas (Gambar 1.3, Tabel 1.3)")
code(r"""
with catat_waktu("5_distribusi_kelas"):
    rows = []
    for flag, mod, size in MODALITAS:
        for split in ("train", "val", "test"):
            y = DATASET[(flag, size)][split][1]
            for k, nama in enumerate(NAMA_ID[flag]):
                rows.append({"dataset": INFO[flag]["python_class"], "split": split, "kelas": nama, "jumlah": int((y == k).sum())})
    DIST = pd.DataFrame(rows); DIST.to_csv(OUT / "tabel_1_3_distribusi_kelas.csv", index=False)
    rasio = (DIST[DIST.split == "train"].groupby("dataset")["jumlah"].agg(lambda s: s.max() / max(s.min(), 1)).rename("rasio_mayoritas_minoritas"))
    rasio.round(1).to_csv(OUT / "tabel_1_3b_rasio_ketidakseimbangan.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    p = DIST[DIST.dataset == "PneumoniaMNIST"].pivot(index="split", columns="kelas", values="jumlah").loc[["train", "val", "test"]]
    p.index = ["latih", "validasi", "uji"]
    p.plot.bar(ax=ax[0], color=["#0B3D5C", "#F28C28"], rot=0); ax[0].set_title("PneumoniaMNIST per split"); ax[0].set_ylabel("jumlah citra"); ax[0].set_xlabel("")
    for c in ax[0].containers: ax[0].bar_label(c, fontsize=8)
    rasio.sort_values().plot.barh(ax=ax[1], color="#0B3D5C"); ax[1].set_title("Rasio kelas terbanyak : tersedikit (data latih)"); ax[1].set_xlabel("rasio"); ax[1].set_ylabel(""); ax[1].set_xlim(0, rasio.max() * 1.12)
    for c in ax[1].containers: ax[1].bar_label(c, fmt="%.1f", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_3_distribusi_kelas.png"); plt.show()
print(rasio.round(1))
""")

md(r"""
## 1.6 Baseline A — regresi logistik pada piksel (28 px)
Setiap citra 28×28 diratakan menjadi 784 angka. Ambang keputusan 0,5.
""")
code(r"""
def metrik(y, p, thr=0.5, n_boot=1000, seed=SEED):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: aucs.append(roc_auc_score(y[i], p[i]))
    return {"n": int(len(y)), "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "akurasi": (tp + tn) / len(y), "sensitivitas": tp / (tp + fn), "spesifisitas": tn / (tn + fp),
            "PPV": tp / (tp + fp) if tp + fp else float("nan"), "NPV": tn / (tn + fn) if tn + fn else float("nan"),
            "AUC": roc_auc_score(y, p), "AUC_CI95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}

with catat_waktu("6_baseline_regresi_logistik"):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [DATASET[("pneumoniamnist", 28)][s] for s in ("train", "val", "test")]
    f = lambda X: X.reshape(len(X), -1).astype(np.float32) / 255.0
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(f(Xtr), ytr)
    WAKTU["6a_latih_regresi_logistik"] = round(time.perf_counter() - t0, 2)
    p_lr = lr.predict_proba(f(Xte))[:, 1]
    M_LR = metrik(yte, p_lr)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_LR.items()}, indent=1))
""")

md(r"""
## 1.7 Baseline B — CNN kecil (64 px, GPU)
Empat blok konvolusi, 10 epoch, Adam, kerugian *binary cross-entropy*. Model terbaik dipilih dari **AUC validasi** (bukan data uji).
Tidak ada *flip* horizontal: pada foto toraks, letak jantung kiri–kanan bermakna klinis.
""")
code(r"""
class CNNKecil(nn.Module):
    def __init__(self):
        super().__init__()
        def blok(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.fitur = nn.Sequential(blok(1, 16), blok(16, 32), blok(32, 64), blok(64, 128))
        self.kepala = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.kepala(self.fitur(x)).squeeze(1)

def ke_tensor(X): return torch.tensor(X, dtype=torch.float32).unsqueeze(1).div(255.0).sub(0.5).div(0.5)

@torch.no_grad()
def prediksi(model, X, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs): out.append(torch.sigmoid(model(X[i:i+bs].to(PERANGKAT))).cpu())
    return torch.cat(out).numpy()

EPOCH = 2 if SMOKE else 10
UKURAN_CNN = 28 if SMOKE else 64
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
(Xtr, ytr), (Xva, yva), (Xte, yte) = [DATASET[("pneumoniamnist", UKURAN_CNN)][s] for s in ("train", "val", "test")]
Ttr, Tva, Tte = ke_tensor(Xtr), ke_tensor(Xva), ke_tensor(Xte)
Ytr = torch.tensor(ytr, dtype=torch.float32)
model = CNNKecil().to(PERANGKAT)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_param = sum(p.numel() for p in model.parameters())
print("parameter:", n_param, "| perangkat:", PERANGKAT)

RIWAYAT, terbaik = [], (-1, None, None)
with catat_waktu("7_latih_cnn_total"):
    for ep in range(1, EPOCH + 1):
        t0 = time.perf_counter(); model.train(); urut = torch.randperm(len(Ttr)); tot = 0.0
        for i in range(0, len(Ttr), 128):
            b = urut[i:i+128]; x, yb = Ttr[b].to(PERANGKAT), Ytr[b].to(PERANGKAT)
            loss = F.binary_cross_entropy_with_logits(model(x), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        auc_va = roc_auc_score(yva, prediksi(model, Tva))
        dt = time.perf_counter() - t0
        RIWAYAT.append({"epoch": ep, "loss_latih": tot / len(Ttr), "auc_validasi": auc_va, "detik": dt})
        print(f"epoch {ep:2d} | loss {tot/len(Ttr):.4f} | AUC val {auc_va:.4f} | {dt:.1f} s")
        if auc_va > terbaik[0]: terbaik = (auc_va, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
model.load_state_dict(terbaik[2])
p_cnn = prediksi(model, Tte)
M_CNN = metrik(yte, p_cnn)
pd.DataFrame(RIWAYAT).to_csv(OUT / "tabel_1_4_riwayat_latih_cnn.csv", index=False)
torch.save(model.state_dict(), OUT / "cnn_kecil_pneumonia_terbaik.pt")
print("epoch terbaik (AUC validasi):", terbaik[1])
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_CNN.items()}, indent=1))
""")

md("## 1.8 Perbandingan baseline: tabel metrik, kurva belajar, ROC, matriks kebingungan (Tabel 1.5, Gambar 1.4–1.6)")
code(r"""
with catat_waktu("8_gambar_evaluasi"):
    TABEL_METRIK = pd.DataFrame([{"model": "Regresi logistik (28 px)", **M_LR}, {"model": f"CNN kecil ({UKURAN_CNN} px)", **M_CNN}])
    TABEL_METRIK.to_csv(OUT / "tabel_1_5_metrik_baseline.csv", index=False)

    R = pd.DataFrame(RIWAYAT)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(R.epoch, R.loss_latih, "o-", color="#0B3D5C"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss latih"); ax[0].set_title("Kerugian data latih")
    ax[1].plot(R.epoch, R.auc_validasi, "o-", color="#F28C28"); ax[1].axvline(terbaik[1], ls="--", c="gray"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("AUC"); ax[1].set_title("AUC data validasi (garis putus = model terpilih)")
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_4_kurva_belajar_cnn.png"); plt.show()

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for nama, p, c in [("Regresi logistik", p_lr, "#0B3D5C"), ("CNN kecil", p_cnn, "#F28C28")]:
        fpr, tpr, _ = roc_curve(yte, p); ax.plot(fpr, tpr, color=c, lw=2, label=f"{nama} (AUC {roc_auc_score(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1); ax.set_xlabel("1 − spesifisitas"); ax.set_ylabel("sensitivitas"); ax.set_title("Kurva ROC — PneumoniaMNIST (uji)"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_5_roc.png"); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for a, (nama, M) in zip(ax, [("Regresi logistik", M_LR), ("CNN kecil", M_CNN)]):
        cm = np.array([[M["TN"], M["FP"]], [M["FN"], M["TP"]]])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["prediksi normal", "prediksi pneumonia"]); a.set_yticks([0, 1], ["normal", "pneumonia"]); a.set_title(nama)
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_6_matriks_kebingungan.png"); plt.show()
TABEL_METRIK.round(3)
""")

md("## 1.9 Contoh salah klasifikasi CNN (Gambar 1.7)")
code(r"""
with catat_waktu("9_salah_klasifikasi"):
    yhat = (p_cnn >= 0.5).astype(int)
    FPi = np.where((yhat == 1) & (yte == 0))[0]; FNi = np.where((yhat == 0) & (yte == 1))[0]
    FPi = FPi[np.argsort(-p_cnn[FPi])][:4]; FNi = FNi[np.argsort(p_cnn[FNi])][:4]
    fig, ax = plt.subplots(2, 4, figsize=(9, 4.8))
    for r, (judul, ids) in enumerate([("Positif palsu (normal → pneumonia)", FPi), ("Negatif palsu (pneumonia → normal)", FNi)]):
        for c in range(4):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            if c < len(ids): a.imshow(Xte[ids[c]], cmap="gray"); a.set_title(f"p = {p_cnn[ids[c]]:.2f}", fontsize=9)
            if c == 0: a.set_ylabel(judul, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "gambar_1_7_salah_klasifikasi.png"); plt.show()
""")

md("## 1.10 Ringkasan, simpan, dan zip")
code(r"""
def bersih(o):
    if isinstance(o, dict): return {k: bersih(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [bersih(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    return o
WAKTU["total_notebook"] = round(time.perf_counter() - T_MULAI, 2)
RINGKASAN = bersih({"smoke": SMOKE, "versi": VERSI, "waktu_detik": WAKTU,
    "dataset": TABEL_DATASET.to_dict(orient="records"),
    "baseline_regresi_logistik_28px": M_LR, "baseline_cnn": {"ukuran_px": UKURAN_CNN, "epoch": EPOCH, "epoch_terbaik": terbaik[1], "parameter": n_param, **M_CNN},
    "riwayat_cnn": RIWAYAT})
(OUT / "ringkasan_bab01.json").write_text(json.dumps(RINGKASAN, indent=1, ensure_ascii=False))
pd.Series(WAKTU, name="detik").to_csv(OUT / "waktu_sel.csv")
arsip = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("Berkas hasil:"); [print(" -", p.name, f"{p.stat().st_size/1e3:.0f} kB") for p in sorted(OUT.iterdir())]
print("ZIP:", arsip, f"{Path(arsip).stat().st_size/1e6:.2f} MB")
print(json.dumps(WAKTU, indent=1))
try:
    from google.colab import files  # unduh otomatis bila di Colab
    files.download(arsip)
except Exception:
    pass
""")

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": "bab01.ipynb"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
p = Path(__file__).with_name("bab01.ipynb")
nbf.write(nb, p); nbf.validate(nb); print("tulis", p, len(C), "sel")
