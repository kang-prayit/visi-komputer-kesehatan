"""Membangun notebook Colab Bab 10 (notebooks/bab10.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 10 — Citra 3D Ringan: CT Organ (OrganMNIST3D) dan Nodul Paru (NoduleMNIST3D)
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi:
1. **OrganMNIST3D** (11 organ dari CT abdomen; sumber LiTS) dan **NoduleMNIST3D** (nodul paru jinak vs ganas; sumber LIDC-IDRI),
   keduanya MedMNIST v2, CC BY 4.0, volume 28×28×28 voksel.
2. Tiga cara memakai volume: **CNN 3D** dari nol; **2.5D** (tiga irisan ortogonal tengah sebagai 3 kanal → ResNet-18 pra-latih);
   **2D per irisan** (ResNet-18 pada setiap irisan aksial, probabilitas dirata-rata per volume). 2 seed.
3. Biaya komputasi: jumlah parameter, detik/epoch, **memori GPU puncak**, dan waktu inferensi per volume.
4. Skala resolusi: CNN 3D pada OrganMNIST3D **64³** vs 28³ (memori dan waktu).

Hasil → `/content/hasil_bab10/` → `/content/hasil_bab10.zip`. Hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 10.0 Lingkungan")
code(sel_lingkungan("10"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nfrom sklearn.metrics import roc_auc_score")

md(r"""
## 10.1 Data volume
Catatan anatomi: OrganMNIST3D membedakan organ **kanan/kiri** (ginjal, femur, paru), sehingga augmentasi flip kiri–kanan **tidak boleh** dipakai
untuk dataset ini. Untuk NoduleMNIST3D, flip pada ketiga sumbu aman.
""")
code(r"""
DS = {"organmnist3d": {"k": 11, "flip": False}, "nodulemnist3d": {"k": 2, "flip": True}}
def muat3d(flag, uk):
    assert INFO[flag]["license"] == "CC BY 4.0"
    d = npz_medmnist(flag, uk)
    out = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE: out = {s: (x[:120], y[:120]) for s, (x, y) in out.items()}
    return out
with catat_waktu("2_muat_data"):
    D3 = {f: muat3d(f, 28) for f in DS}
T101 = pd.DataFrame([{"dataset": f, "split": s, "n": len(y), "bentuk_volume": str(x.shape[1:]), **{f"kelas_{k}": int((y == k).sum()) for k in range(DS[f]["k"])}}
                     for f, d in D3.items() for s, (x, y) in d.items()])
T101.to_csv(OUT / "tabel_10_1_data.csv", index=False)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for r_, f in enumerate(DS):
    v = D3[f]["train"][0][0]; lbl = INFO[f]["label"][str(D3[f]["train"][1][0])]
    ax[r_, 0].imshow(v[14], cmap="gray"); ax[r_, 0].set_title(f"{f}: aksial (z=14)\nlabel: {lbl}", fontsize=8)
    ax[r_, 1].imshow(v[:, 14], cmap="gray"); ax[r_, 1].set_title("koronal (y=14)", fontsize=8)
    ax[r_, 2].imshow(v[:, :, 14], cmap="gray"); ax[r_, 2].set_title("sagital (x=14)", fontsize=8)
    ax[r_, 3].imshow(v.max(0), cmap="gray"); ax[r_, 3].set_title("proyeksi intensitas maksimum", fontsize=8)
    for a in ax[r_]: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_10_1_irisan_ortogonal.png", bbox_inches="tight"); plt.show()
T101
""")

md(r"""
## 10.2 Tiga arsitektur
- **CNN 3D**: 4 blok Conv3d–BN–ReLU (32→64→128→256) dengan *max-pooling* 3D, *global average pooling*, dropout, linear.
- **2.5D**: ambil irisan aksial, koronal, sagital tengah → tensor 3 kanal 2D → ResNet-18 pra-latih (diperbesar ke 112 px).
- **2D per irisan**: setiap irisan aksial menjadi satu sampel berlabel sama dengan volumenya; prediksi volume = rerata softmax 28 irisan.
Augmentasi acak (flip sumbu yang diizinkan, intensitas ±10 %) dilakukan di dalam modul `Aug3D` hanya saat `model.train()`.
""")
code(r"""
class Aug3D(nn.Module):
    def __init__(self, flip): super().__init__(); self.flip = flip
    def forward(self, x):
        if not self.training: return x
        if self.flip:
            for dim in (2, 3, 4):
                f = (torch.rand(x.shape[0], device=x.device) < 0.5).view(-1, 1, 1, 1, 1); x = torch.where(f, x.flip(dim), x)
        return (x * (1 + (torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * 2 - 1) * 0.1)).clamp(0, 1)
def b3(ci, co): return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.ReLU(inplace=True))
class CNN3D(nn.Module):
    def __init__(self, k, flip):
        super().__init__(); self.aug = Aug3D(flip)
        self.fitur = nn.Sequential(b3(1, 32), nn.MaxPool3d(2), b3(32, 64), nn.MaxPool3d(2), b3(64, 128), nn.MaxPool3d(2), b3(128, 256), nn.AdaptiveAvgPool3d(1))
        self.kepala = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, k))
    def forward(self, x): return self.kepala(self.fitur(self.aug(x)))
class Ortho25D(ResNet18):            # turunan ResNet18 agar tahap "beku" tetap mengenali kepala net.fc
    def __init__(self, k, flip):
        super().__init__(k, True, uk=112); self.aug = Aug3D(flip)
    def forward(self, v):
        v = self.aug(v)[:, 0]; c = v.shape[1] // 2
        return super().forward(torch.stack([v[:, c], v[:, :, c], v[:, :, :, c]], 1))
def ke_vol(x): return torch.as_tensor(x)[:, None].contiguous().to(DEV)          # (N,1,D,H,W) uint8
for nm, m in [("CNN 3D", CNN3D(11, False)), ("2.5D ResNet-18", Ortho25D(11, False)), ("2D per irisan ResNet-18", ResNet18(11, False, uk=56))]:
    print(f"{nm}: {n_param(m)/1e6:.2f} juta parameter")
""")

md(r"""
## 10.3 Pelatihan dan evaluasi (2 seed)
Skor *early stopping*: akurasi validasi (OrganMNIST3D) dan AUC validasi (NoduleMNIST3D). Metrik uji: akurasi dan AUC
(makro *one-vs-rest* untuk 11 kelas).
""")
code(r"""
SEEDS = [0] if SMOKE else [0, 1]
def skor(f):
    if DS[f]["k"] == 2: return lambda lg, y: roc_auc_score(y, torch.softmax(lg, 1)[:, 1].cpu().numpy()) if len(np.unique(y)) > 1 else 0.0
    return lambda lg, y: float((lg.argmax(1).cpu().numpy() == y).mean())
def metrik(f, prob, y):
    auc = roc_auc_score(y, prob[:, 1]) if DS[f]["k"] == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=range(DS[f]["k"]))
    return {"akurasi_uji": float((prob.argmax(1) == y).mean()), "AUC_uji": float(auc)}
def mulai_mem():
    if DEV.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
def mem_puncak(): return torch.cuda.max_memory_allocated() / 2 ** 20 if DEV.type == "cuda" else np.nan
def waktu_inferensi(m, X, bs=64):
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); logit_semua(m, X, bs=bs)
    if DEV.type == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(X) * 1000
HASIL, LOG = [], {}
for f, cfg in DS.items():
    k = cfg["k"]; (xtr, ytr), (xv, yv), (xte, yte) = D3[f]["train"], D3[f]["val"], D3[f]["test"]
    V = {"train": ke_vol(xtr), "val": ke_vol(xv), "test": ke_vol(xte)}
    # 2D per irisan aksial
    S = {s: torch.as_tensor(x.reshape(-1, 28, 28))[:, None].contiguous().to(DEV) for s, x in (("train", xtr), ("val", xv), ("test", xte))}
    YS = {s: np.repeat(y, 28) for s, y in (("train", ytr), ("val", yv), ("test", yte))}
    aug2d = dict(flip_h=cfg["flip"], flip_v=cfg["flip"], kecerahan=0.1, kontras=0.1)
    for s in SEEDS:
        for nm in ("CNN 3D", "2.5D (ortogonal) ResNet-18", "2D per irisan ResNet-18"):
            mulai_mem()
            with catat_waktu(f"{f}_{nm[:6]}_s{s}") as _:
                t0 = time.perf_counter()
                if nm == "CNN 3D":
                    m, lg, ep = latih(f"{f} | {nm} | s{s}", CNN3D(k, cfg["flip"]), V["train"], ytr, V["val"], yv, F.cross_entropy, skor(f), maks_epoch=40, patience=8, lr=1e-3, bs=32, seed=s)
                    prob = torch.softmax(logit_semua(m, V["test"], bs=64), 1).cpu().numpy(); ms = waktu_inferensi(m, V["test"])
                elif nm.startswith("2.5D"):
                    m, lg, ep = latih(f"{f} | {nm} | s{s}", Ortho25D(k, cfg["flip"]), V["train"], ytr, V["val"], yv, F.cross_entropy, skor(f),
                                      tahap=[(2, 1e-3, True), (25, 1e-4, False)], patience=6, bs=64, seed=s)
                    prob = torch.softmax(logit_semua(m, V["test"], bs=64), 1).cpu().numpy(); ms = waktu_inferensi(m, V["test"])
                else:
                    m, lg, ep = latih(f"{f} | {nm} | s{s}", ResNet18(k, True, uk=56), S["train"], YS["train"], S["val"], YS["val"], F.cross_entropy, skor(f),
                                      tahap=[(1, 1e-3, True), (10, 1e-4, False)], patience=3, aug=aug2d, bs=256, seed=s)
                    ps = torch.softmax(logit_semua(m, S["test"], bs=1024), 1).cpu().numpy(); prob = ps.reshape(-1, 28, k).mean(1); ms = waktu_inferensi(m, S["test"], bs=1024) * 28
                dt = time.perf_counter() - t0
            LOG[(f, nm, s)] = lg
            HASIL.append({"dataset": f, "model": nm, "seed": s, "n_param_juta": n_param(m) / 1e6, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()),
                          "detik_per_epoch": float(lg.detik_epoch.mean()), "detik_total_latih": dt, "memori_GPU_puncak_MB": mem_puncak(),
                          "ms_inferensi_per_volume": ms, "skor_val_terbaik": float(lg.skor_val.max()), **metrik(f, prob, yte)})
            print(HASIL[-1]); del m
T102 = pd.DataFrame(HASIL); T102.to_csv(OUT / "tabel_10_2_per_seed.csv", index=False); pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
T103 = T102.groupby(["dataset", "model"])[["akurasi_uji", "AUC_uji", "n_param_juta", "detik_per_epoch", "detik_total_latih", "memori_GPU_puncak_MB", "ms_inferensi_per_volume"]].agg(["mean", "std"])
T103.columns = ["_".join(c) for c in T103.columns]; T103.to_csv(OUT / "tabel_10_3_ringkas.csv"); T103
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, f in zip(ax, DS):
    d = T102[T102.dataset == f]; met = "akurasi_uji" if f == "organmnist3d" else "AUC_uji"
    g = d.groupby("model")[met].agg(["mean", "std"]); a.bar(range(len(g)), g["mean"], yerr=g["std"].fillna(0), capsize=4)
    a.set_xticks(range(len(g))); a.set_xticklabels([s.replace(" ResNet-18", "\nResNet-18") for s in g.index], fontsize=7); a.set_ylabel(met); a.set_title(f)
    a.set_ylim(max(0, g["mean"].min() - 0.15), 1)
plt.tight_layout(); plt.savefig(OUT / "gambar_10_2_kinerja.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 10.4 Skala resolusi: CNN 3D pada 64³ vs 28³ (OrganMNIST3D)
Volume 64³ memuat ±12 kali lebih banyak voksel daripada 28³. Kita ukur memori GPU puncak, detik/epoch, dan akurasi (1 seed).
Berkas `organmnist3d_64.npz` (±361 MB) diunduh sekali ke Drive.
""")
code(r"""
UK_BESAR = 28 if SMOKE else 64
with catat_waktu("3_muat_64"):
    d64 = muat3d("organmnist3d", UK_BESAR)
V64 = {s: ke_vol(x) for s, (x, _) in d64.items()}; Y64 = {s: y for s, (_, y) in d64.items()}
baris = [r for r in HASIL if r["dataset"] == "organmnist3d" and r["model"] == "CNN 3D" and r["seed"] == 0]
mulai_mem()
with catat_waktu("latih_cnn3d_64"):
    t0 = time.perf_counter()
    m, lg, ep = latih(f"organmnist3d | CNN 3D {UK_BESAR}³", CNN3D(11, False), V64["train"], Y64["train"], V64["val"], Y64["val"], F.cross_entropy, skor("organmnist3d"),
                      maks_epoch=30, patience=6, lr=1e-3, bs=16, seed=0)
    dt = time.perf_counter() - t0
prob = torch.softmax(logit_semua(m, V64["test"], bs=16), 1).cpu().numpy()
baris.append({"dataset": "organmnist3d", "model": f"CNN 3D {UK_BESAR}³", "seed": 0, "n_param_juta": n_param(m) / 1e6, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()),
              "detik_per_epoch": float(lg.detik_epoch.mean()), "detik_total_latih": dt, "memori_GPU_puncak_MB": mem_puncak(),
              "ms_inferensi_per_volume": waktu_inferensi(m, V64["test"], bs=16), **metrik("organmnist3d", prob, Y64["test"])})
baris[0] = {**baris[0], "model": "CNN 3D 28³"}
T104 = pd.DataFrame(baris)[["model", "n_param_juta", "detik_per_epoch", "detik_total_latih", "memori_GPU_puncak_MB", "ms_inferensi_per_volume", "akurasi_uji", "AUC_uji"]]
T104.to_csv(OUT / "tabel_10_4_skala_resolusi.csv", index=False); T104
""")

md("## 10.5 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("10", 'dict(data=T101.to_dict("records"), per_seed=T102.to_dict("records"), skala_resolusi=T104.to_dict("records"))'))
simpan(C, "bab10.ipynb", Path(__file__).parent / "bab10.ipynb")
