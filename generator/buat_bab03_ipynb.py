"""Membangun notebook Colab Bab 3 (notebooks/bab03.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 3 — Citra Digital, Praproses, dan CNN: dari Nol vs *Transfer Learning*
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4** (*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

Isi:
1. Citra sebagai larik: kanal, histogram, statistik normalisasi (BloodMNIST 64 px, 8 kelas sel darah).
2. Praproses: CLAHE dan ekualisasi histogram pada X-ray (PneumoniaMNIST), koreksi warna *gray-world* pada citra mikroskopi.
3. Augmentasi yang masuk akal secara klinis (flip/rotasi 90° boleh untuk sel darah; flip kiri–kanan **tidak** dianjurkan untuk X-ray dada).
4. Pelatihan dan perbandingan: CNN kecil dari nol, ResNet-18 bobot acak, ResNet-18 pra-latih ImageNet (dua tahap: beku lalu *fine-tune*),
   serta CNN tanpa augmentasi & tanpa *early stopping* untuk memperlihatkan *overfitting*.
5. *Learning curve*, *early stopping* (patience 5, maks. 30 epoch), matriks kebingungan, recall per kelas, efisiensi data (10 % data latih).

Semua hasil → `/content/hasil_bab03/` → `/content/hasil_bab03.zip`.

> Data: MedMNIST v2 (CC BY 4.0) — Yang dkk. (2023), *Scientific Data* 10, 41, https://doi.org/10.1038/s41597-022-01721-8.
> BloodMNIST berasal dari Acevedo dkk. (2020), *Data in Brief* 30, 105474. Bobot ImageNet ResNet-18 diunduh otomatis oleh `torchvision` dari `download.pytorch.org`.
> Model hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")

md(r"""
## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`
""")
code(kode_sel_standar())

md("## 3.0 Lingkungan dan pencatat waktu")
code(sel_lingkungan("03"))
code(r"""
with catat_waktu("1_pip_install"):
    jalankan('pip install -q "medmnist==3.0.2"')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib, matplotlib.ticker; import matplotlib.pyplot as plt
import torchvision, skimage, sklearn, medmnist
from skimage import exposure
from PIL import Image
from medmnist import INFO
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); AMP = DEV.type == "cuda"
VERSI = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
         "numpy": np.__version__, "scikit-image": skimage.__version__, "scikit-learn": sklearn.__version__,
         "medmnist": medmnist.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
print(json.dumps(VERSI, indent=1))
""" + "\n" + SEED_FN)

md("## 3.1 Data dan citra sebagai larik")
code(r"""
UK = 28 if SMOKE else 64            # buku: 64 px; uji cepat penulis memakai 28 px
NAMA = ["basofil", "eosinofil", "eritroblas", "granulosit imatur", "limfosit", "monosit", "neutrofil", "trombosit"]
def npz_medmnist(flag, size):
    fn = berkas_medmnist(flag, size); folder = pastikan_dataset("medmnist", [fn]); return np.load(folder / fn)
with catat_waktu("2_muat_data"):
    assert INFO["bloodmnist"]["license"] == "CC BY 4.0"
    b = npz_medmnist("bloodmnist", UK)
    D = {s: (b[f"{s}_images"], b[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0)
        D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), {"train": 400, "val": 160, "test": 160}[s], replace=False)]}
    pn = npz_medmnist("pneumoniamnist", 128)
T31 = pd.DataFrame({s: np.bincount(D[s][1], minlength=8) for s in D}, index=NAMA); T31.loc["total"] = T31.sum()
T31.to_csv(OUT / "tabel_3_1_distribusi_kelas.csv"); print({s: D[s][0].shape for s in D}, D["train"][0].dtype); T31
""")
code(r"""
xtr = D["train"][0]
st = pd.DataFrame({"kanal": ["R", "G", "B"], "rerata_0_255": xtr.reshape(-1, 3).mean(0), "simpangan_baku_0_255": xtr.reshape(-1, 3).std(0)})
st["rerata_0_1"] = st.rerata_0_255 / 255; st["sb_0_1"] = st.simpangan_baku_0_255 / 255
st.to_csv(OUT / "tabel_3_2_statistik_kanal.csv", index=False)
contoh = xtr[np.where(D["train"][1] == 6)[0][0]]            # satu neutrofil
fig, ax = plt.subplots(1, 5, figsize=(13, 2.8))
ax[0].imshow(contoh); ax[0].set_title(f"RGB {contoh.shape}, {contoh.dtype}")
for k, (nm, cm) in enumerate([("R", "Reds"), ("G", "Greens"), ("B", "Blues")]):
    ax[k + 1].imshow(contoh[..., k], cmap=cm, vmin=0, vmax=255); ax[k + 1].set_title(f"kanal {nm}: [{contoh[..., k].min()}, {contoh[..., k].max()}]")
for k, c in enumerate("rgb"): ax[4].hist(contoh[..., k].ravel(), bins=32, range=(0, 255), color=c, alpha=0.5)
ax[4].set_title("histogram per kanal"); [a.axis("off") for a in ax[:4]]
plt.tight_layout(); plt.savefig(OUT / "gambar_3_1_larik_citra.png", bbox_inches="tight"); plt.show()
fig, ax = plt.subplots(2, 8, figsize=(12, 3.3)); r = np.random.default_rng(SEED)
for c in range(8):
    for rr in range(2):
        i = r.choice(np.where(D["train"][1] == c)[0]); ax[rr, c].imshow(xtr[i]); ax[rr, c].axis("off")
    ax[0, c].set_title(NAMA[c], fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_3_2_galeri_kelas.png", bbox_inches="tight"); plt.show()
st
""")

md("## 3.2 Praproses: CLAHE, ekualisasi histogram, *gray-world*")
code(r"""
xr = pn["test_images"][0].astype(np.float32) / 255
def gray_world(img):
    f = img.astype(np.float32); m = f.reshape(-1, 3).mean(0); return np.clip(f * (m.mean() / m), 0, 255).astype(np.uint8)
fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
for a, (judul, im) in zip(ax[0], [("X-ray asli", xr), ("ekualisasi histogram", exposure.equalize_hist(xr)),
                                   ("CLAHE clip 0,01", exposure.equalize_adapthist(xr, clip_limit=0.01)),
                                   ("CLAHE clip 0,03", exposure.equalize_adapthist(xr, clip_limit=0.03))]):
    a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(judul); a.axis("off")
idx = [np.where(D["train"][1] == c)[0][0] for c in (1, 4)]
for j, i in enumerate(idx):
    ax[1, 2 * j].imshow(xtr[i]); ax[1, 2 * j].set_title(f"{NAMA[D['train'][1][i]]}: asli")
    ax[1, 2 * j + 1].imshow(gray_world(xtr[i])); ax[1, 2 * j + 1].set_title("gray-world")
for a in ax[1]: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_3_3_praproses.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 3.3 Augmentasi yang masuk akal secara klinis
Sel darah pada apusan tidak punya arah "atas" baku, jadi flip dan rotasi 90° aman. Pada X-ray dada, flip kiri–kanan memindahkan
bayangan jantung ke sisi kanan (menyerupai dekstrokardia) sehingga **tidak dianjurkan**; rotasi kecil dan perubahan kecerahan ringan lebih aman.
""")
code(r"""
def augmentasi_gpu(x, gen=None):
    # x: (N,3,H,W) float di [0,1]. Flip H/V acak per sampel, rotasi 90°·k per batch, jitter kecerahan/kontras ±10 %.
    n = x.shape[0]
    fh = torch.rand(n, device=x.device) < 0.5; x = torch.where(fh[:, None, None, None], x.flip(3), x)
    fv = torch.rand(n, device=x.device) < 0.5; x = torch.where(fv[:, None, None, None], x.flip(2), x)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), (2, 3))
    kb = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2; kc = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * kc + m) * kb
    return x.clamp(0, 1)
torch.manual_seed(SEED)
x0 = torch.tensor(xtr[idx[0]]).permute(2, 0, 1)[None].float().div(255).repeat(7, 1, 1, 1)
xa = torch.cat([augmentasi_gpu(x0[k:k + 1]) for k in range(7)])   # per sampel agar rotasinya beragam
fig, ax = plt.subplots(2, 8, figsize=(13, 3.6))
ax[0, 0].imshow(xtr[idx[0]]); ax[0, 0].set_title("asli", fontsize=8)
for k in range(7): ax[0, k + 1].imshow(xa[k].permute(1, 2, 0).numpy()); ax[0, k + 1].set_title(f"augmentasi {k+1}", fontsize=8)
xr8 = pn["test_images"][0]
pilihan = [("asli", xr8, "aman"), ("rotasi +7°", np.array(Image.fromarray(xr8).rotate(7, resample=Image.BILINEAR)), "aman"),
           ("rotasi −7°", np.array(Image.fromarray(xr8).rotate(-7, resample=Image.BILINEAR)), "aman"),
           ("kecerahan +10 %", np.clip(xr8 * 1.1, 0, 255), "aman"), ("kontras −10 %", np.clip((xr8 - xr8.mean()) * 0.9 + xr8.mean(), 0, 255), "aman"),
           ("geser 6 px", np.roll(xr8, 6, axis=1), "aman"), ("flip kiri–kanan", xr8[:, ::-1], "TIDAK dianjurkan"),
           ("flip atas–bawah", xr8[::-1], "TIDAK dianjurkan")]
for k, (j, im, ket) in enumerate(pilihan):
    ax[1, k].imshow(im, cmap="gray", vmin=0, vmax=255); ax[1, k].set_title(f"{j}\n({ket})", fontsize=7, color="red" if "TIDAK" in ket else "black")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_3_4_augmentasi.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 3.4 Model dan *training loop* dengan *early stopping*
- **CNN kecil**: 4 blok Conv-BN-ReLU-MaxPool (32-64-128-256), GAP, dropout; masukan 64 px.
- **ResNet-18** (`torchvision`): masukan diperbesar ke 128 px di GPU dan dinormalisasi dengan statistik ImageNet; kepala diganti 8 kelas.
- Optimasi Adam, *mixed precision* (AMP) di GPU, *early stopping* pada *validation loss* (patience 5, maks. 30 epoch), bobot terbaik dipulihkan.
""")
code(r"""
MEAN_IN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); SD_IN = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
class CNNKecil(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        def blok(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.fitur = nn.Sequential(blok(3, 32), blok(32, 64), blok(64, 128), blok(128, 256))
        self.kepala = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, k))
    def forward(self, x): return self.kepala(self.fitur((x - 0.5) / 0.5))
class ResNet18(nn.Module):
    def __init__(self, pretrain, k=8, uk=128):
        super().__init__(); self.uk = uk
        w = None
        if pretrain:
            try: w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1; self.net = torchvision.models.resnet18(weights=w)
            except Exception as e:
                if not SMOKE: raise
                print("SMOKE: bobot ImageNet tidak dapat diunduh, memakai bobot acak:", e); self.net = torchvision.models.resnet18(weights=None)
        else: self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, k)
    def forward(self, x):
        x = F.interpolate(x, size=(self.uk, self.uk), mode="bilinear", align_corners=False)
        return self.net((x - MEAN_IN.to(x.device)) / SD_IN.to(x.device))

XT = {s: torch.tensor(D[s][0]).permute(0, 3, 1, 2).contiguous() for s in D}       # uint8 di CPU
YT = {s: torch.tensor(D[s][1]) for s in D}
if DEV.type == "cuda": XT = {s: v.to(DEV) for s, v in XT.items()}; YT = {s: v.to(DEV) for s, v in YT.items()}   # data kecil: simpan di GPU
@torch.no_grad()
def evaluasi(model, s, idx=None, bs=1024):
    model.eval(); X, Y = XT[s], YT[s]; n = len(Y); out = []
    for i in range(0, n, bs):
        x = X[i:i + bs].to(DEV).float().div(255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): out.append(model(x).float())
    lg = torch.cat(out); loss = F.cross_entropy(lg, Y.to(DEV)).item()
    return loss, torch.softmax(lg, 1).cpu().numpy()

def latih(nama, model, idx_latih=None, maks_epoch=30, patience=5, aug=True, es=True, lr=1e-3, bs=128, tahap=None, seed=SEED):
    seed_semua(seed); model = model.to(DEV)
    n_all = len(YT["train"]); idx_latih = np.arange(n_all) if idx_latih is None else idx_latih
    if SMOKE: maks_epoch, patience = min(maks_epoch, 3), 1
    scaler = torch.amp.GradScaler("cuda", enabled=AMP); log = []; terbaik = (np.inf, None, -1); ep_global = 0
    for (n_ep, lr_t, beku) in (tahap or [(maks_epoch, lr, False)]):
        for nm, p in model.named_parameters(): p.requires_grad = (not beku) or nm.startswith("net.fc")
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_t); tunggu = 0
        for ep in range(n_ep if not SMOKE else min(n_ep, 3)):
            model.train(); t0 = time.perf_counter(); perm = np.random.permutation(idx_latih); tl, tb, n = 0.0, 0, 0
            for i in range(0, len(perm), bs):
                b = torch.as_tensor(perm[i:i + bs], device=XT["train"].device)
                x = XT["train"][b].to(DEV).float().div(255); y = YT["train"][b].to(DEV)
                if aug: x = augmentasi_gpu(x)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): lg = model(x); loss = F.cross_entropy(lg, y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tl += loss.item() * len(b); tb += (lg.argmax(1) == y).sum().item(); n += len(b)
            if DEV.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            vl, vp = evaluasi(model, "val"); va = accuracy_score(D["val"][1], vp.argmax(1)); ep_global += 1
            log.append({"model": nama, "epoch": ep_global, "tahap_beku": beku, "loss_latih": tl / n, "akurasi_latih": tb / n,
                        "loss_val": vl, "akurasi_val": va, "detik_epoch": dt})
            print(f"{nama} ep{ep_global:02d} loss {tl/n:.4f}/{vl:.4f} acc {tb/n:.4f}/{va:.4f} {dt:.1f}s")
            if vl < terbaik[0]:
                terbaik = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep_global); tunggu = 0
            else:
                tunggu += 1
                if es and not beku and tunggu >= patience: print(f"  early stopping: tidak membaik {patience} epoch"); break
    if es and terbaik[1] is not None: model.load_state_dict(terbaik[1])
    return model, pd.DataFrame(log), (terbaik[2] if es else ep_global)

def metrik_uji(nama, model, ep_terbaik, log, catatan=""):
    _, p = evaluasi(model, "test"); y = D["test"][1]; yh = p.argmax(1)
    try: auc = roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError: auc = float("nan")
    return {"model": nama, "parameter": sum(q.numel() for q in model.parameters()), "epoch_dijalankan": int(log.epoch.max()),
            "epoch_terbaik": int(ep_terbaik), "detik_per_epoch": float(log.detik_epoch.mean()), "detik_total": float(log.detik_epoch.sum()),
            "akurasi": accuracy_score(y, yh), "akurasi_seimbang": balanced_accuracy_score(y, yh), "F1_makro": f1_score(y, yh, average="macro"),
            "AUC_makro_OvR": auc, "catatan": catatan}, p
""")

md("## 3.5 Eksperimen utama")
code(r"""
HASIL, LOG, PROB, MODEL = [], [], {}, {}
def jalankan_eksperimen(nama, buat_model, catatan, **kw):
    with catat_waktu(f"latih_{nama}"):
        m, lg, ep = latih(nama, buat_model(), **kw)
    r, p = metrik_uji(nama, m, ep, lg, catatan); HASIL.append(r); LOG.append(lg); PROB[nama] = p; MODEL[nama] = m
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
jalankan_eksperimen("CNN kecil (dari nol)", CNNKecil, "augmentasi + early stopping")
jalankan_eksperimen("CNN kecil tanpa aug & tanpa ES", CNNKecil, "30 epoch penuh, bobot epoch terakhir", aug=False, es=False)
jalankan_eksperimen("ResNet-18 bobot acak", lambda: ResNet18(False), "augmentasi + early stopping")
jalankan_eksperimen("ResNet-18 pra-latih ImageNet", lambda: ResNet18(True), "tahap 1: 3 epoch beku (hanya fc, lr 1e-3); tahap 2: fine-tune penuh lr 1e-4 + ES",
                    tahap=[(3, 1e-3, True), (30, 1e-4, False)])
T33 = pd.DataFrame(HASIL); T33.to_csv(OUT / "tabel_3_3_perbandingan_model.csv", index=False)
pd.concat(LOG).to_csv(OUT / "log_epoch_semua_model.csv", index=False)
torch.save(MODEL["CNN kecil (dari nol)"].state_dict(), OUT / "cnn_kecil_blood_terbaik.pt")
T33
""")

md("## 3.6 *Learning curve* dan *early stopping*")
code(r"""
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for a, lg in zip(ax.ravel(), LOG):
    nama = lg.model.iloc[0]; r = next(h for h in HASIL if h["model"] == nama)
    a.plot(lg.epoch, lg.loss_latih, "o-", ms=3, label="loss latih"); a.plot(lg.epoch, lg.loss_val, "s-", ms=3, label="loss validasi")
    a2 = a.twinx(); a2.plot(lg.epoch, lg.akurasi_val, "^-", ms=3, color="tab:green", alpha=0.7, label="akurasi validasi"); a2.set_ylim(0, 1)
    a2.set_ylabel("akurasi val", color="tab:green")
    if r["catatan"].startswith("30 epoch") is False: a.axvline(r["epoch_terbaik"], color="red", ls="--", lw=1, label=f"bobot terbaik (ep {r['epoch_terbaik']})")
    if lg.tahap_beku.any(): a.axvspan(0.5, lg[lg.tahap_beku].epoch.max() + 0.5, color="gray", alpha=0.15, label="tahap beku")
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); a.set_title(nama, fontsize=9); a.set_xlabel("epoch"); a.set_ylabel("cross-entropy"); a.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(OUT / "gambar_3_5_learning_curve.png", bbox_inches="tight"); plt.show()
""")

md("## 3.7 Matriks kebingungan dan recall per kelas")
code(r"""
y = D["test"][1]; pilih = ["CNN kecil (dari nol)", "ResNet-18 pra-latih ImageNet"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, nm in zip(ax, pilih):
    cm = confusion_matrix(y, PROB[nm].argmax(1), labels=range(8)); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8): a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.6 else "black")
    a.set_xticks(range(8)); a.set_yticks(range(8)); a.set_xticklabels(NAMA, rotation=45, ha="right", fontsize=7); a.set_yticklabels(NAMA, fontsize=7)
    a.set_xlabel("prediksi"); a.set_ylabel("sebenarnya"); a.set_title(nm, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "gambar_3_6_matriks_kebingungan.png", bbox_inches="tight"); plt.show()
T34 = pd.DataFrame({nm: recall_score(y, PROB[nm].argmax(1), labels=range(8), average=None, zero_division=0) for nm in PROB}, index=NAMA)
T34.insert(0, "n_uji", np.bincount(y, minlength=8)); T34.to_csv(OUT / "tabel_3_4_recall_per_kelas.csv"); T34.round(4)
""")

md("## 3.8 Efisiensi data: hanya 10 % data latih")
code(r"""
from sklearn.model_selection import train_test_split
idx10, _ = train_test_split(np.arange(len(D["train"][1])), train_size=0.10, stratify=D["train"][1], random_state=SEED)
HASIL_10 = []
for nama, bm, kw in [("CNN kecil (dari nol)", CNNKecil, {}),
                     ("ResNet-18 pra-latih ImageNet", lambda: ResNet18(True), {"tahap": [(3, 1e-3, True), (30, 1e-4, False)]})]:
    with catat_waktu(f"latih10_{nama}"):
        m, lg, ep = latih(nama + " [10%]", bm(), idx_latih=idx10, **kw)
    r, _ = metrik_uji(nama, m, ep, lg, "10 % data latih"); r["fraksi_latih"] = 0.10; r["n_latih"] = len(idx10); HASIL_10.append(r); LOG.append(lg)
    full = next(h for h in HASIL if h["model"] == nama); full = dict(full, fraksi_latih=1.0, n_latih=len(D["train"][1])); HASIL_10.append(full)
T35 = pd.DataFrame(HASIL_10)[["model", "fraksi_latih", "n_latih", "epoch_terbaik", "akurasi", "akurasi_seimbang", "F1_makro", "AUC_makro_OvR", "detik_total"]]
T35.to_csv(OUT / "tabel_3_5_efisiensi_data.csv", index=False)
pd.concat(LOG).to_csv(OUT / "log_epoch_semua_model.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 3.4))
for nm, mk in [("CNN kecil (dari nol)", "o-"), ("ResNet-18 pra-latih ImageNet", "s-")]:
    d = T35[T35.model == nm].sort_values("fraksi_latih"); ax.plot(d.n_latih, d.F1_makro, mk, label=nm)
    for _, rr in d.iterrows(): ax.annotate(f"{rr.F1_makro:.3f}", (rr.n_latih, rr.F1_makro), textcoords="offset points", xytext=(4, -10), fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("jumlah citra latih (skala log)"); ax.set_ylabel("F1 makro (uji)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "gambar_3_7_efisiensi_data.png", bbox_inches="tight"); plt.show()
T35
""")

md("## 3.9 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("03", 'dict(ukuran_px=UK, n={s: int(len(D[s][1])) for s in D}, perbandingan=T33.to_dict("records"), efisiensi_data=T35.to_dict("records"), statistik_kanal=st.to_dict("records"))'))

simpan(C, "bab03.ipynb", Path(__file__).parent / "bab03.ipynb")
