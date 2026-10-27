"""Sel-sel pola yang sama untuk notebook bab 02 dst. (mengikuti pola bab01)."""
import nbformat as nbf

def sel_lingkungan(nn):
    return r'''
import os, sys, time, json, platform, subprocess, contextlib, shutil, random
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # perlu untuk algoritme deterministik di GPU
T_MULAI = time.perf_counter()
SMOKE = os.environ.get("BAB@@_SMOKE") == "1"          # uji cepat penulis di luar Colab (CPU, subset kecil); di Colab = False
OUT = Path(os.environ.get("BAB@@_OUT", "/content/hasil_bab@@"))
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

WAKTU = {}
@contextlib.contextmanager
def catat_waktu(nama):
    t0 = time.perf_counter()
    yield
    WAKTU[nama] = round(time.perf_counter() - t0, 2)
    print(f"[waktu] {nama}: {WAKTU[nama]:.1f} s")

def jalankan(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return f"(gagal: {e})"

with catat_waktu("0_info_lingkungan"):
    smi = jalankan("nvidia-smi")
    print(smi if smi.strip() else "nvidia-smi tidak tersedia (tidak ada GPU?). Di Colab: Runtime > Change runtime type > T4 GPU.")
    info_cpu = jalankan("lscpu | grep -E 'Model name|^CPU\\(s\\)'"); info_mem = jalankan("free -h | head -2"); info_disk = jalankan("df -h /content 2>/dev/null || df -h /")
    print(info_cpu, info_mem, info_disk)
    (OUT / "info_lingkungan.txt").write_text(f"Python: {sys.version}\nPlatform: {platform.platform()}\n\n{smi}\n{info_cpu}\n{info_mem}\n{info_disk}")
print("SMOKE =", SMOKE, "| OUT =", OUT)
'''.replace('@@', nn).strip()

SEED_FN = r'''
SEED = 42
def seed_semua(seed=SEED):
    """Seed Python, NumPy, PyTorch (CPU+GPU) dan minta algoritme deterministik."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
seed_semua()
'''.strip()

def sel_akhir(nn, ringkasan_expr):
    return (r'''
def bersih(o):
    if isinstance(o, dict): return {str(k): bersih(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [bersih(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 6)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
WAKTU["total_notebook"] = round(time.perf_counter() - T_MULAI, 2)
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada GPU (CPU)"
RINGKASAN = bersih(dict(smoke=SMOKE, gpu=GPU, versi=VERSI, waktu_detik=WAKTU, **(@@R@@)))
(OUT / "ringkasan_bab@@.json").write_text(json.dumps(RINGKASAN, indent=1, ensure_ascii=False))
pd.Series(WAKTU, name="detik").to_csv(OUT / "waktu_sel.csv")
arsip = shutil.make_archive(str(OUT), "zip", root_dir=OUT)
print("Berkas hasil:"); [print(" -", p.name, f"{p.stat().st_size/1e3:.0f} kB") for p in sorted(OUT.iterdir())]
print("ZIP:", arsip, f"{Path(arsip).stat().st_size/1e6:.2f} MB")
print(json.dumps(WAKTU, indent=1))
print(f"TOTAL WAKTU NOTEBOOK: {WAKTU['total_notebook']:.1f} s ({WAKTU['total_notebook']/60:.1f} menit) | GPU: {GPU}")
try:
    from google.colab import files  # unduh otomatis bila di Colab
    files.download(arsip)
except Exception:
    pass
''').replace('@@R@@', ringkasan_expr).replace('@@', nn).strip()

def simpan(cells, nama, path):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": nama},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    nbf.validate(nb); nbf.write(nb, path); print("tulis", path, len(cells), "sel")

KODE_LATIH = r'''
# ---------- Perangkat latih bersama (sama di notebook bab 04 dst.) ----------
MEAN_IN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); SD_IN = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AMP = DEV.type == "cuda"

def ke_tensor_uint8(X):
    """(N,H,W) atau (N,H,W,C) uint8 -> tensor (N,C,H,W) uint8; disimpan di GPU bila ada (data kecil)."""
    t = torch.as_tensor(X)
    t = t[:, None] if t.ndim == 3 else t.permute(0, 3, 1, 2)
    return t.contiguous().to(DEV)

def augmentasi_gpu(x, flip_h=False, flip_v=False, rot90=False, rot_derajat=0.0, geser=0.0, kecerahan=0.1, kontras=0.1, warna=0.0):
    """x float (N,C,H,W) di [0,1]. Semua acak per sampel kecuali rot90 (per batch)."""
    n = x.shape[0]; d = x.device
    if flip_h: x = torch.where((torch.rand(n, device=d) < 0.5)[:, None, None, None], x.flip(3), x)
    if flip_v: x = torch.where((torch.rand(n, device=d) < 0.5)[:, None, None, None], x.flip(2), x)
    if rot90: x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), (2, 3))
    if rot_derajat or geser:
        a = (torch.rand(n, device=d) * 2 - 1) * rot_derajat * np.pi / 180
        t = (torch.rand(n, 2, device=d) * 2 - 1) * geser * 2
        th = torch.stack([torch.stack([a.cos(), -a.sin(), t[:, 0]], 1), torch.stack([a.sin(), a.cos(), t[:, 1]], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    if warna and x.shape[1] == 3: x = x * (1 + (torch.rand(n, 3, 1, 1, device=d) * 2 - 1) * warna)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * (1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * kontras) + m) * (1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * kecerahan)
    return x.clamp(0, 1)

class ResNet18(nn.Module):
    """ResNet-18 torchvision; masukan 1 atau 3 kanal [0,1], opsional diperbesar ke `uk` piksel, normalisasi ImageNet."""
    def __init__(self, k, pretrain=True, uk=None, p_drop=0.0):
        super().__init__(); self.uk = uk
        w = None
        if pretrain:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            try: self.net = torchvision.models.resnet18(weights=w)
            except Exception as e:
                if not SMOKE: raise
                print("SMOKE: bobot ImageNet gagal diunduh, pakai bobot acak:", e); self.net = torchvision.models.resnet18(weights=None)
        else: self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = nn.Sequential(nn.Dropout(p_drop), nn.Linear(self.net.fc.in_features, k))
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        if self.uk and x.shape[-1] != self.uk: x = F.interpolate(x, size=(self.uk, self.uk), mode="bilinear", align_corners=False)
        return self.net((x - MEAN_IN.to(x.device)) / SD_IN.to(x.device))

@torch.no_grad()
def logit_semua(model, X, bs=512, praproses=None):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(DEV).float().div(255)
        if praproses: x = praproses(x)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): out.append(model(x).float())
    return torch.cat(out)

def latih(nama, model, X, Y, Xv, Yv, fungsi_loss, skor_val, maks_epoch=30, patience=5, aug=None, lr=1e-3, bs=128,
          tahap=None, bobot_sampel=None, praproses=None, seed=SEED, wd=1e-4):
    """Latih dengan AMP; early stopping memantau skor_val(logit_val, y_val) (lebih tinggi lebih baik).
    tahap: daftar (n_epoch, lr, beku_backbone). bobot_sampel: oversampling (sampling dengan pengembalian)."""
    seed_semua(seed); model = model.to(DEV); Yd = torch.as_tensor(Y).to(DEV); Yvd = torch.as_tensor(Yv).to(DEV)
    if SMOKE: maks_epoch, patience = min(maks_epoch, 2), 1
    scaler = torch.amp.GradScaler("cuda", enabled=AMP); log = []; best = (-np.inf, None, 0); ep_g = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for n_ep, lr_t, beku in (tahap or [(maks_epoch, lr, False)]):
        if SMOKE: n_ep = min(n_ep, 2)
        for nm, p in model.named_parameters(): p.requires_grad = (not beku) or nm.startswith("net.fc")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_t, weight_decay=wd); tunggu = 0
        for ep in range(n_ep):
            model.train(); t0 = time.perf_counter()
            if bobot_sampel is not None: perm = torch.multinomial(torch.as_tensor(bobot_sampel, dtype=torch.float), len(Y), replacement=True, generator=g)
            else: perm = torch.randperm(len(Y), generator=g)
            perm = perm.to(X.device); tl = 0.0; n = 0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]; x = X[b].to(DEV).float().div(255)
                if aug: x = augmentasi_gpu(x, **aug)
                if praproses: x = praproses(x)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    lg = model(x)
                loss = fungsi_loss(lg.float(), Yd[b.to(DEV)])
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tl += loss.item() * len(b); n += len(b)
            if DEV.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0; ep_g += 1
            lv = logit_semua(model, Xv, praproses=praproses); vl = fungsi_loss(lv, Yvd).item(); sv = float(skor_val(lv, Yv))
            log.append({"model": nama, "epoch": ep_g, "tahap_beku": beku, "loss_latih": tl / n, "loss_val": vl, "skor_val": sv, "detik_epoch": dt})
            print(f"{nama} ep{ep_g:02d} loss {tl/n:.4f}/{vl:.4f} skor_val {sv:.4f} {dt:.1f}s")
            if sv > best[0]: best = (sv, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep_g); tunggu = 0
            else:
                tunggu += 1
                if not beku and tunggu >= patience: print(f"  early stopping (patience {patience})"); break
    model.load_state_dict(best[1])
    return model, pd.DataFrame(log), best[2]

def n_param(m): return sum(p.numel() for p in m.parameters())

def ci_bootstrap(fn, y, p, n_boot=1000, seed=SEED, strata=True):
    """CI 95 % persentil; resampling bertingkat per kelas agar setiap sampel memuat semua kelas."""
    r = np.random.default_rng(seed); y = np.asarray(y); idx_k = [np.where(y == k)[0] for k in np.unique(y)]; v = []
    for _ in range(n_boot):
        i = np.concatenate([r.choice(ik, len(ik)) for ik in idx_k]) if strata else r.choice(len(y), len(y))
        try: v.append(fn(y[i], p[i]))
        except ValueError: pass
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def gambar_kurva(LOG, judul_skor, path):
    k = len(LOG); fig, ax = plt.subplots(1, k, figsize=(4.2 * k, 3.2), squeeze=False); ax = ax[0]
    for a, lg in zip(ax, LOG):
        a.plot(lg.epoch, lg.loss_latih, "o-", ms=3, label="loss latih"); a.plot(lg.epoch, lg.loss_val, "s-", ms=3, label="loss val")
        a2 = a.twinx(); a2.plot(lg.epoch, lg.skor_val, "^-", ms=3, color="tab:green", label=judul_skor); a2.set_ylabel(judul_skor, color="tab:green")
        eb = int(lg.loc[lg.skor_val.idxmax(), "epoch"]); a.axvline(eb, color="red", ls="--", lw=1, label=f"terbaik ep {eb}")
        if lg.tahap_beku.any(): a.axvspan(0.5, lg[lg.tahap_beku].epoch.max() + 0.5, color="gray", alpha=0.15, label="tahap beku")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); a.set_title(lg.model.iloc[0], fontsize=8); a.set_xlabel("epoch"); a.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.show()
'''.strip()

IMPOR_UMUM = r'''
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib, matplotlib.ticker; import matplotlib.pyplot as plt
import torchvision, sklearn, medmnist
from medmnist import INFO
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSI = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
         "numpy": np.__version__, "scikit-learn": sklearn.__version__, "medmnist": medmnist.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
print(json.dumps(VERSI, indent=1))
def npz_medmnist(flag, size):
    fn = berkas_medmnist(flag, size); folder = pastikan_dataset("medmnist", [fn]); return np.load(folder / fn)
'''.strip()

KODE_HC18 = r'''
# ---------- Pemuat HC18 (CC BY 4.0; van den Heuvel dkk., 2018) ----------
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
def isi_mask(outline):
    m = ndi.binary_fill_holes(outline)
    if m.sum() < outline.sum() * 3:                 # garis elips tidak tertutup rapat -> tutup dulu
        m = ndi.binary_fill_holes(ndi.binary_closing(outline, iterations=3))
    return m
def keliling_elips_mm(mask, pixel_mm):
    """Paskan elips (momen orde-2) pada komponen terbesar, keliling Ramanujan, dikali ukuran piksel (mm)."""
    lab = measure.label(mask)
    if lab.max() == 0: return np.nan
    p = max(measure.regionprops(lab), key=lambda r: r.area); a, b = p.axis_major_length / 2, p.axis_minor_length / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))) * pixel_mm)
def muat_hc18(uk, maks=None):
    folder = pastikan_dataset("hc18", ["training_set.zip", "training_set_pixel_size_and_HC.csv"])
    akar = ekstrak_lokal("hc18", "training_set.zip") / "training_set"
    meta = pd.read_csv(folder / "training_set_pixel_size_and_HC.csv"); meta.columns = ["berkas", "pixel_mm", "hc_mm"]
    if maks: meta = meta.iloc[:maks].copy()
    meta["subjek"] = meta.berkas.str.extract(r"^(\d+)")[0]      # 014_HC, 014_2HC, 014_3HC -> subjek 014 (asumsi)
    def baca(fn):
        im = Image.open(akar / fn).convert("L"); w, h = im.size
        m = isi_mask(np.array(Image.open(akar / fn.replace(".png", "_Annotation.png")).convert("L")) > 127)
        return (np.array(im.resize((uk, uk), Image.BILINEAR)), np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((uk, uk), Image.NEAREST)) > 127, w, h)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex: hasil = list(ex.map(baca, meta.berkas))
    X = np.stack([r[0] for r in hasil]); M = np.stack([r[1] for r in hasil]); meta["lebar"] = [r[2] for r in hasil]; meta["tinggi"] = [r[3] for r in hasil]
    return X, M, meta.reset_index(drop=True), akar
def bagi_subjek(meta, seed=42, uji=0.2, val=0.1):
    from sklearn.model_selection import GroupShuffleSplit
    g = meta.subjek.values; i = np.arange(len(meta))
    a, t = next(GroupShuffleSplit(1, test_size=uji, random_state=seed).split(i, groups=g))
    b, v = next(GroupShuffleSplit(1, test_size=val / (1 - uji), random_state=seed).split(a, groups=g[a]))
    return {"train": a[b], "val": a[v], "test": t}
'''.strip()
