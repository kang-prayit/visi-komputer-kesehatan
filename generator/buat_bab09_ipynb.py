"""Membangun notebook Colab Bab 9 (notebooks/bab09.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM, KODE_HC18

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 9 — Segmentasi Semantik dengan U-Net: Kepala Janin pada USG (HC18)
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi:
1. **HC18** (CC BY 4.0): mask kepala dibuat dengan mengisi garis elips anotasi; pembagian per subjek sama dengan Bab 7 (seed 42).
2. **U-Net** dari nol: *loss* BCE vs Dice + BCE; dan U-Net dengan *encoder* ResNet-18 pra-latih ImageNet (2 seed).
3. Metrik per kasus pada resolusi asli 800×540: **Dice, IoU, HD95 (mm)**; visualisasi tumpang-susun terbaik/median/terburuk.
4. **Dari mask ke ukuran klinis**: paskan elips → keliling (Ramanujan) × ukuran piksel → HC (mm), dibandingkan dengan HC acuan dan dengan regresi langsung Bab 7.

Hasil → `/content/hasil_bab09/` → `/content/hasil_bab09.zip`. Hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 9.0 Lingkungan")
code(sel_lingkungan("09"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH + "\n" + KODE_HC18 +
     "\nfrom scipy.ndimage import distance_transform_edt, binary_erosion")

md("## 9.1 Data dan mask")
code(r"""
UK = 64 if SMOKE else 256
with catat_waktu("2_muat_hc18"):
    XH, MH, META, AKAR = muat_hc18(UK, maks=120 if SMOKE else None)
    SPL = bagi_subjek(META, seed=SEED)
XT = ke_tensor_uint8(XH); MT = torch.as_tensor(MH[:, None]).to(DEV)
print({s: len(i) for s, i in SPL.items()}, "fraksi piksel kepala (latih):", float(MH[SPL["train"]].mean()))
# Batas bawah metode: elips dari mask ACUAN (resolusi asli) vs HC di CSV
def mask_asli(fn): return isi_mask(np.array(Image.open(AKAR / fn.replace(".png", "_Annotation.png")).convert("L")) > 127)
with catat_waktu("3_mask_asli_uji"):
    M_ASLI = {i: mask_asli(META.berkas[i]) for i in SPL["test"]}
HC_GT_ELIPS = np.array([keliling_elips_mm(M_ASLI[i], META.pixel_mm[i]) for i in SPL["test"]])
print("Elips dari mask acuan vs HC CSV: MAE %.2f mm, bias %.2f mm" % (np.nanmean(np.abs(HC_GT_ELIPS - META.hc_mm.values[SPL["test"]])), np.nanmean(HC_GT_ELIPS - META.hc_mm.values[SPL["test"]])))
""")

md(r"""
## 9.2 Arsitektur U-Net
Encoder (turun 4 kali) – *bottleneck* – decoder (naik 4 kali) dengan **skip connection** yang menyambungkan fitur encoder ke decoder
pada resolusi yang sama. Varian kedua memakai blok ResNet-18 pra-latih sebagai encoder.
""")
code(r"""
def blok(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                                       nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__(); ch = [c, 2 * c, 4 * c, 8 * c, 16 * c]
        self.enc = nn.ModuleList([blok(1, ch[0])] + [blok(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([blok(2 * ch[i], ch[i]) for i in reversed(range(4))]); self.akhir = nn.Conv2d(ch[0], 1, 1)
    def forward(self, x):
        skip = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2));
            if i < 4: skip.append(x)
        for u, d, s in zip(self.up, self.dec, reversed(skip)): x = d(torch.cat([u(x), s], 1))
        return self.akhir(x)
class UNetResNet18(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        try: r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrain else None)
        except Exception as e:
            if not SMOKE: raise
            print("SMOKE: bobot acak", e); r = torchvision.models.resnet18(weights=None)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu); self.l1 = nn.Sequential(r.maxpool, r.layer1); self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4
        self.d4 = blok(512 + 256, 256); self.d3 = blok(256 + 128, 128); self.d2 = blok(128 + 64, 64); self.d1 = blok(64 + 64, 64); self.d0 = blok(64 + 1, 32)
        self.akhir = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x0 = x; x = (x.repeat(1, 3, 1, 1) - MEAN_IN.to(x.device)) / SD_IN.to(x.device)
        f0 = self.l0(x); f1 = self.l1(f0); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d4(torch.cat([up(f4, f3), f3], 1)); y = self.d3(torch.cat([up(y, f2), f2], 1)); y = self.d2(torch.cat([up(y, f1), f1], 1))
        y = self.d1(torch.cat([up(y, f0), f0], 1)); y = self.d0(torch.cat([up(y, x0), x0], 1)); return self.akhir(y)
for nm, m in [("U-Net c=32", UNet()), ("U-Net encoder ResNet-18", UNetResNet18(False))]:
    print(nm, f"{n_param(m)/1e6:.2f} juta parameter", tuple(m(torch.zeros(1, 1, UK, UK)).shape))
""")

md(r"""
## 9.3 *Loss* dan pelatihan
- **BCE** per piksel; **Dice loss** `1 − 2|P∩G| / (|P|+|G|)` (versi lunak, per citra); kombinasi Dice + BCE.
- Augmentasi **bersama** citra–mask: flip horizontal/vertikal, rotasi ±15°, geser ±5 %; kecerahan/kontras ±15 % hanya pada citra.
- AdamW lr 1e-3 (encoder pra-latih: 3e-4), kosinus, maks 40 epoch, *early stopping* pada Dice validasi (patience 8).
""")
code(r"""
def dice_lunak(lg, m, eps=1.0):
    p = torch.sigmoid(lg); i = (p * m).sum((1, 2, 3)); return 1 - ((2 * i + eps) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + eps)).mean()
LOSS = {"BCE": lambda lg, m: F.binary_cross_entropy_with_logits(lg, m),
        "Dice+BCE": lambda lg, m: F.binary_cross_entropy_with_logits(lg, m) + dice_lunak(lg, m)}
def aug_bersama(x, m):
    n = x.shape[0]; d = x.device
    f = (torch.rand(n, device=d) < 0.5)[:, None, None, None]; x, m = torch.where(f, x.flip(3), x), torch.where(f, m.flip(3), m)
    f = (torch.rand(n, device=d) < 0.5)[:, None, None, None]; x, m = torch.where(f, x.flip(2), x), torch.where(f, m.flip(2), m)
    a = (torch.rand(n, device=d) * 2 - 1) * 15 * np.pi / 180; t = (torch.rand(n, 2, device=d) * 2 - 1) * 0.1
    th = torch.stack([torch.stack([a.cos(), -a.sin(), t[:, 0]], 1), torch.stack([a.sin(), a.cos(), t[:, 1]], 1)], 1)
    gr = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, gr, padding_mode="zeros", align_corners=False); m = (F.grid_sample(m, gr, mode="nearest", align_corners=False) > 0.5).float()
    mu = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mu) * (1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * 0.15) + mu) * (1 + (torch.rand(n, 1, 1, 1, device=d) * 2 - 1) * 0.15)
    return x.clamp(0, 1), m
@torch.no_grad()
def prob_semua(model, idx, bs=32):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        x = XT[idx[i:i + bs]].float().div(255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): out.append(torch.sigmoid(model(x).float()))
    return torch.cat(out)
def dice_keras(p, m): p = p > 0.5; m = m > 0.5; return ((2 * (p & m).sum((1, 2, 3)) + 1e-6) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1e-6)).float()
def latih_seg(nama, model, fungsi_loss, lr=1e-3, maks_epoch=40, patience=8, bs=16, seed=SEED):
    seed_semua(seed); model = model.to(DEV); g = torch.Generator().manual_seed(seed); itr, iv = SPL["train"], SPL["val"]
    if SMOKE: maks_epoch, patience = 2, 1
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, maks_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP); best = (-1, None, 0); log = []; tunggu = 0
    for ep in range(1, maks_epoch + 1):
        model.train(); t0 = time.perf_counter(); perm = itr[torch.randperm(len(itr), generator=g).numpy()]; tl = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; x, m = aug_bersama(XT[b].float().div(255), MT[b].float())
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): lg = model(x)
            loss = fungsi_loss(lg.float(), m); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item() * len(b)
        sch.step()
        if DEV.type == "cuda": torch.cuda.synchronize()
        dt = time.perf_counter() - t0; dv = float(dice_keras(prob_semua(model, iv), MT[iv].float()).mean())
        log.append({"model": nama, "epoch": ep, "loss_latih": tl / len(perm), "dice_val": dv, "detik_epoch": dt}); print(log[-1])
        if dv > best[0]: best = (dv, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep); tunggu = 0
        else:
            tunggu += 1
            if tunggu >= patience: print("  early stopping"); break
    model.load_state_dict(best[1]); return model, pd.DataFrame(log), best[2]
KONFIG = {"U-Net (BCE)": (lambda: UNet(), "BCE", 1e-3), "U-Net (Dice+BCE)": (lambda: UNet(), "Dice+BCE", 1e-3),
          "U-Net encoder ResNet-18 pra-latih (Dice+BCE)": (lambda: UNetResNet18(True), "Dice+BCE", 3e-4)}
SEEDS = [0] if SMOKE else [0, 1]
MODEL, LOG = {}, {}
for nk, (fm, nl, lr) in KONFIG.items():
    for s in SEEDS:
        with catat_waktu(f"latih_{nk[:18]}_s{s}"): MODEL[(nk, s)], LOG[(nk, s)], _ = latih_seg(f"{nk} | seed {s}", fm(), LOSS[nl], lr=lr, seed=s)
pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for (nk, s), lg in LOG.items(): ax[0].plot(lg.epoch, lg.loss_latih, label=f"{nk} s{s}"); ax[1].plot(lg.epoch, lg.dice_val, label=f"{nk} s{s}")
ax[0].set_ylabel("loss latih"); ax[1].set_ylabel("Dice val (UK×UK)")
for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=6); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.tight_layout(); plt.savefig(OUT / "gambar_9_1_kurva_latih.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 9.4 Evaluasi per kasus pada resolusi asli: Dice, IoU, HD95, dan HC dari elips
Probabilitas diperbesar kembali ke 800×540, diambang 0,5, lalu **pascaproses**: komponen terbesar + isi lubang.
HD95 = persentil ke-95 jarak permukaan dua arah, dikali ukuran piksel (mm).
""")
code(r"""
def pascaproses(b):
    lab = measure.label(b)
    if lab.max() == 0: return b
    return ndi.binary_fill_holes(lab == (np.bincount(lab.ravel())[1:].argmax() + 1))
def hd95(a, b, px):
    if a.sum() == 0 or b.sum() == 0: return np.nan
    ta = a ^ binary_erosion(a); tb = b ^ binary_erosion(b)
    da = distance_transform_edt(~tb)[ta]; db = distance_transform_edt(~ta)[tb]
    return float(np.percentile(np.concatenate([da, db]), 95) * px)
def evaluasi(model):
    P = prob_semua(model, SPL["test"]).cpu(); baris = []
    for k, i in enumerate(SPL["test"]):
        g = M_ASLI[i]; H, W = g.shape
        p = pascaproses(F.interpolate(P[k:k + 1], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy() > 0.5)
        it = (p & g).sum(); un = (p | g).sum()
        baris.append({"berkas": META.berkas[i], "dice": 2 * it / (p.sum() + g.sum() + 1e-9), "iou": it / (un + 1e-9), "hd95_mm": hd95(p, g, META.pixel_mm[i]),
                      "hc_acuan_mm": META.hc_mm[i], "hc_elips_pred_mm": keliling_elips_mm(p, META.pixel_mm[i])})
    return pd.DataFrame(baris)
HASIL, PER_KASUS = [], {}
with catat_waktu("4_evaluasi"):
    for (nk, s), m in MODEL.items():
        d = evaluasi(m); d["galat_hc_mm"] = d.hc_elips_pred_mm - d.hc_acuan_mm; PER_KASUS[(nk, s)] = d
        HASIL.append({"konfigurasi": nk, "seed": s, "n_param_juta": n_param(m) / 1e6, "epoch_terbaik": int(LOG[(nk, s)].dice_val.idxmax() + 1),
                      "detik_per_epoch": float(LOG[(nk, s)].detik_epoch.mean()), "dice_val_UK": float(LOG[(nk, s)].dice_val.max()),
                      "dice_uji_rerata": d.dice.mean(), "dice_uji_median": d.dice.median(), "iou_uji_rerata": d.iou.mean(), "hd95_uji_median_mm": d.hd95_mm.median(),
                      "MAE_HC_uji_mm": d.galat_hc_mm.abs().mean(), "bias_HC_uji_mm": d.galat_hc_mm.mean(), "persen_kasus_dice<0.9": float((d.dice < 0.9).mean() * 100)})
        print(HASIL[-1])
T91 = pd.DataFrame(HASIL); T91.to_csv(OUT / "tabel_9_1_per_seed.csv", index=False)
T92 = T91.groupby("konfigurasi")[["dice_uji_rerata", "iou_uji_rerata", "hd95_uji_median_mm", "MAE_HC_uji_mm", "bias_HC_uji_mm", "persen_kasus_dice<0.9", "n_param_juta", "detik_per_epoch"]].agg(["mean", "std"])
T92.columns = ["_".join(c) for c in T92.columns]
T92.loc["acuan: elips dari mask anotasi", "MAE_HC_uji_mm_mean"] = float(np.nanmean(np.abs(HC_GT_ELIPS - META.hc_mm.values[SPL["test"]])))
T92.to_csv(OUT / "tabel_9_2_ringkas.csv"); T92
""")

md("## 9.5 Visualisasi: sebaran Dice, kasus terbaik/median/terburuk, dan Bland–Altman HC")
code(r"""
PIL = max(MODEL, key=lambda k: T91[(T91.konfigurasi == k[0]) & (T91.seed == k[1])].dice_val_UK.iloc[0]); d = PER_KASUS[PIL]
d.to_csv(OUT / "per_kasus_model_terpilih.csv", index=False); print("model terpilih (Dice val):", PIL)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.boxplot([PER_KASUS[(nk, SEEDS[0])].dice for nk in KONFIG], tick_labels=[nk.replace(" (", "\n(") for nk in KONFIG], showfliers=True)
ax.set_ylabel("Dice per kasus (uji, resolusi asli)"); plt.tight_layout(); plt.savefig(OUT / "gambar_9_2_boxplot_dice.png", bbox_inches="tight"); plt.show()
urut = d.dice.argsort().values; pilih = {"terburuk": urut[0], "median": urut[len(urut) // 2], "terbaik": urut[-1]}
P = prob_semua(MODEL[PIL], SPL["test"]).cpu()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (nm, k) in zip(ax, pilih.items()):
    i = SPL["test"][k]; img = np.array(Image.open(AKAR / META.berkas[i]).convert("L")); g = M_ASLI[i]
    p = pascaproses(F.interpolate(P[k:k + 1], size=g.shape, mode="bilinear", align_corners=False)[0, 0].numpy() > 0.5)
    a.imshow(img, cmap="gray"); a.contour(g, [0.5], colors="lime", linewidths=1); a.contour(p, [0.5], colors="red", linewidths=1, linestyles="--")
    a.set_title(f"{nm}: Dice {d.dice.iloc[k]:.3f}, HD95 {d.hd95_mm.iloc[k]:.1f} mm", fontsize=8); a.axis("off")
fig.text(0.5, 0.01, "hijau = anotasi, merah putus-putus = prediksi", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_9_3_tumpang_susun.png", bbox_inches="tight"); plt.show()
e = d.galat_hc_mm.values; r_ = (d.hc_elips_pred_mm + d.hc_acuan_mm).values / 2; bias = np.nanmean(e); sd = np.nanstd(e, ddof=1)
BA = {"model": f"{PIL[0]} | seed {PIL[1]}", "MAE_HC_mm": float(np.nanmean(np.abs(e))), "bias_mm": float(bias), "LoA_bawah_mm": float(bias - 1.96 * sd), "LoA_atas_mm": float(bias + 1.96 * sd),
      "MAE_HC_CI95": ci_bootstrap(lambda y, p: np.nanmean(np.abs(p - y)), d.hc_acuan_mm.values, d.hc_elips_pred_mm.values, n_boot=200 if SMOKE else 1000, strata=False)}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(d.hc_acuan_mm, d.hc_elips_pred_mm, s=8, alpha=0.6); lim = [d.hc_acuan_mm.min() - 10, d.hc_acuan_mm.max() + 10]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("HC acuan (mm)"); ax[0].set_ylabel("HC dari elips prediksi (mm)"); ax[0].set_title(f"MAE {BA['MAE_HC_mm']:.2f} mm")
ax[1].scatter(r_, e, s=8, alpha=0.6); ax[1].axhline(bias, color="r")
for v in (BA["LoA_bawah_mm"], BA["LoA_atas_mm"]): ax[1].axhline(v, color="r", ls="--")
ax[1].set_xlabel("rerata (mm)"); ax[1].set_ylabel("elips prediksi − acuan (mm)"); ax[1].set_title(f"Bland–Altman: bias {bias:.2f}, LoA [{BA['LoA_bawah_mm']:.1f}, {BA['LoA_atas_mm']:.1f}] mm")
plt.tight_layout(); plt.savefig(OUT / "gambar_9_4_hc_dari_mask.png", bbox_inches="tight"); plt.show()
torch.save(MODEL[PIL].state_dict(), OUT / "unet_hc18_terpilih.pt"); BA
""")

md("## 9.6 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("09", 'dict(ukuran_px=UK, split={s: len(i) for s, i in SPL.items()}, per_seed=T91.to_dict("records"), elips_mask_acuan_MAE_mm=float(np.nanmean(np.abs(HC_GT_ELIPS - META.hc_mm.values[SPL["test"]]))), model_terpilih=list(PIL), bland_altman_hc=BA)'))
simpan(C, "bab09.ipynb", Path(__file__).parent / "bab09.ipynb")
