"""Membangun notebook Colab Bab 12 (notebooks/bab12.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 12 — AI yang Dapat Dijelaskan, *Shortcut Learning*, dan Ketidakpastian
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi (PneumoniaMNIST 128 px, CC BY 4.0):
1. **Grad-CAM** (ditulis sendiri dengan *hook*), *saliency* gradien, dan *occlusion*; uji kewarasan acak-bobot (Adebayo dkk., 2018).
2. **Demo *shortcut learning***: penanda persegi **sintetis** ditempel pada 95 % citra pneumonia dan 5 % citra normal di set latih/validasi.
   Model tampak hebat pada data "serumah", lalu runtuh saat penanda dihapus atau dibalik — dan Grad-CAM menyorot penanda.
3. **Kinerja per subkelompok**: PneumoniaMNIST tidak memuat data demografi; sebagai peragaan dipakai subkelompok teknis (tertil kecerahan citra).
4. **Ketidakpastian**: MC dropout (T = 30) dan *deep ensemble* (4 model) — AUC, ECE, NLL, kurva *selective prediction*,
   dan entropi pada citra di luar distribusi (irisan CT OrganAMNIST).

Hasil → `/content/hasil_bab12/` → `/content/hasil_bab12.zip`. Hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 12.0 Lingkungan")
code(sel_lingkungan("12"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nfrom sklearn.metrics import roc_auc_score")

md("## 12.1 Data dan model bersih (ResNet-18, dropout 0,3 di kepala)")
code(r"""
UK = 28 if SMOKE else 128
with catat_waktu("2_muat_data"):
    d = npz_medmnist("pneumoniamnist", UK)
    D = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0); D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), min(len(y), 300), replace=False)]}
XT = {s: ke_tensor_uint8(D[s][0]) for s in D}; Y = {s: D[s][1] for s in D}; yt = Y["test"]
loss_bce = lambda lg, y: F.binary_cross_entropy_with_logits(lg[:, 0], y.float())
def skor_auc(lg, y): return roc_auc_score(y, lg[:, 0].cpu().numpy())
AUG = dict(rot_derajat=7, geser=0.05, kecerahan=0.1, kontras=0.1)
TAHAP = [(2, 1e-3, True), (10, 1e-4, False)]
def latih_cepat(nama, X, Yy, Xv, Yv, seed): return latih(nama, ResNet18(1, True, p_drop=0.3), X, Yy, Xv, Yv, loss_bce, skor_auc, patience=4, aug=AUG, tahap=TAHAP, bs=64, seed=seed)[0]
with catat_waktu("latih_bersih_s0"): M_BERSIH = latih_cepat("bersih | seed 0", XT["train"], Y["train"], XT["val"], Y["val"], 0)
sig = lambda z: 1 / (1 + np.exp(-z))
P_BERSIH = sig(logit_semua(M_BERSIH, XT["test"])[:, 0].cpu().numpy().astype(np.float64)); print("AUC uji model bersih:", roc_auc_score(yt, P_BERSIH))
""")

md(r"""
## 12.2 Grad-CAM, *saliency*, dan *occlusion*
Grad-CAM: bobot kanal = rerata gradien logit terhadap peta aktivasi `layer3` (8×8 pada masukan 128 px; `layer4` hanya 4×4 sehingga terlalu kasar); peta = ReLU(Σ bobot × aktivasi), diperbesar ke ukuran citra.
Untuk prediksi normal, gradien diambil dari **−logit** (bukti ke arah kelas normal).
""")
code(r"""
def gradcam(model, x, layer):
    simpan_ = {}
    def fwd(m, i, o): simpan_["a"] = o; o.register_hook(lambda g: simpan_.__setitem__("g", g))
    h = layer.register_forward_hook(fwd); model.eval()
    x = x.float().div(255).requires_grad_(True); lg = model(x)[:, 0]
    tanda = torch.where(lg >= 0, 1.0, -1.0); model.zero_grad(); (lg * tanda).sum().backward(); h.remove()
    w = simpan_["g"].mean((2, 3), keepdim=True); cam = F.relu((w * simpan_["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8); return cam.detach().cpu().numpy(), torch.sigmoid(lg).detach().cpu().numpy()
def saliency(model, x):
    model.eval(); x = x.float().div(255).requires_grad_(True); lg = model(x)[:, 0]; tanda = torch.where(lg >= 0, 1.0, -1.0)
    (lg * tanda).sum().backward(); s = x.grad.abs()[:, 0]; return (s / (s.amax((1, 2), keepdim=True) + 1e-8)).cpu().numpy()
@torch.no_grad()
def occlusion(model, x, patch=None, stride=None):
    model.eval(); n, _, H, W = x.shape; patch = patch or H // 8; stride = stride or patch // 2; x = x.float().div(255)
    base = torch.sigmoid(model(x)[:, 0]); peta = torch.zeros(n, H, W, device=x.device); hit = torch.zeros(H, W, device=x.device)
    for i in range(0, H - patch + 1, stride):
        for j in range(0, W - patch + 1, stride):
            xo = x.clone(); xo[:, :, i:i + patch, j:j + patch] = x.mean((2, 3), keepdim=True)
            dp = base - torch.sigmoid(model(xo)[:, 0]); peta[:, i:i + patch, j:j + patch] += dp.abs()[:, None, None]; hit[i:i + patch, j:j + patch] += 1
    peta = peta / hit.clamp(min=1); return (peta / (peta.amax((1, 2), keepdim=True) + 1e-8)).cpu().numpy()
yh = P_BERSIH >= 0.5
pilih = {"TP": np.where(yh & (yt == 1))[0][0], "TN": np.where(~yh & (yt == 0))[0][0],
         "FP": np.where(yh & (yt == 0))[0][0] if (yh & (yt == 0)).any() else 0, "FN": np.where(~yh & (yt == 1))[0][0] if (~yh & (yt == 1)).any() else 1}
idx = list(pilih.values()); xs = XT["test"][idx]
with catat_waktu("3_xai"):
    cam, pp = gradcam(M_BERSIH, xs, M_BERSIH.net.layer3); sal = saliency(M_BERSIH, xs); occ = occlusion(M_BERSIH, xs)
    m_acak = ResNet18(1, False, p_drop=0.3).to(DEV); cam_acak, _ = gradcam(m_acak, xs, m_acak.net.layer3)    # uji kewarasan: bobot acak
fig, ax = plt.subplots(4, 5, figsize=(12, 10))
for r_, (nm, i) in enumerate(pilih.items()):
    img = XT["test"][i, 0].cpu().numpy()
    for c_, (judul, peta) in enumerate([("citra", None), ("Grad-CAM", cam[r_]), ("saliency", sal[r_]), ("occlusion", occ[r_]), ("Grad-CAM bobot ACAK", cam_acak[r_])]):
        a = ax[r_, c_]; a.imshow(img, cmap="gray"); a.axis("off")
        if peta is not None: a.imshow(peta, cmap="jet", alpha=0.45)
        a.set_title(f"{nm}: p={pp[r_]:.2f}" if c_ == 0 else judul, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_12_1_xai_bersih.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 12.3 Demo *shortcut learning* dengan penanda sintetis
Penanda = persegi putih bertepi hitam di pojok kiri atas (mirip penanda lateralitas/teks alat). Ini **sintetis**, ditambahkan oleh kode ini.
- Set latih & validasi "bias": penanda pada 95 % pneumonia dan 5 % normal.
- Set uji: (a) **bersih** tanpa penanda, (b) **bias sama** seperti latih, (c) **dibalik** (penanda pada 95 % normal dan 5 % pneumonia).
""")
code(r"""
S = max(4, UK // 10); O = max(2, UK // 32)
def tempel(X, mask):
    X = X.clone(); i = torch.as_tensor(np.where(mask)[0], device=X.device)
    X[i, :, O:O + S, O:O + S] = 255; X[i, :, O + S // 4:O + S - S // 4, O + S // 4:O + S - S // 4] = 0; return X
def mask_bias(y, p_pos, p_neg, seed): r = np.random.default_rng(seed); u = r.random(len(y)); return np.where(y == 1, u < p_pos, u < p_neg)
XB = {"train": tempel(XT["train"], mask_bias(Y["train"], 0.95, 0.05, 1)), "val": tempel(XT["val"], mask_bias(Y["val"], 0.95, 0.05, 2))}
UJI = {"bersih": XT["test"], "bias sama": tempel(XT["test"], mask_bias(yt, 0.95, 0.05, 3)), "dibalik": tempel(XT["test"], mask_bias(yt, 0.05, 0.95, 3))}
with catat_waktu("latih_bias"): M_BIAS = latih_cepat("bias penanda | seed 0", XB["train"], Y["train"], XB["val"], Y["val"], 0)
baris = []
for nm_m, m in [("model bersih", M_BERSIH), ("model latih-bias", M_BIAS)]:
    for nm_u, X in UJI.items():
        p = sig(logit_semua(m, X)[:, 0].cpu().numpy().astype(np.float64)); yh_ = p >= 0.5
        baris.append({"model": nm_m, "set_uji": nm_u, "AUC": roc_auc_score(yt, p), "akurasi@0.5": float((yh_ == yt).mean()),
                      "sensitivitas@0.5": float(yh_[yt == 1].mean()), "spesifisitas@0.5": float((~yh_[yt == 0]).mean())})
T121 = pd.DataFrame(baris); T121.to_csv(OUT / "tabel_12_1_shortcut.csv", index=False)
# Fraksi "energi" Grad-CAM di dalam area penanda (citra uji bias sama yang berpenanda)
ber = np.where(mask_bias(yt, 0.95, 0.05, 3))[0][:64]; kotak = np.zeros((UK, UK), bool); kotak[O:O + S, O:O + S] = True
EN = {}
for nm_m, m in [("model bersih", M_BERSIH), ("model latih-bias", M_BIAS)]:
    c_, _ = gradcam(m, UJI["bias sama"][ber], m.net.layer3); EN[nm_m] = float((c_[:, kotak].sum(1) / (c_.reshape(len(ber), -1).sum(1) + 1e-8)).mean())
EN["luas_penanda_relatif"] = float(kotak.mean()); print("fraksi energi Grad-CAM di area penanda:", EN)
fig, ax = plt.subplots(2, 4, figsize=(11, 5.6)); contoh = ber[:4]
for c_, i in enumerate(contoh):
    for r_, (nm_m, m) in enumerate([("bersih", M_BERSIH), ("latih-bias", M_BIAS)]):
        cm, pp = gradcam(m, UJI["bias sama"][i:i + 1], m.net.layer3)
        ax[r_, c_].imshow(UJI["bias sama"][i, 0].cpu(), cmap="gray"); ax[r_, c_].imshow(cm[0], cmap="jet", alpha=0.45); ax[r_, c_].axis("off")
        ax[r_, c_].set_title(f"model {nm_m}: p={pp[0]:.2f} (label {yt[i]})", fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gambar_12_2_shortcut_gradcam.png", bbox_inches="tight"); plt.show()
fig, a = plt.subplots(figsize=(6, 3.2)); w = 0.35
for k, nm_m in enumerate(["model bersih", "model latih-bias"]):
    d_ = T121[T121.model == nm_m]; a.bar(np.arange(3) + k * w, d_.AUC, w, label=nm_m)
a.set_xticks(np.arange(3) + w / 2); a.set_xticklabels(list(UJI)); a.set_ylabel("AUC uji"); a.set_ylim(0, 1); a.axhline(0.5, color="gray", ls=":"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gambar_12_3_shortcut_auc.png", bbox_inches="tight"); plt.show(); T121
""")

md(r"""
## 12.4 Kinerja per subkelompok (peragaan)
Tanpa atribut demografis, kita **tidak dapat** menilai keadilan terhadap jenis kelamin, usia, atau etnis pada dataset ini.
Sebagai peragaan cara kerjanya, set uji dibagi menjadi tertil **kecerahan rerata citra** (proksi kasar kondisi akuisisi).
""")
code(r"""
kec = XT["test"].float().mean((1, 2, 3)).cpu().numpy(); ter = pd.qcut(kec, 3, labels=["gelap", "sedang", "terang"])
baris = []
for g in ["gelap", "sedang", "terang"]:
    i = np.where(ter == g)[0]; y_, p_ = yt[i], P_BERSIH[i]; yh_ = p_ >= 0.5
    baris.append({"subkelompok": g, "n": len(i), "prevalensi": float(y_.mean()), "AUC": roc_auc_score(y_, p_) if len(np.unique(y_)) > 1 else np.nan,
                  "AUC_CI95": ci_bootstrap(roc_auc_score, y_, p_, n_boot=200 if SMOKE else 1000) if len(np.unique(y_)) > 1 else None,
                  "sensitivitas@0.5": float(yh_[y_ == 1].mean()) if (y_ == 1).any() else np.nan, "spesifisitas@0.5": float((~yh_[y_ == 0]).mean()) if (y_ == 0).any() else np.nan})
T122 = pd.DataFrame(baris); T122.to_csv(OUT / "tabel_12_2_subkelompok.csv", index=False); T122
""")

md(r"""
## 12.5 Ketidakpastian: MC dropout dan *deep ensemble*
- **Model tunggal**: dropout nonaktif saat inferensi.
- **MC dropout**: dropout di kepala tetap aktif, 30 kali *forward*, probabilitas dirata-rata.
- **Deep ensemble**: 4 model identik dengan seed berbeda (0–3), probabilitas dirata-rata.
Ketidakpastian = entropi prediktif biner. *Selective prediction*: kasus paling tidak pasti dirujuk ke klinisi, akurasi dihitung pada sisanya.
""")
code(r"""
@torch.no_grad()
def mc_dropout(model, X, T=30):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.Dropout): mod.train()
    p = np.mean([sig(logit_semua_mc(model, X)) for _ in range(T)], 0); model.eval(); return p
@torch.no_grad()
def logit_semua_mc(model, X, bs=512):          # seperti logit_semua tetapi TIDAK memanggil model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP): out.append(model(X[i:i + bs].float().div(255)).float())
    return torch.cat(out)[:, 0].cpu().numpy().astype(np.float64)
ANGGOTA = [M_BERSIH]
for s in ([1] if SMOKE else [1, 2, 3]):
    with catat_waktu(f"latih_ensemble_s{s}"): ANGGOTA.append(latih_cepat(f"ensemble | seed {s}", XT["train"], Y["train"], XT["val"], Y["val"], s))
def ece(y, p, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1); return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any()))
def nll(y, p): p = np.clip(p, 1e-7, 1 - 1e-7); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def entropi(p): p = np.clip(p, 1e-7, 1 - 1e-7); return -(p * np.log(p) + (1 - p) * np.log(1 - p))
with catat_waktu("4_ketidakpastian"):
    PRED = {"model tunggal": P_BERSIH, "MC dropout (T=30)": mc_dropout(M_BERSIH, XT["test"]),
            f"deep ensemble ({len(ANGGOTA)} model)": np.mean([sig(logit_semua(m, XT["test"])[:, 0].cpu().numpy().astype(np.float64)) for m in ANGGOTA], 0)}
    # OOD: irisan CT OrganAMNIST (CC BY 4.0) diperbesar ke UK
    do = npz_medmnist("organamnist", 28); xo = torch.as_tensor(do["test_images"][:500])[:, None].float()
    XOOD = F.interpolate(xo, size=(UK, UK), mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8).to(DEV)
    POOD = {"model tunggal": sig(logit_semua(M_BERSIH, XOOD)[:, 0].cpu().numpy().astype(np.float64)), "MC dropout (T=30)": mc_dropout(M_BERSIH, XOOD),
            f"deep ensemble ({len(ANGGOTA)} model)": np.mean([sig(logit_semua(m, XOOD)[:, 0].cpu().numpy().astype(np.float64)) for m in ANGGOTA], 0)}
baris = []; fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); cakupan = np.linspace(0.5, 1, 11)
for nm, p in PRED.items():
    u = entropi(p); salah = ((p >= 0.5) != yt); urut = np.argsort(u)
    akur = [1 - salah[urut[:max(1, int(round(c * len(p))))]].mean() for c in cakupan]
    uo = entropi(POOD[nm]); auc_ood = roc_auc_score(np.r_[np.zeros(len(u)), np.ones(len(uo))], np.r_[u, uo])
    baris.append({"metode": nm, "AUC": roc_auc_score(yt, p), "akurasi@0.5": float(1 - salah.mean()), "ECE": ece(yt, p), "NLL": nll(yt, p),
                  "AUROC_ketidakpastian_vs_salah": roc_auc_score(salah, u) if 0 < salah.sum() < len(salah) else np.nan,
                  "akurasi_cakupan_80%": float(akur[list(np.round(cakupan, 2)).index(0.8)]), "entropi_rerata_ID": float(u.mean()), "entropi_rerata_OOD": float(uo.mean()),
                  "AUROC_deteksi_OOD_entropi": float(auc_ood), "persen_OOD_diprediksi_pneumonia": float((POOD[nm] >= 0.5).mean() * 100)})
    ax[0].plot(cakupan, akur, "o-", ms=3, label=nm); ax[1].hist(uo, bins=30, alpha=0.4, label=f"OOD – {nm}", density=True)
ax[1].hist(entropi(PRED["model tunggal"]), bins=30, alpha=0.4, label="ID (uji) – model tunggal", density=True, histtype="step", color="k")
ax[0].set_xlabel("cakupan (fraksi kasus yang diputuskan model)"); ax[0].set_ylabel("akurasi pada kasus yang diputuskan"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("entropi prediktif"); ax[1].set_ylabel("kepadatan"); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "gambar_12_4_ketidakpastian.png", bbox_inches="tight"); plt.show()
T123 = pd.DataFrame(baris); T123.to_csv(OUT / "tabel_12_3_ketidakpastian.csv", index=False); T123
""")

md("## 12.6 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("12", 'dict(ukuran_px=UK, shortcut=T121.to_dict("records"), energi_gradcam_penanda=EN, subkelompok=T122.to_dict("records"), ketidakpastian=T123.to_dict("records"))'))
simpan(C, "bab12.ipynb", Path(__file__).parent / "bab12.ipynb")
