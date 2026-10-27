"""Membangun notebook Colab Bab 11 (notebooks/bab11.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM, KODE_HC18

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 11 — Evaluasi ala Klinis
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Notebook ini melatih **cepat** beberapa model di dalam notebook (tidak bergantung pada hasil bab lain), lalu mengevaluasinya seperti peneliti klinis:
1. PneumoniaMNIST 128 px (CC BY 4.0): model A = ResNet-18 pra-latih, model A' = ResNet-18 dengan bobot kelas, model B = CNN kecil dari nol.
2. **CI 95 % bootstrap** (AUC, sensitivitas, spesifisitas) dan **uji DeLong** untuk selisih AUC dua model berpasangan.
3. **Kalibrasi**: *reliability diagram*, **ECE**, **Brier score**, dan *temperature scaling* yang dipasang pada set validasi.
4. **Decision curve analysis** (net benefit) pada prevalensi set uji dan pada prevalensi rendah hasil subsampel (simulasi).
5. **Pergeseran distribusi sintetis** (derau, blur, kontras): AUC dan ECE turun seberapa jauh?
6. Regresi HC18 cepat (128 px) → **Bland–Altman**, **ICC(2,1)/ICC(3,1)**, dan **CCC Lin**.

Hasil → `/content/hasil_bab11/` → `/content/hasil_bab11.zip`. Hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 11.0 Lingkungan")
code(sel_lingkungan("11"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH + "\n" + KODE_HC18 +
     "\nfrom sklearn.metrics import roc_auc_score, roc_curve\nfrom scipy.stats import norm\nfrom scipy.optimize import minimize_scalar")

md("## 11.1 Data dan tiga model cepat")
code(r"""
UK = 28 if SMOKE else 128
with catat_waktu("2_muat_data"):
    d = npz_medmnist("pneumoniamnist", UK)
    D = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0); D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), min(len(y), 300), replace=False)]}
XT = {s: ke_tensor_uint8(D[s][0]) for s in D}; Y = {s: D[s][1] for s in D}
print({s: (len(Y[s]), round(float(Y[s].mean()), 3)) for s in Y})
class CNNKecil(nn.Module):
    def __init__(self):
        super().__init__()
        def b(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(b(1, 16), b(16, 32), b(32, 64), b(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.f(x)
n0, n1 = int((Y["train"] == 0).sum()), int((Y["train"] == 1).sum())
def bce(pw=1.0):
    t = torch.tensor([pw], device=DEV); return lambda lg, y: F.binary_cross_entropy_with_logits(lg[:, 0], y.float(), pos_weight=t)
def skor_auc(lg, y): return roc_auc_score(y, lg[:, 0].cpu().numpy())
AUG = dict(rot_derajat=7, geser=0.05, kecerahan=0.1, kontras=0.1)
MODEL_DEF = {"A: ResNet-18 pra-latih": (lambda: ResNet18(1, True), 1.0, [(2, 1e-3, True), (10, 1e-4, False)]),
             "A': ResNet-18 + bobot kelas": (lambda: ResNet18(1, True), n0 / n1, [(2, 1e-3, True), (10, 1e-4, False)]),
             "B: CNN kecil dari nol": (lambda: CNNKecil(), 1.0, [(15, 1e-3, False)])}
Z, MODEL = {}, {}           # logit val/uji
for nm, (fm, pw, tahap) in MODEL_DEF.items():
    with catat_waktu(f"latih_{nm[:2]}"):
        m, lg, ep = latih(nm, fm(), XT["train"], Y["train"], XT["val"], Y["val"], bce(pw), skor_auc, patience=4, aug=AUG, tahap=tahap, bs=64, seed=0)
    MODEL[nm] = m; Z[nm] = {s: logit_semua(m, XT[s])[:, 0].cpu().numpy().astype(np.float64) for s in ("val", "test")}
sig = lambda z: 1 / (1 + np.exp(-z))
yv, yt = Y["val"], Y["test"]
""")

md(r"""
## 11.2 CI bootstrap dan uji DeLong
Ambang tiap model = titik Youden pada set **validasi**. Bootstrap 2000 kali, bertingkat per kelas, persentil 2,5–97,5.
Uji DeLong (implementasi cepat Sun & Xu, 2014) membandingkan AUC dua model pada **sampel uji yang sama** (berpasangan).
""")
code(r"""
def _midrank(x):
    J = np.argsort(x, kind="mergesort"); Zs = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Zs[j] == Zs[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out
def delong(y, pa, pb):
    P = np.vstack([pa, pb]); X = P[:, y == 1]; Yn = P[:, y == 0]; m, n = X.shape[1], Yn.shape[1]
    tx = np.array([_midrank(r) for r in X]); ty = np.array([_midrank(r) for r in Yn]); tz = np.array([_midrank(r) for r in np.hstack([X, Yn])])
    auc = tz[:, :m].sum(1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var); return {"AUC_a": auc[0], "AUC_b": auc[1], "selisih": auc[0] - auc[1], "SE": float(np.sqrt(var)),
                                                  "CI95_selisih": (float(auc[0] - auc[1] - 1.96 * np.sqrt(var)), float(auc[0] - auc[1] + 1.96 * np.sqrt(var))), "z": float(z), "p": float(2 * norm.sf(abs(z)))}
NB = 300 if SMOKE else 2000; baris = []
with catat_waktu("3_bootstrap_delong"):
    for nm in MODEL:
        pv, pt = sig(Z[nm]["val"]), sig(Z[nm]["test"]); fpr, tpr, thr = roc_curve(yv, pv); t = float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))
        sens = lambda y, p: ((p >= t) & (y == 1)).sum() / (y == 1).sum(); spes = lambda y, p: ((p < t) & (y == 0)).sum() / (y == 0).sum()
        b = {"model": nm, "ambang_Youden_val": t}
        for k, fn in [("AUC", roc_auc_score), ("sensitivitas", sens), ("spesifisitas", spes)]:
            b[k] = float(fn(yt, pt)); b[f"{k}_CI95"] = ci_bootstrap(fn, yt, pt, n_boot=NB)
        baris.append(b)
    T111 = pd.DataFrame(baris); T111.to_csv(OUT / "tabel_11_1_ci_bootstrap.csv", index=False)
    nmA, nmA2, nmB = list(MODEL)
    DL = {f"{nmA} vs {nmB}": delong(yt, sig(Z[nmA]["test"]), sig(Z[nmB]["test"])), f"{nmA} vs {nmA2}": delong(yt, sig(Z[nmA]["test"]), sig(Z[nmA2]["test"]))}
    # pembanding: bootstrap berpasangan untuk selisih AUC A − B
    r = np.random.default_rng(SEED); ik = [np.where(yt == k)[0] for k in (0, 1)]; dd = []
    for _ in range(NB):
        i = np.concatenate([r.choice(a, len(a)) for a in ik]); dd.append(roc_auc_score(yt[i], Z[nmA]["test"][i]) - roc_auc_score(yt[i], Z[nmB]["test"][i]))
    DL[f"{nmA} vs {nmB}"]["CI95_bootstrap_berpasangan"] = (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
T112 = pd.DataFrame(DL).T; T112.to_csv(OUT / "tabel_11_2_delong.csv"); display(T111); T112
""")

md(r"""
## 11.3 Kalibrasi: reliability diagram, ECE, Brier, dan *temperature scaling*
ECE di sini = rerata berbobot |frekuensi positif teramati − rerata probabilitas| pada 15 bin lebar sama (kalibrasi probabilitas kelas positif).
*Temperature scaling*: `p = σ(z / T)`, T dipilih untuk meminimalkan *negative log-likelihood* pada set **validasi**.
Bobot kelas (model A') mengubah *prior* yang dipelajari model sehingga probabilitasnya bergeser — lihat dampaknya pada kalibrasi.
""")
code(r"""
def ece(y, p, nb=15):
    b = np.minimum((p * nb).astype(int), nb - 1); return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(nb) if (b == k).any()))
def brier(y, p): return float(np.mean((p - y) ** 2))
def nll(y, p): p = np.clip(p, 1e-7, 1 - 1e-7); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def pasang_T(z, y): return float(np.exp(minimize_scalar(lambda lt: nll(y, sig(z / np.exp(lt))), bounds=(-3, 3), method="bounded").x))
TEMP = {nm: pasang_T(Z[nm]["val"], yv) for nm in MODEL}
baris = []
for nm in MODEL:
    for versi, T in [("asli", 1.0), ("temperature scaling", TEMP[nm])]:
        p = sig(Z[nm]["test"] / T)
        baris.append({"model": nm, "versi": versi, "T": T, "AUC": roc_auc_score(yt, p), "ECE": ece(yt, p), "Brier": brier(yt, p), "NLL": nll(yt, p),
                      "ECE_CI95": ci_bootstrap(ece, yt, p, n_boot=NB // 4), "Brier_CI95": ci_bootstrap(brier, yt, p, n_boot=NB // 4)})
T113 = pd.DataFrame(baris); T113.to_csv(OUT / "tabel_11_3_kalibrasi.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, nm in zip(ax, MODEL):
    for versi, T, gaya in [("asli", 1.0, "o-"), (f"TS (T={TEMP[nm]:.2f})", TEMP[nm], "s--")]:
        p = sig(Z[nm]["test"] / T); b = np.minimum((p * 10).astype(int), 9)
        x = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 5]; yy = [yt[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
        a.plot(x, yy, gaya, ms=4, label=f"{versi}: ECE {ece(yt, p):.3f}")
    a.plot([0, 1], [0, 1], "k:", lw=1); a2 = a.twinx(); a2.hist(sig(Z[nm]["test"]), bins=20, range=(0, 1), alpha=0.2, color="gray"); a2.set_yticks([])
    a.set_xlabel("probabilitas diprediksi"); a.set_ylabel("frekuensi pneumonia teramati"); a.set_title(nm, fontsize=9); a.legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "gambar_11_1_reliability.png", bbox_inches="tight"); plt.show(); T113
""")

md(r"""
## 11.4 *Decision curve analysis* (DCA)
Net benefit pada ambang probabilitas `pt`: `NB = TP/n − FP/n × pt/(1 − pt)`. Pembanding: **rawat semua** dan **rawat tidak seorang pun** (NB = 0).
Panel kanan: prevalensi diturunkan ke 20 % dengan **subsampel acak** kasus positif di set uji (simulasi, bukan data populasi nyata);
model tidak dikalibrasi ulang untuk prevalensi baru.
""")
code(r"""
PT = np.linspace(0.01, 0.95, 95)
def net_benefit(y, p, pt): n = len(y); return np.array([((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in pt])
def rawat_semua(y, pt): prev = y.mean(); return prev - (1 - prev) * pt / (1 - pt)
r = np.random.default_rng(SEED); neg = np.where(yt == 0)[0]; pos = np.where(yt == 1)[0]
n_pos20 = int(round(len(neg) * 0.2 / 0.8)); idx20 = np.concatenate([neg, r.choice(pos, min(n_pos20, len(pos)), replace=False)])
fig, ax = plt.subplots(1, 2, figsize=(12, 4)); DCA = []
for a, (judul, idx) in zip(ax, [(f"set uji (prevalensi {yt.mean():.2f})", np.arange(len(yt))), (f"subsampel (prevalensi {yt[idx20].mean():.2f}, simulasi)", idx20)]):
    y_ = yt[idx]
    for nm in MODEL:
        nb = net_benefit(y_, sig(Z[nm]["test"][idx] / TEMP[nm]), PT); a.plot(PT, nb, label=f"{nm} (TS)")
        DCA += [{"skenario": judul, "model": nm, "pt": t, "net_benefit": v} for t, v in zip(PT, nb)]
    a.plot(PT, rawat_semua(y_, PT), "k--", label="rawat semua"); a.axhline(0, color="gray", lw=1, label="rawat tidak seorang pun")
    a.set_ylim(-0.05, y_.mean() + 0.05); a.set_xlabel("ambang probabilitas pt"); a.set_ylabel("net benefit"); a.set_title(judul, fontsize=9); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gambar_11_2_dca.png", bbox_inches="tight"); plt.show()
pd.DataFrame(DCA).to_csv(OUT / "tabel_11_4_dca.csv", index=False)
""")

md(r"""
## 11.5 Pergeseran distribusi sintetis
Model A dievaluasi pada set uji yang diberi gangguan dengan tingkat keparahan meningkat. Ini **simulasi** pergeseran (mis. alat/protokol berbeda),
bukan pengganti validasi eksternal pada data rumah sakit lain.
""")
code(r"""
def ganggu(x, jenis, s):
    x = x.float().div(255); g = torch.Generator(device="cpu").manual_seed(0)
    if jenis == "derau Gaussian": x = x + torch.randn(x.shape, generator=g).to(x.device) * s
    elif jenis == "blur Gaussian" and s > 0: k = int(2 * round(3 * s) + 1); x = torchvision.transforms.functional.gaussian_blur(x, [k, k], [s, s])
    elif jenis == "kontras turun": x = (x - x.mean((2, 3), keepdim=True)) * (1 - s) + x.mean((2, 3), keepdim=True)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)
LEVEL = {"derau Gaussian": [0, 0.05, 0.1, 0.2], "blur Gaussian": [0, 0.5, 1.0, 2.0], "kontras turun": [0, 0.3, 0.5, 0.7]}
baris = []
with catat_waktu("4_pergeseran"):
    for jenis, lv in LEVEL.items():
        for s in lv:
            p = sig(logit_semua(MODEL[nmA], ganggu(XT["test"], jenis, s))[:, 0].cpu().numpy().astype(np.float64) / TEMP[nmA])
            baris.append({"gangguan": jenis, "tingkat": s, "AUC": roc_auc_score(yt, p), "ECE": ece(yt, p), "proporsi_prediksi_positif@0.5": float((p >= 0.5).mean())})
T115 = pd.DataFrame(baris); T115.to_csv(OUT / "tabel_11_5_pergeseran.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for jenis in LEVEL:
    d_ = T115[T115.gangguan == jenis]; ax[0].plot(range(4), d_.AUC, "o-", label=jenis); ax[1].plot(range(4), d_.ECE, "o-", label=jenis)
for a, t in zip(ax, ["AUC", "ECE"]): a.set_xticks(range(4)); a.set_xticklabels(["0", "ringan", "sedang", "berat"]); a.set_ylabel(t); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gambar_11_3_pergeseran.png", bbox_inches="tight"); plt.show()
fig, ax = plt.subplots(1, 4, figsize=(10, 2.6))
for a, (jenis, s) in zip(ax, [("asli", 0), ("derau Gaussian", 0.2), ("blur Gaussian", 2.0), ("kontras turun", 0.7)]):
    a.imshow((ganggu(XT["test"][:1], jenis, s) if jenis != "asli" else XT["test"][:1])[0, 0].cpu(), cmap="gray", vmin=0, vmax=255); a.set_title(f"{jenis} {s if s else ''}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_11_4_contoh_gangguan.png", bbox_inches="tight"); plt.show(); T115
""")

md(r"""
## 11.6 Kesesuaian pengukuran (regresi HC18 cepat): Bland–Altman, ICC, CCC
Model regresi HC dilatih singkat (128 px, ≤15 epoch) — lebih lemah dari Bab 7 — cukup untuk memperagakan analisis kesesuaian.
ICC dihitung dari ANOVA dua arah (Shrout & Fleiss, 1979; McGraw & Wong, 1996): ICC(2,1) = kesepakatan absolut, ICC(3,1) = konsistensi.
""")
code(r"""
UKH = 64 if SMOKE else 128
with catat_waktu("5_hc18"):
    XH, _, META, _ = muat_hc18(UKH, maks=120 if SMOKE else None); SPL = bagi_subjek(META, seed=SEED)
    SK = (META.pixel_mm * META.lebar).values; TR = META.hc_mm.values / SK; mu, sd = TR[SPL["train"]].mean(), TR[SPL["train"]].std(); ZH = ((TR - mu) / sd).astype(np.float32)
    XHT = ke_tensor_uint8(XH)
    skor_hc = lambda lv, y: -float(np.abs(lv[:, 0].cpu().numpy() - y).mean())
    mh, _, _ = latih("HC18 regresi cepat", ResNet18(1, True), XHT[SPL["train"]], torch.as_tensor(ZH[SPL["train"]]), XHT[SPL["val"]], ZH[SPL["val"]],
                     lambda lg, y: F.smooth_l1_loss(lg[:, 0], y, beta=0.5), skor_hc, tahap=[(2, 1e-3, True), (13, 3e-4, False)], patience=5,
                     aug=dict(flip_h=True, flip_v=True, kecerahan=0.15, kontras=0.15), bs=32, seed=0)
    pred = (logit_semua(mh, XHT[SPL["test"]])[:, 0].cpu().numpy() * sd + mu) * SK[SPL["test"]]; acuan = META.hc_mm.values[SPL["test"]]
def icc(Yk):
    n, k = Yk.shape; g = Yk.mean(); ssr = k * ((Yk.mean(1) - g) ** 2).sum(); ssc = n * ((Yk.mean(0) - g) ** 2).sum(); sse = ((Yk - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / (msr + (k - 1) * mse)
def ccc(a, b): return 2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)
e = pred - acuan; rata = (pred + acuan) / 2; bias = e.mean(); sdd = e.std(ddof=1); kem = np.polyfit(rata, e, 1)
i2, i3 = icc(np.c_[pred, acuan])
KESESUAIAN = {"n_uji": len(e), "MAE_mm": float(np.abs(e).mean()), "bias_mm": float(bias), "LoA_bawah_mm": float(bias - 1.96 * sdd), "LoA_atas_mm": float(bias + 1.96 * sdd),
              "kemiringan_bias_proporsional": float(kem[0]), "ICC_2_1": float(i2), "ICC_3_1": float(i3), "CCC_Lin": float(ccc(pred, acuan)),
              "r_Pearson": float(np.corrcoef(pred, acuan)[0, 1])}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(acuan, pred, s=8, alpha=0.6); lim = [acuan.min() - 10, acuan.max() + 10]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("HC acuan (mm)"); ax[0].set_ylabel("HC model (mm)"); ax[0].set_title(f"r={KESESUAIAN['r_Pearson']:.3f}, CCC={KESESUAIAN['CCC_Lin']:.3f}, ICC(2,1)={i2:.3f}", fontsize=8)
ax[1].scatter(rata, e, s=8, alpha=0.6); ax[1].axhline(bias, color="r")
for v in (bias - 1.96 * sdd, bias + 1.96 * sdd): ax[1].axhline(v, color="r", ls="--")
xx = np.linspace(rata.min(), rata.max(), 10); ax[1].plot(xx, np.polyval(kem, xx), "g:", label=f"tren selisih (kemiringan {kem[0]:.3f})")
ax[1].set_xlabel("rerata (mm)"); ax[1].set_ylabel("model − acuan (mm)"); ax[1].set_title("Bland–Altman", fontsize=9); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gambar_11_5_bland_altman_icc.png", bbox_inches="tight"); plt.show(); KESESUAIAN
""")

md("## 11.7 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("11", 'dict(ci_bootstrap=T111.to_dict("records"), delong=DL, kalibrasi=T113.to_dict("records"), temperatur=TEMP, pergeseran=T115.to_dict("records"), kesesuaian_hc=KESESUAIAN)'))
simpan(C, "bab11.ipynb", Path(__file__).parent / "bab11.ipynb")
