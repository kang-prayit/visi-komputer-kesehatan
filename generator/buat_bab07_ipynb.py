"""Membangun notebook Colab Bab 7 (notebooks/bab07.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM, KODE_HC18

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 7 — Regresi: Lingkar Kepala Janin (HC18) dan Derajat Retinopati Ordinal (RetinaMNIST)
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi:
1. **HC18** (van den Heuvel dkk., 2018; CC BY 4.0): 999 citra USG berlabel lingkar kepala (HC, mm) dan ukuran piksel (mm/piksel).
   Pembagian 70/10/20 **per subjek** (berkas `014_HC`, `014_2HC`, … dianggap satu subjek).
2. Regresi langsung HC dengan ResNet-18 (pra-latih vs acak, 3 seed), target dinormalisasi; MAE/RMSE dalam **mm**, CI 95 % bootstrap,
   plot prediksi vs acuan dan **Bland–Altman** pendahuluan.
3. **RetinaMNIST 128 px** (MedMNIST v2, CC BY 4.0; sumber DeepDRiD): derajat retinopati 0–4 sebagai masalah **ordinal** —
   klasifikasi biasa (CE) vs regresi (MSE + pembulatan) vs ordinal kumulatif (gaya CORAL); metrik **quadratic weighted kappa (QWK)**.

Hasil → `/content/hasil_bab07/` → `/content/hasil_bab07.zip`. Model hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 7.0 Lingkungan")
code(sel_lingkungan("07"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH + "\n" + KODE_HC18 +
     "\nfrom sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error")

md(r"""
## 7.1 Data HC18
Citra asli 800×540 piksel diubah ke kotak `UK×UK`. Karena ukuran piksel (mm) berbeda antarcitra, target yang dipelajari model adalah
**HC dalam satuan lebar citra**: `t = HC_mm / (pixel_mm × lebar_px)`; prediksi dikembalikan ke mm dengan mengalikan `pixel_mm × lebar_px`.
Target kemudian distandardisasi (z-score dari set latih).
""")
code(r"""
UK = 64 if SMOKE else 256
with catat_waktu("2_muat_hc18"):
    XH, MH, META, AKAR = muat_hc18(UK, maks=120 if SMOKE else None)
    SPL = bagi_subjek(META, seed=SEED)
    assert not (set(META.subjek[SPL["train"]]) & set(META.subjek[SPL["test"]])) and not (set(META.subjek[SPL["val"]]) & set(META.subjek[SPL["test"]]))
SKALA = (META.pixel_mm * META.lebar).values                    # mm per "satu lebar citra"
T_REL = META.hc_mm.values / SKALA
MU, SD = T_REL[SPL["train"]].mean(), T_REL[SPL["train"]].std()
Z = ((T_REL - MU) / SD).astype(np.float32)
XHT = ke_tensor_uint8(XH)
T71 = pd.DataFrame([{"split": s, "n_citra": len(i), "n_subjek": META.subjek[i].nunique(), "HC_mm_rerata": META.hc_mm[i].mean(),
                     "HC_mm_sd": META.hc_mm[i].std(), "HC_mm_min": META.hc_mm[i].min(), "HC_mm_maks": META.hc_mm[i].max(),
                     "pixel_mm_min": META.pixel_mm[i].min(), "pixel_mm_maks": META.pixel_mm[i].max()} for s, i in SPL.items()])
T71.to_csv(OUT / "tabel_7_1_data_hc18.csv", index=False)
fig, ax = plt.subplots(2, 5, figsize=(13, 4.6)); r = np.random.default_rng(SEED)
for j, i in enumerate(r.choice(SPL["train"], 5, replace=False)):
    ax[0, j].imshow(XH[i], cmap="gray"); ax[0, j].set_title(f"HC {META.hc_mm[i]:.1f} mm", fontsize=8)
    ax[1, j].imshow(XH[i], cmap="gray"); ax[1, j].contour(MH[i], [0.5], colors="lime", linewidths=1)
    ax[0, j].axis("off"); ax[1, j].axis("off")
ax[1, 0].set_title("anotasi elips (diisi)", fontsize=8, loc="left")
plt.tight_layout(); plt.savefig(OUT / "gambar_7_1_contoh_hc18.png", bbox_inches="tight"); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(META.hc_mm, bins=40); ax[0].set_xlabel("HC (mm)"); ax[0].set_ylabel("jumlah citra")
ax[1].scatter(META.pixel_mm, META.hc_mm, s=4, alpha=0.5); ax[1].set_xlabel("ukuran piksel (mm)"); ax[1].set_ylabel("HC (mm)")
plt.tight_layout(); plt.savefig(OUT / "gambar_7_2_sebaran_hc.png", bbox_inches="tight"); plt.show(); T71
""")

md(r"""
## 7.2 Regresi langsung HC: ResNet-18 pra-latih vs acak (3 seed)
*Loss*: Smooth-L1 pada target z. *Early stopping*: MAE validasi (mm), patience 8. Augmentasi: flip horizontal/vertikal, geser ±5 %,
kecerahan/kontras ±15 % — **tanpa zoom/skala**, karena memperbesar citra akan mengubah HC yang seharusnya diprediksi.
""")
code(r"""
def ke_mm(z, idx): return (z * SD + MU) * SKALA[idx]
def skor_mae_val(lv, yv): return -float(np.abs(ke_mm(lv[:, 0].cpu().numpy(), SPL["val"]) - META.hc_mm.values[SPL["val"]]).mean())
loss_hc = lambda lg, y: F.smooth_l1_loss(lg[:, 0], y, beta=0.5)
AUG_HC = dict(flip_h=True, flip_v=True, geser=0.05, kecerahan=0.15, kontras=0.15)
SEEDS = [0] if SMOKE else [0, 1, 2]
Xtr, Xv, Xte = XHT[SPL["train"]], XHT[SPL["val"]], XHT[SPL["test"]]
ytr, yv = torch.as_tensor(Z[SPL["train"]]), Z[SPL["val"]]
KONFIG_HC = {"ResNet-18 pra-latih (2 tahap)": (True, [(3, 1e-3, True), (40, 3e-4, False)]),
             "ResNet-18 acak": (False, [(40, 1e-3, False)])}
HASIL_HC, LOG_HC, PRED_HC = [], {}, {}
yte_mm = META.hc_mm.values[SPL["test"]]
for nk, (pre, tahap) in KONFIG_HC.items():
    for s in SEEDS:
        with catat_waktu(f"latih_hc_{'pre' if pre else 'acak'}_s{s}"):
            m, lg, ep = latih(f"{nk} | seed {s}", ResNet18(1, pre), Xtr, ytr, Xv, yv, loss_hc, skor_mae_val, patience=8, aug=AUG_HC, tahap=tahap, bs=32, seed=s)
        p_mm = ke_mm(logit_semua(m, Xte)[:, 0].cpu().numpy(), SPL["test"]); PRED_HC[(nk, s)] = p_mm; LOG_HC[(nk, s)] = lg
        e = p_mm - yte_mm
        HASIL_HC.append({"konfigurasi": nk, "seed": s, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()), "detik_per_epoch": float(lg.detik_epoch.mean()),
                         "MAE_val_mm": -lg.skor_val.max(), "MAE_uji_mm": float(np.abs(e).mean()), "RMSE_uji_mm": float(np.sqrt((e ** 2).mean())),
                         "bias_uji_mm": float(e.mean()), "MAPE_uji_persen": float(np.mean(np.abs(e) / yte_mm) * 100)})
        print(HASIL_HC[-1])
        if pre and s == SEEDS[0]: torch.save(m.state_dict(), OUT / "resnet18_hc18_regresi.pt")
base = META.hc_mm.values[SPL["train"]].mean()
T72 = pd.DataFrame(HASIL_HC); T72.to_csv(OUT / "tabel_7_2_hc_per_seed.csv", index=False)
T73 = T72.groupby("konfigurasi")[["MAE_val_mm", "MAE_uji_mm", "RMSE_uji_mm", "bias_uji_mm", "MAPE_uji_persen", "epoch_terbaik", "detik_per_epoch"]].agg(["mean", "std"])
T73.columns = ["_".join(c) for c in T73.columns]
T73.loc["pembanding: selalu menebak rerata latih", "MAE_uji_mm_mean"] = float(np.abs(base - yte_mm).mean())
T73.loc["pembanding: selalu menebak rerata latih", "RMSE_uji_mm_mean"] = float(np.sqrt(((base - yte_mm) ** 2).mean()))
T73.to_csv(OUT / "tabel_7_3_hc_ringkas.csv"); pd.concat(LOG_HC.values()).to_csv(OUT / "log_epoch_hc.csv", index=False)
gambar_kurva([LOG_HC[(k, SEEDS[0])] for k in KONFIG_HC], "−MAE val (mm)", OUT / "gambar_7_3_learning_curve_hc.png")
T73
""")

md(r"""
## 7.3 Prediksi vs acuan, Bland–Altman, dan galat per rentang HC
Model yang dianalisis: konfigurasi pra-latih dengan **MAE validasi terkecil** (dipilih tanpa melihat set uji).
""")
code(r"""
kand = [(k, s) for (k, s) in PRED_HC if "pra-latih" in k]
PIL_HC = min(kand, key=lambda ks: T72[(T72.konfigurasi == ks[0]) & (T72.seed == ks[1])].MAE_val_mm.iloc[0]); p_mm = PRED_HC[PIL_HC]
e = p_mm - yte_mm; rata = (p_mm + yte_mm) / 2; bias = e.mean(); loa = (bias - 1.96 * e.std(ddof=1), bias + 1.96 * e.std(ddof=1))
mae_ci = ci_bootstrap(lambda y, p: np.abs(p - y).mean(), yte_mm, p_mm, n_boot=200 if SMOKE else 1000, strata=False)
BA_HC = {"model": f"{PIL_HC[0]} | seed {PIL_HC[1]}", "n_uji": len(e), "MAE_mm": float(np.abs(e).mean()), "MAE_CI95": mae_ci,
         "bias_mm": float(bias), "LoA_bawah_mm": float(loa[0]), "LoA_atas_mm": float(loa[1]),
         "persen_|galat|<=3mm": float((np.abs(e) <= 3).mean() * 100), "persen_|galat|<=5mm": float((np.abs(e) <= 5).mean() * 100)}
print(BA_HC)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(yte_mm, p_mm, s=8, alpha=0.6); lim = [yte_mm.min() - 10, yte_mm.max() + 10]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("HC acuan (mm)"); ax[0].set_ylabel("HC prediksi (mm)"); ax[0].set_title(f"Uji: MAE {BA_HC['MAE_mm']:.2f} mm")
ax[1].scatter(rata, e, s=8, alpha=0.6); ax[1].axhline(bias, color="r", label=f"bias {bias:.2f} mm")
for v in loa: ax[1].axhline(v, color="r", ls="--")
ax[1].set_xlabel("rerata (prediksi, acuan) (mm)"); ax[1].set_ylabel("prediksi − acuan (mm)"); ax[1].set_title(f"Bland–Altman, LoA [{loa[0]:.1f}, {loa[1]:.1f}] mm"); ax[1].legend(fontsize=7)
bins = pd.cut(yte_mm, [0, 100, 150, 200, 250, 400]); g = pd.DataFrame({"rentang": bins, "abs": np.abs(e)}).groupby("rentang", observed=True)["abs"].agg(["mean", "count"])
ax[2].bar(g.index.astype(str), g["mean"]); [ax[2].text(i, v, f"n={c}", ha="center", va="bottom", fontsize=7) for i, (v, c) in enumerate(zip(g["mean"], g["count"]))]
ax[2].set_xlabel("rentang HC acuan (mm)"); ax[2].set_ylabel("MAE (mm)"); ax[2].set_title("Galat per rentang HC")
plt.tight_layout(); plt.savefig(OUT / "gambar_7_4_prediksi_bland_altman.png", bbox_inches="tight"); plt.show()
g.to_csv(OUT / "tabel_7_4_mae_per_rentang.csv")
idx_buruk = np.argsort(-np.abs(e))[:5]
fig, ax = plt.subplots(1, 5, figsize=(13, 2.8))
for a, i in zip(ax, idx_buruk):
    a.imshow(XH[SPL["test"][i]], cmap="gray"); a.set_title(f"acuan {yte_mm[i]:.0f} / pred {p_mm[i]:.0f} mm", fontsize=7); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_7_5_galat_terbesar.png", bbox_inches="tight"); plt.show()
pd.DataFrame({"berkas": META.berkas.values[SPL["test"]], "hc_acuan_mm": yte_mm, "hc_prediksi_mm": p_mm}).to_csv(OUT / "prediksi_uji_hc.csv", index=False)
""")

md(r"""
## 7.4 RetinaMNIST: derajat retinopati sebagai masalah ordinal
Tiga formulasi dengan ResNet-18 pra-latih (2 tahap) dan augmentasi yang sama:
- **CE**: 5 kelas nominal, prediksi = argmax.
- **Regresi**: 1 keluaran, MSE terhadap derajat 0–4; prediksi = pembulatan (dipotong ke 0–4).
- **Ordinal kumulatif (gaya CORAL)**: 4 keluaran biner `P(y > k)`, k = 0..3; prediksi = jumlah `P > 0,5`.

Metrik utama **QWK** (`cohen_kappa_score(weights="quadratic")`): kesalahan dua derajat dihukum lebih berat daripada satu derajat.
Catatan: set validasi RetinaMNIST hanya 120 citra, sehingga *early stopping* berbasis QWK validasi cukup bising.
""")
code(r"""
UKR = 28 if SMOKE else 128
with catat_waktu("3_muat_retina"):
    assert INFO["retinamnist"]["license"] == "CC BY 4.0"
    d = npz_medmnist("retinamnist", UKR)
    DR = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE: DR = {s: (x[:200], y[:200]) for s, (x, y) in DR.items()}
XR = {s: ke_tensor_uint8(DR[s][0]) for s in DR}; YR = {s: DR[s][1] for s in DR}
T75 = pd.DataFrame([{"split": s, "n": len(YR[s]), **{f"derajat_{k}": int((YR[s] == k).sum()) for k in range(5)}} for s in DR]); T75.to_csv(OUT / "tabel_7_5_data_retina.csv", index=False)
def pred_ce(lg): return lg.argmax(1).cpu().numpy()
def pred_reg(lg): return np.clip(np.rint(lg[:, 0].cpu().numpy()), 0, 4).astype(int)
def pred_ord(lg): return (torch.sigmoid(lg) > 0.5).sum(1).cpu().numpy()
def loss_ord(lg, y): return F.binary_cross_entropy_with_logits(lg, (y[:, None] > torch.arange(4, device=y.device)[None]).float())
FORMULASI = {"CE (nominal)": (5, lambda lg, y: F.cross_entropy(lg, y), pred_ce),
             "regresi MSE + pembulatan": (1, lambda lg, y: F.mse_loss(lg[:, 0], y.float()), pred_reg),
             "ordinal kumulatif (CORAL)": (4, loss_ord, pred_ord)}
def qwk(y, yh): return cohen_kappa_score(y, yh, weights="quadratic")
AUG_R = dict(flip_h=True, flip_v=True, rot_derajat=15, kecerahan=0.1, kontras=0.1, warna=0.05)
HASIL_R, PRED_R, LOG_R = [], {}, {}
for nf, (k, fl, fp) in FORMULASI.items():
    for s in SEEDS:
        with catat_waktu(f"latih_retina_{nf[:5]}_s{s}"):
            m, lg, ep = latih(f"{nf} | seed {s}", ResNet18(k, True), XR["train"], YR["train"], XR["val"], YR["val"], fl,
                              lambda lv, yv, fp=fp: qwk(yv, fp(lv)), patience=6, aug=AUG_R, tahap=[(2, 1e-3, True), (30, 1e-4, False)], bs=64, seed=s)
        yh = fp(logit_semua(m, XR["test"])); PRED_R[(nf, s)] = yh; LOG_R[(nf, s)] = lg; yt = YR["test"]
        HASIL_R.append({"formulasi": nf, "seed": s, "epoch_terbaik": ep, "QWK_val": lg.skor_val.max(), "QWK_uji": qwk(yt, yh),
                        "akurasi_uji": float((yh == yt).mean()), "MAE_derajat_uji": float(np.abs(yh - yt).mean()),
                        "persen_galat>=2_derajat": float((np.abs(yh - yt) >= 2).mean() * 100)})
        print(HASIL_R[-1])
T76 = pd.DataFrame(HASIL_R); T76.to_csv(OUT / "tabel_7_6_retina_per_seed.csv", index=False)
T77 = T76.groupby("formulasi")[["QWK_val", "QWK_uji", "akurasi_uji", "MAE_derajat_uji", "persen_galat>=2_derajat", "epoch_terbaik"]].agg(["mean", "std"])
T77.columns = ["_".join(c) for c in T77.columns]; T77.to_csv(OUT / "tabel_7_7_retina_ringkas.csv"); pd.concat(LOG_R.values()).to_csv(OUT / "log_epoch_retina.csv", index=False)
T77
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, nf in zip(ax, FORMULASI):
    s = max(SEEDS, key=lambda s: T76[(T76.formulasi == nf) & (T76.seed == s)].QWK_val.iloc[0])
    cm = confusion_matrix(YR["test"], PRED_R[(nf, s)], labels=range(5)); a.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xlabel("prediksi"); a.set_ylabel("acuan"); a.set_title(f"{nf}\nseed {s}: QWK uji {qwk(YR['test'], PRED_R[(nf, s)]):.3f}", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_7_6_konfusi_retina.png", bbox_inches="tight"); plt.show()
# Ilustrasi bobot QWK: penalti kuadratik (i-j)^2/(K-1)^2
W = np.array([[(i - j) ** 2 / 16 for j in range(5)] for i in range(5)]); pd.DataFrame(W).round(3).to_csv(OUT / "tabel_7_8_bobot_qwk.csv")
""")

md("## 7.5 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("07", 'dict(ukuran_hc=UK, ukuran_retina=UKR, data_hc=T71.to_dict("records"), hc_per_seed=T72.to_dict("records"), hc_terpilih=BA_HC, data_retina=T75.to_dict("records"), retina_per_seed=T76.to_dict("records"))'))
simpan(C, "bab07.ipynb", Path(__file__).parent / "bab07.ipynb")
