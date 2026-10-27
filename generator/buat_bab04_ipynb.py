"""Membangun notebook Colab Bab 4 (notebooks/bab04.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 4 — Klasifikasi Biner: Deteksi Pneumonia pada X-ray Dada
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4** (*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

Isi:
1. PneumoniaMNIST **224 px** (MedMNIST v2, CC BY 4.0; sumber asli Kermany dkk., 2018): distribusi kelas dan prevalensi per split.
2. ResNet-18 pra-latih ImageNet (dua tahap: beku → *fine-tune*), augmentasi tanpa flip kiri–kanan, **tanpa vs dengan bobot kelas**, 3 seed.
3. Metrik klinis: sensitivitas, spesifisitas, PPV, NPV, AUC, dengan CI 95 % bootstrap.
4. **Memilih ambang (*operating point*)** pada set validasi: 0,5; Youden; sensitivitas ≥ 95 % (skrining); spesifisitas ≥ 95 % (konfirmasi) — lalu diuji pada set uji.
5. PPV/NPV sebagai fungsi prevalensi, dan analisis kesalahan (FP/FN paling yakin).

Hasil → `/content/hasil_bab04/` → `/content/hasil_bab04.zip`. Model hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 4.0 Lingkungan")
code(sel_lingkungan("04"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nfrom sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix")

md("## 4.1 Data")
code(r"""
UK = 28 if SMOKE else 224
with catat_waktu("2_muat_data"):
    assert INFO["pneumoniamnist"]["license"] == "CC BY 4.0"
    d = npz_medmnist("pneumoniamnist", UK)
    D = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0); D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), min(len(y), 300), replace=False)]}
XT = {s: ke_tensor_uint8(D[s][0]) for s in D}; Y = {s: D[s][1] for s in D}
T41 = pd.DataFrame([{"split": s, "n": len(Y[s]), "normal": int((Y[s] == 0).sum()), "pneumonia": int((Y[s] == 1).sum()),
                     "prevalensi_pneumonia": float(Y[s].mean())} for s in D])
T41.to_csv(OUT / "tabel_4_1_data.csv", index=False)
fig, ax = plt.subplots(2, 6, figsize=(12, 4.4)); r = np.random.default_rng(SEED)
for k in range(2):
    for j, i in enumerate(r.choice(np.where(Y["train"] == k)[0], 6, replace=False)):
        ax[k, j].imshow(D["train"][0][i], cmap="gray"); ax[k, j].axis("off")
    ax[k, 0].set_title(["normal", "pneumonia"][k], loc="left")
plt.tight_layout(); plt.savefig(OUT / "gambar_4_1_contoh.png", bbox_inches="tight"); plt.show(); T41
""")

md(r"""
## 4.2 Pelatihan: tanpa vs dengan bobot kelas (3 seed)
Kelas pneumonia lebih banyak di set latih. Dengan **bobot kelas**, *loss* sampel normal diberi bobot lebih besar
(setara `pos_weight = n_normal / n_pneumonia` pada BCE). Augmentasi: rotasi ±7°, geser ±5 %, kecerahan/kontras ±10 %, **tanpa flip**.
*Early stopping* memantau AUC validasi.
""")
code(r"""
n0, n1 = int((Y["train"] == 0).sum()), int((Y["train"] == 1).sum())
AUG = dict(rot_derajat=7, geser=0.05, kecerahan=0.1, kontras=0.1)
def skor_auc(lg, y): return roc_auc_score(y, lg[:, 0].cpu().numpy()) if len(np.unique(y)) > 1 else 0.0
def buat_loss(pos_weight):
    pw = torch.tensor([pos_weight], device=DEV)
    return lambda lg, y: F.binary_cross_entropy_with_logits(lg[:, 0], y.float(), pos_weight=pw)
KONFIG = {"tanpa bobot kelas": 1.0, "dengan bobot kelas": n0 / n1}
SEEDS = [0] if SMOKE else [0, 1, 2]
TAHAP = [(2, 1e-3, True), (20, 1e-4, False)]
HASIL, LOG, PROB = [], {}, {}
for nk, pw in KONFIG.items():
    for s in SEEDS:
        nama = f"{nk} | seed {s}"
        with catat_waktu(f"latih_{nk}_s{s}"):
            m, lg, ep = latih(nama, ResNet18(1, True), XT["train"], Y["train"], XT["val"], Y["val"], buat_loss(pw), skor_auc,
                              patience=4, aug=AUG, tahap=TAHAP, bs=64, seed=s)
        pv = torch.sigmoid(logit_semua(m, XT["val"])[:, 0]).cpu().numpy(); pt = torch.sigmoid(logit_semua(m, XT["test"])[:, 0]).cpu().numpy()
        PROB[(nk, s)] = (pv, pt); LOG[(nk, s)] = lg
        yt = Y["test"]; yh = pt >= 0.5
        HASIL.append({"konfigurasi": nk, "seed": s, "pos_weight": pw, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()),
                      "detik_per_epoch": float(lg.detik_epoch.mean()), "AUC_val": roc_auc_score(Y["val"], pv), "AUC_uji": roc_auc_score(yt, pt),
                      "sensitivitas_uji@0.5": float(yh[yt == 1].mean()), "spesifisitas_uji@0.5": float((~yh[yt == 0]).mean())})
        print(HASIL[-1])
T42 = pd.DataFrame(HASIL); T42.to_csv(OUT / "tabel_4_2_per_seed.csv", index=False)
T43 = T42.groupby("konfigurasi")[["AUC_val", "AUC_uji", "sensitivitas_uji@0.5", "spesifisitas_uji@0.5", "epoch_terbaik", "detik_per_epoch"]].agg(["mean", "std"])
T43.columns = ["_".join(c) for c in T43.columns]; T43.to_csv(OUT / "tabel_4_3_ringkas_konfigurasi.csv")
pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
gambar_kurva([LOG[(k, SEEDS[0])] for k in KONFIG], "AUC val", OUT / "gambar_4_2_learning_curve.png")
T43
""")

md(r"""
## 4.3 Memilih *operating point* pada set validasi
Model yang dianalisis: konfigurasi **dengan bobot kelas**, seed dengan **AUC validasi tertinggi** (dipilih tanpa melihat set uji).
Ambang ditentukan dari set **validasi**, lalu diterapkan apa adanya pada set **uji**.
""")
code(r"""
kandidat = [(k, s) for (k, s) in PROB if k == "dengan bobot kelas"]
PILIH = max(kandidat, key=lambda ks: roc_auc_score(Y["val"], PROB[ks][0])); pv, pt = PROB[PILIH]; yv, yt = Y["val"], Y["test"]
fpr, tpr, thr = roc_curve(yv, pv); thr = np.clip(thr, 0, 1)
def ambang_sens(target):   # ambang TERTINGGI yang masih memberi sensitivitas val >= target
    ok = tpr >= target; return float(thr[ok][0]) if ok.any() else 0.0
def ambang_spes(target):   # ambang TERENDAH yang masih memberi spesifisitas val >= target
    ok = (1 - fpr) >= target; return float(thr[ok][-1]) if ok.any() else 1.0
AMBANG = {"0,5 (bawaan)": 0.5, "Youden (maks sens+spes−1)": float(thr[np.argmax(tpr - fpr)]),
          "skrining: sens val ≥ 95 %": ambang_sens(0.95), "konfirmasi: spes val ≥ 95 %": ambang_spes(0.95)}
def metrik_ambang(y, p, t):
    yh = p >= t; tp = int((yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, sensitivitas=tp / max(tp + fn, 1), spesifisitas=tn / max(tn + fp, 1),
                PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1), akurasi=(tp + tn) / len(y))
baris = []
with catat_waktu("3_operating_point_bootstrap"):
    for nama, t in AMBANG.items():
        mv, mt = metrik_ambang(yv, pv, t), metrik_ambang(yt, pt, t)
        b = {"ambang_nama": nama, "ambang": t, "sens_val": mv["sensitivitas"], "spes_val": mv["spesifisitas"], **{f"{k}_uji": v for k, v in mt.items()}}
        for k in ["sensitivitas", "spesifisitas", "PPV", "NPV"]:
            lo, hi = ci_bootstrap(lambda y, p, k=k: metrik_ambang(y, p, t)[k], yt, pt, n_boot=200 if SMOKE else 1000)
            b[f"{k}_uji_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
        baris.append(b)
    auc_ci = ci_bootstrap(roc_auc_score, yt, pt, n_boot=200 if SMOKE else 1000)
T44 = pd.DataFrame(baris); T44.to_csv(OUT / "tabel_4_4_operating_point.csv", index=False)
INFO_PILIH = {"konfigurasi": PILIH[0], "seed": PILIH[1], "AUC_val": roc_auc_score(yv, pv), "AUC_uji": roc_auc_score(yt, pt), "AUC_uji_CI95": auc_ci,
              "prevalensi_val": float(yv.mean()), "prevalensi_uji": float(yt.mean())}
print(INFO_PILIH); T44
""")

code(r"""
fpr_t, tpr_t, _ = roc_curve(yt, pt)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(fpr_t, tpr_t, label=f"uji AUC {INFO_PILIH['AUC_uji']:.3f}"); ax[0].plot(fpr, tpr, alpha=0.6, label=f"val AUC {INFO_PILIH['AUC_val']:.3f}")
mk = ["o", "s", "^", "D"]
for (nama, t), m_ in zip(AMBANG.items(), mk):
    mt = metrik_ambang(yt, pt, t); ax[0].plot(1 - mt["spesifisitas"], mt["sensitivitas"], m_, ms=8, label=f"{nama}: t={t:.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel("1 − spesifisitas"); ax[0].set_ylabel("sensitivitas"); ax[0].legend(fontsize=6); ax[0].set_title("ROC + operating point (pada uji)")
for k, c in [(0, "tab:blue"), (1, "tab:red")]: ax[1].hist(pt[yt == k], bins=30, range=(0, 1), alpha=0.5, color=c, label=["normal", "pneumonia"][k])
for (nama, t), ls in zip(AMBANG.items(), ["-", "--", "-.", ":"]): ax[1].axvline(t, color="k", ls=ls, lw=1, label=nama)
ax[1].set_xlabel("probabilitas pneumonia (uji)"); ax[1].legend(fontsize=6); ax[1].set_title("Sebaran skor per kelas")
prev = np.linspace(0.01, 0.6, 60)
for nama in ["skrining: sens val ≥ 95 %", "konfirmasi: spes val ≥ 95 %"]:
    mt = metrik_ambang(yt, pt, AMBANG[nama]); se, sp = mt["sensitivitas"], mt["spesifisitas"]
    ppv = se * prev / (se * prev + (1 - sp) * (1 - prev)); npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
    ax[2].plot(prev, ppv, label=f"PPV – {nama.split(':')[0]}"); ax[2].plot(prev, npv, "--", label=f"NPV – {nama.split(':')[0]}")
ax[2].set_xlabel("prevalensi"); ax[2].set_ylabel("nilai prediktif"); ax[2].legend(fontsize=6); ax[2].set_title("PPV/NPV vs prevalensi (teorema Bayes)"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "gambar_4_3_roc_ambang_prevalensi.png", bbox_inches="tight"); plt.show()
""")

md("## 4.4 Analisis kesalahan: FP dan FN paling yakin (ambang Youden)")
code(r"""
t = AMBANG["Youden (maks sens+spes−1)"]
fp_idx = np.where((pt >= t) & (yt == 0))[0]; fn_idx = np.where((pt < t) & (yt == 1))[0]
fp_idx = fp_idx[np.argsort(-pt[fp_idx])][:6]; fn_idx = fn_idx[np.argsort(pt[fn_idx])][:6]
fig, ax = plt.subplots(2, 6, figsize=(12, 4.6))
for r_, (idx, judul) in enumerate([(fp_idx, "FP (normal → diprediksi pneumonia)"), (fn_idx, "FN (pneumonia → diprediksi normal)")]):
    for j in range(6):
        ax[r_, j].axis("off")
        if j < len(idx): ax[r_, j].imshow(D["test"][0][idx[j]], cmap="gray"); ax[r_, j].set_title(f"p={pt[idx[j]]:.3f}", fontsize=8)
    ax[r_, 0].text(0, -0.25, judul, transform=ax[r_, 0].transAxes, fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "gambar_4_4_fp_fn.png", bbox_inches="tight"); plt.show()
KESALAHAN = {"ambang_youden": t, "n_FP": int(((pt >= t) & (yt == 0)).sum()), "n_FN": int(((pt < t) & (yt == 1)).sum())}; KESALAHAN
""")

md("## 4.5 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("04", 'dict(ukuran_px=UK, data=T41.to_dict("records"), per_seed=T42.to_dict("records"), model_dianalisis=INFO_PILIH, operating_point=T44.to_dict("records"), kesalahan=KESALAHAN)'))
simpan(C, "bab04.ipynb", Path(__file__).parent / "bab04.ipynb")
