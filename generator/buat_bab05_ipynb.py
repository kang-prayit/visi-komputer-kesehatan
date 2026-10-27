"""Membangun notebook Colab Bab 5 (notebooks/bab05.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 5 — Klasifikasi Multikelas dan Data Tidak Seimbang: Lesi Kulit
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4** (*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

> **Lisensi data.** DermaMNIST (MedMNIST v2) berasal dari HAM10000 (Tschandl dkk., 2018) dan berlisensi **CC BY-NC 4.0**:
> boleh dipakai dan dibagikan dengan atribusi **untuk tujuan non-komersial**. Gambar dari dataset ini di buku/hasil tidak boleh dipakai komersial.

Isi:
1. DermaMNIST 64 px (7 kelas, sangat tidak seimbang): distribusi kelas.
2. ResNet-18 pra-latih ImageNet (masukan diperbesar ke 128 px) dengan empat strategi ketidakseimbangan:
   **CE biasa**, **CE berbobot kelas**, ***focal loss*** (γ = 2), dan ***oversampling*** (sampling seimbang). 2 seed per strategi.
3. Metrik multikelas: akurasi, *balanced accuracy*, F1 makro/berbobot/mikro, AUC makro *one-vs-rest*, recall per kelas (terutama **melanoma**).
4. Matriks kebingungan ternormalisasi.

Hasil → `/content/hasil_bab05/` → `/content/hasil_bab05.zip`. Model hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 5.0 Lingkungan")
code(sel_lingkungan("05"))
code('with catat_waktu("1_pip_install"):\n    jalankan(\'pip install -q "medmnist==3.0.2"\')\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nfrom sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix")

md("## 5.1 Data dan ketidakseimbangan kelas")
code(r"""
UK = 28 if SMOKE else 64
NAMA = ["keratosis aktinik/IEC", "karsinoma sel basal", "lesi mirip keratosis jinak", "dermatofibroma", "melanoma", "nevus melanositik", "lesi vaskular"]
K = len(NAMA); MEL = 4
with catat_waktu("2_muat_data"):
    assert INFO["dermamnist"]["license"] == "CC BY-NC 4.0" and len(INFO["dermamnist"]["label"]) == K
    d = npz_medmnist("dermamnist", UK)
    D = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0); D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), min(len(y), 400), replace=False)]}
XT = {s: ke_tensor_uint8(D[s][0]) for s in D}; Y = {s: D[s][1] for s in D}
T51 = pd.DataFrame({s: np.bincount(Y[s], minlength=K) for s in D}, index=NAMA)
T51["proporsi_latih"] = T51["train"] / T51["train"].sum(); T51.loc["total"] = T51.sum(); T51.to_csv(OUT / "tabel_5_1_distribusi_kelas.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
cnt = np.bincount(Y["train"], minlength=K); ax[0].barh(NAMA, cnt, color=["tab:red" if i == MEL else "tab:blue" for i in range(K)])
for i, c in enumerate(cnt): ax[0].text(c, i, f" {c} ({c/cnt.sum():.1%})", va="center", fontsize=7)
ax[0].set_xscale("log"); ax[0].set_xlabel("jumlah citra latih (log)"); ax[0].set_title("Distribusi kelas (latih)")
ax[1].axis("off"); r = np.random.default_rng(SEED); sub = ax[1].inset_axes([0, 0, 1, 1]); sub.axis("off")
grid = np.concatenate([np.concatenate([D["train"][0][r.choice(np.where(Y["train"] == k)[0])] for _ in range(3)], 0) for k in range(K)], 1)
sub.imshow(grid); sub.set_title("Contoh per kelas (kolom = " + ", ".join(str(i) for i in range(K)) + ") — DermaMNIST, CC BY-NC 4.0", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_5_1_distribusi_contoh.png", bbox_inches="tight"); plt.show(); T51
""")

md(r"""
## 5.2 Empat strategi ketidakseimbangan
- **CE biasa** — *cross-entropy* standar.
- **CE berbobot** — bobot kelas $w_k = N / (K\,n_k)$.
- **Focal loss** (Lin dkk., 2017) — $(1-p_t)^{\gamma}\,\mathrm{CE}$ dengan γ = 2, menekan contoh yang sudah mudah.
- **Oversampling** — setiap sampel diambil dengan peluang ∝ $1/n_k$ (kelas seimbang di setiap epoch, dengan pengembalian).
Augmentasi: flip H/V, rotasi 90°, kecerahan/kontras ±10 %, warna ±5 %. *Early stopping* memantau *balanced accuracy* validasi.
""")
code(r"""
n_k = np.bincount(Y["train"], minlength=K).astype(np.float64)
w_k = torch.tensor(len(Y["train"]) / (K * np.maximum(n_k, 1)), dtype=torch.float32, device=DEV)
def ce(lg, y): return F.cross_entropy(lg, y)
def ce_bobot(lg, y): return F.cross_entropy(lg, y, weight=w_k)
def focal(lg, y, gamma=2.0):
    l = F.cross_entropy(lg, y, reduction="none"); return ((1 - torch.exp(-l)) ** gamma * l).mean()
def skor_bacc(lg, y): return balanced_accuracy_score(y, lg.argmax(1).cpu().numpy())
bobot_os = (1.0 / np.maximum(n_k, 1))[Y["train"]]
STRATEGI = {"CE biasa": dict(fungsi_loss=ce), "CE berbobot": dict(fungsi_loss=ce_bobot),
            "focal loss (γ=2)": dict(fungsi_loss=focal), "oversampling": dict(fungsi_loss=ce, bobot_sampel=bobot_os)}
AUG = dict(flip_h=True, flip_v=True, rot90=True, kecerahan=0.1, kontras=0.1, warna=0.05)
TAHAP = [(2, 1e-3, True), (25, 1e-4, False)]
SEEDS = [0] if SMOKE else [0, 1]
HASIL, LOG, PROB = [], {}, {}
def ringkas(y, p):
    yh = p.argmax(1)
    try: auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(K)))
    except ValueError: auc = float("nan")
    return {"akurasi": accuracy_score(y, yh), "akurasi_seimbang": balanced_accuracy_score(y, yh), "F1_makro": f1_score(y, yh, average="macro", labels=list(range(K)), zero_division=0),
            "F1_berbobot": f1_score(y, yh, average="weighted", zero_division=0), "F1_mikro": f1_score(y, yh, average="micro"), "AUC_makro_OvR": auc,
            "recall_melanoma": recall_score(y, yh, labels=[MEL], average=None, zero_division=0)[0]}
for nk, kw in STRATEGI.items():
    for s in SEEDS:
        with catat_waktu(f"latih_{nk}_s{s}"):
            m, lg, ep = latih(f"{nk} | seed {s}", ResNet18(K, True, uk=128), XT["train"], Y["train"], XT["val"], Y["val"],
                              skor_val=skor_bacc, patience=5, aug=AUG, tahap=TAHAP, bs=128, seed=s, **kw)
        p = torch.softmax(logit_semua(m, XT["test"]), 1).cpu().numpy(); pv = torch.softmax(logit_semua(m, XT["val"]), 1).cpu().numpy()
        PROB[(nk, s)] = (pv, p); LOG[(nk, s)] = lg
        HASIL.append({"strategi": nk, "seed": s, "epoch_terbaik": ep, "epoch_dijalankan": int(lg.epoch.max()), "detik_per_epoch": float(lg.detik_epoch.mean()),
                      "bacc_val": balanced_accuracy_score(Y["val"], pv.argmax(1)), **ringkas(Y["test"], p)})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in HASIL[-1].items()})
T52 = pd.DataFrame(HASIL); T52.to_csv(OUT / "tabel_5_2_per_seed.csv", index=False)
kol = ["akurasi", "akurasi_seimbang", "F1_makro", "F1_berbobot", "AUC_makro_OvR", "recall_melanoma", "epoch_terbaik", "detik_per_epoch"]
T53 = T52.groupby("strategi", sort=False)[kol].agg(["mean", "std"]); T53.columns = ["_".join(c) for c in T53.columns]
T53.to_csv(OUT / "tabel_5_3_ringkas_strategi.csv"); pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
gambar_kurva([LOG[(k, SEEDS[0])] for k in STRATEGI], "balanced acc val", OUT / "gambar_5_2_learning_curve.png")
T53
""")

md("## 5.3 Recall per kelas dan matriks kebingungan")
code(r"""
s0 = SEEDS[0]
T54 = pd.DataFrame({nk: recall_score(Y["test"], PROB[(nk, s0)][1].argmax(1), labels=list(range(K)), average=None, zero_division=0) for nk in STRATEGI}, index=NAMA)
T54.insert(0, "n_uji", np.bincount(Y["test"], minlength=K)); T54.to_csv(OUT / "tabel_5_4_recall_per_kelas_seed0.csv")
terbaik = max(STRATEGI, key=lambda k: T52[T52.strategi == k].bacc_val.mean())     # dipilih dari validasi
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a, nk in zip(ax, ["CE biasa", terbaik if terbaik != "CE biasa" else "CE berbobot"]):
    cm = confusion_matrix(Y["test"], PROB[(nk, s0)][1].argmax(1), labels=range(K)); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K): a.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.6 else "black")
    a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(NAMA, rotation=40, ha="right", fontsize=7); a.set_yticklabels(NAMA, fontsize=7)
    a.set_xlabel("prediksi"); a.set_ylabel("sebenarnya"); a.set_title(f"{nk} (seed {s0}) — baris dinormalisasi", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "gambar_5_3_matriks_kebingungan.png", bbox_inches="tight"); plt.show()
fig, ax = plt.subplots(figsize=(10, 3.6)); w = 0.8 / len(STRATEGI)
for j, nk in enumerate(STRATEGI): ax.bar(np.arange(K) + (j - 1.5) * w, T54[nk], w, label=nk)
ax.set_xticks(range(K)); ax.set_xticklabels(NAMA, rotation=20, ha="right", fontsize=7); ax.set_ylabel("recall (uji)"); ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=4); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "gambar_5_4_recall_per_kelas.png", bbox_inches="tight"); plt.show()
print("strategi terbaik menurut balanced accuracy validasi:", terbaik); T54.round(3)
""")

md("## 5.4 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("05", 'dict(ukuran_px=UK, lisensi_data="CC BY-NC 4.0", distribusi=T51.to_dict(), per_seed=T52.to_dict("records"), terbaik_menurut_val=terbaik)'))
simpan(C, "bab05.ipynb", Path(__file__).parent / "bab05.ipynb")
