"""Membangun notebook Colab Bab 6 (notebooks/bab06.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 6 — Citra Mikroskopi: Malaria, Split per Pasien, dan Agregasi Tingkat Pasien
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4** (*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

Isi:
1. NIH Malaria Cell Images (Rajaraman dkk., 2018; 27.558 sel) dibagi **per kelompok/pasien** menjadi latih/validasi/uji (70/10/20 % kelompok).
2. CNN **ringan** (konvolusi *depthwise-separable*) dengan tiga perlakuan warna: tanpa augmentasi warna, **augmentasi warna**, **normalisasi gray-world**;
   pembanding: MobileNetV3-Small pra-latih ImageNet (torchvision). 3 seed.
3. Metrik tingkat sel (AUC, sensitivitas, spesifisitas).
4. **Agregasi tingkat pasien**: dari skor sel ke skor pasien (rerata probabilitas; proporsi sel positif), AUC dan ambang dari validasi.
5. Kesiapan perangkat murah: jumlah parameter, ukuran berkas, latensi CPU.

Kode kelompok diambil dari awalan nama berkas `C<n>` — diasumsikan mewakili satu pasien (lihat Bab 2).
Hasil → `/content/hasil_bab06/` → `/content/hasil_bab06.zip`. Model hanya untuk pembelajaran, **bukan** untuk penggunaan klinis.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 6.0 Lingkungan")
code(sel_lingkungan("06"))
code('with catat_waktu("1_pip_install"):\n    pass   # tidak perlu paket tambahan (medmnist tidak dipakai di bab ini)\n' +
     IMPOR_UMUM.replace("import torchvision, sklearn, medmnist\nfrom medmnist import INFO", "import torchvision, sklearn\nmedmnist = type('m', (), {'__version__': '-'})")
     + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nimport re\nfrom PIL import Image\nfrom concurrent.futures import ThreadPoolExecutor\nfrom sklearn.model_selection import GroupShuffleSplit\nfrom sklearn.metrics import roc_auc_score, roc_curve, accuracy_score")

md("## 6.1 Data dan pembagian per pasien")
code(r"""
with catat_waktu("2_malaria_siapkan"):
    pastikan_dataset("malaria"); akar = ekstrak_lokal("malaria", "cell_images.zip")
    berkas = [(p, lab, re.match(r"^(C\d+)", p.name).group(1)) for kelas, lab in [("Parasitized", 1), ("Uninfected", 0)] for p in sorted(akar.rglob(f"{kelas}/*.png"))]
    DF = pd.DataFrame(berkas, columns=["path", "label", "kelompok"])
    if SMOKE:
        pk = DF.groupby("kelompok").label.max(); kel = list(pk[pk == 1].index[:25]) + list(pk[pk == 0].index[:15]); DF = DF[DF.kelompok.isin(kel)].groupby(["kelompok", "label"]).head(10).reset_index(drop=True)
    UK = 64
    def baca(p): return np.asarray(Image.open(p).convert("RGB").resize((UK, UK), Image.BILINEAR), dtype=np.uint8)
    with ThreadPoolExecutor(8) as ex: XM = np.stack(list(ex.map(baca, DF.path)))
YM = DF.label.values.astype(np.int64); GM = DF.kelompok.values
g1 = GroupShuffleSplit(1, test_size=0.2, random_state=SEED); i_sisa, i_uji = next(g1.split(XM, YM, GM))
g2 = GroupShuffleSplit(1, test_size=0.125, random_state=SEED); a, b = next(g2.split(i_sisa, YM[i_sisa], GM[i_sisa])); i_lat, i_val = i_sisa[a], i_sisa[b]
IDX = {"train": i_lat, "val": i_val, "test": i_uji}
assert not (set(GM[i_lat]) & set(GM[i_uji])) and not (set(GM[i_lat]) & set(GM[i_val]))
PASIEN_POS = DF.groupby("kelompok").label.max()          # kelompok berisi sel Parasitized = pasien terinfeksi
T61 = pd.DataFrame([{"split": s, "sel": len(i), "sel_parasitized": int(YM[i].sum()), "kelompok": len(set(GM[i])),
                     "kelompok_terinfeksi": int(PASIEN_POS[sorted(set(GM[i]))].sum()), "kelompok_sehat": int((PASIEN_POS[sorted(set(GM[i]))] == 0).sum())} for s, i in IDX.items()])
T61.to_csv(OUT / "tabel_6_1_split_per_pasien.csv", index=False)
XT = {s: ke_tensor_uint8(XM[i]) for s, i in IDX.items()}; Y = {s: YM[i] for s, i in IDX.items()}
print(XM.shape); T61
""")

md(r"""
## 6.2 Variasi pewarnaan: augmentasi warna dan normalisasi *gray-world*
Warna apusan Giemsa berbeda antarlaboratorium dan antarslide. Dua cara umum: **augmentasi warna** (model dilatih melihat variasi)
dan **normalisasi** (citra disamakan sebelum masuk model). *Gray-world* menyamakan rerata tiap kanal; metode khusus histopatologi
(Macenko, Reinhard) dibahas di teks.
""")
code(r"""
def gray_world_t(x):   # x (N,3,H,W) [0,1]
    m = x.mean((2, 3), keepdim=True); return (x * (m.mean(1, keepdim=True) / m.clamp(min=1e-3))).clamp(0, 1)
AUG_DASAR = dict(flip_h=True, flip_v=True, rot90=True, kecerahan=0.05, kontras=0.05)
AUG_WARNA = dict(flip_h=True, flip_v=True, rot90=True, kecerahan=0.2, kontras=0.2, warna=0.15)
torch.manual_seed(SEED)
r = np.random.default_rng(SEED); pil = r.choice(np.where(Y["train"] == 1)[0], 4, replace=False)
x0 = XT["train"][torch.as_tensor(pil, device=XT["train"].device)].float().div(255)
fig, ax = plt.subplots(4, 6, figsize=(9, 6.4))
for i in range(4):
    kol = [("asli", x0[i:i + 1]), ("gray-world", gray_world_t(x0[i:i + 1]))] + [(f"aug warna {k+1}", augmentasi_gpu(x0[i:i + 1], **AUG_WARNA)) for k in range(4)]
    for j, (judul, im) in enumerate(kol):
        ax[i, j].imshow(im[0].permute(1, 2, 0).cpu().numpy()); ax[i, j].axis("off")
        if i == 0: ax[i, j].set_title(judul, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "gambar_6_1_warna.png", bbox_inches="tight"); plt.show()
""")

md("## 6.3 Model ringan dan pelatihan (3 seed)")
code(r"""
class CNNRingan(nn.Module):
    # konvolusi depthwise-separable (gagasan MobileNet): depthwise 3x3 per kanal + pointwise 1x1
    def __init__(self):
        super().__init__()
        def ds(i, o, s): return nn.Sequential(nn.Conv2d(i, i, 3, s, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                              nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                               ds(16, 32, 2), ds(32, 64, 2), ds(64, 96, 2), ds(96, 128, 1))
        self.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, x): return self.h(self.f((x - 0.5) / 0.5))
class MobileNetV3S(nn.Module):
    def __init__(self):
        super().__init__()
        try: self.net = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception as e:
            if not SMOKE: raise
            print("SMOKE: bobot gagal diunduh:", e); self.net = torchvision.models.mobilenet_v3_small(weights=None)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 1)
    def forward(self, x):
        x = F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False)
        return self.net((x - MEAN_IN.to(x.device)) / SD_IN.to(x.device))
def bce(lg, y): return F.binary_cross_entropy_with_logits(lg[:, 0], y.float())
def skor_auc(lg, y): return roc_auc_score(y, lg[:, 0].cpu().numpy()) if len(np.unique(y)) > 1 else 0.0
KONFIG = {"ringan, tanpa aug warna": dict(model=CNNRingan, aug=AUG_DASAR, praproses=None, lr=2e-3),
          "ringan + aug warna": dict(model=CNNRingan, aug=AUG_WARNA, praproses=None, lr=2e-3),
          "ringan + gray-world": dict(model=CNNRingan, aug=AUG_DASAR, praproses=gray_world_t, lr=2e-3),
          "MobileNetV3-Small pra-latih + aug warna": dict(model=MobileNetV3S, aug=AUG_WARNA, praproses=None, lr=5e-4)}
SEEDS = [0] if SMOKE else [0, 1, 2]
HASIL, LOG, PROB, MODEL = [], {}, {}, {}
for nk, kf in KONFIG.items():
    for s in SEEDS:
        with catat_waktu(f"latih_{nk}_s{s}"):
            m, lg, ep = latih(f"{nk} | seed {s}", kf["model"](), XT["train"], Y["train"], XT["val"], Y["val"], bce, skor_auc,
                              maks_epoch=15, patience=4, aug=kf["aug"], praproses=kf["praproses"], lr=kf["lr"], bs=256, seed=s)
        pv = torch.sigmoid(logit_semua(m, XT["val"], praproses=kf["praproses"])[:, 0]).cpu().numpy()
        pt = torch.sigmoid(logit_semua(m, XT["test"], praproses=kf["praproses"])[:, 0]).cpu().numpy()
        PROB[(nk, s)] = (pv, pt); LOG[(nk, s)] = lg; MODEL[(nk, s)] = m
        yt = Y["test"]; yh = pt >= 0.5
        HASIL.append({"konfigurasi": nk, "seed": s, "parameter": n_param(m), "epoch_terbaik": ep, "detik_per_epoch": float(lg.detik_epoch.mean()),
                      "AUC_val": roc_auc_score(Y["val"], pv), "AUC_uji_sel": roc_auc_score(yt, pt), "akurasi_uji_sel": accuracy_score(yt, yh),
                      "sensitivitas_uji_sel": float(yh[yt == 1].mean()), "spesifisitas_uji_sel": float((~yh[yt == 0]).mean())})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in HASIL[-1].items()})
T62 = pd.DataFrame(HASIL); T62.to_csv(OUT / "tabel_6_2_per_seed.csv", index=False)
T63 = T62.groupby("konfigurasi", sort=False)[["parameter", "AUC_val", "AUC_uji_sel", "akurasi_uji_sel", "sensitivitas_uji_sel", "spesifisitas_uji_sel", "epoch_terbaik", "detik_per_epoch"]].agg(["mean", "std"])
T63.columns = ["_".join(c) for c in T63.columns]; T63.to_csv(OUT / "tabel_6_3_ringkas_konfigurasi.csv")
pd.concat(LOG.values()).to_csv(OUT / "log_epoch.csv", index=False)
gambar_kurva([LOG[(k, SEEDS[0])] for k in KONFIG], "AUC val", OUT / "gambar_6_2_learning_curve.png")
T63
""")

md(r"""
## 6.4 Dari sel ke pasien: agregasi
Keputusan klinis dibuat per **pasien**, bukan per sel. Untuk setiap kelompok/pasien di set uji dihitung
(a) **rerata probabilitas** sel dan (b) **proporsi sel positif** (p ≥ 0,5). Label pasien = 1 bila kelompok memuat sel *Parasitized*.
Ambang tingkat pasien dipilih pada pasien **validasi** (Youden), lalu diterapkan pada pasien uji.
*Catatan:* sel di dataset ini sudah dipilih dan dipotong oleh penyusunnya, sehingga proporsi sel positif **tidak sama** dengan parasitemia sebenarnya.
""")
code(r"""
PILIH = max([k for k in PROB if k[0].startswith("ringan")], key=lambda k: roc_auc_score(Y["val"], PROB[k][0]))   # dipilih dari validasi
pv, pt = PROB[PILIH]
def agregasi(split, p):
    g = GM[IDX[split]]; d = pd.DataFrame({"kelompok": g, "p": p, "pos": p >= 0.5})
    a = d.groupby("kelompok").agg(n_sel=("p", "size"), rerata_p=("p", "mean"), proporsi_positif=("pos", "mean")).reset_index()
    a["label_pasien"] = PASIEN_POS[a.kelompok].values; return a
AV, AT = agregasi("val", pv), agregasi("test", pt)
baris = []
for skor in ["rerata_p", "proporsi_positif"]:
    if AV.label_pasien.nunique() < 2 or AT.label_pasien.nunique() < 2: baris.append({"skor_pasien": skor, "catatan": "kelas pasien tunggal (SMOKE)"}); continue
    f, t_, th = roc_curve(AV.label_pasien, AV[skor]); thr = float(np.clip(th[np.argmax(t_ - f)], 0, 1))
    yh = AT[skor] >= thr; y = AT.label_pasien.values.astype(bool)
    baris.append({"skor_pasien": skor, "AUC_pasien_val": roc_auc_score(AV.label_pasien, AV[skor]), "AUC_pasien_uji": roc_auc_score(y, AT[skor]),
                  "ambang_dari_val": thr, "pasien_uji": len(y), "pasien_uji_terinfeksi": int(y.sum()),
                  "sensitivitas_pasien_uji": float(yh[y].mean()), "spesifisitas_pasien_uji": float((~yh[~y]).mean()),
                  "TP": int((yh & y).sum()), "FN": int((~yh & y).sum()), "TN": int((~yh & ~y).sum()), "FP": int((yh & ~y).sum())})
T64 = pd.DataFrame(baris); T64.to_csv(OUT / "tabel_6_4_agregasi_pasien.csv", index=False)
AT.to_csv(OUT / "tabel_6_5_skor_per_pasien_uji.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, c, lab in [(0, "tab:blue", "pasien sehat"), (1, "tab:red", "pasien terinfeksi")]:
    d = AT[AT.label_pasien == k]; ax[0].scatter(d.n_sel, d.proporsi_positif, c=c, s=18, label=lab, alpha=0.8)
    ax[1].hist(d.rerata_p, bins=20, range=(0, 1), color=c, alpha=0.5, label=lab)
ax[0].set_xlabel("jumlah sel per pasien"); ax[0].set_ylabel("proporsi sel diprediksi positif"); ax[0].legend(fontsize=7); ax[0].set_title("Pasien uji")
ax[1].set_xlabel("rerata probabilitas sel"); ax[1].legend(fontsize=7); ax[1].set_title("Skor pasien (rerata)")
plt.tight_layout(); plt.savefig(OUT / "gambar_6_3_agregasi_pasien.png", bbox_inches="tight"); plt.show()
print("model dianalisis:", PILIH); T64
""")

md("## 6.5 Kesiapan perangkat murah: ukuran dan latensi CPU")
code(r"""
import io
baris = []
with catat_waktu("4_latensi_cpu"):
    torch.set_num_threads(2)
    for nk in KONFIG:
        if "gray" in nk or "tanpa" in nk: continue
        m = MODEL[(nk, SEEDS[0])]; mc = type(m)() if not isinstance(m, MobileNetV3S) else m
        mc.load_state_dict(m.state_dict()); mc = mc.cpu().eval(); buf = io.BytesIO(); torch.save(mc.state_dict(), buf)
        for bs in (1, 64):
            x = torch.rand(bs, 3, UK, UK)
            with torch.inference_mode():
                for _ in range(3): mc(x)
                t0 = time.perf_counter(); ul = 5 if SMOKE else 30
                for _ in range(ul): mc(x)
            dt = (time.perf_counter() - t0) / ul
            baris.append({"model": nk.split(" +")[0], "parameter": n_param(mc), "ukuran_MB_fp32": len(buf.getvalue()) / 1e6, "batch": bs,
                          "ms_per_batch": dt * 1e3, "ms_per_sel": dt * 1e3 / bs, "thread_cpu": torch.get_num_threads()})
    MODEL[PILIH].to(DEV)
T66 = pd.DataFrame(baris); T66.to_csv(OUT / "tabel_6_6_latensi_cpu.csv", index=False)
torch.save(MODEL[PILIH].state_dict(), OUT / "cnn_ringan_malaria_terbaik.pt")
print(jalankan("lscpu | grep 'Model name'")); T66
""")

md("## 6.6 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("06", 'dict(split=T61.to_dict("records"), per_seed=T62.to_dict("records"), model_agregasi=list(PILIH), agregasi=T64.to_dict("records"), latensi=T66.to_dict("records"))'))
simpan(C, "bab06.ipynb", Path(__file__).parent / "bab06.ipynb")
