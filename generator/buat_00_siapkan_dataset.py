"""Membangun notebooks/00_siapkan_dataset.ipynb: sekali jalan mengisi Drive dengan semua dataset buku."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 00 — Siapkan semua dataset buku di Google Drive
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*

Jalankan notebook ini **sekali** (Colab, *Runtime → Run all*; GPU tidak diperlukan). Semua dataset diunduh dari
**sumber resminya** langsung ke `MyDrive/Buku Visi Komputer Kesehatan/dataset/<nama_dataset>/`. Berkas yang sudah ada dan
lolos cek (ukuran, MD5 bila sumber menyediakan, atau jumlah berkas di dalam zip) **tidak diunduh ulang**, sehingga
notebook aman dijalankan berkali-kali (mis. setelah koneksi terputus).

Perkiraan total ruang Drive bila semua pilihan bawaan aktif: ± 2,6 GB (lihat tabel ringkasan di akhir).
Notebook bab (`bab01.ipynb`, dst.) memakai fungsi yang sama (`pastikan_dataset`), jadi menjalankan notebook ini lebih dulu
hanya mempercepat; bukan syarat.

> **Lisensi.** Mengunduh = menerima lisensi tiap sumber. DermaMNIST (CC BY-NC 4.0) dan Kvasir-SEG (riset/pendidikan)
> **tidak untuk penggunaan komersial**. Montgomery County CXR Set **tidak boleh didistribusikan ulang** dan nonaktif bawaan.
> Jangan membagikan folder `dataset/` secara publik; bagikan notebook, bukan datanya.
""")

md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())

md("## 1 Lingkungan dan pencatat waktu")
code(r"""
import contextlib, platform
WAKTU = {}
@contextlib.contextmanager
def catat_waktu(nama):
    t0 = time.perf_counter(); yield
    WAKTU[nama] = round(time.perf_counter() - t0, 1); print(f"[waktu] {nama}: {WAKTU[nama]} s")
smi = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout
print(smi if smi.strip() else "nvidia-smi: tidak ada GPU (tidak masalah untuk notebook ini)")
print("Python", platform.python_version())
""")

md(r"""
## 2 Pilih dataset
Ubah `True`/`False` bila ruang Drive terbatas. Kolom komentar menunjukkan bab yang memakainya.
""")
code(r"""
MEDMNIST_BUKU = [  # (flag, ukuran px)  — bab pemakai
    ("pneumoniamnist", 28), ("pneumoniamnist", 64), ("pneumoniamnist", 128), ("pneumoniamnist", 224),  # Bab 1, 4, 12
    ("organamnist", 64),                                                                                  # Bab 1, 5
    ("breastmnist", 28), ("breastmnist", 128),                                                            # Bab 1, 3
    ("bloodmnist", 28), ("bloodmnist", 64),                                                               # Bab 1, 3, 12
    ("retinamnist", 28), ("retinamnist", 128), ("retinamnist", 224),                                      # Bab 1, 7
    ("octmnist", 64),                                                                                     # Bab 1, 5
    ("organmnist3d", 28), ("organmnist3d", 64),                                                           # Bab 10
    ("nodulemnist3d", 28), ("nodulemnist3d", 64),                                                         # Bab 10
]
PAKAI_DERMAMNIST = True     # Bab 5; lisensi CC BY-NC 4.0 (nonkomersial)
DERMA = [("dermamnist", 28), ("dermamnist", 64)]
PILIH = {
    "medmnist":   True,   # Zenodo 10519652, CC BY 4.0 (kecuali DermaMNIST)
    "malaria":    True,   # NIH/NLM, Bab 2 & 6 (353 MB, 27.558 PNG di dalam zip)
    "hc18":       True,   # Zenodo 1327317, CC BY 4.0, Bab 7 & 9 (± 177 MB)
    "kvasir_seg": True,   # Simula, riset/pendidikan, Bab 9 (46 MB)
    "bccd":       True,   # GitHub Shenggan/BCCD_Dataset, MIT, Bab 8 (± 8 MB)
    "montgomery": False,  # NLM, opsional; dilarang didistribusikan ulang — baca ReadMe dulu
}
""")

md("## 3 Unduh atau verifikasi")
code(r"""
STATUS = []
def jalankan(nama, berkas=None):
    try:
        with catat_waktu(f"pastikan_{nama}"):
            pastikan_dataset(nama, berkas)
        STATUS.append((nama, "OK"))
    except Exception as e:   # satu sumber gagal tidak menghentikan yang lain
        print(f"!! {nama}: {type(e).__name__}: {e}"); STATUS.append((nama, f"GAGAL: {e}"))

if PILIH["medmnist"]:
    daftar = MEDMNIST_BUKU + (DERMA if PAKAI_DERMAMNIST else [])
    jalankan("medmnist", [berkas_medmnist(f, s) for f, s in daftar])
for nama in ("malaria", "hc18", "kvasir_seg", "bccd"):
    if PILIH[nama]: jalankan(nama)
""")

md(r"""
## 4 (Opsional) Montgomery County CXR Set
NLM tidak menyediakan satu arsip; sel ini mengunduh 138 citra PNG + 2×138 masker paru per berkas ke disk lokal, lalu
mengemasnya menjadi `montgomery_set.zip` di Drive (cek: 414 PNG). Data klinis (`ClinicalReadings`) sengaja tidak diunduh.
Buku tidak memakai dataset ini sebagai jalur utama. **Larangan redistribusi**: simpan hanya di Drive pribadi.
""")
code(r"""
BASIS_MC = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"
SUB_MC = ["CXR_png/", "ManualMask/leftMask/", "ManualMask/rightMask/"]
def pastikan_montgomery():
    tujuan = DATASET_DIR / "montgomery" / "montgomery_set.zip"
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    if tujuan.exists():
        n = sum(1 for x in zipfile.ZipFile(tujuan).namelist() if x.endswith(".png"))
        if n == 414: print(f"Dataset montgomery ditemukan di Drive ({tujuan.stat().st_size/1e6:.0f} MB, {n} PNG)."); return
        print(f"montgomery_set.zip berisi {n} PNG (seharusnya 414); mengunduh ulang...")
    else:
        print("Dataset montgomery belum ada, mengunduh per berkas dari NLM...")
    lokal = LOKAL / "montgomery_unduh"
    for sub in SUB_MC:
        html = requests.get(BASIS_MC + sub + "index.html", timeout=60).text
        nama = sorted(set(re.findall(r"href='([^'/]+\.png)'", html)))
        print(f"  {sub}: {len(nama)} berkas")
        (lokal / sub).mkdir(parents=True, exist_ok=True)
        for fn in nama:
            p = lokal / sub / fn
            if not p.exists():
                r = requests.get(BASIS_MC + sub + fn, timeout=120); r.raise_for_status(); p.write_bytes(r.content)
    r = requests.get(BASIS_MC + "NLM-MontgomeryCXRSet-ReadMe.pdf", timeout=60); (lokal / "NLM-MontgomeryCXRSet-ReadMe.pdf").write_bytes(r.content)
    arsip = LOKAL / "montgomery_set.zip"
    with zipfile.ZipFile(arsip, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(lokal.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(lokal))
    shutil.copyfile(arsip, tujuan); print("  tersimpan:", tujuan)

if PILIH["montgomery"]:
    with catat_waktu("pastikan_montgomery"):
        pastikan_montgomery()
else:
    print("Montgomery dilewati (PILIH['montgomery'] = False).")
""")

md(r"""
## 5 Dataset yang memerlukan akun atau registrasi (hanya petunjuk)
Notebook ini **tidak** mengunduhnya otomatis dan **tidak menyimpan kunci API apa pun**.

| Dataset | Mengapa tidak otomatis | Pengganti di buku |
|---|---|---|
| APTOS 2019 Blindness Detection (Kaggle) | perlu akun Kaggle + menyetujui aturan kompetisi + token API | RetinaMNIST (Bab 7) |
| Chest X-Ray Images (Pneumonia), mirror Kaggle | perlu akun Kaggle; versi asli Kermany dkk. ada di Mendeley Data tanpa login, tetapi arsipnya ± 8,4 GB (CXR + OCT) dan lisensinya perlu dikonfirmasi | PneumoniaMNIST (Bab 4) |
| Kvasir-SEG (bila unduhan langsung diblokir) | Simula dapat meminta formulir | HC18 segmentasi (Bab 9) |

Bila Anda tetap ingin memakai Kaggle di Colab: simpan token di **Colab → ikon kunci (Secrets)** dengan nama
`KAGGLE_USERNAME` dan `KAGGLE_KEY`, lalu baca dengan `google.colab.userdata.get(...)` di sel Anda sendiri.
Jangan menuliskan token di notebook, jangan mengunggah `kaggle.json` ke folder yang dibagikan.
""")

md("## 6 Ringkasan isi Drive")
code(r"""
import pandas as pd
baris = []
for d in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
    berkas = [p for p in d.rglob("*") if p.is_file() and not p.name.endswith((".md5ok", ".isiok"))]
    isi_zip = 0
    for p in berkas:
        if p.suffix == ".zip":
            try: isi_zip += sum(1 for x in zipfile.ZipFile(p).namelist() if not x.endswith("/"))
            except zipfile.BadZipFile: pass
    baris.append({"dataset": d.name, "jumlah_berkas_di_drive": len(berkas),
                  "berkas_di_dalam_zip": isi_zip, "ukuran_MB": round(sum(p.stat().st_size for p in berkas) / 1e6, 1),
                  "terverifikasi_MD5": sum(1 for p in d.rglob("*.md5ok"))})
RINGKAS = pd.DataFrame(baris)
RINGKAS.loc[len(RINGKAS)] = {"dataset": "TOTAL", "jumlah_berkas_di_drive": RINGKAS.jumlah_berkas_di_drive.sum(),
                             "berkas_di_dalam_zip": RINGKAS.berkas_di_dalam_zip.sum(),
                             "ukuran_MB": round(RINGKAS.ukuran_MB.sum(), 1), "terverifikasi_MD5": RINGKAS.terverifikasi_MD5.sum()}
RINGKAS.to_csv(DATASET_DIR / "ringkasan_dataset.csv", index=False)
print("Status:", STATUS); print("Waktu (s):", WAKTU)
RINGKAS
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "name": "00_siapkan_dataset.ipynb"}}
out = Path(__file__).parent / "00_siapkan_dataset.ipynb"
nbf.write(nb, out); print("tulis", out, len(C), "sel")
