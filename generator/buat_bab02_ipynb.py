"""Membangun notebook Colab Bab 2 (notebooks/bab02.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 2 — Lingkungan Kerja dan Data Medis
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4** (*Runtime → Change runtime type → T4 GPU*), lalu **Runtime → Run all**.

Isi:
1. Lingkungan kerja: CPU vs GPU (perkalian matriks), RAM, disk, versi paket.
2. **DICOM sintetis** dengan `pydicom` dari citra MedMNIST (X-ray PneumoniaMNIST dan irisan OrganMNIST3D): tag, *pixel data*, *rescale* dan *window/level*.
3. Perbandingan format: DICOM 12 bit vs PNG 16 bit vs JPEG 8 bit.
4. **NIfTI** dengan `nibabel` dari volume OrganMNIST3D: *affine*, *pixdim*, orientasi, irisan ortogonal.
5. **Anonimisasi DICOM** (subset Basic Application Confidentiality Profile, DICOM PS3.15 Lampiran E) pada data yang **SELURUHNYA FIKTIF**.
6. **Demonstrasi kebocoran data** pada NIH Malaria Cell Images: pembagian acak per sel vs pembagian per kelompok-pasien.
7. Reproduktibilitas: seed, uji pengulangan, daftar versi paket, kartu model.

Semua hasil disimpan ke `/content/hasil_bab02/` dan dizip menjadi `/content/hasil_bab02.zip`.

> **Penting.** Semua identitas pasien di notebook ini **FIKTIF** (dibuat-buat untuk latihan). Citra MedMNIST dan NIH Malaria sudah dianonimkan oleh penyedianya.
> Nilai "HU" pada contoh CT adalah **HU semu** hasil pemetaan linear untuk latihan, **bukan** nilai HU asli pasien.
""")

md(r"""
## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`
Sama di semua notebook buku. Dataset disimpan permanen di `MyDrive/Buku Visi Komputer Kesehatan/dataset/<nama>/` dan hanya diunduh dari sumber resmi bila belum ada atau rusak.
""")
code(kode_sel_standar())

md("## 2.0 Lingkungan dan pencatat waktu")
code(sel_lingkungan("02"))

md("Paket: `pydicom` biasanya belum ada di Colab, jadi dipasang di sini. `nibabel` umumnya sudah ada; bila belum, ikut dipasang.")
code(r"""
with catat_waktu("1_pip_install"):
    import importlib.util
    paket = ["medmnist==3.0.2", "pydicom>=3.0,<4"]
    if importlib.util.find_spec("nibabel") is None: paket.append("nibabel")
    jalankan("pip install -q " + " ".join(f'"{p}"' for p in paket))

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; import matplotlib.pyplot as plt
import pydicom, nibabel as nib, PIL, sklearn, medmnist
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from sklearn.metrics import roc_auc_score, accuracy_score
from medmnist import INFO
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSI = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
         "pydicom": pydicom.__version__, "nibabel": nib.__version__, "medmnist": medmnist.__version__,
         "scikit-learn": sklearn.__version__, "pillow": PIL.__version__, "matplotlib": matplotlib.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
print(json.dumps(VERSI, indent=1))
""" + "\n" + SEED_FN)

md(r"""
## 2.1 CPU vs GPU: mengapa runtime GPU penting
Perkalian matriks persegi float32 diulang beberapa kali; waktu dicatat setelah sinkronisasi GPU. Angka bergantung pada mesin Colab yang diperoleh.
""")
code(r"""
def ukur_matmul(n, dev, ulang=5):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    (a @ b); torch.cuda.synchronize() if dev.type == "cuda" else None   # pemanasan
    t0 = time.perf_counter()
    for _ in range(ulang): c = a @ b
    torch.cuda.synchronize() if dev.type == "cuda" else None
    dt = (time.perf_counter() - t0) / ulang
    return dt, 2 * n**3 / dt / 1e9
baris = []
with catat_waktu("2_cpu_vs_gpu"):
    for n in ([256, 512] if SMOKE else [1024, 2048, 4096]):
        for dev in [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []):
            dt, gf = ukur_matmul(n, dev)
            baris.append({"ukuran_matriks": n, "perangkat": dev.type, "detik_per_matmul": dt, "GFLOPS": gf})
T21 = pd.DataFrame(baris); T21.to_csv(OUT / "tabel_2_1_cpu_vs_gpu.csv", index=False); T21
""")

md(r"""
## 2.2 Membuat DICOM sintetis dengan `pydicom`
Citra MedMNIST (sudah dianonimkan, 8 bit) dibungkus menjadi objek DICOM *Secondary Capture*:
- **X-ray**: PneumoniaMNIST 128 px, diperbesar ke 256 px dengan interpolasi bilinear dan disimpan 12 bit (0–4080) agar mirip radiograf, dengan *WindowCenter/WindowWidth*. Karena sumbernya 8 bit, detail ekstra di 12 bit hanya berasal dari interpolasi.
- **CT**: satu irisan aksial OrganMNIST3D (28³); nilai 0–255 dipetakan linear ke **HU semu** −1000…+1000 lewat *RescaleSlope/RescaleIntercept*. Ini **ilustrasi**, bukan HU asli.
Identitas pasien diisi `ANONIM` (tanpa data pribadi).
""")
code(r"""
DIR_DCM = OUT / "dicom_sintetis"; DIR_DCM.mkdir(exist_ok=True)
def npz_medmnist(flag, size):
    fn = berkas_medmnist(flag, size); folder = pastikan_dataset("medmnist", [fn])
    return np.load(folder / fn)

def buat_dicom(arr, path, modality, judul, slope=1.0, intercept=0.0, wc=None, ww=None, bits=12, identitas=None, tambahan=None):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset(); ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(); ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modality; ds.ConversionType = "WSD"; ds.SeriesDescription = judul
    idt = identitas or {"PatientName": "ANONIM", "PatientID": "ANONIM-0001"}
    for k, v in idt.items(): setattr(ds, k, v)
    ds.Rows, ds.Columns = arr.shape; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16; ds.BitsStored = bits; ds.HighBit = bits - 1; ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]           # nilai ilustratif, bukan ukuran fisik asli
    ds.RescaleSlope = slope; ds.RescaleIntercept = intercept; ds.RescaleType = "US" if intercept == 0 else "HU"
    if wc is not None: ds.WindowCenter = wc; ds.WindowWidth = ww
    for k, v in (tambahan or {}).items(): setattr(ds, k, v)
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path

with catat_waktu("3_dicom_sintetis"):
    pn = npz_medmnist("pneumoniamnist", 128)
    xr8 = pn["test_images"][int(np.where(pn["test_labels"].reshape(-1) == 0)[0][0])]   # citra uji pertama berlabel normal
    # perbesar 128->256 dengan interpolasi bilinear pada nilai float lalu simpan 12 bit (0..4080): nilai antara tetap terjaga
    xr12 = np.clip(np.round(np.asarray(Image.fromarray(xr8.astype(np.float32)).resize((256, 256), Image.BILINEAR)) * 16), 0, 4080).astype(np.uint16)
    p_xr = buat_dicom(xr12, DIR_DCM / "xray_sintetis.dcm", "DX", "X-ray sintetis dari PneumoniaMNIST", wc=2040, ww=4080)
    o3 = npz_medmnist("organmnist3d", 28)
    vol = o3["test_images"][0].astype(np.float32)            # (28,28,28), nilai 0..1 atau 0..255
    vol8 = (vol / vol.max() * 255).astype(np.uint8) if vol.max() > 0 else vol.astype(np.uint8)
    irisan = vol8[vol8.shape[0] // 2]
    ct_px = (irisan.astype(np.float32) / 255 * 2000).round().astype(np.uint16)   # 0..2000 -> HU semu = px - 1000
    p_ct = buat_dicom(ct_px, DIR_DCM / "ct_sintetis.dcm", "CT", "CT sintetis (HU semu) dari OrganMNIST3D",
                      slope=1.0, intercept=-1000.0, wc=40, ww=400)
    print("rentang nilai volume OrganMNIST3D asli:", float(vol.min()), float(vol.max()), vol.dtype)

def tabel_tag(ds, maks=60):
    rows = []
    for el in ds.file_meta: rows.append({"tag": str(el.tag), "keyword": el.keyword, "VR": el.VR, "nilai": str(el.value)[:60]})
    for el in ds:
        if el.keyword == "PixelData": rows.append({"tag": str(el.tag), "keyword": el.keyword, "VR": el.VR, "nilai": f"<{len(el.value)} byte>"})
        else: rows.append({"tag": str(el.tag), "keyword": el.keyword, "VR": el.VR, "nilai": str(el.value)[:60]})
    return pd.DataFrame(rows[:maks])
ds_xr = pydicom.dcmread(p_xr); ds_ct = pydicom.dcmread(p_ct)
T22 = tabel_tag(ds_xr); T22.to_csv(OUT / "tabel_2_2_tag_dicom_xray.csv", index=False)
tabel_tag(ds_ct).to_csv(OUT / "tabel_2_3_tag_dicom_ct.csv", index=False)
T22
""")

md(r"""
## 2.3 *Rescale* dan *window/level*
`apply_modality_lut` mengubah nilai piksel tersimpan menjadi nilai fisik (HU semu); *window/level* memilih rentang yang ditampilkan.
Jendela umum: paru (L −600, W 1500), jaringan lunak (L 40, W 400), tulang (L 400, W 1800).
""")
code(r"""
def jendela(hu, L, W):
    lo, hi = L - W / 2, L + W / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)
hu = apply_modality_lut(ds_ct.pixel_array, ds_ct).astype(np.float32)
xr_voi = apply_voi_lut(ds_xr.pixel_array, ds_xr).astype(np.float32)
fig, ax = plt.subplots(1, 5, figsize=(13, 3))
ax[0].imshow(xr_voi, cmap="gray"); ax[0].set_title("X-ray DICOM 12 bit\n(WC 2040 / WW 4080)")
ax[1].imshow(hu, cmap="gray"); ax[1].set_title(f"CT: HU semu mentah\n[{hu.min():.0f}, {hu.max():.0f}]")
for a, (nama, L, W) in zip(ax[2:], [("paru", -600, 1500), ("jaringan lunak", 40, 400), ("tulang", 400, 1800)]):
    a.imshow(jendela(hu, L, W), cmap="gray", vmin=0, vmax=1); a.set_title(f"jendela {nama}\nL {L} / W {W}")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "gambar_2_1_dicom_window.png", bbox_inches="tight"); plt.show()
""")

md("## 2.4 Format penyimpanan: DICOM vs PNG 16 bit vs JPEG 8 bit")
code(r"""
DIR_FMT = OUT / "format"; DIR_FMT.mkdir(exist_ok=True)
Image.fromarray(xr12.astype(np.uint16)).save(DIR_FMT / "xray_16bit.png")
xr_as8 = (xr12 / 4080 * 255).round().astype(np.uint8)
Image.fromarray(xr_as8).save(DIR_FMT / "xray_8bit.png")
Image.fromarray(xr_as8).save(DIR_FMT / "xray_q90.jpg", quality=90)
baris = []
for nama, p in [("DICOM (12 bit dalam 16 bit)", p_xr), ("PNG 16 bit (lossless)", DIR_FMT / "xray_16bit.png"),
                ("PNG 8 bit (kuantisasi)", DIR_FMT / "xray_8bit.png"), ("JPEG 8 bit q=90 (lossy)", DIR_FMT / "xray_q90.jpg")]:
    if str(p).endswith(".dcm"): arr = pydicom.dcmread(p).pixel_array.astype(np.float64)
    else: arr = np.array(Image.open(p)).astype(np.float64)
    if arr.max() <= 255 and not str(p).endswith(".dcm") and "16" not in nama: arr = arr / 255 * 4080
    err = np.abs(arr - xr12.astype(np.float64))
    baris.append({"format": nama, "ukuran_byte": Path(p).stat().st_size, "nilai_unik": int(len(np.unique(arr))),
                  "galat_maks_vs_asli_12bit": float(err.max()), "galat_rerata": float(err.mean()), "metadata_klinis": "ya" if str(p).endswith(".dcm") else "tidak"})
T24 = pd.DataFrame(baris); T24.to_csv(OUT / "tabel_2_4_format.csv", index=False); T24
""")

md(r"""
## 2.5 NIfTI dengan `nibabel`
Volume OrganMNIST3D (28³) disimpan sebagai `.nii.gz` dengan *affine* diagonal (spasi voksel **ilustratif** 2,0 × 2,0 × 2,5 mm; MedMNIST tidak menyimpan spasi asli).
""")
code(r"""
with catat_waktu("4_nifti"):
    spasi = (2.0, 2.0, 2.5)
    aff = np.diag([*spasi, 1.0]); aff[:3, 3] = [-spasi[i] * vol8.shape[i] / 2 for i in range(3)]   # pusat volume di (0,0,0)
    img = nib.Nifti1Image(np.transpose(vol8, (2, 1, 0)).astype(np.int16), aff)   # MedMNIST (z,y,x) -> NIfTI (x,y,z)
    img.header.set_xyzt_units("mm"); img.set_qform(aff, code=1); img.set_sform(aff, code=1)
    p_nii = OUT / "organ_sintetis.nii.gz"; nib.save(img, p_nii)
    im2 = nib.load(p_nii); h = im2.header; data = np.asanyarray(im2.dataobj)
    vox = np.array([14, 14, 14]); dunia = nib.affines.apply_affine(im2.affine, vox)
    T25 = pd.DataFrame([
        {"field": "shape", "nilai": str(data.shape)}, {"field": "datatype", "nilai": str(h.get_data_dtype())},
        {"field": "pixdim (mm)", "nilai": str(tuple(float(v) for v in h.get_zooms()))},
        {"field": "orientasi (aff2axcodes)", "nilai": "".join(nib.aff2axcodes(im2.affine))},
        {"field": "qform_code / sform_code", "nilai": f"{int(h['qform_code'])} / {int(h['sform_code'])}"},
        {"field": "affine baris 1", "nilai": str(np.round(im2.affine[0], 2).tolist())},
        {"field": "affine baris 2", "nilai": str(np.round(im2.affine[1], 2).tolist())},
        {"field": "affine baris 3", "nilai": str(np.round(im2.affine[2], 2).tolist())},
        {"field": "voksel (14,14,14) -> dunia (mm)", "nilai": str(np.round(dunia, 2).tolist())},
        {"field": "ukuran berkas (byte)", "nilai": str(p_nii.stat().st_size)}])
    T25.to_csv(OUT / "tabel_2_5_nifti_header.csv", index=False)
    zx, zy, zz = h.get_zooms()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower", aspect=zy / zx); ax[0].set_title("aksial (x–y)")
    ax[1].imshow(data[:, data.shape[1] // 2, :].T, cmap="gray", origin="lower", aspect=zz / zx); ax[1].set_title("koronal (x–z)")
    ax[2].imshow(data[data.shape[0] // 2, :, :].T, cmap="gray", origin="lower", aspect=zz / zy); ax[2].set_title("sagital (y–z)")
    for a in ax: a.axis("off")
    plt.suptitle("Irisan ortogonal volume NIfTI (OrganMNIST3D, spasi ilustratif)"); plt.tight_layout()
    plt.savefig(OUT / "gambar_2_2_nifti_ortogonal.png", bbox_inches="tight"); plt.show()
T25
""")

md(r"""
## 2.6 Anonimisasi DICOM (data FIKTIF)
Kita membuat satu DICOM yang sengaja diisi **identitas FIKTIF** (nama, ID, tanggal lahir, institusi, dokter perujuk), sebuah *private tag*,
dan teks identitas FIKTIF yang "tertanam" pada piksel (*burned-in annotation*). Lalu diterapkan subset aturan **Basic Application Confidentiality
Profile** (DICOM PS3.15 Lampiran E, Tabel E.1-1): **Z** = kosongkan, **X** = hapus, **U** = ganti UID, ditambah penghapusan *private tag*,
penandaan `PatientIdentityRemoved = YES`, dan penutupan wilayah teks pada piksel. Daftar aturan ini **tidak lengkap** — untuk data nyata gunakan
profil penuh dan alat yang tervalidasi.
""")
code(r"""
IDENTITAS_FIKTIF = {"PatientName": "FIKTIF^CONTOH", "PatientID": "FIKTIF-000123", "PatientBirthDate": "19800101",
                    "PatientSex": "O", "InstitutionName": "RUMAH SAKIT FIKTIF", "ReferringPhysicianName": "FIKTIF^DOKTER",
                    "AccessionNumber": "FIKTIF-ACC-1", "StudyDate": "20260101", "StudyTime": "101500"}
with catat_waktu("5_anonimisasi"):
    besar = np.array(Image.fromarray(xr8).resize((512, 512), Image.BILINEAR))
    kanvas = Image.fromarray(besar); dr = ImageDraw.Draw(kanvas)
    KOTAK = (8, 8, 330, 52)
    dr.rectangle(KOTAK, fill=0)
    dr.text((14, 12), "FIKTIF^CONTOH  ID FIKTIF-000123", fill=255); dr.text((14, 30), "LAHIR 1980-01-01  RS FIKTIF", fill=255)
    px_phi = (np.array(kanvas).astype(np.uint16) * 16)
    p_phi = buat_dicom(px_phi, DIR_DCM / "fiktif_sebelum.dcm", "DX", "Contoh dengan identitas FIKTIF", wc=2040, ww=4080,
                       identitas=IDENTITAS_FIKTIF, tambahan={"BurnedInAnnotation": "YES"})
    ds = pydicom.dcmread(p_phi)
    ds.private_block(0x0011, "CONTOH_VENDOR_FIKTIF", create=True).add_new(0x01, "LO", "catatan internal FIKTIF")
    ds.save_as(p_phi)

    ATURAN = {"PatientName": "Z", "PatientID": "Z", "PatientBirthDate": "Z", "PatientSex": "Z", "ReferringPhysicianName": "Z",
              "AccessionNumber": "Z", "StudyDate": "Z", "StudyTime": "Z", "InstitutionName": "X", "SeriesDescription": "X",
              "StudyInstanceUID": "U", "SeriesInstanceUID": "U", "SOPInstanceUID": "U"}
    def anonimkan(ds, kotak_teks=None):
        import copy; ds = copy.deepcopy(ds)   # salinan, aslinya tidak diubah
        log = []
        for kw, aksi in ATURAN.items():
            if kw not in ds: continue
            lama = str(ds.data_element(kw).value)
            if aksi == "Z": ds.data_element(kw).value = ""
            elif aksi == "X": delattr(ds, kw)
            elif aksi == "U":
                baru = generate_uid(); setattr(ds, kw, baru)
                if kw == "SOPInstanceUID": ds.file_meta.MediaStorageSOPInstanceUID = baru
            log.append({"keyword": kw, "aksi": aksi, "sebelum": lama[:40], "sesudah": (str(ds.data_element(kw).value)[:40] if kw in ds else "(dihapus)")})
        n_priv = sum(1 for el in ds if el.tag.is_private); ds.remove_private_tags()
        log.append({"keyword": "(private tags)", "aksi": "X", "sebelum": f"{n_priv} elemen", "sesudah": "(dihapus)"})
        if kotak_teks is not None:
            arr = ds.pixel_array.copy(); x0, y0, x1, y1 = kotak_teks; arr[y0:y1 + 1, x0:x1 + 1] = 0
            ds.PixelData = arr.astype(np.uint16).tobytes()
            log.append({"keyword": "PixelData (wilayah teks)", "aksi": "tutup", "sebelum": "teks identitas FIKTIF", "sesudah": "ditutup hitam"})
            ds.BurnedInAnnotation = "NO"
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "Subset PS3.15 Annex E Basic Profile (demo buku)"
        kode = Dataset(); kode.CodeValue = "113100"; kode.CodingSchemeDesignator = "DCM"; kode.CodeMeaning = "Basic Application Confidentiality Profile"
        ds.DeidentificationMethodCodeSequence = [kode]
        return ds, pd.DataFrame(log)
    ds_anon, T26 = anonimkan(ds, KOTAK)
    p_anon = DIR_DCM / "fiktif_sesudah.dcm"; ds_anon.save_as(p_anon)
    T26.to_csv(OUT / "tabel_2_6_anonimisasi.csv", index=False)
    # verifikasi: tidak ada string FIKTIF tersisa di header
    sisa = [el.keyword for el in pydicom.dcmread(p_anon) if el.keyword != "PixelData" and "FIKTIF" in str(el.value)]
    print("Elemen header yang masih memuat 'FIKTIF':", sisa or "tidak ada")
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(pydicom.dcmread(p_phi).pixel_array, cmap="gray"); ax[0].set_title("Sebelum: teks identitas FIKTIF tertanam")
    ax[1].imshow(ds_anon.pixel_array, cmap="gray"); ax[1].set_title("Sesudah: wilayah teks ditutup")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "gambar_2_3_anonimisasi.png", bbox_inches="tight"); plt.show()
ANON_SISA = sisa
T26
""")

md(r"""
## 2.7 Kebocoran data: pembagian acak per sel vs per pasien (NIH Malaria)
Dataset NIH *Malaria Cell Images* berisi 27.558 citra sel (13.779 *Parasitized*, 13.779 *Uninfected*) dari apusan darah tipis.
Awalan nama berkas (mis. `C100P61ThinF_...`) diperlakukan sebagai **kode kelompok/pasien** — ini **asumsi** dari pola nama berkas
yang dicocokkan dengan deskripsi dataset (150 pasien terinfeksi + 50 sehat); hasil hitungan di bawah dipakai untuk mengecek asumsi itu.

Rancangan:
1. Sisihkan **20 % kelompok** sebagai **set eksternal** ("pasien baru") — tidak pernah dipakai melatih.
2. Pada 80 % sisanya, bandingkan dua cara membagi latih/uji internal (80/20):
   **A. acak per sel** (sel dari pasien yang sama bisa masuk latih dan uji) dan **B. per kelompok** (satu pasien hanya di satu sisi).
3. CNN kecil yang sama (64 px), epoch tetap, beberapa seed. Bandingkan skor uji internal vs eksternal.
Kebocoran terlihat bila skema A memberi skor internal yang lebih optimistis daripada kinerjanya pada pasien baru.
""")
code(r"""
import re, zipfile
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
with catat_waktu("6_malaria_siapkan"):
    pastikan_dataset("malaria")
    akar = ekstrak_lokal("malaria", "cell_images.zip")
    berkas = []
    for kelas, lab in [("Parasitized", 1), ("Uninfected", 0)]:
        for p in sorted(akar.rglob(f"{kelas}/*.png")): berkas.append((p, lab, re.match(r"^(C\d+)", p.name).group(1)))
    DF = pd.DataFrame(berkas, columns=["path", "label", "kelompok"])
    if SMOKE:
        kel = sorted(DF.kelompok.unique())[:30]; DF = DF[DF.kelompok.isin(kel)].groupby("kelompok").head(20).reset_index(drop=True)
    UK = 64
    def baca(p): return np.asarray(Image.open(p).convert("RGB").resize((UK, UK), Image.BILINEAR), dtype=np.uint8)
    with ThreadPoolExecutor(8) as ex: XM = np.stack(list(ex.map(baca, DF.path)))
    YM = DF.label.values.astype(np.int64); GM = DF.kelompok.values
per = DF.groupby("kelompok").label.agg(["count", "sum"])
T27 = pd.DataFrame([
    {"besaran": "jumlah citra sel", "nilai": len(DF)},
    {"besaran": "Parasitized / Uninfected", "nilai": f"{int(YM.sum())} / {int((YM == 0).sum())}"},
    {"besaran": "jumlah kelompok (awalan C<n>)", "nilai": per.shape[0]},
    {"besaran": "kelompok berisi kedua kelas", "nilai": int(((per["sum"] > 0) & (per["sum"] < per["count"])).sum())},
    {"besaran": "kelompok hanya Uninfected", "nilai": int((per["sum"] == 0).sum())},
    {"besaran": "kelompok hanya Parasitized", "nilai": int((per["sum"] == per["count"]).sum())},
    {"besaran": "sel per kelompok: min / median / maks", "nilai": f"{per['count'].min()} / {per['count'].median():.0f} / {per['count'].max()}"}])
T27.to_csv(OUT / "tabel_2_7_malaria_kelompok.csv", index=False)
rng = np.random.default_rng(SEED)
fig, ax = plt.subplots(2, 8, figsize=(12, 3.4))
for r, lab in enumerate([1, 0]):
    for c, i in enumerate(rng.choice(np.where(YM == lab)[0], 8, replace=False)):
        ax[r, c].imshow(XM[i]); ax[r, c].axis("off")
    ax[r, 0].set_title(["Uninfected", "Parasitized"][lab], loc="left", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "gambar_2_4_malaria_contoh.png", bbox_inches="tight"); plt.show()
print("bentuk XM:", XM.shape); T27
""")

code(r"""
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
idx_dalam, idx_eks = next(gss.split(XM, YM, GM))
def bagi(skema, seed):
    if skema == "A_acak_per_sel":
        tr, te = train_test_split(idx_dalam, test_size=0.2, random_state=seed, stratify=YM[idx_dalam])
    else:
        g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(g.split(idx_dalam, YM[idx_dalam], GM[idx_dalam])); tr, te = idx_dalam[a], idx_dalam[b]
    return tr, te

class CNNKecil(nn.Module):
    def __init__(self):
        super().__init__()
        def blok(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.fitur = nn.Sequential(blok(3, 32), blok(32, 64), blok(64, 128), blok(128, 128))
        self.kepala = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x): return self.kepala(self.fitur(x)).squeeze(1)

XT = torch.tensor(XM).permute(0, 3, 1, 2).contiguous()          # uint8 (N,3,64,64) di CPU
def batch_ke_dev(idx):
    return XT[idx].to(DEV, non_blocking=True).float().div(255).sub(0.5).div(0.5)
@torch.no_grad()
def skor(model, idx, bs=1024):
    model.eval(); out = []
    for i in range(0, len(idx), bs): out.append(torch.sigmoid(model(batch_ke_dev(idx[i:i + bs]))).float().cpu())
    return torch.cat(out).numpy()
def latih(tr, seed, epoch, bs=256):
    seed_semua(seed); model = CNNKecil().to(DEV); opt = torch.optim.Adam(model.parameters(), 1e-3)
    yt = torch.tensor(YM, dtype=torch.float32); g = torch.Generator().manual_seed(seed)
    for ep in range(epoch):
        model.train(); perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; x = batch_ke_dev(b)
            if torch.rand(1, generator=g).item() < 0.5: x = x.flip(3)
            loss = F.binary_cross_entropy_with_logits(model(x), yt[b].to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model
def ukur(y, p):
    return {"AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"), "akurasi": accuracy_score(y, p >= 0.5)}

EPOCH = 2 if SMOKE else 8
SEEDS = [0] if SMOKE else [0, 1, 2]
hasil = []
with catat_waktu("7_malaria_latih_kebocoran"):
    for skema in ["A_acak_per_sel", "B_per_kelompok"]:
        for s in SEEDS:
            tr, te = bagi(skema, s); t0 = time.perf_counter()
            m = latih(tr, s, EPOCH); dt = time.perf_counter() - t0
            irisan_kel = len(set(GM[tr]) & set(GM[te]))
            ri, re_ = ukur(YM[te], skor(m, te)), ukur(YM[idx_eks], skor(m, idx_eks))
            hasil.append({"skema": skema, "seed": s, "n_latih": len(tr), "n_uji_internal": len(te), "n_eksternal": len(idx_eks),
                          "kelompok_uji_yang_juga_di_latih": irisan_kel, "kelompok_uji_internal": len(set(GM[te])),
                          "AUC_internal": ri["AUC"], "AUC_eksternal": re_["AUC"], "akurasi_internal": ri["akurasi"],
                          "akurasi_eksternal": re_["akurasi"], "detik_latih": dt})
            print(hasil[-1])
T28 = pd.DataFrame(hasil); T28["selisih_akurasi_internal_minus_eksternal"] = T28.akurasi_internal - T28.akurasi_eksternal
T28.to_csv(OUT / "tabel_2_8_kebocoran_per_seed.csv", index=False)
T29 = T28.groupby("skema")[["AUC_internal", "AUC_eksternal", "akurasi_internal", "akurasi_eksternal",
                             "selisih_akurasi_internal_minus_eksternal", "kelompok_uji_yang_juga_di_latih"]].agg(["mean", "std"])
T29.columns = ["_".join(c) for c in T29.columns]; T29.to_csv(OUT / "tabel_2_9_kebocoran_ringkas.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, met in zip(ax, ["akurasi", "AUC"]):
    xs = np.arange(2); w = 0.35
    for j, (bag, warna) in enumerate([("internal", "#4c72b0"), ("eksternal", "#dd8452")]):
        mu = [T28[T28.skema == s][f"{met}_{bag}"].mean() for s in ["A_acak_per_sel", "B_per_kelompok"]]
        sd = [T28[T28.skema == s][f"{met}_{bag}"].std() if len(SEEDS) > 1 else 0 for s in ["A_acak_per_sel", "B_per_kelompok"]]
        a.bar(xs + (j - 0.5) * w, mu, w, yerr=sd, capsize=3, label=f"uji {bag}", color=warna)
        for x, m_ in zip(xs + (j - 0.5) * w, mu): a.text(x, m_, f"{m_:.3f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(xs); a.set_xticklabels(["A: acak per sel", "B: per kelompok"]); a.set_title(met)
    lo = min(T28[f"{met}_internal"].min(), T28[f"{met}_eksternal"].min()); a.set_ylim(max(0, lo - 0.05), 1.0)
ax[0].legend(loc="lower right"); plt.suptitle(f"Kebocoran data: rerata ± SD {len(SEEDS)} seed, {EPOCH} epoch")
plt.tight_layout(); plt.savefig(OUT / "gambar_2_5_kebocoran.png", bbox_inches="tight"); plt.show()
T29
""")

md(r"""
## 2.8 Reproduktibilitas
Model skema B (seed 0) dilatih **dua kali** dengan seed dan pengaturan deterministik yang sama; bobot dan skor dibandingkan.
Lalu dicatat versi paket dan dibuat **kartu model** ringkas berisi angka dari jalannya notebook ini.
""")
code(r"""
with catat_waktu("8_reproduktibilitas"):
    tr, te = bagi("B_per_kelompok", 0); ep_r = 1 if SMOKE else 2
    m1 = latih(tr, 0, ep_r); m2 = latih(tr, 0, ep_r)
    s1, s2 = skor(m1, te), skor(m2, te)
    beda_bobot = max(float((a - b).abs().max()) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()) if a.is_floating_point())
    m3 = latih(tr, 1, ep_r); s3 = skor(m3, te)
REPRO = {"epoch": ep_r, "selisih_bobot_maks_seed_sama": beda_bobot, "selisih_skor_maks_seed_sama": float(np.abs(s1 - s2).max()),
         "identik_seed_sama": bool(beda_bobot == 0.0), "AUC_seed0_run1": ukur(YM[te], s1)["AUC"], "AUC_seed0_run2": ukur(YM[te], s2)["AUC"],
         "AUC_seed1": ukur(YM[te], s3)["AUC"], "selisih_skor_maks_seed_beda": float(np.abs(s1 - s3).max())}
pd.Series(REPRO).to_csv(OUT / "tabel_2_10_reproduktibilitas.csv", header=["nilai"])
(OUT / "requirements_bab02.txt").write_text("\n".join(f"{k}=={v}" for k, v in VERSI.items() if k not in ("gpu", "cuda", "python")) +
                                         f"\n# python {VERSI['python']} | CUDA {VERSI['cuda']} | GPU {VERSI['gpu']}\n")
rB = T28[T28.skema == "B_per_kelompok"]
kartu = f'''# Kartu Model — CNN kecil deteksi sel malaria (demo Bab 2)
- **Tujuan**: demonstrasi pendidikan tentang kebocoran data; **bukan** untuk penggunaan klinis.
- **Data**: NIH Malaria Cell Images (Rajaraman dkk., 2018), {len(DF)} citra sel, {per.shape[0]} kelompok (awalan nama berkas, diasumsikan pasien).
- **Pembagian**: 20 % kelompok disisihkan sebagai set eksternal; latih/uji internal per kelompok (skema B).
- **Arsitektur**: 4 blok Conv-BN-ReLU-MaxPool (32-64-128-128), GAP, dropout 0,3; masukan RGB 64×64; {sum(p.numel() for p in m1.parameters()):,} parameter.
- **Pelatihan**: Adam lr 1e-3, batch 256, {EPOCH} epoch, flip horizontal acak; seed {SEEDS}.
- **Kinerja (rerata seed, skema B)**: AUC internal {rB.AUC_internal.mean():.4f}, AUC eksternal {rB.AUC_eksternal.mean():.4f}, akurasi eksternal {rB.akurasi_eksternal.mean():.4f} (ambang 0,5).
- **Lingkungan**: {VERSI['gpu']}, torch {VERSI['torch']}, CUDA {VERSI['cuda']}, Python {VERSI['python']}.
- **Reproduktibilitas**: dua kali latih seed sama → selisih bobot maks {beda_bobot:.3g}.
- **Keterbatasan**: satu sumber data; asumsi kelompok=pasien dari nama berkas; tanpa kalibrasi; tanpa validasi klinis.
'''
(OUT / "kartu_model_malaria.md").write_text(kartu); print(kartu); REPRO
""")

md("## 2.9 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("02", 'dict(t21=T21.to_dict("records"), anonimisasi_sisa=ANON_SISA, malaria=T27.to_dict("records"), kebocoran=T28.to_dict("records"), reproduktibilitas=REPRO, kelompok_eksternal=int(len(set(GM[idx_eks]))), sel_eksternal=int(len(idx_eks)))'))

simpan(C, "bab02.ipynb", Path(__file__).parent / "bab02.ipynb")
