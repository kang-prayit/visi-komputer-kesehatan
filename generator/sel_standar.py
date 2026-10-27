"""Sel standar yang disisipkan di awal SETIAP notebook buku (mount Drive + pastikan_dataset).
Katalog ukuran/MD5 MedMNIST & HC18 diambil dari API Zenodo (hasil/zenodo_*.json)."""
import json
from pathlib import Path
R = Path(__file__).parent  # berisi zenodo/*.json (metadata publik API Zenodo)

def _katalog():
    z1 = json.load(open(R / 'zenodo/zenodo_10519652.json'))
    z2 = json.load(open(R / 'zenodo/zenodo_1327317.json'))
    med = {f['key']: [f['links']['self'], f['size'], f['checksum'].split(':')[1]] for f in z1['files']}
    hc = {f['key']: [f['links']['self'], f['size'], f['checksum'].split(':')[1]] for f in z2['files']}
    return med, hc

def kode_sel_standar():
    med, hc = _katalog()
    med_txt = ',\n'.join('    %r: %r' % (k, v) for k, v in sorted(med.items()))
    hc_txt = ',\n'.join('    %r: %r' % (k, v) for k, v in sorted(hc.items()))
    return SEL.replace('@@MEDMNIST@@', med_txt).replace('@@HC18@@', hc_txt)

SEL = r'''
# ===== SEL STANDAR BUKU: Drive + pastikan_dataset() (sama di semua notebook) =====
import os, sys, time, json, hashlib, shutil, zipfile, re, subprocess
from pathlib import Path
try:
    import requests, certifi
except ImportError:  # sudah ada di Colab; dipasang bila notebook dijalankan di tempat lain
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "requests", "certifi"], check=True)
    import requests, certifi
DI_COLAB = "google.colab" in sys.modules
if DI_COLAB:
    from google.colab import drive
    # Lokasi folder buku di Drive dapat diubah lewat variabel lingkungan BUKU_DRIVE_SUBPATH
    BUKU_DRIVE_SUBPATH = os.environ.get("BUKU_DRIVE_SUBPATH", "MyDrive/Buku Visi Komputer Kesehatan")
    for _coba in range(3):  # popup izin Drive kadang lambat: ulangi, dengan force_remount
        try:
            drive.mount("/content/drive", force_remount=_coba > 0); break
        except Exception as e:
            print("drive.mount gagal (", e, "), mencoba lagi ..."); time.sleep(5)
    AKAR_BUKU = Path("/content/drive") / BUKU_DRIVE_SUBPATH
    LOKAL = Path("/content/data_lokal")
else:  # dijalankan di luar Colab: arahkan ke folder lain lewat variabel lingkungan
    AKAR_BUKU = Path(os.environ.get("BUKU_DRIVE", str(Path.home() / "Buku Visi Komputer Kesehatan")))
    LOKAL = Path(os.environ.get("BUKU_LOKAL", "/tmp/data_lokal"))
DATASET_DIR = AKAR_BUKU / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True); LOKAL.mkdir(parents=True, exist_ok=True)
print("Folder dataset di Drive:", DATASET_DIR)

# Katalog sumber resmi: nama berkas -> [URL, ukuran byte, MD5]. MD5 None = sumber tidak menyediakan
# (MD5 malaria/kvasir dicatat penulis dari unduhan 27-10-2026; bila sumber memperbarui berkas, cek ukuran yang dipakai).
MEDMNIST = {  # Zenodo 10.5281/zenodo.10519652 (CC BY 4.0 kecuali DermaMNIST CC BY-NC 4.0)
@@MEDMNIST@@
}
HC18 = {      # Zenodo 10.5281/zenodo.1327317 (CC BY 4.0)
@@HC18@@
}
KATALOG = {
    "medmnist":   {"berkas": MEDMNIST},
    "hc18":       {"berkas": HC18},
    "malaria":    {"berkas": {"cell_images.zip": ["https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip", 353452851, "2f2e94a91d413e5b0bab6342fddb6de6"]},
                   "isi": {"cell_images.zip": {".png": 27558}}},
    "kvasir_seg": {"berkas": {"kvasir-seg.zip": ["https://datasets.simula.no/downloads/kvasir-seg.zip", 46227172, "6323d9094df93b35d43069a566ee1ca3"]},
                   "isi": {"kvasir-seg.zip": {".jpg": 2000}}},
    "bccd":       {"berkas": {"BCCD_Dataset-master.zip": ["https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip", None, None]},
                   "isi": {"BCCD_Dataset-master.zip": {"/JPEGImages/.jpg": 364, "/Annotations/.xml": 364}}},
}
# Rantai sertifikat: server datasets.simula.no tidak mengirim sertifikat perantara (Sectigo).
# Kita TIDAK mematikan verifikasi TLS; sertifikat perantara publik diambil dari crt.sh dan ditambahkan ke bundel CA.
CRT_PERANTARA = {"datasets.simula.no": "https://crt.sh/?d=4267304690"}

def md5_berkas(p, blok=8 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(blok), b""): h.update(b)
    return h.hexdigest()

def _cek_isi_zip(p, aturan):
    nama = zipfile.ZipFile(p).namelist()
    for pola, n in aturan.items():
        folder, ext = (pola.rsplit("/", 1)[0] + "/", pola.rsplit("/", 1)[1]) if "/" in pola else ("", pola)
        jml = sum(1 for x in nama if x.lower().endswith(ext) and folder in x and "__MACOSX" not in x)
        if jml != n: return f"isi zip {pola}: {jml} berkas, seharusnya {n}"
    return None

def berkas_lengkap(nama, fn):
    """Cek berkas di Drive: ada, ukuran sesuai, MD5 sesuai (MD5 dihitung sekali lalu dicatat di <fn>.md5ok)."""
    url, ukuran, md5 = KATALOG[nama]["berkas"][fn]
    p = DATASET_DIR / nama / fn
    if not p.exists(): return False, "belum ada"
    if ukuran and p.stat().st_size != ukuran: return False, f"ukuran {p.stat().st_size} B, seharusnya {ukuran} B"
    tanda = p.with_name(p.name + ".md5ok")
    if md5 and not (tanda.exists() and tanda.read_text().strip() == md5):
        m = md5_berkas(p)
        if m != md5: return False, f"MD5 {m} tidak cocok ({md5})"
        tanda.write_text(md5)
    aturan = KATALOG[nama].get("isi", {}).get(fn)
    if aturan:
        tanda_isi = p.with_name(p.name + ".isiok")
        if not tanda_isi.exists():
            salah = _cek_isi_zip(p, aturan)
            if salah: return False, salah
            tanda_isi.write_text(json.dumps(aturan))
    return True, "lengkap"

def _bundel_ca(host):
    import certifi, requests
    pem = requests.get(CRT_PERANTARA[host], timeout=60).text
    assert "BEGIN CERTIFICATE" in pem
    b = LOKAL / f"ca_{host}.pem"; b.write_text(open(certifi.where()).read() + "\n" + pem); return str(b)

def unduh(url, tujuan):
    import requests
    from urllib.parse import urlparse
    tmp = LOKAL / "_unduh" / tujuan.name; tmp.parent.mkdir(parents=True, exist_ok=True)
    host = urlparse(url).hostname; verify = True
    for coba in range(2):
        try:
            with requests.get(url, stream=True, timeout=120, verify=verify) as r:
                r.raise_for_status(); total = int(r.headers.get("content-length", 0)); n = 0; t0 = time.time()
                with open(tmp, "wb") as f:
                    for b in r.iter_content(4 << 20):
                        f.write(b); n += len(b)
                        if time.time() - t0 > 15: print(f"   ... {n/1e6:.0f}/{total/1e6:.0f} MB"); t0 = time.time()
            break
        except requests.exceptions.SSLError:
            if host in CRT_PERANTARA and verify is True:
                print("   rantai sertifikat server tidak lengkap; menambahkan sertifikat perantara resmi ke bundel CA")
                verify = _bundel_ca(host)
            else:
                raise
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp, tujuan)          # salin ke Drive (permanen)
    return tmp                            # salinan lokal tetap ada untuk dipakai sesi ini

def pastikan_dataset(nama, berkas=None):
    """Pastikan dataset ada & lengkap di Drive; bila belum, unduh dari sumber resmi ke Drive. Kembalikan folder Drive-nya."""
    daftar = berkas or list(KATALOG[nama]["berkas"])
    for fn in daftar:
        ok, alasan = berkas_lengkap(nama, fn)
        if ok:
            print(f"Dataset {nama}/{fn} ditemukan di Drive ({(DATASET_DIR/nama/fn).stat().st_size/1e6:.1f} MB).")
            continue
        print(f"Dataset {nama}/{fn} belum ada atau tidak lengkap ({alasan}); mengunduh dari sumber resmi...")
        t0 = time.time(); unduh(KATALOG[nama]["berkas"][fn][0], DATASET_DIR / nama / fn)
        for t in (".md5ok", ".isiok"):
            q = DATASET_DIR / nama / (fn + t)
            if q.exists(): q.unlink()
        ok, alasan = berkas_lengkap(nama, fn)
        if not ok: raise RuntimeError(f"{nama}/{fn} gagal diverifikasi setelah diunduh: {alasan}")
        print(f"   selesai {time.time()-t0:.0f} s, tersimpan di Drive dan terverifikasi.")
    return DATASET_DIR / nama

def ekstrak_lokal(nama, fn):
    """Ekstrak arsip dari Drive ke disk lokal Colab (/content), sekali per sesi.
    Alasan: membaca ribuan berkas kecil langsung dari Drive (FUSE) sangat lambat; disk lokal jauh lebih cepat."""
    tujuan = LOKAL / nama; tanda = tujuan / (".ekstrak_" + fn)
    if tanda.exists(): print(f"{fn} sudah diekstrak di {tujuan}"); return tujuan
    t0 = time.time(); src = LOKAL / "_unduh" / fn
    if not src.exists() or src.stat().st_size != (DATASET_DIR / nama / fn).stat().st_size:
        src.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(DATASET_DIR / nama / fn, src)
    with zipfile.ZipFile(src) as z: z.extractall(tujuan)
    tanda.write_text("ok"); print(f"{fn} diekstrak ke {tujuan} dalam {time.time()-t0:.0f} s"); return tujuan

def berkas_medmnist(flag, size):
    return f"{flag}.npz" if size == 28 else f"{flag}_{size}.npz"
# ===== akhir sel standar =====
'''.strip()
