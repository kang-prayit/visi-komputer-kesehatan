# Visi Komputer untuk Kesehatan — Kode Buku

Kode pendamping buku **_Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python_**.

**Penulis:**
1. Prayitno (Politeknik Negeri Semarang)
2. Karisma Trinanda Putra (Universitas Muhammadiyah Yogyakarta)
3. Eko Prasetyo (Universitas Muhammadiyah Yogyakarta)
4. Sunardi (Universitas Muhammadiyah Yogyakarta)
5. Liliek Triyono (Politeknik Negeri Semarang)

> **English summary.** These are the companion Google Colab notebooks for the Indonesian textbook *Computer Vision for Health*. The book covers classification, regression, detection and segmentation of medical images in Python/PyTorch, and every notebook uses public datasets that are downloaded automatically into your Google Drive. The notebooks are for **education and research only**. They are **not a medical device** and must not be used for clinical decisions. Code licence: MIT (proposed). Dataset licences vary; see the table below. DermaMNIST/HAM10000 is **CC BY-NC 4.0**, which means non-commercial use only.

> ⚠️ **Penafian.** Repositori ini hanya untuk pendidikan dan penelitian. Model di dalamnya **bukan alat kesehatan** dan **tidak boleh dipakai untuk keputusan klinis**. Model ini juga belum memiliki izin edar (lihat KMK HK.01.07/MENKES/951/2026). Semua data yang dipakai adalah dataset publik, tanpa data pasien yang dapat dikenali.

## Isi repositori

| Folder/berkas | Isi |
|---|---|
| `notebooks/` | 14 notebook Colab: `00_siapkan_dataset.ipynb` dan `bab01`–`bab13`. Keluaran sel sudah dihapus agar repositori ringan. |
| `generator/` | Skrip Python pembangun notebook (`buat_babNN_ipynb.py`), sel standar Drive + unduh dataset (`sel_standar.py`), pola bersama (`pola_bab.py`), dan metadata publik Zenodo (`zenodo/`). Bagian ini opsional; pembaca cukup memakai `notebooks/`. |
| `requirements.txt` | Paket tambahan untuk menjalankan notebook di luar Colab. |

**Catatan Bab 8.** `notebooks/bab08.ipynb` adalah versi 2 notebook Bab 8 (di naskah kerja bernama `bab08_v2.ipynb`). Isi ilmunya sama, tetapi biayanya jauh lebih ringan di T4. Versi 1 tidak disertakan karena terlalu lambat.

## Notebook per bab

Waktu eksekusi berasal dari satu run nyata di Google Colab (GPU Tesla T4, Python 3.13, PyTorch 2.11), 27–28 Oktober 2026. Waktu ini tidak termasuk unduhan pertama dataset oleh notebook 00.

| Notebook | Bab dan tugas | Dataset | Model/metode | Waktu T4 |
|---|---|---|---|---|
| 00_siapkan_dataset | Mengunduh semua dataset ke Drive sekali saja, dengan cek ukuran dan MD5 | MedMNIST v2, NIH Malaria, HC18, BCCD | — | ±30 menit (unduhan ±2,6 GB) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab01.ipynb) bab01 | 1. Pengantar: eksplorasi modalitas dan baseline klasifikasi biner | MedMNIST (PneumoniaMNIST, BloodMNIST, BreastMNIST, OCTMNIST, OrganAMNIST, RetinaMNIST) | CNN kecil vs regresi logistik | ±2 menit (128,8 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab02.ipynb) bab02 | 2. Lingkungan kerja dan data medis: DICOM/NIfTI sintetis, de-identifikasi (identitas fiktif), demo kebocoran data | PneumoniaMNIST, OrganMNIST3D, NIH Malaria | CNN kecil | ±6 menit (337,5 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab03.ipynb) bab03 | 3. Praproses dan CNN: dari nol vs *transfer learning* | BloodMNIST, PneumoniaMNIST | CNN, ResNet-18 | ±7 menit (423,7 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab04.ipynb) bab04 | 4. Klasifikasi biner pneumonia, kurva ROC/PR, ambang operasi | PneumoniaMNIST | ResNet-18 | ±10 menit (571,1 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab05.ipynb) bab05 | 5. Multikelas dan data tidak seimbang (bobot kelas, oversampling, focal loss) | DermaMNIST (**CC BY-NC 4.0**) | ResNet-18 | ±11 menit (662,4 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab06.ipynb) bab06 | 6. Mikroskopi malaria, split per pasien, agregasi tingkat pasien, latensi CPU | NIH Malaria | ResNet-18, MobileNetV3-Small | ±6 menit (362,5 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab07.ipynb) bab07 | 7. Regresi lingkar kepala janin dan derajat retinopati ordinal | HC18, RetinaMNIST | ResNet-18 (regresi) | ±9 menit (510,1 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab08.ipynb) bab08 | 8. Deteksi dan penghitungan sel darah (mAP) | BCCD | Faster R-CNN MobileNetV3-Large FPN (torchvision) | ±5 menit (304,4 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab09.ipynb) bab09 | 9. Segmentasi semantik kepala janin (Dice, IoU, HC dari elips) | HC18 | U-Net (acak vs encoder ResNet-18 pra-latih), 2 seed | ±54 menit (3.234,4 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab10.ipynb) bab10 | 10. Citra volumetrik 3D: organ CT dan nodul paru | OrganMNIST3D, NoduleMNIST3D | CNN 3D (Conv3d) vs 2,5D (ResNet-18) | ±6 menit (359,0 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab11.ipynb) bab11 | 11. Evaluasi ala klinis: interval kepercayaan bootstrap, kalibrasi, *decision curve*, Bland–Altman | PneumoniaMNIST, HC18 | ResNet-18 | ±3 menit (178,4 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab12.ipynb) bab12 | 12. XAI (Grad-CAM), demo *shortcut learning* dengan penanda sintetis, ketidakpastian (MC dropout, ensemble), OOD | PneumoniaMNIST, OrganAMNIST (OOD) | ResNet-18 | ±4 menit (214,3 s) |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/bab13.ipynb) bab13 | 13. Ekspor ONNX, kuantisasi INT8, latensi CPU, demo Gradio lokal (`share=False`), *model card* | PneumoniaMNIST | MobileNetV3-Small, ResNet-18 | ±4 menit (233,4 s) |

Total semua notebook bab sekitar 2 jam (7.519 s).

## Cara menjalankan di Google Colab

1. Klik lencana **Open in Colab** pada tabel di atas, atau buka `https://colab.research.google.com/github/kang-prayit/visi-komputer-kesehatan/blob/main/notebooks/<nama>.ipynb`.
2. Pilih **Runtime → Change runtime type → T4 GPU**. Notebook 00 tidak memerlukan GPU.
3. Jalankan **`00_siapkan_dataset.ipynb` sekali**. Semua dataset diunduh dari sumber resminya ke `MyDrive/Buku Visi Komputer Kesehatan/dataset/`. Langkah ini opsional: setiap notebook bab juga mengunduh sendiri dataset yang belum ada lewat `pastikan_dataset()`.
4. Jalankan notebook bab dengan **Runtime → Run all**. Sel pertama akan meminta izin **mount Google Drive**.
   - **Tips mount gagal.** Popup izin Drive kadang lambat sehingga `drive.mount` gagal pada percobaan pertama. Sel standar sudah mencoba ulang hingga 3 kali dengan `force_remount=True`. Bila masih gagal, jalankan ulang sel tersebut, atau jalankan manual `from google.colab import drive; drive.mount("/content/drive", force_remount=True)`.
   - **Lokasi folder lain.** Atur variabel lingkungan sebelum sel standar, misalnya `import os; os.environ["BUKU_DRIVE_SUBPATH"] = "MyDrive/folder-saya"`. Nilai bawaan: `MyDrive/Buku Visi Komputer Kesehatan` (dataset di subfolder `dataset/`).
   - **Di luar Colab.** Pakai `BUKU_DRIVE` (folder akar) dan `BUKU_LOKAL` (folder cache), lalu `pip install -r requirements.txt`.
5. Hasil (tabel CSV, gambar, ringkasan JSON waktu eksekusi) disimpan di folder `hasil/` milik notebook. Angka Anda bisa sedikit berbeda dari buku karena perbedaan versi pustaka dan perangkat GPU.

**Keamanan.** Tidak ada token, kunci API, atau kata sandi di repositori ini. Dataset yang memerlukan akun (misalnya Kaggle) tidak dipakai. Bila Anda tetap memakainya, simpan token di **Colab Secrets**, jangan di notebook.

## Dataset dan lisensi

Repositori ini **tidak** menyertakan dataset, bobot model, atau hasil. Semua diunduh dari sumber resmi saat dijalankan. Patuhi lisensi masing-masing dan cantumkan atribusinya.

| Dataset | Bab | Sumber | Lisensi |
|---|---|---|---|
| MedMNIST v2 (PneumoniaMNIST, BloodMNIST, BreastMNIST, OCTMNIST, OrganAMNIST, RetinaMNIST, OrganMNIST3D, NoduleMNIST3D) | 1–4, 7, 10–13 | https://doi.org/10.5281/zenodo.10519652 (Yang dkk., 2023) | CC BY 4.0 |
| DermaMNIST (dari HAM10000) | 5 | https://doi.org/10.5281/zenodo.10519652 | ⚠️ **CC BY-NC 4.0, non-komersial**. Jangan dipakai untuk tujuan komersial. |
| NIH Malaria Cell Images (NLM/LHNCBC) | 2, 6 | https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html (Rajaraman dkk., 2018) | Data pemerintah AS; pernyataan lisensi eksplisit belum ditemukan **[VERIFY]**. Wajib atribusi. |
| HC18 (Grand Challenge) | 7, 9, 11 | https://doi.org/10.5281/zenodo.1327317 (van den Heuvel dkk., 2018) | CC BY 4.0 |
| BCCD | 8 | https://github.com/Shenggan/BCCD_Dataset | Repositori MIT; asal citra mentah **[VERIFY]** |
| RetinaMNIST (sumber DeepDRiD) | 1, 7 | via MedMNIST | CC BY 4.0 (via MedMNIST); lisensi sumber DeepDRiD **[VERIFY]** |
| OrganMNIST3D (sumber LiTS) | 2, 10 | via MedMNIST | CC BY 4.0 (via MedMNIST); lisensi sumber LiTS **[VERIFY]** |

Bobot pra-latih ImageNet torchvision (BSD-3) diunduh saat runtime dan tidak didistribusikan di sini. Dataset ImageNet sendiri berlisensi riset non-komersial.

## Sitasi

APA 7:

> Prayitno, Putra, K. T., Prasetyo, E., Sunardi, & Triyono, L. (2026). *Visi komputer untuk kesehatan: Klasifikasi, regresi, deteksi, dan segmentasi citra medis dengan Python* [Kode sumber]. [Penerbit: TO CONFIRM]. ISBN menyusul. https://github.com/kang-prayit/visi-komputer-kesehatan

Bila memakai dataset, sitasi juga makalah dataset di atas.

## Lisensi

Kode di repositori ini menggunakan lisensi **MIT** (lihat `LICENSE`). **Lisensi ini masih usulan dan menunggu konfirmasi penulis.** Lisensi MIT hanya berlaku untuk kode, tidak untuk dataset, naskah buku, atau gambar dari dataset.
