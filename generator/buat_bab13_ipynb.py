"""Membangun notebook Colab Bab 13 (notebooks/bab13.ipynb)."""
import nbformat as nbf, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sel_standar import kode_sel_standar
from pola_bab import sel_lingkungan, SEED_FN, sel_akhir, simpan, KODE_LATIH, IMPOR_UMUM

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bab 13 — Dari Notebook ke Aplikasi: ONNX, Kuantisasi, Latensi CPU, dan Demo Gradio
**Buku:** *Visi Komputer untuk Kesehatan: Klasifikasi, Regresi, Deteksi, dan Segmentasi Citra Medis dengan Python*
**Penulis:** Prayitno, Karisma Trinanda Putra, Eko Prasetyo, Sunardi, Liliek Triyono

Jalankan di **Google Colab dengan GPU T4**, lalu **Runtime → Run all**.

Isi:
1. Latih cepat dua model PneumoniaMNIST 128 px (CC BY 4.0): ResNet-18 dan MobileNetV3-Small (pra-latih ImageNet).
2. Simpan/muat `state_dict`; **ekspor ONNX** (pra-pemrosesan normalisasi ikut di dalam graf) dan verifikasi keluaran dengan **ONNX Runtime**.
3. **Kuantisasi INT8** ONNX Runtime: dinamis (bobot) dan statis QDQ (kalibrasi 256 citra latih) — ukuran berkas, AUC, kesepakatan prediksi.
4. **Latensi CPU** (batch 1; 1 dan 2 *thread*): PyTorch FP32 vs ONNX FP32 vs ONNX INT8.
5. **Demo Gradio** untuk riset/edukasi — dijalankan dengan `share=False` (**tidak** membuat tautan publik), disertai peringatan "bukan alat diagnosis".
6. Artefak siap-bagi: `model_card.md`, `requirements.txt`, `app.py`.

Hasil → `/content/hasil_bab13/` → `/content/hasil_bab13.zip`.
""")
md("## 0 Sel standar: mount Google Drive dan `pastikan_dataset()`")
code(kode_sel_standar())
md("## 13.0 Lingkungan")
code(sel_lingkungan("13"))
code('os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"          # matikan telemetri Gradio\n'
     'with catat_waktu("1_pip_install"):\n    # pip langsung (bukan jalankan(): batas 60 s di sana dapat memutus pemasangan gradio/onnx di tengah jalan)\n    r_pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist==3.0.2", "onnx", "onnxruntime", "onnxscript", "gradio"], capture_output=True, text=True, timeout=900)\n    print("pip selesai, kode keluar", r_pip.returncode); print(r_pip.stderr[-2000:]) if r_pip.returncode else None\n    assert r_pip.returncode == 0, "pip install gagal; lihat pesan di atas"\n' + IMPOR_UMUM + "\n" + SEED_FN + "\n" + KODE_LATIH +
     "\nfrom sklearn.metrics import roc_auc_score\nimport onnx, onnxruntime as ort, gradio as gr\n"
     "VERSI.update({'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'gradio': gr.__version__}); print(VERSI)")

md("## 13.1 Data dan dua model cepat")
code(r"""
UK = 28 if SMOKE else 128
with catat_waktu("2_muat_data"):
    d = npz_medmnist("pneumoniamnist", UK)
    D = {s: (d[f"{s}_images"], d[f"{s}_labels"].reshape(-1).astype(np.int64)) for s in ("train", "val", "test")}
    if SMOKE:
        r = np.random.default_rng(0); D = {s: (x[i], y[i]) for s, (x, y) in D.items() for i in [r.choice(len(y), min(len(y), 300), replace=False)]}
XT = {s: ke_tensor_uint8(D[s][0]) for s in D}; Y = {s: D[s][1] for s in D}; yt = Y["test"]
class MobileNetV3S(nn.Module):
    def __init__(self):
        super().__init__()
        try: self.net = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception as e:
            if not SMOKE: raise
            print("SMOKE: bobot acak", e); self.net = torchvision.models.mobilenet_v3_small(weights=None)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 1)
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1); return self.net((x - MEAN_IN.to(x.device)) / SD_IN.to(x.device))
loss_bce = lambda lg, y: F.binary_cross_entropy_with_logits(lg[:, 0], y.float())
def skor_auc(lg, y): return roc_auc_score(y, lg[:, 0].cpu().numpy())
AUG = dict(rot_derajat=7, geser=0.05, kecerahan=0.1, kontras=0.1)
MODEL = {}
for nm, fm, tahap in [("ResNet-18", lambda: ResNet18(1, True), [(2, 1e-3, True), (8, 1e-4, False)]), ("MobileNetV3-Small", lambda: MobileNetV3S(), [(10, 1e-3, False)])]:
    with catat_waktu(f"latih_{nm}"):
        MODEL[nm] = latih(nm, fm(), XT["train"], Y["train"], XT["val"], Y["val"], loss_bce, skor_auc, patience=3, aug=AUG, tahap=tahap, bs=64, seed=0)[0]
    print(nm, "AUC uji (PyTorch, GPU):", roc_auc_score(yt, logit_semua(MODEL[nm], XT["test"])[:, 0].cpu().numpy()))
""")

md(r"""
## 13.2 Menyimpan, memuat, dan mengekspor ke ONNX
Praktik baik: simpan **`state_dict`** (bukan seluruh objek) beserta metadata (kelas, ukuran masukan, normalisasi, versi pustaka).
Ekspor ONNX memakai masukan float32 `[batch, 1, UK, UK]` bernilai 0–1; normalisasi ImageNet sudah di dalam graf. Sumbu *batch* dibuat dinamis.
""")
code(r"""
X_UJI = XT["test"].float().div(255).cpu()
def ekspor(nm, model):
    model = model.float().cpu().eval(); torch.save({"state_dict": model.state_dict(), "kelas": ["normal", "pneumonia"], "ukuran": UK, "versi": VERSI}, OUT / f"{nm}.pt")
    contoh = torch.rand(1, 1, UK, UK); f = OUT / f"{nm}_fp32.onnx"
    try:
        torch.onnx.export(model, (contoh,), str(f), input_names=["citra"], output_names=["logit"], dynamic_axes={"citra": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
        jalur = "TorchScript (dynamo=False)"
    except Exception as e:
        print("Eksportir lama gagal, beralih ke dynamo:", repr(e)[:200])
        torch.onnx.export(model, (contoh,), str(f), input_names=["citra"], output_names=["logit"], dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}}, dynamo=True)
        jalur = "torch.export (dynamo=True)"
    onnx.checker.check_model(onnx.load(str(f))); return f, jalur
def sesi(f, thread=None):
    so = ort.SessionOptions()
    if thread: so.intra_op_num_threads = thread; so.inter_op_num_threads = 1
    return ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
def ort_logit(s, X, bs=256): return np.concatenate([s.run(None, {"citra": X[i:i + bs].numpy()})[0][:, 0] for i in range(0, len(X), bs)])
ONNX, baris = {}, []
with catat_waktu("3_ekspor_onnx"):
    for nm, m in MODEL.items():
        f, jalur = ekspor(nm, m); ONNX[nm] = {"FP32": f}
        with torch.no_grad(): lt = m(X_UJI[:256]).numpy()[:, 0]
        lo = ort_logit(sesi(f), X_UJI[:256])
        baris.append({"model": nm, "jalur_ekspor": jalur, "maks_|beda_logit|_torch_vs_onnx": float(np.abs(lt - lo).max()), "ukuran_MB": f.stat().st_size / 2 ** 20})
T131 = pd.DataFrame(baris); T131.to_csv(OUT / "tabel_13_1_ekspor_onnx.csv", index=False); T131
""")

md(r"""
## 13.3 Kuantisasi INT8 dengan ONNX Runtime
- **Dinamis**: bobot dikuantisasi ke INT8, aktivasi dikuantisasi saat inferensi (tanpa data kalibrasi).
- **Statis QDQ**: skala aktivasi ditentukan dari **256 citra latih** (kalibrasi MinMax), bobot per kanal.
Model terkuantisasi dibandingkan dengan FP32: ukuran, AUC uji, dan persentase prediksi (ambang 0,5) yang sama.
""")
code(r"""
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
class PembacaKalibrasi(CalibrationDataReader):
    def __init__(self, X, bs=32): self.it = iter([{"citra": X[i:i + bs].numpy()} for i in range(0, len(X), bs)])
    def get_next(self): return next(self.it, None)
r = np.random.default_rng(SEED); X_KAL = XT["train"][torch.as_tensor(r.choice(len(Y["train"]), min(256, len(Y["train"])), replace=False), device=XT["train"].device)].float().div(255).cpu()
baris = []
with catat_waktu("4_kuantisasi"):
    for nm in MODEL:
        f32 = ONNX[nm]["FP32"]; fpre = OUT / f"{nm}_pre.onnx"; quant_pre_process(str(f32), str(fpre))
        fdyn = OUT / f"{nm}_int8_dinamis.onnx"; quantize_dynamic(str(fpre), str(fdyn), weight_type=QuantType.QInt8)
        fst = OUT / f"{nm}_int8_statis.onnx"
        quantize_static(str(fpre), str(fst), PembacaKalibrasi(X_KAL), quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
        ONNX[nm].update({"INT8 dinamis": fdyn, "INT8 statis": fst}); fpre.unlink()
        p32 = 1 / (1 + np.exp(-ort_logit(sesi(f32), X_UJI)))
        for versi, f in ONNX[nm].items():
            try: p = 1 / (1 + np.exp(-ort_logit(sesi(f), X_UJI))); galat = ""
            except Exception as e: p = np.full(len(yt), np.nan); galat = repr(e)[:120]
            baris.append({"model": nm, "versi": versi, "ukuran_MB": f.stat().st_size / 2 ** 20, "AUC_uji": roc_auc_score(yt, p) if not np.isnan(p).any() else np.nan,
                          "persen_prediksi_sama_dg_FP32": float(((p >= 0.5) == (p32 >= 0.5)).mean() * 100), "maks_|beda_prob|": float(np.nanmax(np.abs(p - p32))), "catatan": galat})
T132 = pd.DataFrame(baris); T132.to_csv(OUT / "tabel_13_2_kuantisasi.csv", index=False); T132
""")

md(r"""
## 13.4 Latensi CPU (batch 1)
Median dari 100 inferensi setelah 10 pemanasan. CPU Colab biasanya 2 vCPU; angka ini **bukan** latensi perangkat tepi (mis. Raspberry Pi), tetapi
memperlihatkan pengaruh format dan kuantisasi. Informasi CPU tersimpan di `info_lingkungan.txt`.
""")
code(r"""
def ukur(fn, n=100, panas=10):
    for _ in range(panas): fn()
    t = []
    for _ in range(n): t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000), float(np.percentile(t, 90) * 1000)
x1 = X_UJI[:1]; N = 10 if SMOKE else 100; baris = []
with catat_waktu("5_latensi"):
    for th in (1, 2):
        torch.set_num_threads(th)
        for nm, m in MODEL.items():
            m = m.float().cpu().eval()
            with torch.no_grad(): med, p90 = ukur(lambda: m(x1), n=N)
            baris.append({"model": nm, "runtime": "PyTorch FP32", "thread": th, "ms_median": med, "ms_p90": p90})
            for versi, f in ONNX[nm].items():
                try:
                    s = sesi(f, th); med, p90 = ukur(lambda: s.run(None, {"citra": x1.numpy()}), n=N)
                except Exception as e: med, p90 = np.nan, np.nan
                baris.append({"model": nm, "runtime": f"ONNX Runtime {versi}", "thread": th, "ms_median": med, "ms_p90": p90})
T133 = pd.DataFrame(baris); T133.to_csv(OUT / "tabel_13_3_latensi_cpu.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for a, th in zip(ax, (1, 2)):
    d_ = T133[T133.thread == th]; lab = [f"{a_}\n{b_}" for a_, b_ in zip(d_.model, d_.runtime.str.replace("ONNX Runtime ", "ORT "))]
    a.barh(range(len(d_)), d_.ms_median); a.set_yticks(range(len(d_))); a.set_yticklabels(lab, fontsize=6); a.set_xlabel("ms per citra (median)"); a.set_title(f"{th} thread CPU")
plt.tight_layout(); plt.savefig(OUT / "gambar_13_1_latensi.png", bbox_inches="tight"); plt.show(); T133
""")

md(r"""
## 13.5 Demo Gradio (riset/edukasi, bukan klinis)
Antarmuka memuat model **ONNX** (tanpa PyTorch) sehingga sama dengan yang akan dipakai di `app.py`. Diluncurkan dengan **`share=False`**:
di Colab antarmuka tampil di bawah sel melalui *proxy* Colab dan **tidak** membuat tautan publik `*.gradio.live`.
Jangan mengunggah citra pasien sungguhan ke demo ini.
""")
code(r"""
PILIH_DEMO = max([n for n in MODEL], key=lambda n: T132[(T132.model == n) & (T132.versi == "FP32")].AUC_uji.iloc[0])
SESI_DEMO = sesi(ONNX[PILIH_DEMO]["FP32"])
PERINGATAN = ("⚠️ Demo riset/edukasi dari buku *Visi Komputer untuk Kesehatan*. BUKAN alat diagnosis dan tidak boleh dipakai untuk keputusan klinis. "
              "Model dilatih pada PneumoniaMNIST (X-ray dada anak, 128 px) dan dapat salah pada citra dari sumber lain.")
def prediksi_demo(img):
    if img is None: return {}
    from PIL import Image as _I
    im = _I.fromarray(np.asarray(img)).convert("L").resize((UK, UK), _I.BILINEAR)
    x = (np.asarray(im, dtype=np.float32) / 255.0)[None, None]
    p = float(1 / (1 + np.exp(-SESI_DEMO.run(None, {"citra": x})[0][0, 0])))
    return {"pneumonia": p, "normal": 1 - p}
CONTOH = []
(OUT / "contoh_demo").mkdir(exist_ok=True)
from PIL import Image as _I
for k, i in enumerate([int(np.where(yt == 0)[0][0]), int(np.where(yt == 1)[0][0])]):
    fn = OUT / "contoh_demo" / f"uji_{i}_{['normal', 'pneumonia'][yt[i]]}.png"; _I.fromarray(D["test"][0][i]).save(fn); CONTOH.append([str(fn)])
print({c[0]: prediksi_demo(np.array(_I.open(c[0]))) for c in CONTOH})
demo = gr.Interface(fn=prediksi_demo, inputs=gr.Image(type="numpy", image_mode="L", label="X-ray dada (PNG/JPG)"), outputs=gr.Label(num_top_classes=2, label="probabilitas"),
                    title=f"Demo edukasi: pneumonia pada X-ray dada ({PILIH_DEMO}, ONNX)", description=PERINGATAN, examples=CONTOH, flagging_mode="never")
demo.launch(share=False, prevent_thread_lock=True, quiet=True, **({"server_name": "127.0.0.1"} if SMOKE else {}))
if SMOKE: demo.close()
""")

md("## 13.6 Artefak siap-bagi: `app.py`, `requirements.txt`, `model_card.md`")
code(r"""
(OUT / "app.py").write_text(f'''# Demo edukasi Gradio — Bab 13 buku Visi Komputer untuk Kesehatan. BUKAN alat diagnosis.
import numpy as np, onnxruntime as ort, gradio as gr
from PIL import Image
UK = {UK}
SESI = ort.InferenceSession("{PILIH_DEMO}_fp32.onnx", providers=["CPUExecutionProvider"])
PERINGATAN = {PERINGATAN!r}
def prediksi(img):
    if img is None: return {{}}
    x = (np.asarray(Image.fromarray(np.asarray(img)).convert("L").resize((UK, UK), Image.BILINEAR), dtype=np.float32) / 255.0)[None, None]
    p = float(1 / (1 + np.exp(-SESI.run(None, {{"citra": x}})[0][0, 0])))
    return {{"pneumonia": p, "normal": 1 - p}}
gr.Interface(fn=prediksi, inputs=gr.Image(type="numpy", image_mode="L"), outputs=gr.Label(num_top_classes=2),
             title="Demo edukasi: pneumonia pada X-ray dada", description=PERINGATAN, flagging_mode="never").launch(share=False)
''')
(OUT / "requirements.txt").write_text(f"numpy\npillow\nonnxruntime=={ort.__version__}\ngradio=={gr.__version__}\n")
r32 = T132[(T132.model == PILIH_DEMO) & (T132.versi == "FP32")].iloc[0]; lat = T133[(T133.model == PILIH_DEMO) & (T133.runtime == "ONNX Runtime FP32")]
(OUT / "model_card.md").write_text(f'''# Kartu Model — {PILIH_DEMO} PneumoniaMNIST (demo edukasi Bab 13)
**Penggunaan yang dimaksud:** pembelajaran dan riset; **bukan** untuk diagnosis, triase, atau keputusan klinis.
**Data:** PneumoniaMNIST {UK} px (MedMNIST v2, CC BY 4.0; sumber Kermany dkk., 2018 — X-ray dada anak, satu pusat). Pembagian resmi train/val/test.
**Arsitektur:** {PILIH_DEMO}, pra-latih ImageNet, keluaran 1 logit (pneumonia). Masukan: 1×{UK}×{UK}, nilai 0–1.
**Kinerja (set uji internal, n={len(yt)}):** AUC FP32 = {r32.AUC_uji:.3f}. Belum ada validasi eksternal.
**Latensi CPU (ONNX FP32, batch 1):** {"; ".join(f"{int(t)} thread: {m:.1f} ms" for t, m in zip(lat.thread, lat.ms_median))} — diukur pada CPU host Colab (spesifikasi di info_lingkungan.txt).
**Keterbatasan:** satu sumber data; anak-anak saja; resolusi rendah; tidak dikalibrasi untuk prevalensi lain; rentan pergeseran distribusi dan *shortcut* (lihat Bab 11–12).
**Risiko & mitigasi (konsep ISO 14971):** salah negatif → bukan pengganti pembacaan radiolog; demo memuat peringatan; tidak menyimpan citra unggahan.
**Keamanan:** demo dijalankan `share=False`; jangan unggah data pasien yang dapat dikenali.
**Versi:** {json.dumps(VERSI)}
''')
print((OUT / "model_card.md").read_text()[:1500])
""")

md("## 13.7 Ringkasan, zip hasil, dan total waktu")
code(sel_akhir("13", 'dict(ukuran_px=UK, ekspor=T131.to_dict("records"), kuantisasi=T132.to_dict("records"), latensi=T133.to_dict("records"), model_demo=PILIH_DEMO)'))
simpan(C, "bab13.ipynb", Path(__file__).parent / "bab13.ipynb")
