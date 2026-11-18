# 🤖 AI-TRADING — Prediksi Sinyal XAUUSD (M5) dengan Deep Learning

Sistem prediksi sinyal **BUY / SELL / HOLD** untuk **XAUUSD (Emas)** pada timeframe **M5**, memakai fitur **multi-timeframe (M15, M30, H1)** dan model deep learning berbasis **CNN + PatchTST (Transformer)** yang dilatih dengan PyTorch. Prediksi live dijalankan langsung dari terminal **MetaTrader 5**.

> ⚠️ **Disclaimer:** Proyek ini dibuat untuk riset dan pembelajaran. Ini **bukan saran finansial**. Trading emas/forex berisiko tinggi dan bisa menyebabkan kerugian. Hasil training atau backtest tidak menjamin performa di pasar nyata. Uji di **akun demo** sebelum memakai uang asli.

---

## ✨ Fitur Utama

- **Dua model terpisah**: satu model untuk sinyal BUY dan satu untuk SELL (klasifikasi biner, loss BCE).
- **Confidence head**: setiap model juga memprediksi tingkat keyakinan, dipakai sebagai filter tambahan.
- **Multi-timeframe**: fitur M5 digabung dengan fitur M15, M30, dan H1. Candle HTF digeser satu periode penuh agar tidak ada *look-ahead bias*.
- **Feature engineering yang kaya**: EMA, RSI, MACD, Bollinger Bands, ATR/NATR, ADX, volatilitas Garman-Klass, volume profile (POC), price density, pivot harian (PP/R1/S1), Fair Value Gap (FVG), sesi pasar (Asia/Eropa/Amerika), regime tren/volatilitas, dan lainnya.
- **Filter kondisi pasar**: sinyal di-skip saat pasar sideways (NATR rendah) atau tidak trending (ADX rendah).
- **Live prediction otomatis**: prediksi dijalankan setiap candle M5 baru selesai (closed), dengan auto-reconnect ke MT5.
- **Fine-tuning**: tersedia notebook untuk fine-tune model dengan data XAUUSD terbaru.

---

## 🧠 Arsitektur Model

```
Input (96 candle × N fitur)
        │
        ▼
 Conv1D + Residual SE-Block (CNN)
        │
        ▼
 PatchTST Encoder
 (Patch Embedding → Positional Embedding → Transformer Blocks → Attention Pooling)
        │
        ├──► Class Head  → probabilitas sinyal (sigmoid)
        └──► Conf Head   → confidence (sigmoid)
```

Pipeline keputusan saat live:

1. Ambil data candle terakhir dari MT5 (M5, M15, M30, H1), buang candle yang masih berjalan.
2. Hitung fitur lalu normalisasi dengan `scaler.pkl`.
3. Cek filter pasar: `NATR >= 0.0005` dan `ADX >= 0.20`.
4. Jalankan model BUY dan SELL, lalu hasilnya dibandingkan dengan threshold.
5. Keluarkan sinyal **BUY**, **SELL**, atau **HOLD**.

| Parameter | Nilai default | Keterangan |
|---|---|---|
| `WINDOW_INF` | 96 | Panjang window input (harus sama dengan saat training) |
| `BUY_THR` / `SELL_THR` | 0.70 | Threshold probabilitas minimum |
| `CONF_BUY_MIN` | 0.60 | Confidence minimum untuk BUY |
| `CONF_SELL_MIN` | 0.55 | Confidence minimum untuk SELL |
| `MARGIN` | 0.20 | Selisih minimum prob BUY vs SELL, kalau lebih kecil maka HOLD |
| `NATR_MIN_INFERENCE` | 0.0005 | Filter volatilitas minimum |
| `ADX_MIN` | 0.20 | Filter kekuatan tren minimum |

---

## 📁 Struktur Repository

```
AI-TRADING/
├── live_prediction.py                 # Script prediksi live (standalone, tanpa Jupyter)
├── prediksi_precision70_..._fix.ipynb # Notebook training utama (multi-timeframe, BCE, GPU)
├── finetune_xauusd_2026.ipynb         # Notebook fine-tuning dengan data 2026
├── finetune_xauusd_2026_1.ipynb       # Variasi notebook fine-tuning
├── scaler.pkl                         # Scaler fitur (hasil training)
├── buy_model.pkl                      # Model BUY (hasil training awal)
├── sell_model.pkl                     # Model SELL (hasil training awal)
├── buy_model_finetuned.pkl            # Model BUY hasil fine-tuning (dipakai live)
├── sell_model_finetuned.pkl           # Model SELL hasil fine-tuning (dipakai live)
└── README.md
```

---

## ⚙️ Persyaratan

- **Windows** (package `MetaTrader5` tidak berjalan di Linux/macOS)
- **MetaTrader 5** terinstal dan sudah login ke akun broker
- **Python 3.9+**
- GPU NVIDIA + CUDA (opsional, tapi disarankan untuk training)

Library Python:

```
MetaTrader5, pandas, numpy, torch, ta, joblib, scikit-learn
```

---

## 🚀 Cara Menjalankan Live Prediction

**1. Clone repository**

```bash
git clone https://github.com/Andra-Fauzi/AI-TRADING.git
cd AI-TRADING
```

**2. Install dependency**

```bash
pip install MetaTrader5 pandas numpy torch ta joblib scikit-learn
```

**3. Siapkan MetaTrader 5**

Buka terminal MT5 dan login ke akun broker (sangat disarankan akun **demo** dulu). Pastikan simbol XAUUSD tersedia di Market Watch.

**4. Atur konfigurasi** di bagian `CONFIG` pada `live_prediction.py`:

```python
MT5_PATH     = None      # contoh: r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN    = None      # contoh: 12345678
MT5_PASSWORD = None
MT5_SERVER   = None      # contoh: "Exness-MT5Trial"

SYMBOL = "XAUUSD"        # sesuaikan dengan nama simbol di broker (XAUUSD.m, GOLD, dll.)

SCALER_PATH     = "scaler.pkl"
BUY_MODEL_PATH  = "buy_model_finetuned.pkl"
SELL_MODEL_PATH = "sell_model_finetuned.pkl"
```

Jika terminal MT5 sudah terbuka dan login, kredensial boleh dibiarkan `None`.

**5. Jalankan**

```bash
python live_prediction.py
```

Contoh output:

```
⏳ Menunggu 142s sampai candle M5 berikutnya close...
📊 prob_buy=0.742 (conf=0.683) prob_sell=0.211 (conf=0.402)
📈 BUY — prob=0.742 conf=0.683
#1 | candle 2026-01-15 09:35:00 | 📈 BUY | conf=74.20%
```

Hentikan dengan `Ctrl + C`, koneksi MT5 akan ditutup otomatis.

---

## 🏋️ Training & Fine-Tuning

| Notebook | Fungsi |
|---|---|
| `prediksi_precision70_binary_3_multiframe_BCE_v5_gpu_optimized_2_best_auc_FIXED_torchcompile_fix.ipynb` | Training utama model BUY & SELL (multi-timeframe, BCE loss, optimasi GPU, `torch.compile`, checkpoint berdasarkan AUC terbaik) |
| `finetune_xauusd_2026.ipynb` / `finetune_xauusd_2026_1.ipynb` | Fine-tuning model dengan data XAUUSD tahun 2026 |

Urutan yang disarankan: jalankan notebook training utama, lalu notebook fine-tuning, kemudian pakai file `*_finetuned.pkl` untuk live prediction.

---

## 📝 Catatan Penting

- **Definisi class model harus identik** dengan saat training, karena `joblib.load()` memuat objek model penuh. Jangan ubah nama atau struktur class arsitektur di `live_prediction.py`.
- **Urutan dan jumlah fitur** (`FEATURE_COLUMNS`) serta `WINDOW_INF` harus sama persis dengan saat training.
- Fitur sesi pasar (Asia/Eropa/Amerika) memakai jam dari data MT5 (waktu server broker), jadi hasilnya bisa berbeda antar broker.
- Script ini **hanya menghasilkan sinyal prediksi** dan tidak mengeksekusi order otomatis.

---

## 🛣️ Rencana Pengembangan

- [ ] Eksekusi order otomatis dengan manajemen risiko (SL/TP, lot sizing)
- [ ] Backtesting terintegrasi
- [ ] Notifikasi sinyal (Telegram)
- [ ] Dukungan simbol dan timeframe lain

---

## 🤝 Kontribusi

Pull request dan issue sangat diterima. Silakan fork repository ini lalu ajukan perubahan lewat PR.

## 📄 Lisensi

Belum ditentukan. Tambahkan file `LICENSE` (misalnya MIT) sesuai kebutuhan.

## 👤 Author

**Andra Fauzi** — [@Andra-Fauzi](https://github.com/Andra-Fauzi)
