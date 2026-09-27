# ⚡ Zenless Zone Zero (ZZZ) Gacha Data Mining & Predictor

Aplikasi web berbasis Data Mining dan Machine Learning untuk analisis riwayat tarikan (*Signal History*) dan prediksi perolehan Karakter S-Rank dan W-Engine S-Rank pada game *Zenless Zone Zero* (ZZZ). Projek ini dibuat untuk memenuhi tugas besar mata kuliah **Penambangan Data (Data Mining) Semester 5**.

---

## 🧠 3 Algoritma Inti yang Digunakan

1. **Rantai Markov Penyerap (*Absorbing Markov Chain*):**
   * Pendekatan stokastik matematis murni menggunakan Matriks Fundamental $N = (I - Q)^{-1}$.
   * Memodelkan setiap langkah dari Pity 0 sampai Pity Max dengan S-Rank sebagai *absorbing state*.
   * Menghitung nilai ekspektasi tarikan tersisa (*expected pulls left*) dan kurva probabilitas kumulatif secara analitik tanpa bias sampel.

2. **Random Forest (Ensemble Bagging):**
   * Menggabungkan 100 *Decision Trees* independen secara paralel.
   * Melakukan tugas Regresi (prediksi estimasi sisa tarikan) dan Tugas Klasifikasi (prediksi rentang pity: *Early*, *Mid*, *Soft/Hard Pity*).
   * Menghasilkan akurasi **91.02%** dan tahan terhadap *overfitting*.

3. **XGBoost (*Extreme Gradient Boosting*):**
   * Model Machine Learning terbaik (**Akurasi 96.70%**, MAE kesalahan prediksi terendah ~**4.37 pull**).
   * Mampu mengoreksi residual error pohon sebelumnya secara bertahap, sangat presisi dalam menangkap lonjakan tajam pada fase *soft pity*.

---

## 🎯 2 Kategori Banner yang Didukung

* **🎭 Karakter Eksklusif (Exclusive Channel):**
  * Hard Pity: **90 pull**
  * Soft Pity: **Tarikan 74 - 89** (Base rate: 0.6%)
  * Aturan: **50:50** (Peluang menang 50%, jika kalah dijamin dapat di S-Rank berikutnya)
* **⚙️ W-Engine Eksklusif (Dissonant Sonata):**
  * Hard Pity: **80 pull**
  * Soft Pity: **Tarikan 65 - 79** (Base rate: 1.0%)
  * Aturan: **75:25** (Peluang menang 75%, jika kalah dijamin dapat di S-Rank berikutnya)

---

## ⚡ Fitur Unggulan

1. **Auto-Extraction Pipeline (PowerShell):**
   * Mendeteksi instalasi ZZZ otomatis via Windows Registry, Active Process, dan File Cache.
   * Menghindari *file-lock* saat game sedang berjalan.
   * Auto-detection status authkey (notifikasi ramah jika expired).

2. **Simulasi Monte Carlo Real-Time (10.000 Percobaan Langsung):**
   * Menjalankan 10.000 simulasi acak langsung di latar belakang setiap kali slider pity atau toggle garansi digeser.
   * Memberikan persentase kepastian: Peluang dapat dalam 10 pull, 20 pull, 30 pull, dan batas 90% aman.

3. **Dashboard Statistik Akun & Riwayat S-Rank:**
   * Tracking pity real-time, winrate 50:50 / 75:25, rata-rata pity akun, dan tabel riwayat S-Rank.

4. **Komparasi Akademik untuk Evaluasi Dosen:**
   * Tabel metrik evaluasi lengkap: Akurasi, Precision, Recall, F1-Score, MAE, RMSE, dan Waktu Komputasi.
   * Visualisasi *Feature Importance* (Analisis faktor paling berpengaruh terhadap tarikan gacha).

---

## 🚀 Cara Menjalankan

1. Double-click file [`run_app.bat`](file:///e:/projek_data_mining/run_app.bat) di folder projek.
2. Web otomatis terbuka di `http://localhost:8501`.
3. Buka tab **Prediksi 3 Algoritma** untuk melihat hasil analisis real-time!
