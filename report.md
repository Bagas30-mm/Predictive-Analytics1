# Laporan Proyek Machine Learning - Analisa Personality

## Domain Proyek

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi tipe kepribadian (Introvert atau Extrovert) berdasarkan pola aktivitas sosial. Pendekatan ini memanfaatkan data terkait waktu sendiri, kehadiran di acara sosial, frekuensi aktivitas di luar rumah, ukuran lingkaran pertemanan, serta frekuensi posting di media sosial.

**Mengapa masalah ini penting untuk diselesaikan?**

Memahami kepribadian individu penting dalam banyak aspek, termasuk pengembangan personal, pendidikan, dan rekrutmen kerja. Dengan memanfaatkan data sosial, model ini dapat memberikan alternatif non-invasif dalam menilai kepribadian, yang sebelumnya memerlukan tes psikologi tradisional seperti MBTI atau Big Five. Ini dapat menghemat waktu, biaya, dan membuka peluang aplikasi skala besar pada platform digital.

**Referensi Pendukung:**

- Srivastava, S., et al. (2003). Development of personality in early adulthood: "Big Five" personality traits. _Journal of Personality and Social Psychology_.
- Kosinski, M., Stillwell, D., & Graepel, T. (2013). Private traits and attributes are predictable from digital records of human behavior. _PNAS_.

## Business Understanding

Pada bagian ini dijelaskan proses klarifikasi masalah dan tujuan proyek.

### Problem Statements

- Bagaimana menentukan tipe kepribadian seseorang berdasarkan pola aktivitas sosial?
- Indikator apa saja yang secara signifikan mendukung klasifikasi kepribadian?

### Goals

- Mengembangkan model klasifikasi yang mampu memprediksi tipe kepribadian dengan akurasi tinggi.
- Mengidentifikasi fitur-fitur utama yang mempengaruhi perbedaan antara individu introvert dan extrovert.

**Solution Statement:**

Pendekatan solusi dimulai dengan eksplorasi berbagai algoritma menggunakan LazyClassifier untuk mengidentifikasi model baseline. Setelah itu, dipilih SVC karena menunjukkan performa terbaik dalam hal akurasi dan generalisasi. Dilakukan juga hyperparameter tuning menggunakan GridSearchCV untuk mengoptimalkan parameter seperti `C` dan `kernel`.

## Data Understanding

Dataset yang digunakan pada proyek ini adalah _personality_dataset.csv_, yang terdiri dari 2.900 entri dengan distribusi kelas Introvert dan Extrovert yang relatif seimbang. Data ini bersumber dari Kaggle dan dapat diunduh melalui tautan berikut: [Extrovert vs Introvert Behavior Data (Kaggle)](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data). Dataset ini menyediakan berbagai fitur terkait perilaku sosial yang relevan untuk analisis kepribadian. data ini mencakup variabel-variabel berikut:

- **Time_spent_Alone:** Waktu (jam per hari) yang dihabiskan sendirian.
- **Stage_fear:** Indikator apakah seseorang memiliki ketakutan tampil di depan umum.
- **Social_event_attendance:** Frekuensi kehadiran dalam acara sosial.
- **Going_outside:** Frekuensi keluar rumah.
- **Friends_circle_size:** Ukuran lingkaran pertemanan.
- **Post_frequency:** Frekuensi posting aktivitas di media sosial.
- **Personality:** Label tipe kepribadian (Introvert atau Extrovert).

**Eksplorasi Visualisasi:**

- **KDE Plot**: Menunjukkan bahwa `Time_spent_Alone` cenderung lebih tinggi pada kelompok Introvert.
- **Countplot**: Menunjukkan bahwa kelompok Extrovert lebih sering menghadiri acara sosial dan lebih aktif memposting di media sosial.

## Data Preparation

Tahapan data preparation yang dilakukan antara lain:

- **Encoding:** Variabel kategori seperti Stage_fear, Drained_after_socializing, dan Personality dikonversikan ke format numerik.
- **Handling Missing Values:** Menghapus nilai yang hilang untuk memastikan integritas data.
- **Seleksi Fitur:** Menghapus kolom yang tidak relevan (Stage_fear dan Drained_after_socializing) untuk pemodelan.

**Penjelasan Langkah-langkah:**

- **Encoding:** LabelEncoding digunakan untuk variabel target `Personality`, dengan 0 = Introvert dan 1 = Extrovert.
- **Missing Value Handling:** Karena data tidak terlalu besar, missing values dihapus (dropna) agar tidak mengintroduksi bias dari imputasi.
- **Feature Selection:** `Stage_fear` dan `Drained_after_socializing` dihapus karena menunjukkan korelasi rendah terhadap label dan menyebabkan multikolinearitas dalam beberapa model awal.

## Modeling

Proses pemodelan dilakukan dalam beberapa langkah:

1. **Pemisahan Data:** Data dibagi menjadi set pelatihan dan pengujian dengan metode train_test_split.
2. **Eksplorasi Algoritma:** LazyClassifier digunakan untuk menguji berbagai algoritma dan memperoleh gambaran performa awal.
3. **Implementasi SVC:** Model Support Vector Classifier (SVC) diterapkan sebagai model final. Pelatihan dan evaluasi dilakukan dengan pembagian data latih dan uji serta pengaturan random_state untuk reproduksibilitas.

- Hasil pengujian dengan LazyPredict menunjukkan bahwa model SVC merupakan model terbaik.
- Model SVC dipilih karena memiliki performa yang lebih baik berdasarkan metrik evaluasi seperti classification report dan confusion matrix.
- Model SVC diimplementasikan dengan menggunakan parameter `random_state` untuk memastikan hasil yang konsisten dan dapat direproduksi pada setiap proses pelatihan.

## Evaluation

Evaluasi model dilakukan dengan cara:

- **Classification Report:** Menampilkan metrik seperti precision, recall, F1 score, dan akurasi.
- **Confusion Matrix:** Visualisasi untuk melihat kesalahan klasifikasi antara kelas Introvert dan Extrovert.
- **Feature Importance:** Mengukur kontribusi masing-masing fitur terhadap performa model.

**Penjelasan Metrik Evaluasi:**

- **Precision**: Seberapa banyak prediksi positif yang benar (TP / (TP + FP)).
- **Recall**: Seberapa banyak kelas positif yang tertangkap (TP / (TP + FN)).
- **F1 Score**: Harmonik rata-rata precision dan recall, cocok untuk data yang tidak seimbang.
- **Confusion Matrix**: Menyediakan informasi kesalahan klasifikasi antar kelas.

**Hasil Evaluasi Model:**

```
              precision    recall  f1-score   support

           0       0.91      0.95      0.93       250
           1       0.94      0.91      0.93       246

    accuracy                           0.93       496
   macro avg       0.93      0.93      0.93       496
weighted avg       0.93      0.93      0.93       496
```

Model SVC yang digunakan berhasil mencapai akurasi 93% dengan nilai precision dan recall yang seimbang pada kedua kelas, menunjukkan performa yang baik dalam membedakan antara Introvert dan Extrovert.

## Usage

Contoh fungsi berikut menunjukkan cara memprediksi tipe kepribadian berdasarkan input fitur:

```python
# filepath: e:\Kerja [report.md](http://_vscodecontentref_/0)
def predict_personality(time_spent_alone, social_event_attendance, going_outside, friends_circle_size, post_frequency):
    """
    Predicts the personality type (Introvert or Extrovert) based on the given features.

    Parameters:
    - time_spent_alone (float): Time spent alone per day.
    - social_event_attendance (float): Frequency of social event attendance.
    - going_outside (float): Frequency of going outside.
    - friends_circle_size (float): Size of friends circle.
    - post_frequency (float): Frequency of posting on social media.

    Returns:
    - str: Predicted personality type ('Introvert' or 'Extrovert').
    """
    import pandas as pd
    # Buat DataFrame dari input
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_spent_alone],
        'Social_event_attendance': [social_event_attendance],
        'Going_outside': [going_outside],
        'Friends_circle_size': [friends_circle_size],
        'Post_frequency': [post_frequency]
    })

    # Melakukan prediksi menggunakan model SVC yang telah dilatih
    prediction = model.predict(input_data)
    personality_type = 'Extrovert' if prediction[0] == 1 else 'Introvert'
    return personality_type

# Contoh penggunaan:
time_spent_alone = 9
social_event_attendance = 0
going_outside = 0
friends_circle_size = 0
post_frequency = 3
predicted_personality = predict_personality(time_spent_alone, social_event_attendance, going_outside, friends_circle_size, post_frequency)
print(f"Predicted personality: {predicted_personality}")
```
