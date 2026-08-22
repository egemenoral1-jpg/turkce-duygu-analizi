# 💬 Türkçe Duygu Analizi — TF-IDF'ten BERT'e

Türkçe müşteri yorumlarında duygu analizi yapan, klasik makine öğrenmesi (TF-IDF + Logistic Regression) ile modern bir transformer modelini (fine-tuned Türkçe BERT) karşılaştıran bir proje.

## 🎯 Proje Özeti

- **Veri seti**: [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) (HuggingFace)
- **Görev**: İkili sınıflandırma (Positive / Negative)
- **Baseline model**: TF-IDF + Logistic Regression
- **İleri model**: Fine-tuned `dbmdz/bert-base-turkish-cased`
- **Demo**: Streamlit ile canlı tahmin arayüzü

## 📊 Sonuçlar

| Model | Macro F1 | Accuracy | Negative Precision |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.86 | 0.91 | 0.69 |
| Fine-tuned BERT | **0.92** | **0.95** | **0.89** |

BERT, özellikle olumsuzlama içeren karmaşık cümlelerde ("Hiç kötü değil, gayet iyiydi.") baseline'ın kaçırdığı bağlamı doğru yakalıyor.

## 🗂️ Veri Hazırlığı Notu

Orijinal veri setinde 3 sınıf vardı (Positive/Negative/Notr). Analiz sırasında **"Notr" etiketinin neredeyse tamamen tek bir kaynaktan (Wikipedia) geldiği** tespit edildi — bu, modelin duygu yerine "yazı stilini" öğrenmesine yol açabilecek bir confounding (karıştırıcı değişken) sorunuydu. Bu yüzden Notr sınıfı çıkarılıp ikili (Positive/Negative) sınıflandırmaya geçildi.

## 🚀 Kurulum ve Çalıştırma

```bash
git clone https://github.com/egemenoral1-jpg/turkce-duygu-analizi.git
cd turkce-duygu-analizi

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Veri setini hazırlama ve baseline modeli eğitme

`notebooks/01_veri_kesfi.ipynb` ve `notebooks/02_baseline_model.ipynb` dosyalarını sırayla çalıştırın.

### BERT fine-tuning (Colab, GPU gerekli)

`notebooks/03_bert_finetune.ipynb` dosyasını [Google Colab](https://colab.research.google.com)'da, T4 GPU ile çalıştırın. Eğitilen model ağırlıklarını indirip `models/bert_finetuned/` klasörüne yerleştirin.

### Demo'yu çalıştırma

```bash
streamlit run app/streamlit_app.py
```

## 📁 Proje Yapısı

turkce-duygu-analizi/
├── notebooks/ # Veri keşfi, baseline, BERT fine-tuning
├── src/ # Yeniden kullanılabilir Python fonksiyonları
├── app/ # Streamlit demo uygulaması
├── models/ # Eğitilmiş model ağırlıkları (git'e dahil değil)
└── data/ # Veri setleri (git'e dahil değil, HuggingFace'ten indirilir)


## 🛠️ Kullanılan Teknolojiler

- Python, pandas, scikit-learn
- HuggingFace `transformers`, `datasets`
- PyTorch
- Streamlit

## 📝 Lisans

Veri seti [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) kaynağına aittir.