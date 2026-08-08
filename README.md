# Makine Öğrenmesi Ara Ödev – Müşteri Churn Sınıflandırması

## Projenin Amacı
Bu proje, temel bir makine öğrenmesi akışını (veri oluşturma → veri inceleme
→ ön işleme → öznitelik mühendisliği → train/validation/test bölme → model
eğitimi → değerlendirme) tek bir Python dosyasında uçtan uca uygulamak için
hazırlanmıştır. Problem, bir müşterinin abonelikten ayrılıp ayrılmayacağını
(churn) tahmin eden bir **ikili sınıflandırma** problemidir.

Derste paylaşılan bir müşteri veri seti bulunmadığından/senaryoya uymadığından,
`musteri_churn_siniflandirma.py` içinde en az 100 satırlık, gerçekçi bir
mantığa dayanan **sentetik müşteri verisi** üretilmiştir (yaş, gelir,
abonelik süresi, destek talebi sayısı, şehir, üyelik tipi ve hedef değişken
olarak churn).

## Dosyalar
- `musteri_churn_siniflandirma.py` — ödevin tüm adımlarını içeren ana Python dosyası
- `requirements.txt` — gerekli kütüphaneler
- `README.md` — bu dosya

## Nasıl Çalıştırılır
```bash
pip install -r requirements.txt
python musteri_churn_siniflandirma.py
```

Betik çalıştırıldığında sırasıyla:
1. Sentetik veri seti oluşturulur (400 satır),
2. Verinin ilk satırları, boyutu ve hedef değişken dağılımı yazdırılır,
3. Eksik değerler kontrol edilip medyan ile doldurulur,
4. Yeni öznitelikler üretilir (`gelir_grubu`, `destek_talebi_var_mi`, `abonelik_yili`),
5. Kategorik değişkenler One-Hot Encoding ile sayısala çevrilir,
6. Sayısal değişkenler `StandardScaler` ile ölçeklenir,
7. Veri stratify edilerek train (%60) / validation (%20) / test (%20) olarak bölünür,
8. Logistic Regression, KNN ve (bonus) Decision Tree modelleri eğitilir,
9. Modeller validation seti üzerinde karşılaştırılır ve en iyi model (F1 skoruna göre) seçilir,
10. Seçilen model test seti üzerinde değerlendirilir: confusion matrix, accuracy, precision, recall, F1-score,
11. Kısa bir sonuç yorumu konsola yazdırılır.

## Kısa Sonuç Yorumu
Kod her çalıştırıldığında, hangi modelin validation performansına göre daha
iyi olduğu ve olası nedeni konsolun sonunda otomatik olarak yazdırılır
(bkz. "12) KISA YORUM" bölümü). Kullanılan sabit `random_state` sayesinde
sonuçlar her çalıştırmada tekrarlanabilirdir.

Genel gözlem: veri setindeki churn ile abonelik süresi / destek talebi
sayısı arasındaki ilişki büyük ölçüde doğrusal kurgulandığından Logistic
Regression genelde tutarlı ve dengeli sonuçlar verirken, KNN ölçeklenmiş
uzayda yerel kümelenmeleri yakalayarak zaman zaman daha yüksek F1 skoruna
ulaşabilmektedir. Decision Tree ise küçük veri setinde biraz daha
değişken (varyansı yüksek) sonuçlar üretmektedir.
