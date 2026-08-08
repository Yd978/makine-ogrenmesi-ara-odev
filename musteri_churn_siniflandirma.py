"""
Makine Öğrenmesi Ara Ödevi - Müşteri Ayrılma (Churn) Tahmini
================================================================

AMAÇ
-----
Bu dosyada, dersteki temel makine öğrenmesi akışı küçük ve anlaşılır bir
sınıflandırma problemi (müşteri churn tahmini) üzerinde uçtan uca uygulanır:
veri oluşturma -> veri inceleme -> ön işleme -> öznitelik üretme ->
train/validation/test bölme -> model eğitimi -> metriklerle değerlendirme.

Hedef değişken: churn (0 = müşteri kalır, 1 = müşteri ayrılır)

KULLANILAN KÜTÜPHANELER
------------------------
- numpy            : sentetik veri üretimi ve sayısal işlemler
- pandas           : veri okuma / DataFrame işlemleri
- scikit-learn      : ön işleme (encoding, scaling), train/val/test bölme,
                      modeller (Logistic Regression, KNN, Decision Tree)
                      ve değerlendirme metrikleri

ÇALIŞTIRMA ADIMLARI
--------------------
1) Gerekli kütüphaneleri kurun:
       pip install -r requirements.txt
2) Betiği çalıştırın:
       python musteri_churn_siniflandirma.py
3) Çalıştırıldığında sırasıyla veri inceleme çıktıları, ön işleme adımları,
   model karşılaştırma sonuçları ve test seti metrikleri konsola yazdırılır.

NOT: Derste paylaşılan hazır bir müşteri veri seti bulunmadığı / ödev
senaryosuna uymadığı için, ödev metninde belirtildiği gibi Python içinde
en az 100 satırlık, gerçekçi bir mantığa dayanan sentetik müşteri verisi
üretilmiştir (bkz. `veri_seti_olustur` fonksiyonu). Sonuçların her
çalıştırmada aynı çıkması için sabit bir random_state kullanılmıştır.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 2) VERİ SETİ OLUŞTURMA
# ---------------------------------------------------------------------------
def veri_seti_olustur(n=400, random_state=RANDOM_STATE):
    """
    En az 100 satırlık, sentetik bir müşteri churn veri seti üretir.

    Sütunlar:
        yas                  : müşteri yaşı
        gelir                : aylık gelir (TL)
        abonelik_suresi      : müşterinin abone olduğu ay sayısı
        destek_talebi_sayisi : son 6 ayda açılan destek talebi sayısı
        sehir                : müşterinin yaşadığı şehir (kategorik)
        uyelik_tipi          : üyelik paketi (kategorik)
        churn                : hedef değişken (0 = kalır, 1 = ayrılır)

    Churn olasılığı; düşük abonelik süresi, yüksek destek talebi sayısı ve
    düşük gelir ile artacak şekilde kurgulanmıştır ki model bir şeyler
    öğrenebilsin (tamamen rastgele değil, gerçekçi bir örüntü var).
    """
    rng = np.random.default_rng(random_state)

    yas = rng.integers(18, 70, size=n).astype(float)
    gelir = rng.normal(loc=18000, scale=7000, size=n)
    gelir = np.clip(gelir, 3000, None).round(2)
    abonelik_suresi = rng.integers(1, 60, size=n).astype(float)  # ay
    destek_talebi_sayisi = rng.poisson(lam=1.5, size=n).astype(float)

    sehirler = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]
    sehir = rng.choice(sehirler, size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])

    uyelik_tipleri = ["Standart", "Premium", "Kurumsal"]
    uyelik_tipi = rng.choice(uyelik_tipleri, size=n, p=[0.5, 0.35, 0.15])

    # Churn olasılığını belirleyen basit bir skor (lojistik fonksiyon)
    skor = (
        -0.05 * abonelik_suresi
        + 0.55 * destek_talebi_sayisi
        - 0.00006 * gelir
        + 0.4 * (uyelik_tipi == "Standart")
        - 0.3 * (uyelik_tipi == "Kurumsal")
    )
    olasilik = 1 / (1 + np.exp(-skor))
    churn = rng.binomial(1, olasilik)

    df = pd.DataFrame(
        {
            "yas": yas,
            "gelir": gelir,
            "abonelik_suresi": abonelik_suresi,
            "destek_talebi_sayisi": destek_talebi_sayisi,
            "sehir": sehir,
            "uyelik_tipi": uyelik_tipi,
            "churn": churn,
        }
    )

    # Gerçekçi olması için bazı hücrelere kasıtlı olarak eksik değer koyuyoruz
    eksik_idx_gelir = rng.choice(df.index, size=int(n * 0.04), replace=False)
    eksik_idx_yas = rng.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[eksik_idx_gelir, "gelir"] = np.nan
    df.loc[eksik_idx_yas, "yas"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 3) TEMEL VERİ İNCELEME
# ---------------------------------------------------------------------------
def veriyi_incele(df):
    print("=" * 70)
    print("3) TEMEL VERİ İNCELEME")
    print("=" * 70)
    print("\nİlk 5 satır:")
    print(df.head())

    print(f"\nSatır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")

    print("\nHedef değişken (churn) dağılımı:")
    print(df["churn"].value_counts())
    print(df["churn"].value_counts(normalize=True).round(3))


# ---------------------------------------------------------------------------
# 4) EKSİK DEĞER KONTROLÜ VE TEMİZLEME
# ---------------------------------------------------------------------------
def eksik_degerleri_temizle(df):
    print("\n" + "=" * 70)
    print("4) EKSİK DEĞER KONTROLÜ")
    print("=" * 70)
    print("\nSütun bazında eksik değer sayısı:")
    print(df.isnull().sum())

    df = df.copy()
    # Sayısal sütunlardaki eksik değerleri medyan ile dolduruyoruz
    # (medyan, aykırı değerlerden ortalamaya göre daha az etkilenir)
    for kolon in ["yas", "gelir"]:
        medyan = df[kolon].median()
        df[kolon] = df[kolon].fillna(medyan)

    print("\nDoldurma sonrası eksik değer sayısı:")
    print(df.isnull().sum())
    return df


# ---------------------------------------------------------------------------
# 7) ÖZNİTELİK ÜRETME (Feature Engineering)
# ---------------------------------------------------------------------------
def oznitelik_uret(df):
    """
    En az 1 yeni öznitelik üretilmesi isteniyor; burada üç tane üretiliyor:
      - gelir_grubu        : gelire göre kategorik grup (dusuk/orta/yuksek)
      - destek_talebi_var_mi : hiç destek talebi açılmış mı (0/1)
      - abonelik_yili       : abonelik süresinin yıl cinsinden hali
    """
    df = df.copy()

    df["gelir_grubu"] = pd.cut(
        df["gelir"],
        bins=[0, 12000, 25000, np.inf],
        labels=["dusuk", "orta", "yuksek"],
    )

    df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)

    df["abonelik_yili"] = (df["abonelik_suresi"] / 12).round(2)

    return df


# ---------------------------------------------------------------------------
# 5) ONE-HOT ENCODING
# ---------------------------------------------------------------------------
def kategorik_encode_et(df):
    kategorik_kolonlar = ["sehir", "uyelik_tipi", "gelir_grubu"]
    df_encoded = pd.get_dummies(df, columns=kategorik_kolonlar, drop_first=True)
    return df_encoded


# ---------------------------------------------------------------------------
# ANA AKIŞ
# ---------------------------------------------------------------------------
def main():
    # 2) Veri setini oluştur
    df = veri_seti_olustur(n=400)

    # 3) Veri inceleme
    veriyi_incele(df)

    # 4) Eksik değer temizleme
    df = eksik_degerleri_temizle(df)

    # 7) Öznitelik üretme
    df = oznitelik_uret(df)

    # 5) One-Hot Encoding
    df = kategorik_encode_et(df)

    # Hedef ve öznitelikleri ayır
    y = df["churn"]
    X = df.drop(columns=["churn"])

    # 8) Train / Validation / Test bölme (stratify ile sınıf oranı korunur)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\n" + "=" * 70)
    print("8) TRAIN / VALIDATION / TEST BÖLME")
    print("=" * 70)
    print(f"Train: {X_train.shape[0]} satır | Val: {X_val.shape[0]} satır | "
          f"Test: {X_test.shape[0]} satır")

    # 6) Sayısal değişkenlerde ölçekleme
    # Sadece sayısal (0/1 dummy olmayan) kolonları ölçekliyoruz
    sayisal_kolonlar = [
        "yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi", "abonelik_yili"
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[sayisal_kolonlar] = scaler.fit_transform(X_train[sayisal_kolonlar])
    X_val_scaled[sayisal_kolonlar] = scaler.transform(X_val[sayisal_kolonlar])
    X_test_scaled[sayisal_kolonlar] = scaler.transform(X_test[sayisal_kolonlar])

    print("\n" + "=" * 70)
    print("6) ÖLÇEKLEME TAMAMLANDI (StandardScaler)")
    print("=" * 70)
    print(f"Ölçeklenen sayısal kolonlar: {sayisal_kolonlar}")

    # 9) En az 2 model eğit: Logistic Regression, KNN (+ bonus: Decision Tree)
    modeller = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (bonus)": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4),
    }

    print("\n" + "=" * 70)
    print("9-10) MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI")
    print("=" * 70)

    val_sonuclari = {}
    for isim, model in modeller.items():
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        val_sonuclari[isim] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"{isim:<25} | Val Accuracy: {acc:.3f} | Val F1: {f1:.3f}")

    # En iyi modeli validation F1 skoruna göre seç
    en_iyi_isim = max(val_sonuclari, key=lambda k: val_sonuclari[k]["f1"])
    en_iyi_model = val_sonuclari[en_iyi_isim]["model"]
    print(f"\n>> Validation performansına göre seçilen model: {en_iyi_isim}")

    # 11) Test seti üzerinde değerlendirme
    y_test_pred = en_iyi_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(f"11) TEST SETİ SONUÇLARI ({en_iyi_isim})")
    print("=" * 70)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # 12) Kısa yorum
    print("\n" + "=" * 70)
    print("12) KISA YORUM")
    print("=" * 70)
    diger_modeller = ", ".join(
        f"{isim} (F1={sonuc['f1']:.3f})"
        for isim, sonuc in val_sonuclari.items()
        if isim != en_iyi_isim
    )

    aciklamalar = {
        "Logistic Regression": (
            "öznitelikler ile churn arasındaki ilişkinin ağırlıklı olarak "
            "doğrusala yakın olması ve az sayıda öznitelikle aşırı öğrenmeden "
            "(overfitting) kaçınarak iyi genelleme yapabilmesi"
        ),
        "KNN": (
            "ölçeklenmiş öznitelik uzayında churn eden ve etmeyen müşterilerin "
            "birbirine yakın kümeler oluşturması; komşuluk tabanlı yaklaşımın "
            "bu tür yerel örüntüleri yakalamada başarılı olması"
        ),
        "Decision Tree (bonus)": (
            "verideki eşik değer tabanlı (ör. belirli bir destek talebi sayısının "
            "üzerinde churn riskinin artması gibi) örüntüleri doğrudan "
            "yakalayabilmesi"
        ),
    }

    yorum = (
        f"Validation sonuçlarına göre en iyi model {en_iyi_isim} oldu "
        f"(F1={val_sonuclari[en_iyi_isim]['f1']:.3f}). Diğer modeller: {diger_modeller}. "
        f"Bunun nedeni muhtemelen {aciklamalar[en_iyi_isim]} olabilir. "
        "Test seti sonuçları da (yukarıda) bu modelin validation performansıyla "
        "tutarlı olup olmadığını gösteriyor; farklı random_state veya veri "
        "büyüklüğüyle bu sonuç değişebilir."
    )
    print(yorum)


if __name__ == "__main__":
    main()
