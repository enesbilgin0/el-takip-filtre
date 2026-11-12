<div align="center">

# ✋ El Takip & Filtre Stüdyosu

**Webcam'den canlı el takibi · Parmak sayma · Parmağınla değişen gerçek zamanlı kamera efektleri**

El aç, parmağını say, kamera modu anında değişsin. Tek dosya, modüler kod, sunuma hazır.

<br>

![Python](https://img.shields.io/badge/Python-3.8--3.12-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logo=google&logoColor=white)
![License](https://img.shields.io/badge/Lisans-MIT-green?style=flat)

</div>

---

## 🎬 Nedir Bu?

Bilgisayarının kamerasına elini gösterirsin; uygulama **MediaPipe** ile elini
yakalar, kaç parmağının açık olduğunu hesaplar ve bu sayıyı bir **kumanda** gibi
kullanarak kameraya anlık efektler uygular. Yumruk yaparsan normal kamera, beş
parmağını açarsan sıcak tonlu stüdyo modu... Arada dört farklı efekt daha var.

Amaç; bilgisayarlı görü (computer vision) ve el tespiti kavramlarını **basit,
çalışan ve eğlenceli** bir örnekle göstermek.

---

## ✨ Özellikler

- 🖐️ **Gerçek zamanlı el takibi** — 21 eklem noktası ve bağlantı çizgileriyle
- 🔢 **Kararlı parmak sayma** — mesafe tabanlı mantık, titremeyi önleyen yumuşatma
- 🎨 **Parmakla değişen 6 kamera modu** — 0'dan 5'e kadar her sayı farklı bir efekt
- 📊 **Şık HUD arayüzü** — parmak sayısı, aktif mod, parmak göstergesi ve canlı FPS
- 📸 **Fotoğraf çekme** — `S` tuşu veya beş parmağı 2 saniye sabit tutarak
- 📄 **Tek dosya, temiz kod** — her adımı Türkçe yorumlarla açıklanmış

---

## 🎨 Parmak Sayısı → Kamera Modu

| Parmak | Mod | Efekt |
|:------:|:----|:------|
| ✊ **0** | Normal Kamera | Orijinal görüntü (yumruk veya el yok) |
| ☝️ **1** | Sinematik Siyah-Beyaz | Grayscale + hafif kontrast |
| ✌️ **2** | Matrix Neon Kenar | `cv2.Canny` kenarları, siyah zeminde neon yeşili çizgiler |
| 🤟 **3** | Termal Kamera | `cv2.applyColorMap` + `COLORMAP_INFERNO` |
| ✋ **4** | Retro Karikatür | Renk sadeleştirme + koyu kenarlar (cartoon efekti) |
| 🖐️ **5** | Stüdyo Portre Odak | Sıcak/sepia ton + hafif bulanıklık, "Stüdyo Modu Aktif" |

> 💡 Landmark çizimleri ve bilgi panelleri **filtrenin üzerine** çizilir; böylece
> hangi modda olursan ol arayüz her zaman net okunur.

---

## 🧠 Parmak Sayma Algoritması Nasıl Çalışır?

MediaPipe her eli **21 eklem noktası** (landmark) ile işaretler. Bilek (0)
noktasını referans alıp her parmağın "açık" mı "kapalı" mı olduğunu ölçeriz.

### Dört parmak (İşaret, Orta, Yüzük, Serçe)

Her parmağın iki noktasını kullanırız:

| Parmak | Uç (tip) | Orta eklem (PIP) |
|:-------|:--------:|:----------------:|
| İşaret | 8 | 6 |
| Orta | 12 | 10 |
| Yüzük | 16 | 14 |
| Serçe | 20 | 18 |

**Kural:** Parmak ucunun bileğe (0) mesafesi, orta eklemin (PIP) bileğe
mesafesinden **büyükse** parmak uzanmış (açık) kabul edilir.

> **Neden mesafe?** El eğildiğinde ya da döndüğünde bile, uç nokta uzanmışsa
> bilekten uzaklaşır. Mesafe karşılaştırması bu yüzden basit "yukarıda mı?"
> kuralından çok daha kararlıdır.

### Başparmak (farklı davranır) 👍

Başparmak **yana** açıldığı için yukarı/aşağı kuralı işe yaramaz. Bunun yerine:

- Başparmak ucunun (4) **serçe köküne (17)** olan mesafesini,
- İşaret köküne (5) serçe köküne olan mesafesiyle (elin genişlik referansı)
  karşılaştırırız.

Başparmak ucu, bu referansın **%30'undan fazla** uzaklaşmışsa "açık" sayılır.
Böylece avuç içine kapanmış başparmağı yanlışlıkla açık saymayız.

### Titreme Önleme (Smoothing) 🎯

Kamera gürültüsüyle sayının anlık zıpladığını önlemek için son **7 karenin** en
sık görülen değerini (mod) alırız. `collections.deque` + `Counter` ile yapılan
bu basit filtre, sonucu stabil tutar.

---

## 🖥️ Arayüz (HUD)

```
┌─────────────────────────────┐              ┌──────────────┐
│  Açık Parmak: 2             │              │  FPS: 29.4   │
│  Mod: 2 - Matrix Neon Kenar │              │  [Q] Çıkış   │
│  [Bas][İsa][Ort][Yüz][Ser]  │              └──────────────┘
└─────────────────────────────┘
```

- **Sol üst:** Yarı saydam koyu kart içinde
  - Büyük puntoyla `Açık Parmak: X`
  - Aktif mod adı
  - 5'li parmak göstergesi — açık = 🟢 yeşil, kapalı = ⬛ koyu gri
- **Sağ üst:** Anlık `FPS` ve `[Q] Çıkış  [S] Foto` bilgisi
- **5 parmakta:** Ekranda "Stüdyo Modu Aktif" vurgusu

---

## 📸 Fotoğraf Çekme

İşlenmiş kare `captures/` klasörüne zaman damgalı olarak kaydedilir ve ekranda
kısa süreli **"Fotoğraf Çekildi!"** bildirimi çıkar. İki yolu var:

- ⌨️ Klavyeden **`S`** tuşuna basmak
- 🖐️ **Beş parmağı 2 saniye** sabit tutmak (otomatik deklanşör)

---

## 🚀 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/enesbilgin0/el-takip-filtre.git
cd el-takip-filtre

# (Önerilir) Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Çalıştır 🎉
python main.py
```

**Kontroller:** `Q` → çıkış · `S` → fotoğraf çek

> ⚠️ **Not:** `mediapipe` Python **3.8 – 3.12** sürümleriyle en uyumlu çalışır.
> Kameran başka bir uygulama tarafından kullanılıyorsa açılmayabilir.

---

## 📁 Dosya Düzeni

```
el-takip-filtre/
├── main.py            # Tüm uygulama — modüler fonksiyonlar, Türkçe yorumlar
├── requirements.txt   # opencv-python, mediapipe, numpy
├── .gitignore
└── README.md
```

Kod altı mantıklı bölüme ayrılmıştır: `Sabitler → Parmak Sayma → Filtreler →
HUD → Fotoğraf → Ana Döngü`.

---

<div align="center">

**🛠️ OpenCV · MediaPipe · NumPy ile yapıldı**

</div>
