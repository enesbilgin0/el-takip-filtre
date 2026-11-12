"""
El Takibi + Parmak Sayma + Canli Kamera Filtreleri
---------------------------------------------------
Webcam'den canli goruntu alir, MediaPipe ile eli tespit eder,
acik parmak sayisini hesaplar ve bu sayiya gore gercek zamanli
OpenCV filtreleri uygular.

Calistirmak icin:  python main.py
Cikis:             klavyeden [Q]
Foto cekme:        [S] tusu (veya 5 parmak 2 saniye sabit tutulunca)
"""

import time
import os
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp


# =============================================================
# 1) SABITLER VE AYARLAR
# =============================================================

# MediaPipe'in el uzerinde isaretledigi 21 eklemden bizim icin onemli olanlar.
# (Her parmagin ucu ve orta eklemi; bilek ise referans noktamiz.)
WRIST = 0            # Bilek (tum olcumlerde referans / merkez noktamiz)
THUMB_TIP = 4        # Basparmak ucu
THUMB_MCP = 2        # Basparmak kok eklemi (yedek referans)
INDEX_MCP = 5        # Isaret parmagi kokü (basparmak karsilastirmasi icin)
PINKY_MCP = 17       # Serce parmagi kokü (basparmak karsilastirmasi icin)

# Dort parmak icin (basparmak haric) [uc, pip] eslesmeleri:
# Uc noktasi bilege, PIP noktasindan daha uzaksa parmak "acik" demektir.
FINGER_TIPS_PIPS = [
    (8, 6),    # Isaret
    (12, 10),  # Orta
    (16, 14),  # Yuzuk
    (20, 18),  # Serce
]

FINGER_NAMES = ["Bas", "Isaret", "Orta", "Yuzuk", "Serce"]

# Her parmak sayisinin actigi mod ismi (HUD'da gosterilir).
MODE_NAMES = {
    0: "0 - Normal Kamera",
    1: "1 - Sinematik Siyah-Beyaz",
    2: "2 - Matrix Neon Kenar",
    3: "3 - Termal Kamera",
    4: "4 - Retro Karikatur",
    5: "5 - Studyo Portre Odak",
}

# Titreme onleme: son 7 karenin en cok tekrar eden degerini kullaniriz.
SMOOTHING_WINDOW = 7

# 5 parmak bu kadar saniye sabit kalirsa otomatik foto cekilir.
AUTO_CAPTURE_SECONDS = 2.0

CAPTURE_DIR = "captures"


# =============================================================
# 2) PARMAK SAYMA MANTIGI
# =============================================================

def _dist(a, b):
    """Iki landmark arasindaki Oklid mesafesi. Olceklemeden bagimsiz
    karsilastirma yaptigimiz icin bu basit mesafe yeterlidir."""
    return np.hypot(a[0] - b[0], a[1] - b[1])


def parmaklari_say(landmarks):
    """
    21 eklem noktasindan hangi parmaklarin acik oldugunu tespit eder.

    Donus: (acik_sayisi, [bas, isaret, orta, yuzuk, serce])  -> bool listesi

    MANTIK:
    - Dort parmak (isaret/orta/yuzuk/serce): Parmak ucunun bilege mesafesi,
      alt eklemin (PIP) bilege mesafesinden buyukse parmak uzanmis (acik)
      demektir. Mesafe kullanmak, elin donuk/egik durusunda bile kararlidir.
    - Basparmak: Yana acildigi icin dikey kural ise yaramaz. Bu yuzden
      basparmak ucunun (4) serce koküne (17) mesafesini, isaret koküne (5)
      serce koküne mesafesiyle kiyaslariz. Basparmak acilinca uc, avucun
      disina dogru uzaklasir ve bu mesafe belirgin sekilde buyur.
    """
    bilek = landmarks[WRIST]
    durum = []

    # --- Basparmak ---
    bas_uc_mesafe = _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    el_genislik_ref = _dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    # %30 esik: gurultuyu eleyip net acilmayi kabul eder.
    bas_acik = bas_uc_mesafe > el_genislik_ref * 1.3
    durum.append(bool(bas_acik))

    # --- Diger dort parmak ---
    for tip, pip in FINGER_TIPS_PIPS:
        uc_mesafe = _dist(landmarks[tip], bilek)
        pip_mesafe = _dist(landmarks[pip], bilek)
        durum.append(bool(uc_mesafe > pip_mesafe))

    return sum(durum), durum


# =============================================================
# 3) KAMERA FILTRELERI
# =============================================================

def filtre_normal(frame):
    """0 parmak: dokunulmamis orijinal goruntu."""
    return frame


def filtre_siyah_beyaz(frame):
    """1 parmak: sinematik grayscale + hafif kontrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Kontrasti hafif artir (alpha=1.2), parlakligi dusur (beta=-10).
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-10)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filtre_matrix_neon(frame):
    """2 parmak: Canny ile kenarlari bul, siyah zemine neon yesili bas."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kenar = cv2.Canny(gray, 80, 160)
    cikti = np.zeros_like(frame)              # Tamamen siyah tuval
    cikti[kenar > 0] = (0, 255, 70)           # Kenar pikselleri -> neon yesil
    return cikti


def filtre_termal(frame):
    """3 parmak: parlakligi isiya cevirir (termal kamera taklidi)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def filtre_karikatur(frame):
    """4 parmak: renkleri sadelestir + kenarlari koyultarak cizgi film etkisi."""
    # Renk bolgelerini yumusatip sadelestir (kenarlari koruyan filtre).
    renk = cv2.bilateralFilter(frame, d=9, sigmaColor=250, sigmaSpace=250)
    # Siyah kenar maskesi cikar.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kenar = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    kenar = cv2.cvtColor(kenar, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(renk, kenar)


def filtre_studyo(frame):
    """5 parmak: sicak tonlu studyo bakisi + hafif arka plan yumusatma."""
    # Hafif blur ile "portre" hissi.
    yumusak = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    frame = cv2.addWeighted(frame, 0.7, yumusak, 0.3, 0)
    # Sepia/sicak ton: BGR kanallarini matrisle karistir.
    sepia = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    sicak = cv2.transform(frame, sepia)
    return np.clip(sicak, 0, 255).astype(np.uint8)


# Parmak sayisi -> filtre fonksiyonu esleme tablosu.
FILTRELER = {
    0: filtre_normal,
    1: filtre_siyah_beyaz,
    2: filtre_matrix_neon,
    3: filtre_termal,
    4: filtre_karikatur,
    5: filtre_studyo,
}


def filtre_uygula(frame, parmak_sayisi):
    """Parmak sayisina uygun filtreyi secip uygular."""
    return FILTRELER.get(parmak_sayisi, filtre_normal)(frame)


# =============================================================
# 4) ARAYUZ (HUD / OVERLAY)
# =============================================================

def yari_saydam_panel(frame, x, y, w, h, alpha=0.55):
    """Belirtilen dikdortgene koyu yari saydam bir kart cizer."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def hud_ciz(frame, parmak_sayisi, parmak_durum, fps, bildirim=None):
    """
    Sol ust kose: bilgi karti (parmak sayisi, mod, parmak gostergeleri).
    Sag ust kose: FPS ve cikis bilgisi.
    Filtrenin UZERINE cizildigi icin her modda net okunur.
    """
    h, w = frame.shape[:2]

    # --- Sol ust bilgi karti ---
    yari_saydam_panel(frame, 10, 10, 330, 150)

    cv2.putText(frame, f"Acik Parmak: {parmak_sayisi}", (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2, cv2.LINE_AA)

    mod_adi = MODE_NAMES.get(parmak_sayisi, "Normal")
    cv2.putText(frame, f"Mod: {mod_adi}", (24, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # --- 5'li parmak gostergesi (acik=yesil, kapali=koyu gri) ---
    kutu_w, kutu_h, bosluk = 58, 34, 6
    baslangic_x = 24
    gosterge_y = 105
    for i, (ad, acik) in enumerate(zip(FINGER_NAMES, parmak_durum)):
        kx = baslangic_x + i * (kutu_w + bosluk)
        renk = (0, 180, 0) if acik else (70, 70, 70)
        cv2.rectangle(frame, (kx, gosterge_y),
                      (kx + kutu_w, gosterge_y + kutu_h), renk, -1)
        cv2.putText(frame, ad, (kx + 4, gosterge_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Sag ust: FPS + cikis bilgisi ---
    yari_saydam_panel(frame, w - 190, 10, 180, 60)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 175, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "[Q] Cikis  [S] Foto", (w - 175, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Studyo modu vurgusu ---
    if parmak_sayisi == 5:
        cv2.putText(frame, "5 Parmak: Studyo Modu Aktif", (int(w / 2) - 230, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

    # --- Gecici bildirim ( or. "Fotograf Cekildi!") ---
    if bildirim:
        cv2.putText(frame, bildirim, (int(w / 2) - 160, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)


# =============================================================
# 5) YARDIMCI: FOTO KAYDETME
# =============================================================

def foto_kaydet(frame):
    """Islenmis kareyi captures/ klasorune zaman damgali kaydeder."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    dosya = os.path.join(CAPTURE_DIR, f"capture_{int(time.time())}.png")
    cv2.imwrite(dosya, frame)
    return dosya


# =============================================================
# 6) ANA DONGU
# =============================================================

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera acilamadi. Baska bir uygulama kullaniyor olabilir.")
        return

    # Titreme onleme icin son karelerin parmak sayilarini tutan kuyruk.
    gecmis = deque(maxlen=SMOOTHING_WINDOW)

    # FPS hesabi ve zamanlayicilar.
    onceki_zaman = time.time()
    bes_parmak_baslangic = None   # 5 parmagin ne zamandan beri sabit oldugu
    bildirim_metni = None
    bildirim_bitis = 0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Ayna goruntusu: dogal el hareketi hissi icin yatay cevir.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe RGB bekler; OpenCV ise BGR verir.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sonuc = hands.process(rgb)

            parmak_sayisi = 0
            parmak_durum = [False] * 5
            landmark_px = None

            if sonuc.multi_hand_landmarks:
                el = sonuc.multi_hand_landmarks[0]
                # Normalize (0-1) koordinatlari piksele cevir.
                landmark_px = [(lm.x * w, lm.y * h) for lm in el.landmark]
                ham_sayi, parmak_durum = parmaklari_say(landmark_px)
                gecmis.append(ham_sayi)
            else:
                gecmis.append(0)

            # Yumusatma: son karelerin en sik gorulen degeri (mod).
            if gecmis:
                parmak_sayisi = Counter(gecmis).most_common(1)[0][0]

            # 1) Once filtreyi uygula (goruntunun tamamini degistirir).
            cikti = filtre_uygula(frame, parmak_sayisi)

            # 2) Landmark cizimlerini filtrenin UZERINE yap ki her modda gorunur.
            if sonuc.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    cikti,
                    sonuc.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2),
                )

            # 3) 5 parmak 2 saniye sabit kalirsa otomatik foto cek.
            simdi = time.time()
            if parmak_sayisi == 5:
                if bes_parmak_baslangic is None:
                    bes_parmak_baslangic = simdi
                elif simdi - bes_parmak_baslangic >= AUTO_CAPTURE_SECONDS:
                    dosya = foto_kaydet(cikti)
                    print(f"Otomatik foto kaydedildi: {dosya}")
                    bildirim_metni = "Fotograf Cekildi!"
                    bildirim_bitis = simdi + 1.5
                    bes_parmak_baslangic = simdi + 3  # kisa bekleme (spam engel)
            else:
                bes_parmak_baslangic = None

            # FPS hesapla (iki kare arasi gecen sure).
            fps = 1.0 / max(simdi - onceki_zaman, 1e-6)
            onceki_zaman = simdi

            # Bildirim suresi dolduysa temizle.
            aktif_bildirim = bildirim_metni if simdi < bildirim_bitis else None

            # 4) En uste HUD'u ciz.
            hud_ciz(cikti, parmak_sayisi, parmak_durum, fps, aktif_bildirim)

            cv2.imshow("El Takibi - Parmak Sayma", cikti)

            # Klavye kontrolu.
            tus = cv2.waitKey(1) & 0xFF
            if tus == ord('q'):
                break
            elif tus == ord('s'):
                dosya = foto_kaydet(cikti)
                print(f"Foto kaydedildi: {dosya}")
                bildirim_metni = "Fotograf Cekildi!"
                bildirim_bitis = simdi + 1.5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
