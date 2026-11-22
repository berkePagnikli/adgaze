# Reklam Ekranı İzleyici Analitiği: İlk Çalışan Sürüm

Bu sürüm, hocanın önerdiği ilk kombinasyonla kuruldu:
**YOLO26 + ByteTrack + YuNet + 6DRepNet + L2CS-Net**

Kamera ekranın üzerine yerleştirilmiş kabul edilir. Bir kişinin "ekrana bakması", kabaca kameraya bakması demektir.

## 1. Akış

```
Video / kamera (OpenCV)
   │
   ▼
1-2. İnsan tespiti + takip ─── YOLO26s + ByteTrack ──► kişi kutusu + kalıcı ID
   │
   ▼
3.   Yüz tespiti ───────────── YuNet ──► yüz kutusu, kişiye atanır
   │
   ├──► 4A. Baş yönü ───────── 6DRepNet ──► head yaw / pitch / roll
   └──► 4B. Bakış yönü ─────── L2CS-Net ──► gaze yaw / pitch
   │
   ▼
5.   Ekrana bakma kararı ───── eşik ──► BAKIYOR / BAKMIYOR / BELİRSİZ
   │
   ▼
6.   Zamansal filtre ───────── çoğunluk oyu + tutma ──► kalma süresi, izleme süresi
   │
   ▼
7.   Analitik ──────────────── kişi sayısı, izleyici sayısı/oranı, süreler
```

Her adım `pipeline/` klasöründe ayrı ve kısa bir dosyadır:

| Adım | Dosya | Ne yapar |
|---|---|---|
| 1-2 | [pipeline/tracking.py](pipeline/tracking.py) | YOLO26s ile yalnızca "person" sınıfını bulur. Ultralytics'in hazır ByteTrack'i ile her kişiye kalıcı bir ID verir. |
| 3 | [pipeline/face.py](pipeline/face.py) | YuNet tüm karede yüzleri bulur. Merkezi bir kişi kutusunun üst yarısında kalan yüz o kişiye atanır; her kişiye en fazla bir yüz düşer. |
| 4A | [pipeline/head_pose.py](pipeline/head_pose.py) | Yüz kutusu resmi demodaki gibi %20 büyütülür; 6DRepNet baş açılarını verir. |
| 4B | [pipeline/gaze.py](pipeline/gaze.py) | Yüzün etrafından kare bir kırpım alınır; L2CS-Net (Gaze360 ağırlığı) bakış açılarını verir. |
| 5 | [pipeline/decision.py](pipeline/decision.py) | Kare bazında karar verir (ayrıntı aşağıda). |
| 6 | [pipeline/temporal.py](pipeline/temporal.py) | Kararı zamanda yumuşatır; her ID için kalma ve izleme süresini sayar. |
| 7 | [pipeline/analytics.py](pipeline/analytics.py) | Kişi tablosunu ve video özetini çıkarır. |
| – | [pipeline/pipeline.py](pipeline/pipeline.py) | Adımları kare kare sırayla çalıştırır, çıktıları ve görselleştirmeyi yazar. |
| – | [pipeline/config.py](pipeline/config.py) | Tüm parametreler (eşikler vb.) tek yerde. |
| – | [pipeline/weights.py](pipeline/weights.py) | Model ağırlıklarını ilk kullanımda `weights/` klasörüne indirir. |

### Adım 5: Karar kuralı

- Yüz bulunamadıysa sonuç **BELİRSİZ** olur.
- Aşağıdaki iki koşul birlikte sağlanıyorsa **BAKIYOR**, sağlanmıyorsa **BAKMIYOR** olur:
  - **Bakış kameraya yönelik:** `|gaze_yaw| ≤ 20°` ve `|gaze_pitch| ≤ 20°`.
  - **Baş kabaca ekrana dönük:** `|head_yaw| ≤ 35°` ve `|head_pitch| ≤ 30°`. Bu bir güvenlik kontrolüdür, çünkü L2CS yandan görünen yüzlerde güvenilmez.
- Kamera ekranın tam ortasında değilse `gaze_yaw_offset` ve `gaze_pitch_offset` ile hedef yön kaydırılabilir.

### Adım 6: Zamansal filtre ve süreler

- **Yumuşatma:** Bir kişinin son 5 kesin kararının çoğunluğu alınır. Bu, tek karelik titremeleri bastırır.
- **BELİRSİZ kareler:** 6 kareye kadar (24 fps'te 0,25 s) son durum korunur, sonra BAKMIYOR sayılır.
- **Kalma süresi (dwell):** Kişinin takip edildiği kare sayısı / fps.
- **İzleme süresi (watch):** Filtrelenmiş durumu BAKIYOR olan kare sayısı / fps.
- Kalma süresi ile izleme süresi ayrı hesaplanır; akış dokümanındaki öneri de buydu.

### Adım 7: Analitik

- 0,5 s'den kısa izlenen kişiler (yanlış tespit veya anlık geçiş) sayılmaz.
- 0,5 s'den uzun bakan kişi **izleyici** sayılır.
- Video özeti şunları içerir: kişi sayısı, izleyici sayısı, izleyici oranı, toplam/ortalama kalma süresi ve izleyici başına ortalama izleme süresi.

## 2. Klasör yapısı

```
scripts/
├── run_pipeline.py      # pipeline'ı videolarda veya webcam'de çalıştırır
├── evaluate.py          # sonuçları gt/ etiketleriyle karşılaştırır
├── annotate.py          # elle etiketleme aracı (ground truth üretir)
├── requirements.txt
├── pipeline/            # adımlar (yukarıdaki tablo)
├── data/                # videolar
├── gt/                  # etiketler (annotate.py çıktısı)
├── weights/             # model ağırlıkları (otomatik indirilir)
└── runs/                # çalıştırma çıktıları
```

## 3. Kurulum ve çalıştırma

Komutlar tez kök klasöründen de `scripts/` içinden de çalışır.

```powershell
# kurulum (.venv'de hepsi zaten kurulu)
pip install -r scripts/requirements.txt

# data/ altındaki tüm videolar
python scripts/run_pipeline.py

# tek video / webcam
python scripts/run_pipeline.py --source data/man.mp4
python scripts/run_pipeline.py --source 0 --show

# parametre değiştirerek ayrı klasöre çalıştırma
python scripts/run_pipeline.py --set gaze_yaw_max=25 --out runs/yaw25

# etiketlerle karşılaştırma
python scripts/evaluate.py
python scripts/evaluate.py --runs runs/yaw25
```

Her video için `runs/<video>/` altına şunlar yazılır:

| Dosya | İçerik |
|---|---|
| `frames.csv` | Her kare × kişi için bir satır: kişi ve yüz kutusu, baş açıları, bakış açıları, ham karar (`raw`: 1/0/-1) ve filtrelenmiş karar (`looking`: 1/0). |
| `persons.csv` | Her kişi (ID) için: ilk/son kare, kalma süresi, izleme süresi, izleme oranı, yüz görünme oranı, izleyici mi. |
| `summary.json` | Video özeti (Adım 7), hız ve kullanılan parametreler. |
| `annotated.mp4` | Görselleştirme. Yeşil: bakıyor, kırmızı: bakmıyor, beyaz kutu: yüz, sarı ok: L2CS bakış yönü. `?` işareti o karede yüz olmadığını gösterir. |

## 4. Değerlendirme (`evaluate.py`)

- **Eşleştirme:** Her karede etiketteki kişi kutuları, pipeline'ın kişi kutularıyla eşleştirilir (Hungarian, IoU ≥ 0,5). Bu sayede sonuç track ID numaralarından bağımsızdır.
- **Kapsam (coverage):** Etiketli kişi-karelerin ne kadarının takipçi tarafından bulunduğu.
- **Doğruluk / Precision / Recall / F1:** Kare bazında, pozitif sınıf BAKIYOR. Bulunamayan kişi-kare "bakmıyor" sayılır.
- **İzleme süresi hatası:** Her etiketli kişi için gerçek ve tahmin edilen izleme süresi farkının mutlak değeri, video bazında toplanır.
- Etiketi -1 (yok say) olan satırlar hesaba katılmaz.

## 5. İlk sonuçlar (varsayılan parametreler, 6 video, RTX 3050)

| Video | Kapsam | Doğruluk | Precision | Recall | F1 | Gerçek izleme (s) | Tahmin (s) | Süre hatası (s) | Hız (fps) |
|---|---|---|---|---|---|---|---|---|---|
| man | 1,00 | 0,925 | 0,987 | 0,885 | 0,933 | 3,63 | 3,25 | 0,38 | 11,7 |
| man_busstop_1 | 1,00 | 0,984 | 0,976 | 0,988 | 0,982 | 3,50 | 3,54 | 0,04 | 8,2 |
| man_busstop_2 | 1,00 | 0,966 | 1,000 | 0,955 | 0,977 | 5,58 | 5,33 | 0,25 | 8,1 |
| man_phone_1 | 1,00 | 0,990 | 0,939 | 1,000 | 0,969 | 1,29 | 1,38 | 0,08 | 4,9 |
| man_phone_2 | 1,00 | 0,979 | 0,953 | 1,000 | 0,976 | 3,42 | 3,58 | 0,17 | 7,2 |
| woman_busstop_1 | 1,00 | 0,620 | 1,000 | 0,503 | 0,670 | 6,13 | 3,08 | 3,04 | 9,1 |
| **Toplam** | **1,00** | **0,909** | **0,981** | **0,841** | **0,906** | **23,54** | **20,17** | **3,96** | |

Ham tablo: `runs/evaluation.csv`.

**Gözlemler**

- **Beş videoda F1 0,93–0,98.** İzleme süresi hatası video başına 0,4 s'nin altında.
- **Takip kapsamı tam (1,00).** YOLO26 + ByteTrack etiketli tüm kişi-kareleri buldu.
- **`woman_busstop_1` zayıf (recall 0,50). Sorun bakış modelinde.**
  - Kişi kameraya bakarken 6DRepNet baş yönünü doğru ölçüyor (yaw ≈ 0°).
  - L2CS-Net ise kapüşonlu yüzde, gece ışığında bakış yaw'ını −20…−30° tahmin ediyor; bazı karelerde 150°'ye sıçrıyor.
  - Bu durum, akışta önerilen bir sonraki deneyi destekliyor: yalnızca bakış adımını GaT (gaze3d) ile değiştirip aynı videolarda karşılaştırmak.
- **Eşikler elle seçilmiş başlangıç değerleridir.** Yukarıdaki sonuçlar varsayılan parametrelerle alındı. Bu değerler (bakış ±20°, baş ±35°/±30°, 5 karelik yumuşatma, 6 karelik tutma, 0,5 s) veriden, kalibrasyondan veya literatürden türetilmedi; makul görünen yuvarlak sayılar olarak seçildi ve sonuçlar görüldükten sonra değiştirilmedi. Eşiği test videolarına göre ayarlamak sonucu iyimser gösterirdi.
  - Kaba geometri kontrolü: 55" bir ekran (~1,2 m) 2–3 m'den ±11–17° açı kaplar. L2CS'in Gaze360 hatası ~10° eklenince ±20–27° beklenir; 20° bu aralıkta.
  - Tez için önerilen: eşiği değerlendirmede kullanılmayan veriyle seçmek, örneğin videoları dönüşümlü kullanarak (5 videoda seç, kalan 1 videoda ölç) veya ayrı bir kalibrasyon kaydıyla.
- **Hız:** 5–12 fps. Kalabalık sahnelerde (man_phone) her yüz için iki model çalıştığından düşüyor.

## 6. Notlar

- **Track ID'ler:** `gt/<video>.json` içindeki değişim noktaları, etiketlemenin yapıldığı eski çalıştırmanın track ID'lerine bağlı. Yeni bir çalıştırma ID'leri farklı numaralandırabilir. Bu yüzden mevcut bir etiketi `annotate.py` ile yeniden açarken dikkatli olun. `gt/<video>.csv` kutu içerdiği için değerlendirme bundan etkilenmez.
- **Yüzü görünmeyen kişiler:** Uzaktaki veya arkası dönük kişiler de kişi olarak sayılır, ancak yüzleri olmadığı için izleyici sayılmaz. İzleyici oranı bu yüzden "önünden geçenlerin ne kadarı baktı" anlamına gelir.
- **Ağırlıklar ve lisanslar:**
  - YOLO26s: Ultralytics, AGPL-3.0.
  - YuNet: OpenCV Zoo.
  - 6DRepNet: 300W-LP, resmi bağlantı.
  - L2CS-Net Gaze360: resmi ağırlığın py-feat tarafından safetensors olarak değiştirilmeden yeniden yayımlanmış hali.
