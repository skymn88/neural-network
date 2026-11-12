# Horizontal tail actuator · MATLAB + Python

Uçuş verisinden **Current → DDV → RAM → Deflection** dinamiğini öğrenen black-box sistem tanımlama projesi. Bu dal: **`dev_CodexPy`**. Gerçek uçuş verileri repoda yoktur. Sentetik sonuçlar gerçek uçuş doğrulaması değildir.

| Klasör | İçerik |
|---|---|
| `matlab/` | Önceki R2019a uygulaması; fonksiyon adları ve model varsayılanları korunur |
| `python/` | Bağımsız eğitim, NumPy çıkarım, yerel HTML arayüzü, testler ve kayıtlı deneyler |
| `data/synthetic/` | Değişmeyen eski MAT + ayrı darbe/bekleme ham verisi |
| `data/user/` | Yerel verilerin; Git dışında |
| `docs/` | [Modelleme](docs/modeling.md), [gerçek deney sonuçları](docs/experiments.md), [tüm commit geçmişi](docs/changelog.md) |

## Telefonda sentetik sonuçları incele

**[Sentetik sonuç sayfasını aç](https://actuator-synthetic-lab.drawnparrot1.chatgpt.site)** — kurulum, bilgisayar veya `.bat` gerekmez. Aynı ChatGPT hesabınla giriş istenebilir. İki veri setinin 22 uçuşu, dört modelin karşılaştırması, DDV/RAM/açı grafikleri, zaman aralığı seçimi ve XLSX indirme bulunur. Bu sayfa yeni eğitim başlatmaz veya kullanıcı verisi yüklemez.

Sayfa kaynağı `python/viewer/` altındadır. Yeniden üretim: `.venv/bin/python python/viewer/export_results.py /tmp/actuator-viewer/dist`. Yalnız kayıtlı sentetik reference sonuçları dışa aktarılır. Eğitim uygulaması aşağıda ayrı olarak korunur.

## Python: arayüzü aç

**Windows:** Python **3.12** kurulu olsun. ZIP'i çıkar, **`python/start.bat`** dosyasına çift tıkla. İlk açılış CPU bağımlılıklarını indirir; sonraki açılışlar mevcut ortamı kullanır. Tarayıcıda `http://127.0.0.1:8765` açılır. Konsol açık kalmalı; durdurmak için Ctrl+C. Başlatıcı Linux'ta çalıştırılmadı; Python sunucusu ve tarayıcı akışı Linux'ta test edildi.

Linux/macOS, repo kökünde bir kez:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r python/requirements.txt
.venv/bin/python python/app.py
```

Arayüzde:

1. **Kaydedilmiş koşu** seçerek hazır sentetik sonuçları ve uçuş grafiklerini eğitim yapmadan incele.
2. Yeni eğitim için veri kaynağı/yöntem/ayarları seç, **Deneyi başlat**. Varsayılan karşılaştırma OE + dinamik sinir ağıdır. Bunlar gerçek veri için doğrulanmış optimum ayarlar değildir.
3. Kendi verin için **Kendi CSV dosyalarım** seç; her uçuş için bir CSV yükle. **Sütunları göster** ile dört sinyali ve her birinin zaman sütununu eşleştir; **Verileri hazırla** → **Deneyi başlat**.
4. Sonuçtan uçuş/model seç: DDV, RAM ve açı aynı üçlü grafikte. XLSX ve model JSON indirilebilir. Tekrar eğitim gerekmez.

Zaman sütunları **sayısal saniye**, aynı uçuş içinde ortak referansta olmalı. Farklı uzunluktaki sinyal sütunları boş/NaN ile doldurulabilir. 64 Hz current/DDV/RAM/açı kanallarını seç; desired 1 Hz ve RAM 1000 Hz kanallarını bu çalışma için eşleme. Birimler ve işaretler uçuşlar arasında tutarlı olmalı. Python şu an CSV'de virgül/`;`/tab ayırıcı ve noktalı ondalık kabul eder; MATLAB `datetime` otomatik dönüştürülmez.

Aynı split bütün adaylarda korunur. 10 uçuş 7/2/1, **11 uçuş 7/2/2**, 20 uçuş 14/4/2. 1–3 uçuşta kronolojik bölme ve koruma aralıkları vardır; bu bağımsız uçuş validasyonu değildir. Eski MATLAB sonuçlarıyla aynı uçuşları karşılaştırmak için CSV sırasına göre `train,train,...,validation,test` atamalarını açıkça gir; Python ve MATLAB RNG atamaları aynı olmak zorunda değildir.

Seçim yalnız **tam uçuş validation RAM Cascade RMSE** ile yapılır. **Test skorlarını hesapla** varsayılan kapalıdır; geliştirmeyi bitirdikten sonra aç. Test sonucuna bakıp ayar aramaya devam edersen artık bağımsız test değildir. Kullanıcı CSV verileri yerel süreçte/`python/results/` altında kalır; dış servise yüklenmez ve Git'e eklenmez.

## Mevcut sonuç ve sınır

İki sentetik sette validation kazananı **NN_joint**, yani DDV yardımcı hedefiyle ortak eğitilen dinamik ağ oldu. RAM Cascade Fit Train/Val/Test:

| Veri | Train | Validation | Test |
|---|---:|---:|---:|
| Eski sentetik MAT | 77.05 | 87.62 | 77.53 |
| Yeni darbe–bekleme | 85.09 | 79.23 | 87.76 |

%95 hedefi elde edilmedi. Daha uzun eğitim ve monoton ağ her zaman iyileştirmedi; ayrıntılar ve başarısız denemeler [deney kaydında](docs/experiments.md). Gerçek DDV eğitimde hata hedefidir; serbest koşuda model **akım + başlangıç RAM** ile çalışır. Mevcut Python çalışma alanı dört-sinyalli **plant** zinciridir; eski deneysel setpoint/force/pressure/closed dalları MATLAB'da korunmuştur, Python'a birebir taşındıkları iddia edilmez.

Eğitim PyTorch, çıkarım yalnız NumPy kullanır. `models.json` sabit boyutlu ağırlık/durum tanımını içerir. **C/MEX/S-Function henüz üretilmedi.** MATLAB–Python seçimi tek başına C çıktısını garanti etmez; model denklemleri ve hedef arayüz belirleyicidir.

## MATLAB'ı kullanmaya devam et

MATLAB Current Folder: **`matlab/`**. Mevcut yerel adaptörünü de bu klasöre taşı. `data/user/` repo kökünde ortak kalır; örnek adaptörün yolu buna göre güncellendi.

```matlab
start_here
run_project('user')
```

Daha önceki komutlar/ayarlar [MATLAB README](matlab/readme.md) içinde. Bu değişiklikte MATLAB modeli/eğitimi değiştirilmedi; yalnız klasör ve ortak veri yolları uyarlandı. MATLAB çalıştırılmadı; R2019a statik kontrolü yapıldı.

## Geliştirici kontrolü

```bash
PYTHONPATH=python .venv/bin/python -m unittest discover -s python/tests -v
.venv/bin/python python/run_experiment.py --source bundled --steps 400 --horizon 192
```

Windows test karşılığı: önce `set PYTHONPATH=python`, ardından `.venv\Scripts\python -m unittest discover -s python\tests -v`. API testleri için `python/requirements-dev.txt` gerekir. Tarayıcı kontrolü: `python/tests/browser_smoke.py`; ayrıntı [modelleme belgesinde](docs/modeling.md).
