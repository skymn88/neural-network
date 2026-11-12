# Horizontal tail actuator · Python 3.9

Current → DDV → RAM → Deflection zincirini uçuş verisinden öğrenen black-box sistem tanımlama projesi. Dal: **`dev_CodexPy`**. Bu dalda MATLAB kaynakları kaldırıldı; geçmiş MATLAB uygulaması [dev_Codex](https://github.com/skymn88/neural-network/tree/dev_Codex) dalında duruyor. Ortak `.mat` dosyası bir sentetik veri dosyasıdır, MATLAB çalıştırmayı gerektirmez.

**Gerçek uçuş verisi bu ortamda yok.** Sentetik başarı gerçek uçuş doğrulaması değildir.

## Önce kütüphaneleri çevrimdışı kur

Windows **64-bit Python 3.9** gerekir. `start.bat` artık paket indirmez; mevcut ortamı kontrol edip uygulamayı açar. Önceki “Downloading CPU dependencies” mesajı PyTorch CPU ve diğer paketleri internetten indirme girişimiydi; bu davranış kaldırıldı.

Elindeki yerel `.whl` paket klasörüyle, repo kökünde bir kez:

```bat
python\setup_offline.bat "D:\kutuphaneler\wheelhouse"
```

Ardından her kullanımda:

```bat
python\start.bat
```

Kurulum `.venv39` ortamı oluşturur ve **`--no-index --find-links`** kullanır. İnternet kaynağına başvurmaz. Eksik veya yanlış platformlu paket varsa açık hata verir. `py` komutu yoksa Python 3.9'un `python.exe` dosyasıyla önce `-m venv .venv39` çalıştırabilirsin.

Kütüphaneler mevcut başka bir Python 3.9 ortamında zaten kuruluysa yeniden kurmak gerekmez:

```bat
set "ACTUATOR_PYTHON=C:\kendi_python39_ortamim\python.exe"
python\start.bat
```

Gerekli paketler aşağıdadır. **Alt bağımlılıkları da** wheel klasöründe bulunmalı; yalnız bu sekiz dosyayı kopyalamak yeterli olmayabilir.

| Paket | Önerilen / bu ortamda test edilen sürüm | İşlev |
|---|---|---|
| NumPy | 1.26.4 | Sayısal işlemler ve çıkarım |
| SciPy | 1.13.1 | Filtreler, hizalama, MAT okuma |
| pandas | 2.2.3 | CSV ve rapor |
| openpyxl | 3.1.5 | XLSX |
| PyTorch CPU | 2.6.0+cpu | Sinir ağı eğitimi |
| FastAPI / Pydantic | 0.115.12 / 2.10.6 | Yerel arayüz API'si |
| Uvicorn | 0.34.3 | Yerel sunucu |

Sabit sürümler: `python/requirements-cpu.txt`. Wheel dosyaları **CPython 3.9 / Windows x64** ile uyumlu olmalı; Linux paketleri Windows'ta kullanılamaz. Paketleri bir kez hazırlamak gerekirse **internete bağlı başka bir Windows x64 + Python 3.9 bilgisayarda**:

```bat
python\download_wheels.bat "D:\wheelhouse"
```

Bu klasörün tamamını çevrimdışı bilgisayara taşı. Eğitim bilgisayarında `download_wheels.bat` çalıştırma. Ayrıntılı eksik-paket kontrolü: `.venv39\Scripts\python python\check_environment.py`. Windows `.bat` dosyaları bu Linux ortamında çalıştırılmadı; Python 3.9.25 ve paketler gerçekten test edildi.

## Arayüz ve kendi uçuşların

`start.bat`, `http://127.0.0.1:8765` adresini açar. Konsol açık kalmalı; durdurmak için Ctrl+C. Arayüz yereldir, CSV verileri dış servise gönderilmez.

1. **Kaydedilmiş koşu** seç: sonuçları ve DDV/RAM/açı grafiklerini yeniden eğitim yapmadan incele.
2. Kendi verin için **Kendi CSV uçuşlarım** seç; her uçuş için bir CSV yükle.
3. **Sütunları göster** ile dört sinyali ve zaman sütunlarını eşleştir. Sayısal saniye, ortak zaman referansı ve tutarlı birimler kullan.
4. **Verileri hazırla → Deneyi başlat**. Yeni aday için yöntem listesindeki **Yeni: sakinlik + osilasyon null izleyici** seçeneğini kullan. Eski varsayılan karşılaştırma korunur.

64 Hz current/DDV/RAM/açı kanallarını kullan; desired 1 Hz ve RAM 1000 Hz kanallarını bu çalışma için eşleme. CSV virgül/`;`/tab ayırıcılı, noktalı ondalıklı olmalı. Farklı uzunluktaki sütunlar NaN ile doldurulabilir.

10 uçuş 7/2/1, 11 uçuş **7/2/2**, 20 uçuş 14/4/2. 1–3 uçuşta kronolojik bölme ve guard aralıkları uygulanır; bu bağımsız uçuş validasyonu değildir. MATLAB ile aynı uçuşları kullanmak için açık `train,validation,test` atamalarını gir; iki dilin rastgele sıralaması aynı olmak zorunda değildir.

Seçim yalnız tam uçuş **validation RAM Cascade RMSE** ile yapılır. Testi geliştirme bittikten sonra aç; test sonuçlarıyla ayar ararsan bağımsız test olmaktan çıkar. `syntheticTrainFlights=0` varsayılan; >0 yalnız kendi verin için aynı ayarla TRAIN-only sentetik ekleme A/B'si yapar. Gerçek validation/test uçuşları korunur. Bu gerçek-veri A/B'si burada çalıştırılmadı.

## Son sonuç: üç yeni aday

Aynı flightLike verisi ve split'inde validation RAM Cascade Fit **%68,13 → %93,47**. Seçilen: **Rate_envelope** (öğrenilmiş hız haritası + nedensel null izleyici; NN değildir).

| RAM Cascade | Train | Validation | Tarihsel test |
|---|---:|---:|---:|
| Önceki Rate_slow_null | 71.18 | 68.13 | 83.34 |
| Yeni Rate_envelope | 93.50 | 93.47 | 95.18 |

Seçim dondurulduktan sonra üretilen iki ayrı sentetik uçuşta Fit **%93,67**. Tarihsel test daha önce görüldüğü için yeni kör test olarak sunulmaz. Validation %95'e ulaşmadı; üç denemede duruldu. Ham ölçülen DDV ile **RAM_Component Val Fit %48,40**: observer gürültüye hassas. Gerçek veri için doğrulanmış çözüm iddiası yok.

Yeni kayıt: **`20261112_105714_655119`**. [Tüm deneyler](docs/experiments.md) · [Changelog](docs/changelog.md) · [Modelleme](docs/modeling.md).

## Düzenlenebilir sunum

**[PowerPoint sunumunu indir](presentations/actuator_black_box_overview.pptx)** — 16 slayt; metinler, şekiller, tablolar ve grafikler düzenlenebilir. Gerçek uçuş bildirimleri ve çalıştırılmış sentetik sonuçlar ayrıdır. Grafiklerin gömülü Excel verileri PowerPoint'ten değiştirilebilir.

Sunumu açmak için Python gerekmez. Yeniden üretmek istersen yalnız o zaman `python/requirements-presentations.txt` paketlerini yerel kaynaktan kur ve `.venv39\Scripts\python python\build_presentation.py` çalıştır.

## Repo / kontroller

`python/`: eğitim, NumPy-only çıkarım ve HTML arayüzü. `python/reference_results/`: kayıtlı modeller/raporlar. `data/synthetic/`: sabit yardımcı veri. `data/user/`, `.venv39`, `wheelhouse`, yerel sonuçlar Git dışındadır. Büyük yeni ham veriler commit edilmez; üreteç aynı seed ile yeniden oluşturabilir.

```bat
set PYTHONPATH=python
.venv39\Scripts\python -m unittest discover -s python\tests -v
```

Testler için `requirements-dev.txt` ek bağımlılıklarını da yerel kaynaktan kur. Linux karşılığı `PYTHONPATH=python .venv39/bin/python -m unittest discover -s python/tests -v`.

C/MEX/S-Function henüz üretilmedi. JSON modelleri ve sabit boyutlu durum tanımı bu sonraki aşamaya temel sağlar.

[Telefonda eski sentetik sonuç sayfası](https://actuator-synthetic-lab.drawnparrot1.chatgpt.site) önceki iki veri setinin sabit görüntüsüdür. Bu tur yayımlanan sayfa güncellenmedi; yeni sonuçlar yerel uygulamada ve sunumda.

## Valf ofseti dayanıklılık denemesi (Py-0.5)

Eski sentetik validation %93.47 sonucu, uçuş bazlı valf kazancı/ofseti/sürüklenmesi eklenince %69.54'e düşüyor. Aynı zor veride quiet referansı %73.03; üç ek denemeden sonra en iyi %73.56. Bu yalnız sentetik validation; gerçek uçuş başarısı değildir, TEST açılmadı. Varsayılanlar değişmedi.

Paketlerini yüklediğin Python 3.9 ile repo kökünde:

```bat
python python\run_valve_audit.py
python python\run_valve_improvement.py
```

İlk komut üç sabit yöntemi üç sentetik senaryoda karşılaştırır. İkinci komut mixed senaryoda mevcut referans + en fazla üç yeni adayı çalıştırır. İnternet gerekmez; XLSX/JSON `python/results/audits/` altına yazılır. Önceden çalıştırılmış sonuçlar [python/audit_results](python/audit_results) içinde; bunlar yerel web sonuç seçicisinden ayrı stres raporlarıdır. Yeni gate seçeneği bu sınırlı deney komutunda denenir; mevcut web eğitimi otomatik değiştirilmedi.

On adaylık devam deneyi: `python python\run_valve_search.py`. Aynı zor sentetik veride >=%90 validation Fit veya 10 yeni adayda durur; TEST açılmaz. Bu turda 10 aday sonunda en iyi **%73.90** bulundu (önceki %73.56); %90 hedefi sağlanmadı. Kayıtlı XLSX/JSON: `python/audit_results/valve_search_20261112_162852/`. Bu deney web varsayılanlarını değiştirmez.
