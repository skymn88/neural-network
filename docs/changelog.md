# Changelog

Her geliştirme commitinde bu dosya güncellenir: değişiklik, deney/protokol, gerçekten çalıştırılan kontrol ve bilinen sınır kaydedilir. Başarısız denemeler silinmez. Aşağıdaki Git geçmişi eksiksizdir; Git öncesi bilgiler kullanıcı konuşmasından alınmıştır, bağımsız doğrulanmış deney kaydı değildir.

## Py-0.10 — 2026-11-13 — Çok uçuşlu genelleme, veri/UI ve sunum

Önceki commit `da39402` üzerine; kullanıcı token sınırında yeni denemelerin kesilmesini istedi. Yeni varsayılan model/split seçilmedi; değişiklikler opt-in, TEST kapalı, eski sentetik fingerprint ve synthetic_rich.mat korunur.

### 1. Gerçekte çalıştırılan genelleme deneyi
- 18 eğitim: 7/14/21 TRAIN × %99.9 hız kırpma kapalı/açık × 3 seed42/43/44. Her model LSTM24, 800 adım, horizon256. Eski2 + yeni6 validation uçuşu sabit; seed910247/741203 protokolü JSON'da. Hızları ortalayan tek integratör ensemble, RAM Component dahil tüm adayların uçuş/seed sonuçları XLSX CandidateFlights ve JSON'da. TEST değerlendirilmedi.
- Seçim artık en düşük validation uçuş Fit'i, sonra ortalama. Seçilen `n14_q999`: birleşik TRAIN **87.29**, validation **71.08**; validation macro **76.83**, minimum **40.39**; Component macro **51.25**, minimum **37.09**. TRAIN95/VAL90 hedefi sağlanmadı. Önceki iki validation üzerindeki yüksek Fit bağımsız çok-uçuş başarısı sayılmıyor.

| TRAIN | Kırpma | Val min | Val ort | Val birleşik | Component ort | Train–Val macro fark |
|---:|---|---:|---:|---:|---:|---:|
|7|kapalı|29.15|78.67|70.71|52.09|13.68|
|7|açık|36.31|79.08|72.27|48.36|12.68|
|14|kapalı|26.62|76.22|68.00|44.99|12.29|
|14|açık|40.39|76.83|71.08|51.25|12.49|
|21|kapalı|40.06|77.18|73.02|45.97|11.35|
|21|açık|23.23|75.64|68.92|4.72|10.93|

- Toplam **1914.59 s**. Veri artışı farkı biraz küçülttü ama validation'da tutarlı yükselme yok. Sabit optimizer adımı daha büyük veri için eşit epoch değildir. Bazı Component sonuçları çok zayıf: model DDV tahmin hatasına uyum sağlamış olabilir. q999 her durumda iyileştirmedi. Tek tek seed-validation Cascade negatifleri de saklandı (n07_q999 seed43 VAL_NEW_06 −6.13; n14_qoff seed44 −4.87); bunlar ensemble skorları değildir.
- Portable NumPy ve hızlandırılmış ensemble Cascade/Component eşleşmesi gerçekten kontrol edildi. Eski veri/fingerprint, ayrık cohort, TRAIN-only quantile ve causal rate-averaging testleri çalıştı. Sonuçlar `python/audit_results/generalization_20261113_051726/`; web kaydı `20261113_051726_000000`. Yeni genel arama negatif seed skorlarını da eleme kuralına dahil eder; bu geçmiş deneyin ham kayıtları silinmedi.

### 2. Veri ve arayüz
- Multipart doğrudan diske, usecols/chunksize ile kanal bazında anti-aliasing/64 Hz; ham dosya başarılı importtan sonra silinir. `ACTUATOR_MAX_UPLOAD_MB` varsayılan512 MiB/dosya. Yeni çevrimdışı bağımlılık python-multipart0.0.20; start.bat internetten paket indirmez.
- Gerçek HTTP ölçümü: **103,087,576 byte** CSV (101 MB talebinden büyük; 480000 satır), upload **0.380 s**, import **0.518 s**, toplam **0.898 s**. Linux sunucu VmHWM başlangıç **143777792 byte**, tepe **186564608 byte** (177.92 MiB; artış40.80 MiB). Ölçüm API bağımlılıklarını kapsar; Windows performansı iddiası değildir. Sonradan tam101 MiB üretmeye ayarlanan benchmark tekrar çalıştırılmadan kullanıcı durdurma istedi; yukarıdaki rakamlar ilk gerçekten tamamlanan dosyaya aittir.
- Chunk sınırı/anti-aliasing testi başlangıç padding farkını buldu; sabit padding ile düzeltildi ve tekrar geçti. Multipart boyut sınırı/yarım dosya temizliği, usecols ve cache tek-üretim/LRU kontrolleri çalıştı.
- Plot/raw üretim cache'i, TEST dışı arka plan prewarm, yükleniyor/hata/zaman aşımı ve eski yanıt token koruması. Sistem Chromium'da kayıtlı sonuç ilk açılışında3 grafik, gecikmiş uçuş yanıtı, load yarışı, görünür hata/tekrar ve gerçek multipart import **geçti**, JS page error yok. Browser regresyon script'i repoda.

### 3. Güvenilir arama ve sunum
- Opt-in güvenilir seçim/otomatik arama: 6 aile, iki sınırlı preset, 3 seed; aday/dakika/iptal/ilerleme, her tamamlanan aşama sonrası kısmi sonuç ve model kaydı. Validation<3 → TRAIN-only LOFO; TEST her zaman kapalı. Negatif/nonfinite Cascade uçuşları elenir; >25puan genelleme farkı ve Component düşüklüğü uyarılır. Seed min/ort/max ve Component/Cascade CandidateFlights'ta.
- Altı ailenin her biri küçük deterministik veride gerçekten eğitildi (2 optimizer adımlı smoke; stable'ın mevcut600 ön eğitim adımı korunur). LOFO/test izolasyonu, seed ensemble fiziksel hız ortalaması, farklı DDV üyeleri, iptal/kısmi kayıt test edildi. Bu yeni genel otomatik aramanın geniş gerçek-veri performans taraması yapılmadı.
- Sunum yeniden yazıldı: beyaz/siyah, Franklin Gothic Medium, gerçek title placeholder ve bullet/slidenum alanları, editable native chart/şekil/tablo, genel CONTENT ve ayrı NUMBERS, gelecek işler/sözlük/bulgular. `--template` eski slaytları kaldırıp düzenleri korur; font=None korunumu test edildi. Kaynak kayıt/metin kullanır, eğitim çalıştırmaz. Şablon boyutu/nesne sınırları/tema/başlık/nativechart/bullet/alan kontrolleri geçti. PowerPoint/Windows görsel açılış testi yapılmadı.
- Python3.9 gerçek test paketi: **60 test geçti**; Ruff E9/F63/F7/F82 (py39), compileall ve dependency check çalıştırıldı. İlk sandbox TestClient soket kısıtından durduruldu; localhost izinli koşuda testler geçti. Son küçük düzenlemelerden sonra da 60 test yeniden geçti (9.09 s). MATLAB ve gerçek uçuş yok; performans iddiası yalnız belirtilen sentetik deneye aittir.

## Py-0.9 — 2026-11-13 — Her TRAIN uçuşu >90 ve VAL >75: 20 aday

- Kullanıcı hedefi bu turda **her TRAIN uçuşunda RAM Cascade Fit >90**, birleşik validation Fit >75 olarak uygulandı; aynı mixed-valve seed621, 7/2/2 split, donuk DDV/açı ve değişmeyen fingerprint. TEST açılmadı. 20 yeni aday tamamlandı; iki şartı birlikte sağlayan model çıkmadı.
- TCN/Elman/RBF için ilk tam performans deneyleri; LSTM/GRU/TDNN kapasite, pencere, TRAIN-hata ağırlıklı örnekleme, loss katsayısı ve seed karşılaştırmaları. Yeni `refine_sequence_head` donuk recurrent state/readout konum-Jacobian düzeltmesini yalnız TRAIN'de öğrenir; validation optimizere verilmez. 20 aday ile iç optimizer adımı aynı kavram değildir.
- TRAIN-only lider `20_head_best`: birleşik Train 93.99, en düşük TRAIN uçuşu 93.11, Val 72.30; validation şartından elendi. `09_lstm`: Train 92.20, en düşük TRAIN 86.83, Val 78.24; birleşik skorlar iyileşti ama her-uçuş hedefi sağlanmadı. Önceki LSTM Train 90.45/min86.85/Val76.27, validation>75 altında en düşük TRAIN sıralamasında korundu (09 adayıyla min farkı yalnız ~0.02 puan).
- `run_balanced_search.py`: 20 üst sınır, iki koşulda erken durma; tüm adayların uçuş bazlı sonuçları XLSX `CandidateFlights` ve JSON'da. Sonuç `python/audit_results/balanced_search_20261113_044434/`; yerel UI kaydı `20261113_044434_000000`, önceki model + TRAIN-only lider + ortalama skoru iyi alternatif birlikte saklanır. Başlık pooled ortalama yerine **en düşük TRAIN uçuş Fit** gösterir. Normal eski koşu başlıkları/varsayılanları değişmedi.
- `flight_weights=None`, `position_weight=.2` eski eğitim davranışını korur. Pozitif TRAIN ağırlıkları sadece örnekleme içindir, uçuş ID modeli beslemez. Readout düzeltmesi yeni giriş/state eklemez, aynı NumPy ağırlık çıkarımını kullanır. Rate/position clamp Jacobian'ı finite-difference testiyle doğrulandı. Daha büyük ağın otomatik daha iyi olmadığı bu sınırlı taramada görüldü; mimari aileler için genel üstünlük iddiası yok.
- Python3.9: **47 test geçti**, Ruff/compileall geçti; Torch chunk-state ile NumPy eşleşmesi, clipped-Jacobian, donuk recurrent ağırlıklar, TRAIN kaybı iyileşmesi ve kesin hedef eşikleri dahil. Seçilen model final NumPy rollout ile tekrar kontrol edildi. Sistem Chromium ile üç model seçeneği, en düşük TRAIN etiketi, hedef-sağlanmadı bildirimi ve TRAIN uçuşundaki grafikler doğrulandı; ek eğitim başlatılmadı. Adayların eğitim+skor süreleri toplam716.90 s; veri hazırlama/son rapor hariç. Gerçek veri/MATLAB/Windows testi veya yeni bağımsız audit yok. Validation birçok geliştirme turunda tekrar kullanılıyor.

| Aday | Birleşik TRAIN | En düşük TRAIN uçuşu | Birleşik VAL | Süre (s) |
|---|---:|---:|---:|---:|
| 01_tcn | 84.15 | 75.44 | 78.09 | 10.1 |
| 02_elman | 85.76 | 79.00 | 77.42 | 18.1 |
| 03_rbf | 84.97 | 67.01 | 69.64 | 4.5 |
| 04_head | 93.12 | 90.62 | 72.01 | 49.3 |
| 05_lstm | 89.56 | 85.55 | 79.50 | 26.7 |
| 06_lstm | 88.57 | 82.68 | 78.11 | 32.6 |
| 07_lstm | 87.13 | 79.68 | 72.41 | 25.4 |
| 08_head | 93.97 | 93.05 | 72.49 | 68.0 |
| 09_lstm | 92.20 | 86.83 | 78.24 | 30.5 |
| 10_lstm | 90.52 | 79.04 | 75.55 | 34.7 |
| 11_tcn | 89.22 | 78.44 | 76.35 | 27.3 |
| 12_tdnn | 88.78 | 84.20 | 68.83 | 10.4 |
| 13_rbf | 83.93 | 62.53 | 69.49 | 5.3 |
| 14_elman | 81.24 | 67.01 | 80.34 | 32.1 |
| 15_lstm | 78.82 | 62.35 | 72.08 | 33.0 |
| 16_head | 92.99 | 91.40 | 70.76 | 57.8 |
| 17_gru | 89.27 | 77.21 | 75.89 | 132.6 |
| 18_lstm | 90.94 | 82.29 | 78.43 | 27.3 |
| 19_lstm | 85.60 | 74.30 | 66.06 | 45.0 |
| 20_head_best | 93.99 | 93.11 | 72.30 | 46.1 |

## Py-0.8 — 2026-11-12 — TRAIN kapasite hedefi ve LSTM

- Kullanıcı hedefi validation yerine TRAIN RAM Cascade Fit >90, üst sınır 20 yeni aday olarak değişti. Aynı mixed-valve seed-621/split/fingerprint; TRAIN RMSE ile sıralama ve durma, validation report-only, TEST kapalı. Altıncı adayda **TRAIN 90.4487** elde edildi; kalan 14 çalıştırılmadı. Bu TRAIN-seçilmiş tanı modeli, eski validation kazananının otomatik üretim yerine geçmesi değildir.
- TRAIN teşhisi: FLIKE_06'da düşük hızla sürüklenen DDV tahmini öğrenilmiş null gate dışına çıkıyor; RAM sabitken yanlış hız üretilip uzun hold boyunca hata büyüyor. `fit_null_rate(gate_expansion=1.0)` opt-in genişletme; midpoint korunur, yarı genişlik çarpılır. Varsayılan 1 eski davranışı korur; 1.5/2/3 karşılaştırıldı. Truth/TEST kullanılmadı.
- Yeni adaylar Train/Val Fit: GRU12 gate1.5 87.80/81.13; gate2 87.80/81.13; gate3 74.33/38.27; GRU24/800 gate1.5 79.98/68.10; gate2 79.98/68.10; **LSTM24/800 gate1.5 90.45/76.27**. Pencere256, seed42; null-rate+2s OE referansı üzerine hız düzeltmesi. 1.5 ve 2 aynı örneklerde aynı observer kararlarını verdi; aynı sonuçlar ayrı yürütülmüş adaylardır.
- Seçilen LSTM RAM Component Train/Val **57.68/43.11**, RAM Cascade **90.45/76.27**, Angle Cascade **90.74/76.06**. Önceki GRU RAM Cascade 69.06/91.43. FLIKE_06 Train Fit 34.23→92.81. Kapasite hedefi sağlandı; genelleme iyileşmesi iddia edilmiyor, Component hâlâ zayıf.
- `run_train_search.py`: 20 üst sınır ve TRAIN >90 erken durma; her aday sonrası JSON kayıt. `sequence_evaluate.py` Torch double/chunk-state ile hızlandırılmış tam-uçuş skor; GRU/LSTM/Elman için 8192 örnek chunk sınırında NumPy eşleşmesi test edildi. Kazananın tüm Train/Val/Component sonuçları taşınabilir NumPy rollout ile yeniden hesaplandı; TRAIN skor farkı <1e-5 kontrolü geçti.
- API `selectionBasis='validation'|'train'` (varsayılan validation), `nullGateExpansion=1` ve `lstm_rate` opt-in. TRAIN modu rapor/notice içinde açık; test seçimde yok. Yerel UI LSTM preset hidden24/horizon256/steps800/seed42/gate1.5/TRAIN seçim gönderir. Yeni rapor `20261112_170014_000000`, önceki GRU yan yana korunur; başlık doğru TRAIN metriğini gösterir.
- Python3.9: **43 test geçti**, Ruff/compileall geçti; gerçek TRAIN/VAL hedefleri zıt fixture'da seçim ayrımı ve gate/option forwarding testleri dahil. Sistem Chromium ile kayıtlı TRAIN raporu/3 grafik ve LSTM preset isteği (selectionBasis=train, gate1.5) doğrulandı; istek engellenerek ek eğitim başlatılmadı. Dataset SHA korunur. MATLAB/Windows/gerçek uçuş test edilmedi; bu turda yeni bağımsız audit yapılmadı.

## Py-0.7 — 2026-11-12 — Sequence aileleri, GRU ile validation hedefi

- Kullanıcının yöntem listesi `docs/methods.md` içinde denenen/uygun/önceliksiz olarak değerlendirildi. 20 sabit adaylık plan; aynı mixed-valve seed-621 ve 7/2/2 split, validation RAM Cascade Fit >=90 erken durma. TEST açılmadı. TDNN **85.53**, ESN **76.34**, GRU **91.43**; üçüncü adayda duruldu. Kalan 17 tam deney çalıştırılmadı.
- Seçilen GRU: 12 hidden, 400 Adam adımı, 256 örnek TRAIN penceresi, seed 42. Öğrenilmiş referans hızı + GRU düzeltmesi; TRAIN null gate/sınırlar korunur. DDV ve açı donuk. TRAIN RAM yalnız hedef/başlangıç; çıkarımda ara ölçülen geri besleme/Truth yok. Yeni davranış opt-in; eski varsayılanlar değişmedi.
- RAM Cascade Train/Val 69.06/91.43; RMSE 23.14/7.06. RAM Component 56.60/40.16; Angle Cascade 68.79/91.76. Component zayıflığı çözülemedi; güçlü validation sonucu bağımsız genelleme/gerçek uçuş başarısı değildir. Aynı validation birçok tur kullanıldı; yeni blind audit bu turda yok.
- `sequence_inference.py`: NumPy-only GRU/LSTM/Elman/TDNN/TCN/ESN/RBF çıkarımı; açık ağırlıklar ve sabit boyutlu durum, C/MEX yok. `sequence_train.py`: TRAIN ölçekleme, hız + interior konum-pencere kaybı; ESN/RBF ridge readout. LSTM/Elman/TCN/RBF için yalnız küçük deterministik eğitim/parity kontrolleri; performans iddiası yok.
- `run_sequence_search.py`: her aday sonrası sonuç/model checkpoint, 20 üst sınır, >=90 erken durma. İlk üç adayın eğitim+validation süreleri sırasıyla 4.8/6.7/20.9 s (veri hazırlığı/son rapor hariç). Kayıtlı audit XLSX/JSON ve yerel UI için `20261112_164547_000000` sonucu eklendi.
- Yerel UI: mixed-valve sentetik kaynak ve GRU+referans seçimi; preset 12/256/400/42. Kendi CSV verileri aynı opt-in yoluyla eğitilebilir. Referans ham veri cache anahtarı source+seed olarak ayrıldı; varyasyon kaynağı doğru fingerprint ile üretilir. Eski launcher/bağımlılıklar değişmedi; site yeniden yayımlanmadı.
- Python 3.9'da 40 regresyon testi geçti; Torch–NumPy GRU/LSTM/Elman/TCN/TDNN parity, nedensel prefix ve yedi ailenin küçük eğitim/gate testleri dahil. Ruff ve compileall geçti. Sistem Chromium ile kayıtlı GRU raporu, üç grafik ve mixed/GRU preset alanları doğrulandı; bu UI kontrolünde yeniden eğitim başlatılmadı. MATLAB/Windows çalıştırılmadı. Sunum önceki tarihsel deney sürümünde bırakıldı.

## Py-0.6 — 2026-11-12 — On adaylık sınırlı RAM araması

- Kullanıcı hedefi: aynı mixed-valve verisinde validation RAM Cascade Fit >=90 veya en fazla 10 yeni aday. On aday tamamlandı; hedef sağlanmadı. Seed 621, aynı 7/2/2 split ve fingerprint, donuk DDV/açı blokları; TEST skorlanmadı. TRAIN pencereleri validation rollout'unda reset oluşturmaz.
- `fit_null_rate` opt-in `quiet_seconds` ve `quiet_quantile` alır; varsayılan .5 s/.95 ile önceki davranış korunur. Observer geçmişi 1/2/4 s, eşik medyanı ve 2/10/30 s TRAIN konum-penceresi ridge düzeltmeleri karşılaştırıldı. Bu on yeni mimari değil; mevcut aile içinde on ayar/eğitim adayıdır.
- Validation Fit: quiet_1s 73.24; quiet_2s 71.64; quiet_4s 23.84; quiet_1s_q50 72.09; quiet_2s_q50 72.16; quiet_half_q50 73.07; window_2s **73.90**; window_10s 33.15; window_30s 25.86; quiet_2s_window 44.94. Önceki referans 73.56; yeni kazanç yalnız 0.35 puan.
- Seçilen `window_2s`: önceki 81 düğümlü/ridge=.001 stationary/predicted-gate haritasına 2 s TRAIN pencerelerinde konum hatası ridge=.01 düzeltmesi. Ölçülen RAM yalnız TRAIN hedefi ve rollout başlangıcında kullanılır. Kaynak offset/hold belirsizliği çözülmedi; uzun pencereler kötüleşti. Varsayılan web modeli değiştirilmedi.
- `python/run_valve_search.py` aynı sınırlı deneyi tekrarlar; kısmi search/model kayıtları her adayda yazılır, >=90 durumunda erken durur. Sonuçlar `python/results/audits/`; kayıtlı kopyalar `python/audit_results/valve_search_20261112_162852/` içinde. On adayın eğitim+validation süresi bu ortamda toplam 73.77 s (veri üretimi/son toplu rapor hariç).
- Gerçek Python 3.9: **37 test geçti**, Ruff/compileall geçti; yeni pencere/quantile doğrulama ve causal-prefix testi eklendi. Gerçek uçuş/MATLAB/Windows çalıştırması yapılmadı; bağımsız test başarısı iddiası yok. Validation tekrar tekrar kullanıldığı için yeni sonuç bağımsız genelleme kanıtı değildir.

## Py-0.5 — 2026-11-12 — Valf varyasyonu ve sınırlı dayanıklılık araması

- Claude `177c677` değişikliği bütünüyle merge edilmedi: üreteçteki opt-in `valve_variation` fikri uyarlanıp parametre doğrulaması, pozitif gain kontrolü ve tam bias Truth izi eklendi. Varsayılan raw fingerprint ve ortak MAT SHA değişmedi. Kazanç, akım gürültüsünün DDV üzerindeki etkisini de az miktarda değiştirir; downstream dizilerin birebir aynı olduğu iddia edilmez.
- Sabit seed 621, aynı 7/2/2 split. Üç veri koşulunda eski slow-null, quiet ve envelope değerlendirildi; TEST skorlanmadı. Validation RAM Cascade: baseline 68.13/91.01/93.47; mixed gain=.03,bias=.025,drift=.02 için 57.88/73.03/69.54; bias-only=.04 için 54.67/65.89/73.86. Önceki %93.47'nin bu stres koşuluna genellenmediği açıkça görülüyor.
- Kullanıcı isteğiyle mixed koşulda tam **üç ek aday**: TRAIN tahmin-DDV ölçeğinde null gate + stationary (%73.50), aynı gate + envelope (%65.57), stationary + 81 düğüm/ridge=.001 (%73.56). Mevcut quiet referansı %73.03. Son aday validation RMSE ile seçildi; varsayılan model değiştirilmedi. Artış yalnız 0.52 puan; %95 hedefi sağlanmadı.
- Seçilen aday RAM Cascade Train/Val 52.20/73.56, RAM Component 50.21/50.90; Angle Cascade 52.50/74.41. Gerçek veri yok; uçuş ofseti yalnız akımdan her koşulda tanımlanamaz. Sentetik train/validation farkı ve Component zayıflığı sürüyor. Bu turda yeni test/fresh-audit değerlendirilmedi.
- `fit_null_rate(..., gate_source='predicted')` opt-in; eski `'measured'` varsayılanı korundu. Hold maskesi TRAIN RAM'den, gate TRAIN DDV tahmininden; Truth kullanılmaz. TRAIN upstream tahminleri in-sample olmaya devam ediyor (OOF değil).
- Tekrarlanabilir komutlar: `python python/run_valve_audit.py`, `python python/run_valve_improvement.py`. Çıktı `python/results/audits/` altında XLSX/JSON; kayıtlı kopyalar `python/audit_results/`. Yerel web sonuç listesiyle farklı şemalar karıştırılmıyor; web varsayılanları ve çevrimdışı launcher değişmedi.
- Python 3.9'da deneyler gerçekten çalıştı. Deterministik opt-in/zero/fingerprint, varyasyon tekrarlanabilirliği, fizik integrali, geçersiz parametre ve predicted-gate testi eklendi. Tam regresyon: 36 test geçti; Ruff ve compileall geçti. MATLAB/Windows çalıştırması yapılmadı. Sunum bu turda yeniden üretilmedi; önceki deneyin tarihsel çıktısıdır.

## Py-0.4 — 2026-11-12 — Python 3.9, çevrimdışı kurulum ve sunum

- Kullanıcı isteğiyle `matlab/` bu daldan kaldırıldı. Önceki MATLAB kaynakları `dev_Codex` dalında/Git geçmişinde; ortak sentetik MAT veri dosyası korundu.
- Python 3.9 nullable tip uyumu; uyumlu paket sürümleri sabitlendi. `start.bat` otomatik indirme yapmaz. `setup_offline.bat` yalnız yerel wheelhouse, `download_wheels.bat` yalnız ayrı online hazırlık makinesi içindir. Eksik/eski paket ve interpreter kontrolü eklendi; README komutları 3.9'a güncellendi.
- Üç opt-in aday: sakin aralık null kestirimi, osilasyon tepe/çukur merkez izlemesi, 5 s konum-penceresi LS düzeltmesi. Aynı veri/split; Val RAM Cascade 68.13 → 93.47. Train/Val/tarihsel-Test 93.50/93.47/95.18. %95 validation hedefi elde edilmedi; üç denemede duruldu.
- Seçimden sonra iki yeni seed-9621 uçuşunda RAM Cascade 93.67. Ham-DDV Component gürültü hassasiyeti (Val 48.40) açıkça raporlandı; gerçek veri başarı iddiası yok.
- Yeni observer tek yöntem seçeneğiyle yerel arayüzde; eski varsayılan yol `legacy` olarak korunur. Canlı telefon sayfası güncellenmedi.
- Düzenlenebilir 16 slaytlık PowerPoint: süreç, yöntemler, kullanıcı tarafından bildirilen gerçek uçuş sonuçları, çalıştırılan sentetik sonuçlar, grafikler, çevrimdışı kurulum ve sınırlamalar. Kaynak üretici `python/build_presentation.py`; sunum `presentations/actuator_black_box_overview.pptx`.
- Gerçek Python 3.9.25 ortamında 31 regresyon + 2 yeni kontrol geçti; NumPy 1.26.4 / SciPy 1.13.1 / Torch 2.6.0+cpu. Chromium yerel UI kontrolü, Ruff/compileall; PPTX ZIP/XML/gömülü grafik ve slayt sınırı kontrolü. Windows .bat / PowerPoint GUI burada çalıştırılmadı.
- Önceki commit `986124c`. Tüm yeni deney sayıları/protokol [experiments.md](experiments.md) içinde; yeni koşu `20261112_105714_655119`.

## Py-0.3 — 2026-11-12 — FlightLike ve öğrenilmiş null/hız modeli

- Mevcut üretece opt-in `flightLike`: değişken null/band, drift, park/uzun-kısa hold, iki yönlü darbeler, gürültü; RAM integral tutarlılığı ve azalan statik açı. `Truth` veriyle saklanır, öğreniciye verilmez. Eski MAT ve pulse profili korundu.
- Yeni opt-in `null_rate`: TRAIN'den öğrenilen hız haritası, TRAIN sınırları ve causal null observer. Beta sızıntısı yok. NN referansı korunur; yeni ailenin NN olduğu iddia edilmez.
- 10 önceden belirlenmiş aday tamamlandı; %95 validation RAM Cascade eşiğine ulaşılamadı, arama durdu. NN Val %18.87 → kazanan Rate_slow_null %68.13. Kazanan Train/Val/Test 71.18/68.13/83.34. Veri değiştiği için önceki %87 ile doğrudan başarı kıyası yapılmaz. [Tüm 10 deney ve sınırlamalar](experiments.md#py-03--flightlike-10-deney--2026-11-12).
- Gerçek veri için `syntheticTrainFlights=0` varsayılan, >0 TRAIN-only sentetik ekleme A/B'si; kalibrasyon TRAIN-only, gerçek holdout'lar korunur; aynı XLSX'e fark satırı. Gerçek veri A/B performans deneyi yapılmadı.
- Yerel arayüze yeni veri/model seçenekleri, null süresi ve ekleme adedi. Kayıtlı sonuçlar tekrar eğitim olmadan açılır. Canlı mobil sayfa kullanıcı isteğiyle değiştirilmedi, otomatik güncellenmediği README'de belirtildi.
- 28-test regresyon + son eklenen testle 9 yeni-alan kontrolü geçti (29 farklı test); Ruff/compileall/JS sözdizimi ve yerel Chromium okuma akışı. Son üreteç eğitim verisi fingerprint'ini aynen üretiyor. MATLAB değiştirilmedi/çalıştırılmadı.
- Büyük yeni ham/Truth NPZ'ler çalışma alanında korundu; son kullanıcı isteğiyle ZIP hazırlanmadı; Git'te küçük model/rapor/üreteç kayıtları. MATLAB klasörü korunur; repoda bulunması tek başına token tüketmez. Önceki mobil commit: `2413915`.

## Py-0.2 — 2026-11-12 — Mobil sentetik sonuç sayfası

- Sites üzerinde özel erişimli, salt sonuç gösteren mobil HTML sayfası eklendi. Eğitim sunucusu gerektirmez; 2 veri seti × 11 uçuş × 4 model için 88 grafik kaydı, kompakt Fit/RMSE tablosu, zaman aralığı ve XLSX indirme.
- Yeni model/eğitim denemesi yapılmadı. `python/reference_results/` çıktıları aynen kullanıldı; yalnız grafik sayıları 7 anlamlı basamağa yuvarlandı ve mevcut tepe-koruyan önizleme kullanıldı.
- `python/viewer/export_results.py` kullanıcı verisini dışa aktarmayı reddeder. İki yeni unit test geçti; 88 grafik dosyasında boyut/zaman sırası ve iki tam veri seti denetlendi; JavaScript `node --check` ve Ruff geçti.
- Bu yeni Sites sayfası için tarayıcı QA yapılmadı: managed ortamın gerekli `cua_repl.js` aracı yoktu. Mevcut yerel Python uygulamasının önceki Chromium doğrulaması bunun yerine sunulmaz. Sites helper dosyaları ortamda bulunamadığı için kaynak push'u ve statik tar arşivi eşdeğer Git/paketleme adımlarıyla yapıldı; geçici kimlik bilgisi yalnız gizli stdin/bellekte kullanıldı.
- MATLAB, modeller, split ve eski `synthetic_rich.mat` değişmedi. Canlı bağlantı README'de, Site kimliği `python/viewer/site.json` içinde. Önceki Python uygulaması commit'i: `1e7d07c`.

## Py-0.1 — 2026-11-12 — dev_CodexPy

### Added

- Bağımsız Python plant hattı: zaman hizalama/anti-alias, aynı uçuş split'i, DDV ridge bazları, OE, dinamik MLP ve monoton ağ karşılaştırması, validation checkpoint seçimi, test izolasyonu.
- DDV gerçek/tahmin hatası ortak ağın yardımcı kaybına eklendi; RAM kaybı DDV ağırlıklarına da yayılıyor. Serbest çıkarım gerçek DDV/RAM geri beslemesi kullanmıyor; yalnız akım + başlangıç RAM.
- Yerel HTML arayüzü: CSV sütun/zaman eşleme, split, eğitim ayarları, yöntem seçimi, beş sayfalı XLSX, DDV/RAM/açı grafikleri, yeniden eğitmeden kayıtlı uçuş inceleme.
- NumPy-only çıkarım ve JSON ağırlıklar. Ayrı 11 uçuşluk düzensiz darbe–bekleme veri seti/ham NPZ. Mevcut MAT değişmedi.
- Altı farklı sentetik deney + bir tekrar gerçekten çalıştırıldı. Tam protokoller/negatif sonuçlar [experiments.md](experiments.md) ve `python/experiments/` altında. Son testli modeller `python/reference_results/` altında.
- NN_joint validation kazananı: eski sette RAM Fit 77.05/87.62/77.53, darbe setinde 85.09/79.23/87.76 (Train/Val/Test). Bunlar gerçek uçuş sonuçları değildir. %95 hedefi elde edilmedi; uzun eğitim ve monoton ağ denemeleri ek kazanç sağlamadı.

### Changed / Fixed

- MATLAB uygulaması `matlab/` altına taşındı; fonksiyon adları/model ayarları korunarak ortak veri yolları düzeltildi. Eski setpoint/pressure/force deneyleri MATLAB'da tutuldu; Python'ın bunları birebir taşıdığı iddia edilmez.
- Python ve MATLAB belgeleri ayrıldı; eski README korundu. İlk changelog tüm 15 önceki commiti aşağıda listeler.
- CSV'de bilinmeyen split etiketi, eksik kanal, uzun zaman boşluğu ve çok kısa eğitim penceresi açık hatadır. Kullanıcı verisi ve yerel sonuçlar Git dışındadır.

### Validation / Limitations

- 18 deterministik Python testi: gerçek küçük eğitim, test verisini değiştirerek eğitim izolasyonu, OOF dışlama, yoğun/normal-denklem denkliği, PyTorch–NumPy çıkarım denkliği, CSV/API, XLSX sayfaları.
- Chromium ile CSV→eğitim→grafik→XLSX akışı; Ruff/compileall; MATLAB R2019a MISS_HIT: 120 dosya.
- İlk tarayıcı testindeki göreli indirme URL'si test kodunda düzeltildi; uygulama indirme rotası değişmedi. Sandbox'ta yerel test istemcisi socket kısıtına takıldı; yerel socket izniyle testler çalıştı.
- MATLAB/Simulink ve Windows `.bat` başlatıcısı çalıştırılmadı. C/MEX yok. Gerçek veri yok. Test açıldıktan sonra yeni model araması yapılmadı.

## Önceki Git commitleri — 2026-11-09…11

Tüm hash'ler mevcut `dev_Codex` geçmişindendir. Bu bölüm geçmişte her testin çalıştığı iddiasını taşımaz; bugün dosyaların statik kontrolü yapıldı. Kullanıcı tarafından bildirilen gerçek veri sonuçları aşağıda ayrıca etiketlenmiştir.

| Commit | Tarih | Değişiklik / deney |
|---|---|---|
| [44d54dd](https://github.com/skymn88/neural-network/commit/44d54dd) | 11-09 | MATLAB proje düzeni, gerçek uçuş adaptörü, README/docs ve modüler kaynak/test ayrımı. |
| [3802a5e](https://github.com/skymn88/neural-network/commit/3802a5e) | 11-09 | Alignment, component/cascade ayrımı, hold sürüklenmesi, kompakt çıktı teşhisleri. |
| [fb95496](https://github.com/skymn88/neural-network/commit/fb95496) | 11-09 | Uçuş bazlı OOF DDV, bellek sınırlı tüm-örnek OE, null varyantı. |
| [6f8aa77](https://github.com/skymn88/neural-network/commit/6f8aa77) | 11-09 | Hammerstein/yardımcı force-pressure OE ve RAM sınırları. |
| [2af7376](https://github.com/skymn88/neural-network/commit/2af7376) | 11-09 | İzole setpoint-sürülü closed-loop kontrolcü/plant ve raporlama. |
| [5fc68ed](https://github.com/skymn88/neural-network/commit/5fc68ed) | 11-10 | NL_OE odağı, ilk uçuş çıkışlarını otomatik çizme. |
| [0a1d4c9](https://github.com/skymn88/neural-network/commit/0a1d4c9) | 11-10 | Beta/sızıntı teşhisi, sınırlı ortak arama, TRAIN+VAL refit ve nonlinear leakage deneyi. |
| [f920ab6](https://github.com/skymn88/neural-network/commit/f920ab6) | 11-10 | Monoton Wiener RAM, geçiş ağırlıklı OE, RateMap/StepSmoothing; null/baseline bayrakları. |
| [fcfa1e0](https://github.com/skymn88/neural-network/commit/fcfa1e0) | 11-10 | NL_FB filtre bankası, BandGain, kompakt workbook ve aşama süreleri. |
| [b32b104](https://github.com/skymn88/neural-network/commit/b32b104) | 11-10 | RAM filtre bankasına odak, yöntem süreleri; grafikte current yerine DDV. |
| [365c55b](https://github.com/skymn88/neural-network/commit/365c55b) | 11-10 | Duruma bağlı/state-space RAM adayı; workbook/fixture kontrollerini güçlendirme. |
| [6f55332](https://github.com/skymn88/neural-network/commit/6f55332) | 11-11 | Öğrenilmiş pulse-hold RAM deneyi; bağımsız kayıtlı uçuş plot script'i. |
| [3f05766](https://github.com/skymn88/neural-network/commit/3f05766) | 11-11 | Çok adımlı dinamik MLP; tüm DDV örnekleri, tanhScale/filtre başlangıç seçenekleri. |
| [0d6b0bd](https://github.com/skymn88/neural-network/commit/0d6b0bd) | 11-11 | PulseHold/HorizonFit, pencereli OE ve seçim; run sonrası runtime serbest bırakma. |
| [123b04b](https://github.com/skymn88/neural-network/commit/123b04b) | 11-11 | MLP hareket/bekleme kayıp dengesi; TRAIN static-null tanısı ve opt-in null merkezleme. Python dalının tabanı. |

## Kullanıcının bildirdiği önceki deneyler

Kaynak: konuşmada paylaşılan tablolar/gözlemler. Bu ortamda gerçek veriler veya eski tüm XLSX'ler yok; tek tek koşu/hash eşlemesi kesin değil. Aynı model adı farklı ayarlarda kullanıldı; aşağıdaki sayılar birbirinin tekrarı olarak yorumlanmamalı.

- Git öncesi Ver4.1: MLP Coil→RAM Fit 65/49/62; NARX −168/−272/−368. NARX serbest koşu sorunları nedeniyle aktif yoldan çıkarıldı. Bazı Ver4 Fit'leri aşırı negatif, düşük varyans/başlangıç/hafıza sorunları açık kaldı.
- İlk çok-aileli karşılaştırma: MLP recipe, more_training, quasiNewton; FIR_rich, GP_coverage, retrained pair, ardından FIR_OE. Örnek tabloda FIR_OE RAM 68.69/65.56/3.87, açı 69.13/67.30/9.28. Tek adım başarısı serbest koşu başarısı değildi; kötü adaylara odak azaltıldı.
- Sentetik akımın fazla düzenli olduğu plot ile görüldü. 11 uçuş için 7/2/2 istendi. Sentetik veriyi gerçek fizik doğrulaması olarak yorumlamama ve gerçek veri eğitimini önceliklendirme kararı alındı.
- NL_OE: bir koşuda RAM 71/63/38, açı 71/64/42; daha sonra RAM 70/61/39, açı 71/63/42. Calibration ve beta grid kıyaslarının aynı protokolle yapılması gerektiği tartışıldı. Beta −1 ve −.3 farklı koşullarda kazandı; tek bir evrensel beta sonucu çıkarılamadı.
- OOF DDV ve `oeMaxTrainRows=Inf` en yararlı değişiklikler olarak kullanıcı tarafından bildirildi. WH ve geçiş ağırlığı belirgin kazanç vermedi.
- NL_FB: RAM 73/65/40, açı 73/67/43; bir sonraki tabloda DDV 76/68/90, RAM_Component 65/34/38, Angle_Component 92/96/98. Toplam süre önce 665 s, sonra daraltılmış aramada 168 s; RAM araması 29 s bildirildi. Bunlar farklı koşular, kontrollü hız benchmark'ı değil.
- Flight 2: DDV/Angle_Component ~93/98, RAM_Component/RAM_Cascade/Angle_Cascade ~8/10/15. Düşük varyans, farklı hareket tipi, başlangıç/offset ve dağılım farkı olasılıkları; gerçek ham veri olmadan kesin neden saptanmadı.
- Excel otomatik Sheet1/2/3 oluşturması ve `test_workbook` assertion; sonrasında `test_cascade_fixture` hatası kullanıcı tarafından bildirildi. Workbook temizleme ve kontroller güncellendi; eski tüm MATLAB sürümlerinde doğrulandığı iddia edilmez.
- NLSS: toplam 1375 s, RAM NLSS 1130 s; RAM_Component 63/18/38. Kazanç bildirilmedi, süre arttı. Tek yönlü hareketlerde karşı-yön aşımı gözlendi.
- NL_HOLD: kullanıcı negatif Fit bildirdi, tam sayılar yok. Dinamik MLP ve pencereli amaçlara geçildi; tek yönlü darbe ve uzun tutma davranışı hedeflendi.
- Pencereli koşu (`oeWindowSearch`, windowed=10 s, all DDV, steady start, unique poles; MLP kapalı): Angle_Cascade 68/64/46, RAM_Component 55/60/34. Beta −.3/ridge1e−6 için yaklaşık 17/13 RMSE bildirildi; sütun bağlamı tam doğrulanmadı. PulseHold'da eğim/parabol kalması static-null teşhisini tetikledi.
- Son MATLAB static-null/balanced-MLP commitinin gerçek uçuş sonucu henüz bildirilmedi. Python'a geçiş, burada gerçek eğitim/test döngüsü çalıştırabilmek için istendi.

Geriye dönük kayıtlarda bilinmeyen ayar, süre veya başarı uydurulmadı. Sonraki deneyler ham sonuç kimliği, veri fingerprint'i, split, ayar ve gerçekten çalışan kontrollerle kaydedilmelidir.
