# Changelog

Her geliştirme commitinde bu dosya güncellenir: değişiklik, deney/protokol, gerçekten çalıştırılan kontrol ve bilinen sınır kaydedilir. Başarısız denemeler silinmez. Aşağıdaki Git geçmişi eksiksizdir; Git öncesi bilgiler kullanıcı konuşmasından alınmıştır, bağımsız doğrulanmış deney kaydı değildir.

## Py-0.4 — 2026-11-12 — Opt-in valf (akım→DDV) varyasyonu

- `pulse_flights(profile='flightLike', valve_variation={'gain_sd','bias_sd','drift_amp'})`: akım→DDV kazanç/bias'ı uçuşa göre rastgele ve yavaş sürüklenen hale gelir. Gerçek DDV/RAM fiziği değişmez (yalnız akımın kodlaması kayar). Varsayılan `None`: fingerprint ve önceki sonuçlar aynı. Truth'a `valve_gain`, `valve_bias_mean` eklendi (öğrenici okumaz).
- Gerekçe: gerçek figürlerde DDV tutma seviyesi akımdan tahmin edilen değerden 0.04–0.07 farklı (F9 −0.02 vs +0.02, F11). Mevcut sentetikte akım→DDV doğrusal regresyon R²≈0.9999 idi, yani bu hata sentetikte hiç üretilmiyordu.
- Tek seferlik A/B (seed=621, aynı 7/2/2, yalnız numpy `Rate_*` ailesi, validation tam-uçuş Fit; model araması yapılmadı, test okunmadı): baseline DDV 99.6 / static 31.97 / adaptive 60.03 / slow_null 68.13 (Codex değerleri aynen yeniden üretildi). `gain .03, bias .025, drift .02`: DDV 89.33 / 39.53 / 50.67 / 57.88. `bias .04`: DDV 86.73 / 19.82 / 42.78 / 54.67. Tek örnek; kesin sıralama iddiası yok.
- Kontrol: `tests.test_flightlike` 10 test geçti (yeni: opt-in/aynı fiziği koruma). PyTorch/Ruff bu ortamda yok: NN_joint eğitilmedi, Ruff çalışmadı, yalnız compileall.

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
