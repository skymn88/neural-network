# Python deney kaydı — 2026-11-12

Bu kayıt gerçek CPU eğitimlerinin çıktısıdır. Gerçek uçuş verisi kullanılmadı. Tüm aramalar validation ile yapıldı; test ancak aile/koşu kazananları sabitlendikten sonra açıldı. Testi gördükten sonra yeni eğitim/hiperparametre araması yapılmadı.

`bundled`: değiştirilmeyen mevcut MAT, kendi kayıtlı 7/2/2 split’i. `pulse`: ayrı 11 uçuş, seed=123 üretim ve seed=42 7/2/2 split. İki setin RAM birimleri/ölçekleri farklıdır; RMSE’lerini birbirleriyle kıyaslama.

| Deney | Veri | DDV baz | Ön eğitim / ortak adım | Ufuk / nöron | Kazanan | Val RAM RMSE | Koşu süresi (s) |
|---|---|---|---|---|---|---:|---:|
| E1 | bundled | legacy | 100/80 | 64/16 | NN_joint | 0.0048552549 | 22.13 |
| E2 | bundled | legacy | 320/320 | 128/24 | OE | 0.0054797829 | 39.56 |
| E3 | bundled | legacy | 600/400 | 128/12 | OE | 0.0054797829 | 47.48 |
| P1 | pulse | legacy | 240/240 | 128/16 | NN_joint | 4.7671738 | 65.00 |
| E4 | bundled | rich | 100/400 | 192/16 | NN_joint | 0.0037489652 | 62.85 |
| P2 | pulse | rich | 100/600 | 192/16 | NN_joint | 7.5959598 | 76.78 |
| P1-repeat | pulse | legacy | 240/240 | 128/16 | NN_joint | 4.7671738 | 34.97 |

E2: daha büyük/uzun ağın ön eğitimden sonra validation RMSE’si 748.71 oldu; ortak eğitim 0.00897’ye indirdi, yine OE’den kötü. E3 monoton yapı validation’da 0.00613 ile OE’yi geçemedi. E4’te en iyi ortak checkpoint 100/400; sonraki üç checkpoint daha kötü. P2 daha uzun/zengin eğitimle P1’i geçemedi. P1-repeat aynı validation RMSE’sini yeniden üretti. Bu nedenle bu sınırlı araştırma turu durduruldu; küresel optimum iddiası yok.

Bazı keşif koşuları aynı CPU’da örtüştü. Süreler bu makinedeki duvar saati ölçümleridir; yöntemler arasında kontrollü hız benchmark’ı değildir. Son tekrar toplam 34.97 s, ortak ağ eğitimi 18.02 s sürdü.

## Dondurulmuş son değerlendirme

Her ailede en düşük validation RAM RMSE veren kayıt alındı; ardından aile seçildi. `python/reference_results/` içindeki JSON/XLSX ve taşınabilir modeller bunun çıktısıdır. Son raporda her yöntemin hangi koşudan geldiği `model_provenance` ile kayıtlıdır.

| Veri | Seçilen | RAM Cascade Fit Train/Val/Test | Açı Cascade Fit Train/Val/Test |
|---|---|---|---|
| bundled | NN_joint | 77.05 / 87.62 / 77.53 | 77.04 / 87.60 / 77.52 |
| pulse | NN_joint | 85.09 / 79.23 / 87.76 | 84.84 / 78.48 / 87.40 |

Örnekler 64 Hz. İlk 32 örnek metriklerden çıkarılır. Pencereli eğitim kullanılsa da yukarıdaki skorlar tam uçuş serbest koşudur; sadece ilk ölçülen RAM konumu başlangıçta verilir, gerçek DDV/RAM ile düzeltme yapılmaz. Açı Component’in sentetik veride ~100 olması dönüşümün deterministik olmasındandır. Cascade fidelity’sini göstermez.

Kayıtları yeniden üretmek için `python/run_experiment.py` argümanları ve her deneydeki `protocol.json` kullanılır. PyTorch/sürüm/platform farkları bit düzeyinde aynı sonucu garanti etmez. E1–E3 eski fold-ridge protokolü birebir geriye uyarlanmadı; kaydedilmiş modeller ve sonuçlar korunuyor. P1’in sinir ağı sonucu güncel kodla tekrarlandı.

## Çalıştırılan kontroller

- Python unittest: 18 deterministik kontrol; gerçek kısa eğitim, test verisini değiştirme izolasyonu, OOF dışlama, normal denklemler, NumPy/PyTorch çıkarım denkliği, CSV/API ve XLSX sayfa sayısı.
- Chromium: CSV yükleme, açık split, gerçek OE eğitimi, üçlü grafik, yakınlaştırma, XLSX indirme; JavaScript hatası yok.
- Ruff ve compileall; MATLAB MISS_HIT R2019a statik kontrolü: 120 dosya.
- MATLAB/Simulink ve Windows başlatıcısı bu Linux ortamında çalıştırılmadı. C/MEX üretilmedi.
- Eski MAT SHA256: `22fab73b38de0f4b40da981f21f10bf55b02016da40b13ec7f6b4fb09a75c8fb`.


## Py-0.3 · FlightLike, 10 deney — 2026-11-12

Fotoğraflar ve kullanıcı tarafından iletilen yorumlar profil tasarımında kullanıldı; fotoğraftan sayısal veri çıkarılmadı. Gerçek TRAIN istatistikleri elimizde yok. Bu, gerçek uçuşlarla sentetik ekleme A/B'si değil, daha zor bir sentetik yazılım deneyi. Önceki sentetik setle Fit doğrudan karşılaştırılamaz.

Mevcut `pulse_flights` üreteci opt-in `profile='flightLike'` ile genişletildi. 11 uçuş, 2100–2800 s, Fs=64 Hz, seed=621, aynı 7/2/2 split. Uçuşa göre null/ofset/band ve yavaş sürüklenme, park, yüzlerce saniye bekleme, dikdörtgen/yamuk/ters darbeler, titreşim paketleri. RAM ±100, statik azalan açı yaklaşık +33…−25 derece. Birimler gerçek veriden doğrulanmadı.

Üreteç arama başlamadan sabitlendi. RAM hızın integrali; gerçek null bandında hız sıfır, hız ve konum sınırları var. Parktan null'a yalnızca geçmek RAM'i hareket ettiremez: çalışma seviyesine kısa ters darbe eklenir. Yavaş ilk hareket, büyük park DDV'si altında yapay yavaşlama olarak değil, önce near-null komut olarak oluşturuldu. İlk konumlar bu sürümde −90…−60 aralığından; ikisi fotoğraftaki −95…−80 aralığında, birebir fotoğraf eşleştirmesi değildir. İçerideki hız katsayıları öğreniciye verilmedi.

Durma kuralı: **pooled tam-uçuş validation RAM Cascade Fit >=95 veya 10 aday**. Bir adayın optimizer adımları/checkpoint'leri ayrı araştırma iterasyonu sayılmadı. Liste başlangıçta kaydedildi. 10 aday tamamlandı, %95 elde edilmedi ve duruldu. Test yalnız kazanan/durma kararı dondurulduktan sonra hesaplandı. Sonraki kontroller unit test ve rapor/arayüz kontrolleridir; ek performans araması yapılmadı.

| Deney | Aday | Val RAM Cascade Fit | Val RMSE | Eğitim + val süresi (s) |
|---:|---|---:|---:|---:|
| 1 | NN_joint | 18.87 | 66.825 | 22.28 |
| 2 | Rate_static | 31.97 | 56.031 | 2.14 |
| 3 | Rate_adaptive | 60.03 | 32.920 | 2.30 |
| 4 | Rate_fast_null | 13.41 | 71.318 | 2.42 |
| 5 | Rate_dense | 48.79 | 42.179 | 2.31 |
| 6 | Rate_delayed | 46.31 | 44.221 | 2.38 |
| 7 | Rate_smooth | 48.78 | 42.184 | 2.35 |
| 8 | Rate_slow_null | 68.13 | 26.250 | 2.25 |
| 9 | Rate_no_delay | 48.80 | 42.174 | 2.37 |
| 10 | Rate_coarse | 26.02 | 60.936 | 2.29 |

Kazanan `Rate_slow_null`: RAM Cascade **71.18 / 68.13 / 83.34**, açı Cascade **71.59 / 68.42 / 83.49** (Train/Val/Test). DDV **99.62 / 99.60 / 99.63**. RAM Component **70.64 / 66.76 / 84.34**. Açı Component ~99.93: statik geometriyi öğrenmek RAM dinamiğini çözmüş olmak değildir.

Yeni aile sinir ağı değildir: TRAIN'den öğrenilen parçalı doğrusal hız haritası + sızıntısız konum integrali + nedensel null izleyici. Sonuç iyileşmesinin tümünü null'a bağlamıyoruz; mevcut NN referansından farklı olarak TRAIN'den öğrenilmiş konum sınırı da uygulanır. Aynı sınırları kullanan statik-null aile Val Fit %31.97, uyarlanan en iyi aile %68.13 verdi.

Seçili ailede validation uçuşlarının sentetik hold zamanlarının en az %90'ında tahmin hızı sıfır çıktı; buna karşın bazı hareketlerde yanlış seviyeye gidilip o seviyede kalınıyor. Null hata RMSE'si, gate içindeki örneklerde ~0.003–0.007 DDV birimi. Bu maske hareket sırasında gate'den geçişleri de içerebilir; saf gerçek hold maskesi değildir. Gözlenemeyen bağımsız null sürüklenmesini yalnız akımdan kesin bulma iddiası yok: burada holding current gerçek null ile birlikte kayıyor ve gözlenebilir bilgi sağlıyor.

Toplam ölçülen süre **113.40 s** (son workbook yazımı hariç), NN adayı **22.28 s**, seçili hız adayı **2.25 s**, tüm modellerin son değerlendirmesi **56.22 s**. Aynı ortamın duvar saati; gerçek veri hız garantisi değil.

Kayıt: `python/reference_results/20261112_062616_313440/`. Tüm adayların modelleri, Search/Diagnostics ve XLSX repoda. 51 MiB ham kayıt ve 29 MiB Truth NPZ çalışma alanında korundu; son kullanıcı isteğiyle ZIP hazırlanmadı. Git geçmişine büyük ham kopya eklenmedi. Üreteç son koduyla aynı veri fingerprint'ini yeniden verdi. Truth yalnız teşhis ve üretici testleri için; model eğitimi/çıkarımında kullanılmadığı ayrıca sınandı.

Opt-in `syntheticTrainFlights=N` gerçek veri geldiğinde aynı ayarlarla 0 ve N için A/B çalıştırır. Kalibrasyon yalnız gerçek TRAIN, sentetik yalnız TRAIN'e; validation/test aynı gerçek uçuşlar. Son XLSX Diagnostics'te tek A/B fark satırı. **Gerçek uçuş A/B'si bu ortamda yapılmadı.**

Kontroller: 28-test tam regresyon koşusu geçti; ardından ek A/B seçim-sırası testi dahil 9 yeni-alan testi geçti (toplam 29 farklı test). NumPy çıkarım, Truth izolasyonu, integral/neutral tutma, monoton açı, causal observer ve holdout izolasyonu sınandı. Yerel Chromium'da yeni kaydın 10 adayı/üçlü grafiği/uçuş seçimi kontrol edildi, yeni eğitim başlatılmadı. MATLAB ve Windows .bat çalıştırılmadı. Canlı mobil Site bu tur güncellenmedi.


## Py-0.4 · Python 3.9, üç observer denemesi — 2026-11-12

Aynı flightLike TRAIN/VAL, aynı veri fingerprint'i ve önceki donuk DDV/açı blokları. Python **3.9.25**, NumPy **1.26.4**, SciPy **1.13.1**, Torch CPU **2.6.0** ortamında çalıştırıldı. Eski MATLAB kaynakları bu daldan kaldırıldı; modeller/split değiştirilmedi. Başlangıçta üç aday ve %95 validation durma eşiği belirlendi.

| Aday | Val RAM Cascade Fit | Val RMSE | Eğitim + val (s) |
|---|---:|---:|---:|
| Rate_quiet | 91.01 | 7.408 | 4.54 |
| Rate_envelope | 93.47 | 5.380 | 5.31 |
| Rate_windowOE | 91.50 | 6.997 | 3.97 |

Önceki Rate_slow_null Val Fit **68.13**, yeni validation kazananı **Rate_envelope 93.47**. RAM Cascade Train/Val/tarihsel-Test **93.50 / 93.47 / 95.18**; açı Cascade **93.19 / 93.46 / 95.41**. Üç aday tamamlandı, validation %95 olmadı, arama durdu. Tarihsel test önceki turda görüldüğü için yeni kör test diye sunulmaz; bu turdaki seçime dahil edilmedi.

Seçim dondurulduktan sonra seed=9621, süreler 2400/2700 s ile **iki yeni sentetik audit uçuşu** üretildi. RAM Cascade Fit **93.67**, RMSE **5.554**; açı Cascade Fit **93.80**. Audit arama veya yeniden eğitime verilmedi. İki uçuş ve aynı simülatör ailesi gerçek uçuş genellemesi kanıtı değildir.

Önemli açık sorun: **RAM Component Fit 25.06 / 48.40 / 68.02**. Ham DDV ölçüm gürültüsü, tahmin edilen daha düzgün DDV ile kalibre edilmiş sakinlik/tepe izleyicisini bozabiliyor. Cascade kazancını tüm giriş koşullarında sağlamlık diye yorumlamıyoruz. Açı Component ~99.93 olması da RAM'in doğru olduğunu göstermez.

Rate_quiet, null güncellemesini TRAIN'den öğrenilen sakinlik aralıklarına kısıtladı. Rate_envelope, iki yönlü osilasyonların tepe/çukur orta noktasıyla merkez drift'ini izledi. Rate_windowOE, 5 s TRAIN konum pencereleriyle hız haritasını yeniden ağırlıklandırdı; kazananı geçemedi. Simülatör Truth'u hiçbir eğitim girdisi olmadı.

Toplam **47.85 s**, seçili aday **5.31 s**; DDV/açı zaten eğitilmiş olarak donuktu. Sonuç kaydı: `python/reference_results/20261112_105714_655119/` (XLSX, tüm adayların JSON modeli, arama tablosu, yeni audit metrikleri).

Kontroller: Python 3.9'da 31-test regresyon koşusu + 2 çevrimdışı/observer yönlendirme kontrolü geçti. `--no-index` pip kontrolü mevcut sabitlenmiş ortamda ağ kaynağı olmadan geçti; bu Windows wheelhouse kurulumu çalıştırıldı demek değildir. Python 3.9 sunucusunda Chromium ile yeni sonuç/üçlü grafik/yöntem seçeneği doğrulandı; yeni eğitim başlatılmadı. PPTX: 16 slayt, 4 gömülü düzenlenebilir grafik çalışma kitabı, ZIP/XML ve slayt sınırları doğrulandı. Windows .bat ve Microsoft PowerPoint açılışı bu Linux ortamında çalıştırılmadı. Canlı mobil Site değiştirilmedi.
