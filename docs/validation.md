# Test kapsamı

## Çalıştırılabilen kontroller

- MATLAB 2019a dil hedefinde MISS_HIT statik parser/lint. MATLAB'ın kendisi değildir; toolbox davranışını doğrulamaz.
- Repo bütünlüğü: dosya adı çakışmaları, aktif yerel bağımlılıklar, gerekli fixture'lar, Git ignore kuralları.
- Sabit sentetik fixture'ın 11 uçuş, 64 Hz, sonlu değer ve ayrık 7/2/2 ataması kontrolleri. Veri bu düzenlemede yeniden üretilmedi.
- Önceki Python çıkarımından gelen bağımsız cascade/OE fixture'larının dosya/yapı kontrolleri.

## MATLAB'da çalıştırılacaklar

`start_here`, ham test verisini, split politikasını (N=1..20), metrikleri, FIR/GP forward hesabını, integratörü, kalibrasyonu, OE birim/ofset tutarlılığını, nonlinear baz çıkarımını ve kompakt raporu kontrol eder. Ardından iki inference fixture'ını ve kısa smoke eğitimini çalıştırır. Başarısızlıkta durur; log `results/checks_TIMESTAMP/` altında kalır.

Bu bulut oturumunda MATLAB, gerçek MATLAB eğitimi/smoke veya gerçek uçuş validasyonu çalıştırılmadı. Repo içinde gerçek MATLAB PASS iddiası veya doğrulanmış yeni fidelity sonucu yoktur. Statik ve fixture kontrolleri bu eksikliği kapatmaz.

## Veri ve kabul sınırları

- Sentetik verinin 11 uçuşu yalnız test desteğidir; gerçek uçak parametrelerine kalibre değildir.
- Eğitim hedefi %95+ Fit olabilir, fakat tek başına kabul kriteri değildir. Validation/test, fiziksel RMSE, maksimum hata, bekleme sürüklenmesi ve geçiş hatası birlikte incelenir.
- Model/ayar seçimi validation ile yapılır. Teste tekrar tekrar bakarak geliştirme yapılmışsa sonunda yeni bağımsız uçuşlarla değerlendirme gerekir.
- 1–3 uçuşta zaman kesitleri bağımsız uçuş testi değildir.
- Basınç seçeneği ve gerçek zamanlı preprocessing ayrıca doğrulanmalıdır.
- C MEX ve Simulink entegrasyonu bu repo düzenlemesinde üretilmedi/doğrulanmadı.

## İsteğe bağlı aşamaların kontrolleri

Yeni deterministik MATLAB testleri `bb_run_checks` içine eklendi: gecikme işareti/sabit sinyal, hold eğimi, test sütunu gizleme; yoğun/bloklu OE denkliği ve örnek sayısı; OOF hedef sızıntısı; HAM baz/gecikme sırası, force/pressure zorunluluğu ve kıskaç; LOFO raporu; kontrolcü özellikleri, RNG tekrarı, hedef/akım/DDV olmadan kapalı çevrim çıkarımı ve ıraksama. Bunlar bu ortamda çalıştırılmadı.

Gerçekte çalıştırılan: MISS_HIT R2019a statik kontrol, Git diff/bağımlılık/dosya adı kontrolleri, SHA256 kontrolü; bağımsız Python/NumPy/SciPy hesabında bloklu filtre/normal denklem birikiminin yoğun hesapla sayısal denkliği. Python kontrolü MATLAB kodunu çalıştırmaz ve eğitim başarısını doğrulamaz.

MATLAB'da önce `start_here`, sonra gerçek adaptörü bağlayıp varsayılan `run_project('user')`; ardından README'deki bayrakları sırayla açarak aynı split üzerinde karşılaştır. `run_project('user','closed')` için 64 Hz RAMSetpoint gerekir. Mevcut sentetik MAT değiştirilmedi; closed veri eksikliğinde açık hata beklenir. İlk çalıştırma sonuçlarını runtime doğrulaması için paylaş.

NL odak güncellemesinde `test_nl_focus`: kaba kazananın korunması, minimum validation satırının seçimi, tanh ölçeğinin çıkarım denkliği ve üç eksenli ilk-uçuş grafiği için eklendi. MATLAB/graphics çalışma testi burada yapılmadı. 66 MATLAB dosyası statik kontrolden geçti; sentetik MAT hash'i değişmedi.

Yeni opt-in NL testleri: kapalı varsayılanlar, ortak arama bütçesi, beta zaman sabiti/denge, TrainMean x0 bağımsızlığı, sıfır leakage ile eski denklik, validation reddi, hedef olmadan çıkarım, sabit hiperparametreli refit, yalnız test etiketleri ve gruplanmış OOF sızıntı kontrolü. MATLAB bulunmadığından bu testler burada çalıştırılmadı; statik kontrol ve bağımsız Python sayısal kontrolleri ayrı raporlanır.

Bu güncellemede gerçekte geçen kontroller: 74 dosyada R2019a MISS_HIT, bağımlılık/yeni dosya/test bağlantısı kontrolleri ve değişmeyen sentetik SHA256; Python referans hesabında sıfır sızıntı denkliği, nonlinear dengeye yakınsama ve beta=−1 zaman sabiti. MATLAB eğitimi, unit/smoke ve gerçek uçuş validasyonu çalıştırılmadı.

Geçiş/NL_WH testleri: sıfır ve pozitif ağırlıkta yoğun LS–blok denklik, lambda aday kapsamı, monoton/sıfır ankrajlı/asimetrik doygun harita, vectorize rollout–tek-adım denklik ve x0, sabit beta refit, hedef olmadan çıkarım, RateMap örnek sayısı ve StepSmoothing süre hesabı. MATLAB yok; yeni testler yalnız statik olarak kontrol edilir.

Bu güncellemede 81 MATLAB dosyası R2019a statik kontrolden geçti. Bağımsız Python/NumPy/SciPy referansında monoton harita, kararlı beta parametrelemesi, vectorize/rekürsif rollout ve lambda 0/2/5 bloklu ağırlıklı normal denklem denklikleri çalıştırıldı. Bunlar MATLAB çalıştırması veya gerçek veri başarısı değildir. Sentetik MAT SHA256 aynı kaldı.
