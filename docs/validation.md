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
