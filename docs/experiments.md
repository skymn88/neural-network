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
