# İlk repo düzenlemesi

Başlangıç kaynağı yerel Ver10 paketidir. Eğitim adayları, öğrenme denklemleri ve mevcut sentetik verinin içeriği değiştirilmedi. Yol çözümleme merkezi `bb_paths` ile yeni klasör düzenine uyarlandı.

## Aktif repoya alınmayanlar

- Eski v5/v6 çalıştırıcıları: `run_comparison`, `run_clean_comparison`.
- Aktif yolun çağırmadığı `bb_angle_fit`, `bb_evaluate`, `bb_predict`, `bb_rate_beta`.
- Doğrulanmamış eski `bb_sfun` prototipi. C/Simulink daha sonra ayrı geliştirilir.
- Eğitim seçimine katkısı olmayan `Frozen_v6` karşılaştırması ve ağırlıkları.
- Tekrarlanan v6 rate fixture testi; güncel birim testleri, cascade ve OE fixture'ları korundu.
- Tarihsel Python karşılaştırma kopyaları, pickle model arşivleri, results klasörleri, loglar, XLSX deney çıktıları, ZIP'ler ve sürüm bazlı README tekrarları.

Bunlar önceki yerel sürüm klasörlerinde korunuyor; Git reposuna taşınmıyor. Git geçmişi bundan sonraki değişikliklerin kaydı olacak.

## Korunan yardımcı sentetik veri

`data/synthetic/synthetic_rich.mat` mevcut 11-uçuş fixture'ıdır. `generation.json` üretim varsayımlarını taşır. Üretici `tools/generate_synthetic.py` altında bağımsız hale getirildi; bu düzenleme sırasında çalıştırılmadı. Fixture'ı değiştirmek gerçek veride model doğruluğunu artırmak anlamına gelmez.

## Yerel veriler

Gerçek uçuş CSV/MAT dosyaları `data/user/`, özel dosya yolları/sütun eşlemeleri `load_user_flights_local.m`, üretilen çıktılar `results/` altında tutulur ve Git'e eklenmez. Uzak repoya yalnız bilinen sentetik/fixture dosyaları gönderilir.
