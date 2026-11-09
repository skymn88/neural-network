# Horizontal tail actuator — veriye dayalı sistem tanımlama

MATLAB **R2019a + Deep Learning Toolbox** ile uçuş kayıtlarından aktüatör dinamiğini öğrenme projesi.

**Öncelik gerçek uçuş verilerine uygun eğitim, validation ve testtir.** Sentetik veri yalnız yazılım akışını, sayısal davranışı ve kontrollü senaryoları sınayan yardımcı veridir. Gerçek performansın yerine geçmez; mevcut sentetik dosya bu repo düzenlemesinde değiştirilmedi.

```text
Coil Current → DDV modeli → RAM durum modeli → Deflection dönüşümü
```

RAM tarafında aktif adaylar `FIR_rich`, `FIR_OE`, `NL_OE`; `HoldInitial` basit referanstır. DDV ayrı seçilir. NARX aktif eğitimde kullanılmaz. Açı için lineer/kübik geometrik dönüşüm validation'da karşılaştırılır. Ayrıntılar: [modelleme](docs/modeling.md).

## Gerçek uçuşlarınla başla

1. Repoyu indir/clone et ve MATLAB Current Folder'ı repo köküne getir.
2. `examples/load_user_flights_local.m` dosyasını köke **`load_user_flights_local.m`** adıyla kopyala.
3. Yerel dosyadaki dosya listesini ve `readtable` sütunlarını kendi verine göre düzenle. Verileri `data/user/` altında tutabilirsin. Bu klasör ve yerel adaptör Git tarafından yok sayılır.
4. Çalıştır:

```matlab
setup_project
out = run_project('user');
```

`run_project()` de varsayılan olarak kullanıcı verisini kullanır. Gerçek veri henüz bağlanmamışsa açık bir hata verir; otomatik olarak sentetik eğitime geçmez.

Önceden düzenlediğin `load_user_flights.m` varsa içeriğini yerel adaptöre taşı ve fonksiyon adını `load_user_flights_local` yap. Takip edilen `load_user_flights.m` sabit yönlendirici olarak kalsın.

### Veri sözleşmesi

Her uçuş bir `Raw(k)` elemanı; dört zorunlu sinyal:

```matlab
Raw(k).name = 'flight_01';
Raw(k).CoilCurrent = bb_signal(t_current, current);
Raw(k).DDV         = bb_signal(t_ddv, ddv);
Raw(k).RAM         = bb_signal(t_ram, ram);
Raw(k).Deflection  = bb_signal(t_angle, angle);
```

- Her sinyal kendi zaman vektörünü kullanabilir. Aynı uçuşta ortak zaman referansı gerekir; her sinyali ayrı sıfırlama. Sayısal zamanlar saniyedir; `datetime`/`duration` da desteklenir.
- Birimler, sensör sıfır noktaları ve yön işaretleri uçuşlar arasında tutarlı olmalıdır. Otomatik mm↔m dönüşümü yapılmaz.
- Eğitim ekseni 64 Hz'dir; şimdilik RAM'in 64 Hz kaydı tercih edilir. Yüksek hızlı kayıtlar anti-alias filtreyle hazırlanır.
- Force aktif dört-sinyal zincirinde zorunlu değildir. Desired komutlar kullanılmaz.
- Basınç kullanmak için `src/bb_config.m` içinde `usePressure=true` yap ve `SupplyPressure`, `ReturnPressure` alanlarını bağla. Bu özellik ayrıca gerçek veriyle doğrulanmalıdır.

## Bölme, seçim ve hedef

| Uçuş sayısı | Eğitim / validation / test |
|---|---|
| 1–3 | Her uçuşta kronolojik %70/%20/%10, guard aralıklarıyla |
| 4–9 | N−2 / 1 / 1 uçuş |
| 10 | 7 / 2 / 1 |
| **11** | **7 / 2 / 2** |
| Diğer ≥10 | Yaklaşık %70/%20/%10; 20 için 14/4/2 |

Bütün adaylar aynı split'i kullanır. Seçim **validation serbest çalışma hatasına** dayanır; test seçim için kullanılmaz. Tek uçuşun zaman kesitleri bağımsız uçuş validasyonu değildir. Uçuş sayısı değişince seed aynı olsa da atamalar değişebilir; `Split` sayfasını kontrol et.

%95 üzeri training Fit bir geliştirme hedefidir, elde edilmiş başarı iddiası değildir. Training yükselirken validation/test düşük kalırsa model kabul edilmez. Uzun beklemelerin toplam skoru şişirmemesi için geçiş/bekleme hataları ayrıca raporlanır.

## Kontroller ve yardımcı komutlar

| İşlem | MATLAB komutu / dosya |
|---|---|
| Unit kontroller + fixture karşılaştırmaları + kısa smoke | `start_here` veya `start_here.m` → Run |
| Yalnız unit testler | `setup_project; test_comparison; test_rate_model; test_rich_model; test_oe_model; test_compact_summary` |
| Tam sentetik kontrol eğitimi | `run_project('synthetic')` |
| Gerçek/sentetik uçuşları üst üste çizmek | `plot_inputs.m` → Run |
| Eski XLSX'i yeniden eğitim olmadan dar basmak | `print_last_results.m` → Run |

Smoke, azaltılmış eğitim bütçesiyle kod akışını sınar; model doğruluğu karşılaştırması değildir. `start_here` yalnız test/sentetik yolunu kullanır, gerçek CSV'lere erişmez. MATLAB kullanımı için Python kurulumu gerekmez.

Grafikte `syntheticScale` ve `syntheticOffset` yalnız bilinen sensör kalibrasyonları için kullanılmalı. Sentetik birimler A / normalize DDV strok / RAM m / açı derece. Grafiği benzetmek için dönüşüm tahmini yapma.

## Sonuçlar ve ayarlar

Çıktılar Git'e eklenmeyen `results/` altında her çalıştırmada ayrı timestamp'li klasöre yazılır:

- `comparison.xlsx`: `Compact`, `Summary`, `PerFlight`, `Split`, `RunInfo`, `DataAudit`, `EventErrors`, aday seçim sayfaları.
- `summary_short.txt`: Train/Validation/Test yan yana, dar ekran özeti.
- `trained_models.mat`, `selection.mat`: modeller, normalizasyon, seçilen yapı ve ayarlar.

`src/bb_config.m`: örnekleme, veri bölme, beta/OE lag/ridge seçenekleri. `src/bb_rich_config.m`: DDV adaylarının eğitim bütçeleri. Geniş OE araması ve tam eğitim smoke'tan belirgin uzun sürebilir.

## Repo düzeni

```text
readme.md                    kullanım ve proje amacı
run_project.m                gerçek/sentetik tam eğitim girişi
start_here.m                 kontroller + smoke
plot_inputs.m                uçuş karşılaştırması
print_last_results.m         dar rapor
load_user_flights.m          yerel veri adaptörüne yönlendirici
setup_project.m              yalnız src/tests yollarını ekler
src/                         aktif eğitim, inference, preprocessing, raporlama
tests/                       aktif unit/smoke ve iki sabit inference fixture'ı
data/synthetic/              tek sabit 11-uçuş test verisi
data/user/                   yerel uçuş kayıtları; Git dışı
examples/                    yerel adaptör şablonu
docs/                        modelleme, doğrulama ve düzenleme notları
tools/                       isteğe bağlı sentetik üretici; eğitim için gerekmez
results/                     üretilen sonuçlar; Git dışı
```

## Doğrulamanın gerçek durumu

MATLAB R2019a hedefli **statik sözdizimi kontrolü** ve paket/dataset/dependency kontrolleri bu ortamda yapılır. **Gerçek MATLAB unit/smoke/eğitim, gerçek uçuş validasyonu ve Simulink/C MEX çalıştırması burada yapılmadı.** Fixture'lar önceki Python çıkarımından gelir; MATLAB eğitiminin doğrulandığı anlamına gelmez. [Test kapsamı ve sınırlamalar](docs/validation.md).

C/Simulink dağıtımı daha sonraki aşamadır; eski doğrulanmamış S-Function prototipi aktif repodan çıkarıldı.

## Geliştirme akışı

Çalışmalar `dev_Codex` dalına commit/push edilir. Model değişiklikleri gerçek veri teşhisi ve validation kanıtıyla yapılır. Dosya adları küçük harfli ve Windows'ta çakışmayacak şekilde farklıdır. Eski ZIP'ler, sürüm kopyaları ve deney sonuçları repoya yığılmaz. [Bu düzenlemede ayrılan dosyalar](docs/repository_cleanup.md).

Teşhis için `cfg.diagnostics=true`: seçili modelin beş aşaması, `Alignment` sayfası ve hold hata eğimi. `cfg.printTestForAll=false` kompakt tabloda diğer adayların test hücrelerini gizler; ayrıntılı XLSX korunur. Varsayılanlar `false` / `true`. Ayarlar `src/bb_config.m` içindedir.
