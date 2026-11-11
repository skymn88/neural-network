# Horizontal tail actuator — veriye dayalı sistem tanımlama

MATLAB **R2019a + Deep Learning Toolbox** ile uçuş kayıtlarından aktüatör dinamiğini öğrenme projesi.

**Öncelik gerçek uçuş verilerine uygun eğitim, validation ve testtir.** Sentetik veri yalnız yazılım akışını, sayısal davranışı ve kontrollü senaryoları sınayan yardımcı veridir. Gerçek performansın yerine geçmez; mevcut sentetik dosya bu repo düzenlemesinde değiştirilmedi.

```text
Coil Current → DDV modeli → RAM durum modeli → Deflection dönüşümü
```

RAM tarafında varsayılan aday `NL_FB`; `HoldInitial` yalnız `holdBaselines=true` ile açılan referanstır. Eski çok-aileli arama `nlFocus=false` ile korunur. DDV için NL_OE ve NL_FB validation ile karşılaştırılır. NARX aktif eğitimde kullanılmaz. Açı için lineer/kübik geometrik dönüşüm validation'da karşılaştırılır. Ayrıntılar: [modelleme](docs/modeling.md).

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

Cascade eğitimi: `cfg.ramTrainInput='oof'` yalnız TRAIN uçuşları arasında DDV LOFO tahminleriyle OE modellerini eğitir (varsayılan `'oof'`). Üçten az TRAIN uçuşunda uyarıyla ölçülen DDV kullanılır. `cfg.reportLoo=true` yalnız DDV LOFO teşhisini de açabilir. `cfg.oeMaxTrainRows=Inf` tüm uygun örnekleri `oeChunkRows=2048` bloklarıyla işler. `cfg.oeZeroRateAtNull=true` intercept içermeyen ek `FIR_OE_Zero` adayını açar.

Yapısal adaylar (hepsi varsayılan kapalı): `hamEnabled` → HAM_OE; `clampRAM` → TRAIN aralığı + `clampMargin` kadar kıskaç; `useForce` → ek `_F` adayları. `usePressure=true` ile `pressureInteractions=true` ek `_P` adaylarını açar; eski ham basınç yolu korunur, yeni `_P` modelleri yalnız DDV×sqrt(Ps−Pr) kullanır. Eksik kuvvet/basınç kanalı açık hatadır. `hamDeadbands` ve `hamScales` fiziksel DDV birimindedir, kendi sensör ölçeğine göre ayarla.

Kapalı çevrim (izole): adaptöre 64 Hz `RAMSetpoint` bağla ve `run_project('user','closed')` çalıştır. Bu komut `useSetpoint` seçeneğini açar; eksik kanalda hata verir. Varsayılan `run_project('user')` plant akışıdır. Bayrakları dosya düzenlemeden vermek için:
```matlab
opts=struct('diagnostics',true,'ramTrainInput','oof','oeMaxTrainRows',Inf,'hamEnabled',true);
run_project('user','plant',opts);
run_project('user','closed',opts);
```
`ctrlNoiseStd=0` varsayılandır (RAM birimi); `ctrlLags`, `ctrlTaus`, `ctrlRidge` kontrolcü aramasını belirler. `ClosedLoop`/`ClosedSelection` sayfaları ve `closed_models.mat` ayrı üretilir. Sabit sentetik fixture setpoint içermediğinden closed komutu için hazır veri değildir; veri değiştirilmez.

Güncel RAM odağı: `nlFocus=true` varsayılanıyla NL_OE raporlanır; HoldInitial/TrainMean varsayılan kapalıdır (DDV de yalnız NL_OE kullanır). `nlRefine=true`, kaba NL_OE kazananının beta çevresini ve `nlTanhScales=[.5 1 2]` doygunluk ölçeklerini validation'da karşılaştırır; iyileşmezse kaba kazanan korunur. `nlFocus=false` eski çok-aileli aramayı geri açar. Odak modunda eski `hamEnabled`/`oeZeroRateAtNull` seçenekleri RAM aramasını genişletmez; açık force/pressure seçenekleri yalnız NL varyantlarını ekler.

`run_project('user')` sonunda ilk uçuş DDV/RAM/Angle olarak 3×1 çizilir, sonuç klasörüne FIG/PNG kaydedilir. Üç panelde de ölçülen ve tahmin edilen çıktı vardır. `src/bb_plot_flight_outputs.m` içindeki `for i=1` döngüsünü tüm uçuşlar için değiştirebilirsin. Tam-uçuş grafiği split skorunun yerine geçmez; `makePlots=false` çizimi kapatır. Seçilmiş beta/lag/ridge/tanh ölçeği kısa özette görünür.

DDV NL_OE, current geçmişinin doğrusal olmayan bazlarından doğrudan DDV çıktı hatasını ridge LS ile öğrenir (`ddvRidge=1e-3`); RAM integratörü DDV'ye uygulanmaz. Varsayılan eğitim artık Deep Learning Toolbox gerektirmez; önceki MLP aramaları yalnız `nlFocus=false` yolundadır. Açı statik RAM→açı dönüşümüdür.

### Kısa kullanım ve isteğe bağlı NL deneyleri

`run_project()` kullanıcı verisiyle **plant** çalıştırır; setpoint gerekmez. `run_project('user','closed')` yalnız 64 Hz RAMSetpoint bağlandıktan sonra kullanılır. Örnekler `run_project.m` başında da var. Uzun komut yerine `src/bb_config.m` içinde istediğin bayrakları bir kez aç, sonra kısa komutu kullan. Aşağıdaki yeniliklerin tamamı varsayılan **false**:

| Bayrak | comparison.xlsx çıktısı / etkisi |
|---|---|
| `nlDiagnostics` | `BetaCurve`: beta–en iyi validation RMSE noktaları ve kenar uyarısı; `HoldVariability`, `NullEquilibrium`; `NL_Selection` içinde eğitim saniyesi |
| `holdBaselines` | HoldInitial ve TrainMean birlikte geri açılır; ikisi de seçim dışıdır |
| `nlFIRReference` | NL odakta FIR_OE yalnız raporlanır; seçilemez |
| `nlWideBeta` | Mevcut yerel aramaya 2× ve 3× beta büyüklükleri eklenir |
| `nlJointSearch` | beta/lag/ridge/tanhScale ortak yerel araması; en fazla `nlJointBudget=16` ek aday |
| `nlRefitTrainVal` | Seçim sabitlenince TRAIN+VAL ağırlık eğitimi; `RefitTest`, `RefitPerFlight`, `RefitInfo` |
| `nlStateLeak` | TRAIN'de öğrenilen deneysel durum sızıntısı; validation iyileşmezse reddedilir; `StateLeak` |

Teşhis varsayılan açıktır: `run_project()` yeterlidir. İkinci denemede gerekirse `nlWideBeta` ve `nlJointSearch` aç. Refit ve state-leak bağımsız deneylerdir; closed moduna geçmeyi gerektirmez. Mevcut plot/MAT çıktıları korunur, yeni teşhisler için ayrı çıktı dosyası oluşturulmaz. Refit modeli mevcut `trained_models.mat` içine eklenir; testte daha iyi diye orijinal modelin yerine otomatik geçirilmez.

Kaba beta ızgarasını tek seferlik genişletmek için:
```matlab
run_project('user','plant',struct('nlDiagnostics',true, ...
    'betaCandidates',[0 -.003 -.01 -.03 -.1 -.3 -1 -2 -3]));
```
`BetaCurve` bir alt zarf tablosudur: her beta için diğer ayarlar arasındaki en düşük validation hatası; sabit diğer ayarlarla yapılmış kontrollü eğri değildir. İstersen Excel'de Beta/BestValidationRMSE sütunlarını çiz. Eğitim süreleri hata değil maliyet ölçüsüdür, seçimi değiştirmez.

### Geçiş keskinliği deneyi

`run_project()` artık `ramTrainInput='oof'`, `oeMaxTrainRows=Inf`, `holdBaselines=false` ile başlar; uzun override gerekmez. Üçten az TRAIN uçuşunda OOF uyarıyla ölçülen DDV'ye döner. `holdBaselines=true` yalnız istenirse iki sabit referansı geri açar; eski `trainMeanBaseline` bayrağı tek başına onları açmaz.

Yeni seçenekler `src/bb_config.m` içinde:
- `nlDiagnostics=true`: `RateMap` (TRAIN/VAL, 30 kutu) ve `StepSmoothing` (VAL'daki en büyük beş ölçülen geçiş).
- `nlWiener=true`: NL_WH karşılaştırması; `WH_Selection`, `WH_Map`. Varsayılan false. FIR ağırlıkları donuk, monoton harita/beta TRAIN'de öğrenilir; yalnız daha iyi validation RMSE ile seçilir.
- `oeTransitionWeight=[0 2 5]`: lambda=0/2/5 için aynı NL araması. Varsayılan 0. `NL_Selection` ve `EventErrors` lambda bilgisi taşır. Geçiş eşiği TRAIN RAM aralığının `oeTransitionFraction=.01` katı, uzatma `oeTransitionSeconds=1.5` saniyedir.

Kısa kullanım: config'te istediğin seçenekleri aç, `start_here` kontrollerinden sonra `run_project()` çalıştır. Setpoint gerekmez. Smoke, kısa kod kontrolü için 700 örnek bütçesi ve measured DDV kullanır; gerçek run varsayılanlarını değiştirmez. Sabit sentetik MAT dosyası korunur.

### Bant teşhisi, filtre bankası ve sade Excel

Varsayılan workbook artık beş sayfadır: **Summary, PerFlight, Diagnostics, Search, Config**. Eski sayfa isimleri aşağıdaki açıklamalarda artık aynı isimli tablo/blokları ifade eder. `legacySheets=true` eski çok-sayfalı düzeni geri açar. Summary tüm Model×Stage Fit/RMSE değerlerini Train/Val/Test yan yana gösterir; ham metrikler MAT içinde korunur. `print_last_results` yeni düzeni `selection.mat` ile yeniden basar.

- `nlDiagnostics=true`: Diagnostics içinde **BandGain** ve **Alignment** dahil teşhisler. Bantlar 0–0.3, 0.3–1, 1–3, 3–8, 8–32 Hz; validation/test DDV ve RAM component/cascade karşılaştırılır.
- `nlFilterBank=true`: RAM'de **NL_FB** adayı; `ddvFilterBank=true`: DDV'de aynı özellik yapısını karşılaştırır. İkisi de varsayılan true; seçim yalnız validation. DDV aday Fit/RMSE değerleri Diagnostics/DDVPerformance bloğundadır.
- Süreler Command Window sonunda, Summary/Config içinde ve `selection.mat` içindedir. Ana rapor yazımı süreye dahildir; son küçük süre damgası ve MAT güncellemesi ölçümün dışındadır.

Kullanım değişmedi: config'te istediğin bayrakları aç, `start_here`, ardından `run_project()`. OOF, tüm OE örnekleri ve baseline'ların kapalı olması korunur; WH/lambda araması kapalı kalır.

Varsayılan hızlı akış: RAM NL_FB, `betaCandidates=[-.3 0]`, `nlRefine=false`, OOF ve tüm TRAIN satırları. RAM NL_OE karşılaştırması için `nlReference=true`; NL_WH açılırsa ihtiyaç duyduğu NL_OE de eğitilir. WH/lambda araması kapalı kalır. `RAM_NL_FB`, `RAM_NL_OE` ve `DDV_*` süreleri eğitim + validation değerlendirmesini içerir; RAMSearch/DDVTraining alt süreleridir, toplama tekrar eklenmez. Excel yazımı `UseExcel=false` ile boş Excel şablon sayfalarını önler; yeni sonuç klasörüne yazılır.

### Durum bağımlı RAM deneyi
NL_SS başarısız gerçek-veri denemesi sonrasında varsayılan kapalıdır (`nlStateSpace=false`); isteğe bağlı açılabilir. `ssMaxEvals=80` TRAIN serbest-koşu optimizasyon bütçesidir. Seçim yalnız validation; Search/SS_Selection ve RAM_NL_SS süresi raporlanır. Refit, stateLeak ve LOFO ile birlikte kullanımı bu ilk sürümde engellenir. Excel yalnız tamamen boş Sheet1/2/3 sayfalarını kaldırır; dolu sayfalara dokunmaz. MATLAB testleri bu geliştirme ortamında çalıştırılmamıştır; yerelde önce `start_here`, sonra `run_project()` çalıştırın.

### Pulse / hold karşılaştırması ve bağımsız çizim
NL_HOLD başarısız gerçek-veri denemesi sonrası varsayılan kapalıdır (`nlHold=false`); isteğe bağlı açılırsa NL_FB yanında denenir; seçim yalnız validation RAM cascade RMSE. `holdLags=[0 8]`, `holdDeadbandFractions=[0 .02 .05]` (TRAIN DDV std oranı), `holdRidge=[1e-3 .1]`; toplam 12 kapalı-form aday. NL_HOLD beta=0, intercept yok, kısa sonlu DDV hafızası ve asimetrik nonlinear baz kullanır. WH/lambda araması kapalı kalır. Search/Hold_Selection ve RAM_NL_HOLD sürelerini inceleyin.

Yeniden eğitim yapmadan grafik: MATLAB Editor ile `src/plot_saved_flights.m` dosyasını açıp Run. Bu bir script'tir. `for i=1` ilk uçuş, `for i=[2 5]` seçilenler, `for i=1:numel(rawFlights)` hepsini çizer. Boş resultsFolder en son kullanıcı koşusunu bulur; farklı koşu için script başındaki resultsFolder yolunu doldurun. Gerçek veri dosyaları/sırası aynı kalmalıdır; figürler kaydedilmez, açık pencerelerde gösterilir.

### Dinamik MLP ve isteğe bağlı DDV karşılaştırmaları
`run_project()` RAM için NL_FB ve **DMLP** karşılaştırır (`dynamicMLP=true`); NL_SS/NL_HOLD kapalıdır. DMLP küçük tanh ağının bütün ağırlıklarını TRAIN çok-adımlı konum hatasıyla birlikte öğrenir. Varsayılan 4 gizli nöron, 128 örneklik pencere, TRAIN uçuşlarına dağıtılmış en fazla 64 pencere, 200 amaç değerlendirmesi; validation/test tam uçuş serbest koşudur. `Search/DMLP_Selection` pencere hatası, bütçe/çıkış kodu ve validation RMSE; `RAM_DMLP` toplam yöntem süresidir. Pencere hatası tüm-TRAIN Fit değildir. Refit/stateLeak/LOFO deneyi için dynamicMLP kapatılmalıdır. Basınç/force girdileri bu ilk DMLP adayında desteklenmez.

Ek seçenekler **varsayılan false**; eski DDV/FB davranışı korunur:
- `ddvAllSamples=true`: tüm uygun TRAIN satırları uniform, iki geçişli blok normal denklemler; maxTrainRows/mixed örnekleme kullanılmaz. OOF/refit aynı yolu izler. DDVPerformance'ta SampledFit/RMSE ve AllSamplesFit/RMSE yan yana; sampled modeller rapor referansıdır, bu modda seçim all-sample adaylar arasındadır. Ek referans eğitimi süreye dahildir.
- `fbTanhSearch=true`: RAM NL_FB için `nlTanhScales=[.5 1 2]` taranır; varsayılan yalnız 1. Search TanhScale sütunu ve seçili konfigürasyon değeri gösterir.
- `fbSteadyStart=true`: filtre bankası ilk girişin sabit rejiminde başlatılır; bütün sonraki örneklerde durum taşınır.
- `fbUniquePoles=true`: ham lag-0 ile aynı olan sıfır kutbu çıkarılır. Varsayılan sonuçları korumak için bu da opt-in'dir.

Önce `start_here`, sonra `run_project()`. Bağımsız çizim script'i `src/plot_saved_flights.m` değişmedi. DDV gürültüsü için Diagnostics/BandGain yüksek frekans genlik oranlarını da inceleyin; yeni seçeneklerin gürültüyü azaltacağı garanti edilmez.

### Opt-in PulseHold / HorizonFit ve pencereli OE
Model varsayılanları değişmedi: `pulseHold=false`, `horizonFit=false`, `oeWindowSeconds=0`, `oeWindowSearch=false`, `selectionMode='fullFreeRun'`, `holdDroopMax=Inf`. `src/bb_config.m` içinde yalnız istenenleri açıp kısa `run_project()` komutunu kullanın:
- `pulseHold=true`: TRAIN akım std'siyle ölçeklenen `pulseProfile=[.1 2;0 5;-.1 4;0 5]` (genlik çarpanı, saniye) bütün raporlanan RAM adaylarının akım→DDV→RAM→açı zincirine verilir. Diagnostics/PulseHold ve makePlots açıkken tek `pulse_hold.png`. Bu sinyal eğitim verisine eklenmez; fiziksel doğruluk referansı değildir.
- `horizonFit=true`: Diagnostics/HorizonFit; validation/test RAM ve açı için 2/5/10/30 s resetli değerlendirme ve tam serbest koşu. RESET satırlarında her pencerenin başlangıç RAM'i ölçümden alınır, skorlar normal serbest-koşu fidelity olarak yorumlanmamalıdır. Bu teşhis seçim yapmaz.
- `oeWindowSeconds=5`: OE eğitimi 5 s çakışmayan pencereler kullanır. Yalnız integratör resetlenir, giriş hafızası/filtreler kesilmez. `oeWindowSearch=true` ise [2 5 10] s ve beta [0 -.03 -.3] taranır. NL_OE için `nlReference=true`, NL_HOLD için `nlHold=true` ayrıca açılmalıdır; varsayılan NL_FB etkin kalır. NL_HOLD bu deneyde negatif beta da denediğinden bütün varyantları tam tutma garantisine sahip değildir.
- `selectionMode='windowed'`: seçim validation'da `selectionWindowSeconds=10` resetli RAM RMSE ile yapılır. Search içinde FullFreeRunRMSE/ResetWindowRMSE, WindowSeconds, SelectionMode ve adayların Eligibility bilgisi birlikte raporlanır. Test hiçbir seçim hesabına girmez. Tam koşu Summary metrikleri değişmez.
- `holdDroopMax`: sonlu yapılırsa TRAIN ölçekli darbe profilinin beklemelerinde en büyük mutlak **RAM eğimi (fiziksel birim/s)** eşiği aşan aday elenir. Bütün adaylar elenirse eşik sessizce gevşetilmez; reddedilme bilgisi comparison.xlsx/Search içinde yazılıp açık hata verilir. Eşik bir yüzde değildir.

Yeni seçim/eğitim deneyleri NL_OE/NL_FB/NL_HOLD/DMLP/NL_SS yoluyla sınırlıdır; eski çok-aileli arama, FIR referansı, HAM, WH, stateLeak, refit/LOFO ve force/pressure etkileşim deneyiyle birlikte kullanılırsa açık hata verilir. Yeni pencere aramaları ek eğitim/değerlendirme süresi getirir.

Koşu ve kontroller bittiğinde veya hata verdiğinde kullanıcı isteğiyle `bdclose('all')` ve `clear mex` çağrılır; Simulink yoksa ilk adım atlanır, normal grafikler açık kalır. `releaseAfterRun=false` yalnız eğitim koşusunun otomatik temizliğini kapatır; start_here kontrolleri çıkışta temizler. Bu MATLAB kaynaklı kilitlere yöneliktir, dosya sistemi izinlerini değiştirmez.
