# Python modelleme ve veri sözleşmesi

MATLAB kaynakları bu Python dalından kaldırıldı; [dev_Codex dalında](https://github.com/skymn88/neural-network/tree/dev_Codex) korunur. Python yeni bir deney hattıdır; MATLAB ile sayısal eşdeğerlik iddiası yoktur. Hedef fiziksel aktüatör parametrelerini beslemek değil, ölçülen giriş/çıktılardan nedensel dinamiği öğrenmektir.

## Veri → eğitim → seçim

`actuator/data.py`: her uçuş zaman + dört sütun (`CoilCurrent, DDV, RAM, Deflection`). Bağımsız zamanlar temizlenir/sıralanır; ortak aralık 64 Hz'e alınır. Yukarıdan örnekleme öncesi polyphase anti-alias uygulanır. Uzun boşluklar sessizce doldurulmaz. İçe aktarma sayısal saniye ister; birimler otomatik değiştirilmez. Sabit/çok düşük varyanslı hedeflerde Fit yanıltıcı veya tanımsızdır; RMSE ve uçuş bazlı std birlikte okunmalıdır.

Split politikası README'dedir. MAT'in saklanmış split'i aynen kullanılır. Diğer veri kaynaklarında sabit seed ya da açık uçuş etiketleri vardır. 1–3 uçuşta kesitlerin sınırlarından birer saniye çıkarılır. Normalizasyon, katsayılar ve ön eğitim yalnız TRAIN'den; hiperparametre/checkpoint/aile seçimi yalnız validation'dan gelir. Test etiketleri seçim fonksiyonlarına verilmez. Testi değiştiren otomatik kontrol aynı eğitilmiş ağırlıkları doğrular.

## Adaylar

**DDV:** 0..16 akım gecikmesi üzerinde ridge LS. Kompakt baz `z,tanh(z),tanh(z/2),abs(z)`; zengin baz `z,abs(z),tanh(z/s)` (s=.5,1,2,4,8) ve iki soft deadband (.05,.15). TRAIN akım std'siyle ölçeklenir. Ridge `1e-7,1e-5,1e-3`; validation DDV RMSE seçer. Ham geçmiş uçuşun ilk akımıyla doldurulur. Normal denklemler uçuş bazında tüm örneklerle biriktirilir; çok uzun tek uçuşların özellik matrisi yine bellekte tutulur.

**OE referansı:** DDV 0..8 lag + `z,tanh(z),abs(z)` ve sabit terim; `x[k]=(1+beta/64)*x[k-1]+drive[k]/64`. Beta `0,-.03,-.3`, ridge `1e-6,.001,.1`. Konum çıktı hatası kapalı form LS ile öğrenilir. >=3 TRAIN uçuşunda upstream DDV leave-one-flight-out; daha azsa ölçülen DDV ve açık uyarı. Fold ağırlıkları yalnız diğer TRAIN uçuşlarından; aile/ridge seçimi validation'da yapıldığı için bu iç içe bağımsız CV skoru değildir.

**NN_teacher / NN_joint:** üç girdi `[DDVhat/stdDDV, (xhat-meanRAM)/stdRAM, vhat/stdVelocity]`, bir tanh gizli katman ve skaler hız artışı çıktısı. `vhat[k]=vhat[k-1]+stdVelocity*NN(...)`, `xhat[k]=xhat[k-1]+vhat[k]/64`. Fiziksel katsayı/gain verilmez. Ön eğitim ölçülen DDV/RAM/hız ile hız artışını öğrenir. Ortak eğitim TRAIN pencerelerinde DDV tahminini kullanır; RAM/hız bundan sonra tahmin edilerek ilerler. Kayıp: normalize konum + .2 hız + .2 hold hız hatası + **2 DDV hata kaybı**. Gerçek DDV ile tahmin farkı böylece doğrudan öğrenmeye katılır; DDV ağırlıkları da güncellenir. Arayüzde ön eğitim adımı, ortak adım, ufuk, nöron sayısı, baz ve seed ayarlanabilir.

Pencere başlangıç RAM/hızı yalnız TRAIN pencerelerinde ölçümden alınır. Raporlama/validation/test **tam uçuş**, başlangıç RAM ölçümden, başlangıç hız=0. Gerçek ara çıktılarla sürekli düzeltme yapılmaz. Dört validation checkpoint'inden en iyisi saklanır; ön eğitim daha iyiyse `NN_joint` onu korur (`training.teacher_fallback=true`). NN_teacher ayrı raporlanır. Gürültülü gerçek RAM'in farkını almak hız hedefini gürültülü yapabilir; gerçek veri geldiğinde bu ayrıca teşhis edilmeli.

**NN_stable:** iki öğrenilmiş monoton tanh haritası, öğrenilmiş giriş yönü, negatif konum geri beslemesi ve konveks hız hafızası. Simülatör parametreleri verilmez. İsmi bütün koşullarda kararlılık garantisi değildir; sert kısıtlar gerçek pulse-hold davranışını sınırlayabilir. Bu turda genel ağdan iyi olmadı; varsayılan yönteme eklenmedi, isteğe bağlı karşılaştırmadır.

**Açı:** TRAIN RAM–açı derece 1/3 polinomları; derece validation ölçülen RAM hatasıyla seçilir. Component iyi, Cascade kötü olabilir: component RAM hatasını içermez. Skorların ortalaması fidelity değildir; nihai akımdan açı tahmini `Angle_Cascade` ile değerlendirilir.

## Çıkarım ve gelecekte C

`actuator/inference.py` yalnız **NumPy** import eder. PyTorch/SciPy/eğitim verisi gerekmez. `models.json`:

- `ddv`: `scale,lags,basis,weights` (ridge eğitim metadatası).
- `angle`: `scale,weights`, artan polinom derecesi sırası.
- `ram.kind=oe`: `beta,scale,lags,weights`. Durum bir RAM ve 8 DDV gecikmesi.
- `ram.kind=neural`: `W1,b1,W2,b2,x_mean,x_scale,v_scale,ddv_scale`. Durum **RAM ve hız**; DDV için 16 akım gecikmesi. Bu deney önceki tek-RAM-durumu OE'den farklı olarak iki durum kullanır.
- `ram.kind=monotone`: iki haritanın pozitif ağırlıkları (`g1,g2`), `direction,leak,offset,memory`, aynı ölçekler. Durum RAM+hız.

İlk RAM örneği verilen x0'dır, hız sıfırdır. Devamında yalnız güncel akım ve durum/tampon gerekir. Büyük ıraksama NaN olarak işaretlenir; başarıya benzeyen fiziksel kıskaç uygulanmaz. Şu an batch NumPy referans çıkarımı vardır; C tek-adım API, 200 Hz arayüz ve MATLAB/Simulink eşdeğerlik testleri henüz yoktur. NN transferi yalnız ağırlık/durum denklemi gerektirir, PyTorch runtime gerekmez.

## Rapor ve yeniden üretim

`python/results/<UTC timestamp>/`: beş sayfalı `comparison.xlsx` (Summary, PerFlight, Search, Diagnostics, Config), models.json, result.json, hazırlanmış ham flights.npz ve uçuş isimleri. Bu klasör Git dışındadır. PulseHold sentetik tanı, gerçek doğruluk hedefi değildir; beklemede RAM/açı eğimini raporlar, seçimi etkilemez. Metrikler ilk32 örneği atlar. Toplu Fit = `100*(1-sqrt(sum(SSE_flight)/sum(SST_flight)))`; uçuş Fit'lerinin aritmetik ortalaması değildir. Sıfır varyansta Fit boş, ıraksamada RMSE/JSON değeri boş ve coverage düşük olur. Validation ıraksaması adayın seçilmesini engeller.

`python/experiments/`: yapılan yedi keşif/tekrar koşusunun modelleri, sonuçları ve protokolü. `python/reference_results/`: son seçim dondurulduktan sonraki testli raporlar. Arayüz bunları ortak ham veriden çizer. `python/finalize_results.py <run-id> ...` sadece bu halka açık sentetik kayıtları birleştirir; modeli yeniden eğitmez, veri/split ve validation skorunu doğrular, testten önce seçimi sabitler. Yeni gerçek verinin test raporu arayüzde eğitim bittikten sonra açıkça istenir.

Eski MAT SHA ve yeni pulse metadata/ham veri SHA kaydedilir. Yeni pulse jeneratörü düzensiz pozitif/negatif darbeler, değişken bekleme, gürültülü akım, daha az gürültülü DDV, yön asimetrisi, sonlu hareket hızı ve mekanik sınıra benzer duruş üretir. Rezonans/motor/gerçek hidrolik yük yoktur; uçak modelinin fiziksel geçerliliği iddiası taşımaz. Öğreniciler jeneratörün iç parametrelerini veya gizli durumlarını almaz.

## Doğrulama komutları

```bash
PYTHONPATH=python .venv39/bin/python -m unittest discover -s python/tests -v
.venv39/bin/ruff check python --select F
.venv39/bin/ruff format --check python
.venv39/bin/python -m compileall -q python
```

Tarayıcı testi için dev bağımlılıklarını kur, `python -m playwright install chromium`, sunucuyu `python/app.py --no-browser` ile aç, sonra `python/tests/browser_smoke.py` çalıştır. Yerel fixture CSV'leriyle gerçek kısa eğitim yapar. Windows launcher ve MATLAB runtime bu ortamda test edilmedi; bu sınır changelog'da korunur.


## Opt-in flightLike ve null_rate (Py-0.3)

`pulse_flights(profile='flightLike')` aynı üretecin ek profilidir; eski çağrı ve MAT değişmez. `Flight.truth` yalnız audit için. Bu profil drift'in holding current'ta gözlenebildiğini varsayar; akımdan bağımsız bilinmeyen bir DDV ofsetini kesin öğrenme vaadi yoktur. Fotoğraf birimlerini doğrulayamadığımız için RAM ±100 kullanıcı-benzeri birimdir. Park→work kısa ters darbe gerektirir. Üretimde gizli hız katsayısı olsa da modele aktarılmaz.

`null_rate` model alanları: `observer(center,gate,tau,max_step,adaptive)`, `knots,values,bounds,delay,ridge,training_rows`. TRAIN'deki iç bölgede düşük RAM hızı gözlenen örneklerden null gate çıkarılır; uç pozisyonda parked DDV null kabul edilmez. Girdi gate içinde ve adım yeterince küçükse null tahmini nedensel güncellenir. Eğitim ve rollout aynı observer'ı kullanır. Hız haritası, tahmin edilmiş DDV−null ile ölçülen TRAIN hızından ridge PWL regresyonuyla öğrenilir; seyrek hareketler için occupancy ağırlığı kullanılır. Bu yeni ailede upstream DDV TRAIN tahmini in-sample'dır; flight-OOF olduğu iddia edilmez.

`x[k]=clip(x[k−1]+g(DDVhat[k−delay]−chat[k−delay])/64, TRAIN_bounds)`. Beta/ortalama konuma çekme yok; gövde öğrenilmiş nonlinear rate modelidir, NN değildir. Sabit boyutlu durum: RAM, null tahmini, hazır bayrağı, önceki DDV ve delay tamponu; DDV bloğunun akım geçmişi ayrıca. Çıkarım NumPy-only, Truth/future/ölçülen RAM geri beslemesi yoktur. Sıfır hız platosu öğrenilen haritadadır; tüm null hatalarını garantiyle düzeltmez. Küçük hareketi drift sanmak veya yanlış hareket seviyesini tutmak mümkündür.

Yeni aile `methods=['null_rate']` ile opt-in; `nullAdaptive=True`, `nullTau=2` yalnız bu aday seçilince etkilidir. `run_flightlike.py --limit 10` en fazla 10 aday dener; %95 validation RAM Cascade'te erken durur; test yalnız final seçimden sonra açılır. `score` nonfinite çıktıyı başarısız sayar. Eski varsayılan modeller/ayarlar korunur.

`syntheticTrainFlights=N` yalnız `source='user'` için A/B çalıştırır; 0 eski tek koşu. `calibrate_train` sadece TRAIN ölçümlerini görür. Basit akım/DDV eşlemesi, yön, RAM aralığı, hız, null dağılımı ve açı haritası çıkarır; gerçek fizik kalibrasyonu iddiası yoktur. Geçerli iç hold/hareket bulunamazsa sessizce sahte veri üretmek yerine hata verir. Sentetik yeni uçuşlar yalnız TRAIN'e eklenir; iki varyantın seçimi aynı gerçek validation'dadır. Test her iki eğitim bittikten sonra, kullanıcı isterse raporlanır. Summary'deki TRAIN metriği gerçek TRAIN üzerindedir; augmented modelin sentetik eğitim örnekleri buna dahil değildir. Diagnostics `SyntheticTrainAB` satırı val/test Fit farklarını ve validation RMSE farkını verir. Raw/Truth yerel kayıtları Git dışında kalır.


## Python 3.9 ve yeni observer (Py-0.4)

Uygulama Python 3.9.25 ile gerçekten çalıştırıldı; PEP604 nullable tipleri `Optional` olarak değiştirildi. Python 3.9 uyumlu sürümler `requirements.txt` ve `requirements-cpu.txt` içinde sabitlendi. `check_environment.py` sürüm/64-bit/eksik veya eski paket kontrolüdür; kurulum yapmaz. `setup_offline.bat` yalnız yerel wheelhouse kullanır; `start.bat` hiçbir paket kaynağına erişmez. `download_wheels.bat` yalnız ayrı, internete bağlı Windows/Python3.9 makinesinde kullanılacak hazırlık aracıdır.

`nullObserver='legacy'` eski davranışı korur. Opt-in `'stationary'`: 32 örneklik nedensel DDV aralığı yeterince sakin ve TRAIN'den öğrenilmiş null gate içinde olduğunda, sakin bölge girişinde null'ı yeniden kestirir. Hareket rampasını drift sanıp takip etmez. Sakinlik eşiği TRAIN'deki interior-hold örneklerinden; hızın sıfır kabul edildiği bant da TRAIN'den öğrenilir. `'envelope'`: çift yönlü ve gate dışındaki yakın tarihli tepe/çukurların orta noktasıyla, osilasyon sırasında merkez kaymasını izlemeyi ekler. Gerçek null/Truth okunmaz. `nullAdaptive=false` observer güncellemesini kapatır.

Durum: RAM, null, sakinlik bayrağı, 32 DDV örnek tamponu; envelope için tepe/çukur, yaş ve yön. `quiet_window`, `quiet_range`, `envelope_age`, `mode` model spec alanları; RAM `zero_band` öğrenilmiş hız-sıfır bölgesidir. Bu sınırlı gözlemci gürültüye duyarlıdır; Component ve Cascade sonuçları birlikte raporlanır. NumPy-only batch çıkarımı korunur; C tek-adım API henüz yazılmadı.

`refine_window_rate`: TRAIN'de uç sınırlara vurmayan 5 s pencerelerde konum çıktı hatasıyla PWL ağırlıklarını ridge LS ile düzeltir; önceki hız haritasına düzenlileştirme uygular. Sıfır hız bandı korunur. Reset yalnız eğitim hedefinin pencere başlangıcındadır; validation/test rollout'ta reset yoktur. Bu üçüncü aday envelope'ü geçmedi.

`run_improvement.py`: üç önceden belirlenmiş aday; aynı TRAIN/VAL, donuk DDV/açı, %95 validation Fit veya üç adayda durma. Tarihsel test daha önce görüldüğü için ayrı etikettedir. Seed 9621 ile iki yeni uçuş yalnız seçim dondurulduktan sonra üretildi, aramada kullanılmadı. Seçilen observer opt-in yerel yöntem listesinden kullanılabilir. Runtime/package sürümleri `environment-tested-py39.json` içinde.

## Valf varyasyonu / gate ölçeği (Py-0.5)

`pulse_flights(profile='flightLike', valve_variation={'gain_sd':.03,'bias_sd':.025,'drift_amp':.02})`: opt-in sentetik stres profili. Ayrı RNG eski dizileri değiştirmez; None/boş/sıfır eski raw fingerprint'i korur. Uçuşa özgü gain/bias ve yavaş sinüzoidal bias current→DDV eşlemesini değiştirir; gizli trace yalnız Truth'ta. Bunlar gerçek TRAIN'den kalibre edilmiş değerler değildir. Bağımsız bilinmeyen ofsetin yalnız akımdan kesin belirlenmesi mümkün değildir.

`fit_null_rate(..., gate_source='predicted')`: TRAIN hold anlarında **tahmin edilmiş DDV** dağılımından observer center/gate çıkarır. Eski varsayılan measured'dır. RAM struct `gate_source` eğitim metadatasıdır; inference denklemi ve sabit boyutlu durum değişmez. Yeni model ailesi değil, gate ölçeği + PWL düzenlileştirme denemesidir. Üç aday sınırı `run_valve_improvement.py` içinde açıkça tanımlı; test ne seçime ne rapora girer.

## Sınırlı RAM araması (Py-0.6)

`fit_null_rate(..., quiet_seconds=.5, quiet_quantile=.95)` varsayılanları korur. `quiet_window=round(Fs*quiet_seconds)` observer'ın nedensel DDV tampon uzunluğudur; eşik TRAIN hold anlarındaki pencere aralığının belirtilen quantile'ından öğrenilir. Modelde mevcut `quiet_window` ve `quiet_range` alanlarıyla saklanır; yeni inference denklemi yoktur. Daha uzun history bu veride iyileştirmedi. Seçilen adayın mevcut `window_refinement` alanı 2 s/ridge=.01'dir; eğitimde pencere reset'i, çıkarımda ise kesintisiz integrasyon vardır. Web varsayılanları değişmez.

## Sequence-rate (Py-0.7)

`ram.kind='sequence_rate'`: `reference` eski null-rate struct'ı; `sequence` içinde TRAIN `mean,scale,rate_scale,rate_limit`, `family,residual,gate`, head/bias ve açık ağ ağırlıkları. GRU/Elman/ESN `wi,wh,bi,bh,hidden`; GRU gate sırası PyTorch r,z,n. LSTM i,f,g,o ve ek cell durumu. TDNN `offsets,w,b`; TCN nedensel iki katman `layers(w,b,dilation)`; RBF `centers,width2,offsets`. ESN `leak` ve sabit reservoir. Çıkarım yalnız NumPy; Torch yalnız eğitim/testte. C/MEX üretilmedi.

Referans null observer ve hız integraline ek GRU hidden=12 durumları kullanılır. GRU hız düzeltmesi, öğrenilmiş hız sınırları ve null gate ile uygulanır; RAM TRAIN bounds'a clamp edilir. İlk RAM ölçümle başlatılır; ara ölçülen RAM geri beslemesi yok. TRAIN 256 örnek pencerelerde ilk 64 örnek warmup, hız kaybı + tamamen interior pencerelerde .2 konum-cumsum kaybı; validation tüm uçuş boyunca sıfır hidden başlangıcıyla kesintisiz. Pencere reset'i değerlendirmeye taşınmaz. 400 Adam adımı; tek son checkpoint, test seçimi yok.

Arayüz `gru_rate` hem önceki referansı hem GRU'yu validation ile karşılaştırır. Preset hidden=12,horizon=256,steps=400,seed=42. Diğer family uygulamaları küçük deterministik parity/smoke testinden geçti; bu turda sadece TDNN/ESN/GRU tam performans deneyi yapıldı. Yöntem kapsamı: [methods.md](methods.md).

## TRAIN odaklı kapasite deneyi (Py-0.8)

`selectionBasis='train'` yalnız opt-in seçim hedefini TRAIN tam-uçuş RAM Cascade RMSE yapar; validation metrikleri ayrı kalır, test seçime girmez. Normal varsayılan `validation`. TRAIN-seçilmiş sonucun validation başarısı gibi sunulmaması için config/notice/selection_policy ve UI metrik başlığı bunu belirtir. `nullGateExpansion=1` API/experiment bayrağı, `fit_null_rate(gate_expansion=1)` parametresine gider. Gate midpoint sabit, yarı genişlik çarpılır; yeni inference state yoktur. Tüm aralıklar yalnız TRAIN'den gelir. Mevcut null_rate ve sequence referansları aynı flag'i kullanır.

`lstm_rate` seçeneği aynı residual hız modelinde LSTM kullanır; hidden24 + cell24, öğrenilmiş null observer ve RAM durumu; numPy-only export gate sırası i,f,g,o. TRAIN preset: gate1.5, 800 adım, horizon256, seed42. Kısa eğitim pencereleri sıfır recurrent başlangıçla, tam-uçuş değerlendirme kesintisiz state ile çalışır. TRAIN başlangıç ölçümleri değerlendirmede sürekli reset yapılması anlamına gelmez.

`sequence_evaluate.py` yalnız eğitim/arama hızlandırıcısıdır; dağıtım NumPy çıkarımını değiştirmez. Torch double ağırlıklar ve chunklar arasında taşınan hidden/cell state; kazanan raporu tekrar NumPy ile hesaplanır. Raporların gerçek pipeline sonucu olduğu ayrıca doğrulanır.

## Uçuş bazlı dengeleme ve sabit hafıza çıkış düzeltmesi (Py-0.9)

`fit_sequence(flight_weights=None, position_weight=.2)` varsayılanları önceki eğitim yolunu korur. Opt-in flight ağırlıkları yalnız TRAIN hatalarından türetilip pencere örnekleme olasılığına uygulanır; uçuş kimliği modele girdi olmaz. TRAIN hız kaybına ek konum-penceresi kaybı katsayısı ayrı ayarlanabilir. Bu turda her TRAIN uçuşunun Fit >90, birleşik validation Fit >75 olması birlikte aranır; test kapalıdır.

`refine_sequence_head` yalnız TRAIN alır, DDV/açı/recurrent hafıza ağırlıklarını dondurur. RAM çıkış hatasına göre lineer readout/bias, Gauss–Newton normal denklemleriyle düzeltilir. Hız doygunluğunda türev sıfırlanır; RAM sınırında biriken konum Jacobian'ı resetlenir. Başlangıç gerçek RAM yalnız ilk örnektir; eğitim hedefini kolaylaştıran ara konum reset'i yoktur. TRAIN uçuşlarının normalize hatalarıyla ağırlıklandırma ve yalnız TRAIN en kötü uçuş hatasını azaltan backtracking kullanılır. Validation optimizere verilmez; aday kabulünde >75 şartı uygulanır.

Yeni inference state/denklemi yok: güncellenmiş `sequence.head,bias` aynı NumPy çıkarımında kullanılır. `head_refinement` sadece eğitim metadatasıdır. `sequence_evaluate.recurrent_states` hızlandırıcı; chunklar arasında state taşır. İterasyon raporunda aday sayısı ile optimizer adımı ayrıdır; en fazla20 aday sınırı korunur.

## Opt-in genelleme ve güvenilir arama (Py-0.10)

`run_generalization.py` eski mixed-valve seed621, 7/2/2 ve fingerprint'i değiştirmez. Seed910247 ile 6 yeni validation, seed741203 ile 14 ek TRAIN üretilir; Truth öğreniciye verilmez. Validation = eski2 + yeni6; TEST hiçbir aday için açılmaz. 7/14/21 TRAIN × kırpma kapalı/açık × seed42/43/44: 18 LSTM24/800/horizon256 eğitimi. Adım sayısı sabit olduğundan büyüyen TRAIN'de uçuş başına örneklenme azalır; veri büyüklüğü deneyini eşit epoch karşılaştırması gibi yorumlamayın. Seçim en düşük validation Fit, sonra macro ortalama; TRAIN yalnız rapordur. Viewer kaydının fingerprint'i yalnız görüntülenebilir 8 validation uçuşunu kapsar; orijinal fingerprint `protocol.original_sha256` içinde korunur.

`rate_ensemble`: ortak `ddv`, `angle`, `ram.members` sequence struct'ları, `ram.bounds`, `training_seeds`. Her üyenin hız çıktısı ortalanır; tek RAM integratörü ve TRAIN sınırları uygulanır. `map_rate_limit` referans hız haritasını, `correction_limit` ağ düzeltmesini, `output_rate_limit` toplam hızı TRAIN |dRAM/dt| %99.9 sınırına kırpar. Alanlar yoksa eski davranış aynıdır; eski sequence modellerinde zaten var olan legacy hız sınırı kaldırılmaz.

`robust_search.search` / API `reliableSelection=false`, `autoSearch=false`, `maxCandidates=6`, `maxMinutes=30`, `clipRate=false`. 3 seed zorunlu; aileler null_rate, gru_rate, lstm_rate, oe, neural, stable; iki sabit hidden/horizon/steps seti (12/128/200, 24/256/400). OE ve neural ayrı adaylardır, yeni bir OE+NN hibriti iddia edilmez. Validation <3: dış fold yalnız TRAIN uçuşlarından; held-flight model eğitimine/checkpoint seçimine girmez. Her fold içinde tüm katsayılar yalnız kalan TRAIN'de öğrenilir. Seçilen ayar tüm TRAIN'de tekrar eğitilir; CV skorları son refit modelin bağımsız skorları gibi sunulmaz. `validation_source` ve notice bunu belirtir. Genel split değişmez.

`seed_ensemble` diğer ailelerde `ram.members`, `ram.ddv_members` ve ortak açı kullanır. Joint neural üyelerinin DDV ağırlıkları farklı olabilir: her üye kendi DDV tahminini kullanır, rapor DDV'si bunların ortalamasıdır. Hızlar ortak tahmin edilen RAM durumunda hesaplanıp ortalanır; her üyenin hız hafızası ayrıdır. Component'te tüm üyeler ölçülen DDV alır. Taşınabilir çıkarım NumPy'dır; S-Function/C bu turda üretilmedi.

Negatif/sonlu olmayan RAM Cascade seed-validation, ensemble-validation veya ensemble-TRAIN uçuşu adayı eler; isimleri raporlanır. Fit tanımsızsa güvenilir seçimde uygun kabul edilmez. Component düşüklüğü (>20 macro Fit puanı) yalnız uyarıdır, seçim ölçütü değildir. >25 TRAIN–validation farkı da uyarıdır. Her tamamlanan aday/seed sonrası kısmi JSON, sonunda mevcut XLSX sayfaları + CandidateFlights yazılır. Kesinti sonrası otomatik eğitim devamı yoktur; tamamlanan modeller korunur. Süre/iptal işbirlikçidir, tek LS/rollout çağrısını zorla bölmez.

Yükleme arayüzü ham CSV metnini JSON/base64 içine almaz. `SignalReducer` sabit global fazlı, örtüşmeli polyphase anti-aliasing uygular; seçilen kanalın ilk/son kenarında sıfır padding vardır. Her kanal kendi zamanıyla 64 Hz'e indirgenip ortak zaman aralığında hizalanır. Eski JSON API geriye uyumluluk için kalır; büyük dosyalar multipart yolunu kullanmalıdır. Plot cache id/uçuş/model anahtarlıdır; ilk açılışta TEST dışı uçuşlar tek arka plan işçisiyle hazırlanır. Tarayıcı artan istek token'ı, zaman aşımı ve görünür hata kullanır.
