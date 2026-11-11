# Aktif modelleme yaklaşımı

Amaç gerçek fiziksel denklemleri zorunlu tutmadan Current → DDV → RAM → açı zincirini uçuş verisinden tanımlamaktır. Kapalı çevrim kontrolcü deneyi yalnız açıkça seçilen closed modunda çalışır.

## DDV

Varsayılan DDV adayı NL_OE'dir: akım geçmişine z/tanh(z)/abs(z) bazları uygulanır, DDV çıktı hatası ridge LS ile küçültülür. Dinamik RAM integratörü burada kullanılmaz. Eski MLP/FIR/GP araması yalnız nlFocus=false ile korunur. RAM'de başarısız bulunan bir yöntemin DDV'de de başarısız olduğu varsayılmaz. MLP'de MATLAB trainscg/trainbfg kullanılır. Önceki Python Adam/L-BFGS deneyleri ile birebir aynı eğitim değildir.

## RAM

Varsayılan RAM adayı NL_OE; diğer aileler yalnız `nlFocus=false` ile açılan önceki karşılaştırma yolunda korunur. Yapılar:

- **FIR_rich:** DDV giriş geçmişinden residual hız öğrenir; durumu integratörle ilerletir.
- **FIR_OE:** katsayıları doğrudan TRAIN konum trajektorisi hatasıyla öğrenir; fark/türev hedefi kullanmaz.
- **NL_OE:** normalize giriş geçmişinin `z`, `tanh(z)`, `abs(z)` bazlarıyla doğrusal olmayan sonlu hafıza modeli; katsayılar yine konum hatasından öğrenilir. NN veya NARX değildir.
- **HoldInitial (isteğe bağlı):** başlangıç RAM konumunu sabit tutan karşılaştırma; nihai aday olarak seçilmez.

RAM durum denklemi:

```text
xhat(k) = (1 + beta/Fs) * xhat(k-1) + rate(k)/Fs
```

Başlangıçta bir RAM konumu gereklidir. Sonrasında gerçek RAM ile reset/geri besleme yoktur. Beta=0 başlangıç hafızasını korur; negatif beta sızıntılı integratördür. Beta fiziksel parametre kestirimi iddiası taşımaz. Rate bias varsa saf integratörde sürüklenebilir.

FIR_OE en fazla 256, NL_OE en fazla 128 input lag kullanır. Ayarlar validation serbest çalışma RAM RMSE'siyle seçilir. Aktif RAM MLP/GP adayları geri eklenmemiştir. DDV çıktısı ortak tutularak RAM adaylarının farkı karşılaştırılır.

## Açı

RAM-açı dönüşümünün derece 1/3 seçenekleri TRAIN'de eğitilip validation'da seçilir. Gerçek RAM ile iyi açı sonucu, Cascade'in doğru olduğu anlamına gelmez. RAM sabitken bağımsız esneme/titreşim varsa bu statik dönüşüm yeterli değildir.

## Teşhis ve gerçek veri önceliği

Yüksek one-step Fit serbest simülasyonun doğru olduğunu göstermez. Önceki gerçek RAM'i kullanan basit persistence bile yavaş sinyallerde yüksek one-step skor verebilir. `DataAudit` çalışma aralığı değişimini, `PerFlight` uçuş farklarını, `EventErrors` geçiş/bekleme hatalarını gösterir.

Geçiş teşhisi, bir örnekte RAM değişiminin TRAIN aralığının %1'ini aşması ve sonraki 2 saniyedir. Fiziksel etiket değildir; yavaş hareketler bekleme grubuna düşebilir. Bu maske model seçimini değiştirmez.

Gerçek zaman vektörü, birim, sensör sıfırı, işaret, basınç/yük ve başlangıç durumunu doğrulamadan daha büyük model seçilmez. Gerçek data mevcut olmadan yalnızca fotoğraftan kesin fiziksel neden veya %95+ başarı çıkarılamaz. Sentetik veri üretimi birincil geliştirme amacı değildir.

`Alignment` gecikmesi pozitifse çıktı girdiden sonra gelir. Negatif gecikmeyi nedensel FIR temsil edemez. Korelasyon gecikmesi fiziksel gecikme kanıtı veya otomatik hizalama değildir. Hold eğimleri ayrı kesitlerde LS ile hesaplanır: örnek ağırlıklı işaretli ortalama ve en büyük mutlak eğim (RAM birimi/s).

OOF tahminleri yalnız diğer TRAIN uçuşlarıyla eğitilir; validation/test kullanılmaz. DDV aday ailesi validation ile seçilir, dolayısıyla raporlanan TRAIN LOFO skoru bağımsız nihai genelleme kanıtı değildir ve seçimi değiştirmez. OE tüm-örnek yolu normal denklemleri bloklarla biriktirir; bellek özellik sayısının karesiyle büyür. Null varyantında merkezleme ve intercept kaldırılır; sıfır giriş geçmişinde öğrenilen hız sıfırdır (beta sızıntısı ayrıca devam eder).

HAM_OE bir adayda yalnız bir statik skaler harita seçer: deadband veya tanh; ham DDV haritalandıktan sonra geciktirilir. Böylece L+1 ana özellik kullanır (NL_OE: 3(L+1)); d/s seçenekleri birlikte yığılmaz, validation ile seçilir. `_F`: force ve DDV×force, lag 0..8. `_P`: DDV×sqrt(Ps−Pr), lag 0..8; negatif fark hatadır, ham uzun basınç geçmişi yoktur. Bunlar öğrenilen katsayılardır, fiziksel gain beslemesi değildir.

Toolbox'sız model sözleşmesi: `kind='FIR'`, `weights`, `xMean/xStd`, `yMean/yStd`, `inputLags`, `inputNames`, `inputBasis`, `beta`, `Fs`, `rateGain/rateBias`; isteğe bağlı `noIntercept`, `featureSpec(basis,parameter,force,pressure)`, `ramBounds`. HAM/custom modeller için `bb_predict_batch` statik harita ve lag'leri hazırlar; `bb_forward` hazırlanmış özellikleri işler. Çıkarım durumu bir RAM konumu ve en fazla max(inputLags,8) harici giriş geçmişidir; eğitim verisi gerekmez. `oeConfig` yalnız yeniden eğitim/CV metadatasıdır. Kıskaç etkinse açık/kapalı çevrim kabulünde kıskaçsız sonuçlar da ayrıca incelenmelidir.

`reportLoo=true` ayrıca OE RAM adaylarının sabit seçilmiş hiperparametreleriyle TRAIN LOFO cascade RMSE hesaplar (`LOFO_RAM`, kompakt ek sütun). Her fold DDV ve RAM'i yeniden eğitir; OOF modunda iç TRAIN fold'ları da yeniden hesaplanır. Bu iç içe model seçimi değildir; hiperparametreler validation'dan geldiği için yalnız teşhistir, seçim ölçütü değildir.

Kapalı çevrim: `SP_OE` setpoint geçmişinden RAM tahmin eden beta<0 OE temel çizgisidir. `Hybrid` tek ridge kontrolcü + donuk seçilmiş DDV/RAM plant kullanır. Kontrolcü eğitim girdileri e(k)=setpoint(k)−RAM(k−1), e geçmişi, sızıntılı integral, önceki DDV/akım geçmişidir. Simülasyonda RAMhat(k−1), tahmin edilen DDV ve akım kullanılır; gerçek çıktılardan reset yoktur. Başlangıç RAM dışarıdan verilir; kontrolcü/DDV/akım geçmişleri sıfırdır. Bu başlangıç seçimi deploy arayüzünde korunmalıdır.

`bb_ctrl_train` model alanları: FIR normalizasyon/ağırlıkları, `lags`, `taus`, `Fs`, `ridge`, `noiseStd`. `bb_ctrl_features` durumu: error/DDV/current sabit uzunluklu tamponları ve integral vektörü. `bb_closed_loop` plant alanları `ddv`, `rate`, `trainBounds`; sadece setpoint ve modelin gerektirdiği harici force/pressure okunur. İleride C için sonlu tamponlar yeterlidir; burada C/Simulink üretilmez.

Iraksama eşiği 2×max(abs(TRAIN RAM)) (sıfır aralıkta eps); hybrid kıskaç öncesi durumu da kontrol eder. NaN/Inf başarısızlıktır. Iraksayan validation kontrolcüleri seçilmez; tümü başarısızsa ilk kontrolcü açık uyarıyla başarısız teşhis olarak raporlanır, performans iddiası yapılmaz. Closed seçim yalnız validation'dadır; plant seçimi değişmez. Gürültü yalnız kontrolcü eğitimindeki RAM geri beslemesine uygulanır; seed ve RNG durumu korunur.

NL odak geliştirmesi: baz `z, tanh(z/tanhScale), abs(z)` olarak genişletildi. Eski modellerde eksik `tanhScale` alanı 1 kabul edilir. Kaba kazanan korunarak beta × [.5 .75 1 1.25 1.5] ve doygunluk ölçekleri denenir; lag/ridge bu yerel aramada kaba kazananın değeridir. Beta=0 kazananında [0 −.001 −.003] denenir. Bunlar validation karşılaştırmalarıdır, gerçek veri başarısı iddiası değildir.

Yeni NL bayrakları kapalıyken adaylar ve sayısal model hesabı korunur. `nlDiagnostics`: beta<0 için başlangıç hafızası zaman sabiti −1/(Fs·log(1+beta/Fs)); beta=−1, Fs=64 için yaklaşık 0.99 s (bir zaman sabitinde etkinin %37'si kalır). DDV geçmişi sıfır, yardımcı girdiler TRAIN ortalamasındayken lineer denge −rate(0)/beta'dır; kıskaçsız denge raporlanır. Hold değişkenliği her kesitte ayrı hesaplanır; ölçülen varyans sıfırsa oran NaN'dır. TrainMean, örnek ağırlıklı TRAIN RAM ortalamasıdır ve x0'ı kullanmaz; hiçbir hiperparametre seçimine katılmaz.

`nlJointSearch` mevcut kazananı koruyup sınırlı ortak komşuluk dener; lag ortak değerlendirme başlangıcını uzatmaz. `nlWideBeta` beta çarpanlarına 2/3 ekler; Euler katsayısı pozitif kalmalıdır. `TrainSeconds` yalnız model eğitiminin duvar saati süresidir (validation çıkarımı hariç), hız iddiası değildir.

`nlStateLeak`: tek durum güncellemesine −k·tanh((x−center)/scale)/Fs eklenir. center ve ölçek tabanı TRAIN'den, k ise TRAIN serbest koşu hatasını küçülten sınırlı `fminbnd` aramasından gelir; base MATLAB kullanılır. DDV→hız ağırlıkları bu deneyde donuktur. Ölçek faktörleri validation ile karşılaştırılır; orijinal modelden iyi değilse aday seçilmez. Model alanı `stateLeak={k,center,scale}`; `bb_ram_step` açık/kapalı çevrim ve one-step için ortak toolbox'sız hesaptır. Kıskaç ayrıca uygulanır. Bu deneyin LOFO skoru iç içe leakage eğitimi yapılmadığı için açık uyarıyla atlanır.

`nlRefitTrainVal`: aynı seçilmiş DDV ailesi, RAM beta/lag/ridge/tanhScale ve açı derecesi ile TRAIN+VALIDATION ağırlıkları yeniden öğrenilir; validation artık eğitim verisidir. Orijinal TRAIN normalizasyonuyla raporlanan test metrikleri yan yana iki etiketle yazılır. Test girdileri yeniden eğitim fonksiyonuna verilmez. Seçilmiş leakage ve kıskaç parametreleri sabit tutulur. OOF modunda zaman kesitleri aynı kaynak uçuş grubunda dışarıda bırakılır. Refit orijinal seçimi veya grafik modelini değiştirmez; dağıtım seçimini test hatasıyla otomatik yapmaz.

NL_WH: seçilmiş NL_OE giriş bloğunun sürüş çıktısı r (gain/bias sonrası) donuktur. TRAIN r kantillerinden 11 civarı düzensiz düğüm + sıfır kurulur. `wh.knots`, `wh.values` monoton PWL haritayı saklar; artımlar kare parametrelerle negatif olamaz, h(0)=0, uçların dışında sabit doygunluk vardır. İki yön bağımsız eğim/limit öğrenebilir; sıfıra yakın düz bölüm ölü bant temsil eder. Tek durum: x(k)=a*x(k−1)+h(r(k))/Fs. İlk h örneği sıfırlanır; x0*a.^k+filter(...) x(1)=x0 sağlar. Çıkarım `interp1` ve `filter` ile base MATLAB'dır; C/Simulink üretilmedi. `fminsearch` yalnız TRAIN serbest-koşu RMSE üzerinde harita artımları ve kararlı beta'yı optimize eder; bütçe `whMaxEvals=300`, `whMaxIter=150`. FIR ağırlıklarıyla dönüşümlü güncelleme bu ilk deneyde yoktur. Vectorize amaç fonksiyonunu korumak için NL_WH deneyi clampRAM=false ister. Kıskaç/statedependent-leak karıştırılmaz. Refit açıkken harita TRAIN+VAL'da seçilmiş beta sabitken yeniden öğrenilir; iç içe WH LOFO uygulanmadığından açık uyarıyla atlanır.

Geçiş ağırlıkları yalnız TRAIN normal denklemlerine girer: G=Σw*A'*A, b=Σw*A'*y; ölçek ve ridge normalizasyonu Σw ile yapılır. Girdi/çıktı normalizasyonu ağırlıksız TRAIN istatistikleridir. λ=0 eski denklemdir; [0 2 5] aramasında her lambda aynı hiperparametre aramasını ve validation serbest-koşu ölçütünü kullanır. EventErrors her lambda'nın en iyi adayını ayrı adla gösterir; seçilmiş modelin lambda'sı kısa özette ve RunInfo'da eğitim modu/bütçesi görünür. Test seçim için kullanılmaz.

RateMap kutuları TRAIN ölçülen RAM hızından kurulur; VAL aynı sınırları kullanır. `MeanModelDrive` h sonrası sürüş terimidir, beta*x içermez; `MeanPredictedRAMRate` rollout türevidir ve fiziksel hızla doğrudan karşılaştırılır. İkisi de ölçülen DDV girdisiyle component teşhisidir. StepSmoothing: eşik+uzatma maskesindeki bağlı geçişler ölçülen maksimum konum değişimine göre sıralanır; VAL'da ilk 5 alınır. Tepe hız ve 10–90% süre oranları **tahmin/ölçüm** yönündedir; tahmin ölçülen hedef seviyelerini geçmezse süre NaN ve CrossedBothLevels=false olur. Maske/örnekleme sınırı nedeniyle bunlar fiziksel adım testi yerine geçmez.

StepSmoothing ayrıca ters oranları `MeasuredOverPredPeak` / `MeasuredOverPredDuration` sütunlarında verir; oran yönü sütun adında açıktır. Sıfıra bölme veya ulaşılamayan süre NaN olarak kalır.

NL_FB (`nlFilterBank`, `ddvFilterBank`): girdiye lag 0..8, (1−p)/(1−p z⁻¹) birinci derece bölümler (p=0,.5,.8,.9,.95,.98,.99) ve doğal frekans 2/4/8/16 Hz, sönüm .3/.7 olan ikinci derece rezonans bölümleri uygulanır. Karmaşık kutuplar exp(−ζω/Fs)·exp(±jω√(1−ζ²)/Fs); pay DC kazancını 1 yapar. Sonra TRAIN normalizasyonu ve z/tanh(z)/abs(z) bazları kullanılır. RAM aynı bloklu OE normal denklemleriyle, DDV doğrudan ridge çıktı hatasıyla öğrenilir. Sabit özellik bankası dışında öğrenilmiş fiziksel denklem/gain yoktur.

Model `filterBank` alanı: Fs, poles, rawLags, ikinci derece a/b katsayıları. `bb_filter_bank` durumu: raw geçmiş, first (birinci derece gecikmeler), second (iki gecikme/bölüm). IIR durumları uçuş başında sıfır, ham geçmiş ilk girişle doldurulur. Chunk sınırlarında ve closed tek-adım çıkarımında durum korunur. RAM konumu ek tek durumdur; hedef RAM history özelliklere girmez. Aynı boyutlu durumlarla toolbox'sız S-Function'a taşınabilir; burada C üretilmez. Yeni adaylar mevcut değerlendirme penceresini/split'i değiştirmez.

BandGain yalnız offline FFT bant projeksiyonudur; gerçek-zaman filtresi değildir. Ortalama çıkarılır; düşük enerjili bantlar NaN verir. Pozitif BestLagSamples tahminin ölçüme göre gecikmesidir. ±8 aralığında en büyük mutlak korelasyon seçilir, işaret korunur ve örtüşen örneklerde genlik/korelasyon tekrar hesaplanır. Kayan pencere/kazanç düzeltmesi yapılmaz. FFT uç süreksizliği ve dar bant periyodik belirsizlikleri nedeniyle gecikme fiziksel kanıt değildir; düşük bant gücü, std oranı ve korelasyon birlikte okunmalıdır.

Excel Search, farklı aday tablolarının birleşimidir: Stage ve Source sütunları aramayı ayırır. BetaCurve da burada saklanır. Diagnostics başlıklı blokları, Config split/ayarları taşır. Süreler duvar saati ölçümüdür; DDV/RAM arama süreleri validation çıkarımını da içerir, model seçimini etkilemez. Summary Fit iki ondalık, RMSE dört anlamlı basamakla gösterilir; seçim ve MAT metrikleri yuvarlanmaz.

NL_FB bağımsız eğitilir; NL_OE ağırlıklarını kullanmaz. Varsayılan RAM NL_FB, `nlReference=false`; isteğe bağlı NL_OE aynı validation ölçütüyle karşılaştırılır. Beta adayları −0.3 ve 0, yerel beta araması kapalıdır. DDV_NL_OE/DDV_NL_FB ve RAM_NL_OE/RAM_NL_FB süreleri üst aşamaların alt süreleridir. Angle_Component ölçülen RAM ile geometrik dönüşümü, Angle_Cascade tahmin RAM ile uçtan uca çıktıyı değerlendirir; skorların ortalaması kullanılmaz.

NL_SS: donuk NL_FB dış-giriş sürüşüne, tahmin RAM durumu z=(x-center)/scale ve v=tanh(drive/driveScale) ile scale*[1,z,tanh(z),v,tanh(z)*v]*tanh(weights) düzeltmesi eklenir. Model alanı stateSpace: center, scale, driveScale, weights(1x5). İstatistikler yalnız TRAIN; fminsearch tam TRAIN serbest-koşu konum MSE ile öğrenir. Başlangıç sıfır ağırlık tam taban modeldir. Tek RAM durumu ve mevcut filtre durumları korunur, hedef konum yalnız başlangıçta kullanılır. Katsayı sınırı global kararlılık garantisi değildir; TRAIN amaç fonksiyonu 20 standart sapmalık kaçışı reddeder, validation sonlu RMSE ve taban adayla karşılaştırılır. Lineer denge/tau teşhisi bu model için NaN yazılır. BetaCurve taban bankanın aramasıdır, NL_SS optimizasyon eğrisi değildir.

NL_HOLD: model.featureSpec.basis=hold, parameter=TRAIN std tabanlı seçilmiş DDV ölü bandı. Önce sign(u)*max(abs(u)-d,0), ardından 0..L gecikmeler ve [z,tanh(z),abs(z)] uygulanır. xMean=0, intercept yok, yMean=0, beta=0. W ridge LS ile TRAIN tam konum yörüngesi hatasından öğrenilir; OOF DDV desteği korunur. abs(z) iki yönde farklı kazanç öğrenebilir, hiçbir yön/konum miktarı elle verilmez. x(k)=x(k-1)+drive(k)/Fs; sıfır giriş geçmişi L örnekten sonra tam sıfır drive verir. Bu, seçilebilir bir tutma varsayımıdır; gerçek kaçak/kalıcı DDV biası varsa yetersiz olabilir. Genlik taşmasını veya monotonluğu matematiksel olarak garanti etmez. Sabit boyutlu lag durumu+tek RAM durumu, mevcut toolbox'sız çıkarım; yeni fiziksel denklemler yoktur.

DMLP: `dynamicMLP` alanı uMean/uStd(1x3), xCenter/xScale, rateScale, W(Hx4), b(Hx1), v(Hx1), bias içerir. DDV lag0..8'den [anlık DDV, 9 örnek ortalaması, DDV farkı] çıkarılır. Normalleştirilmiş üç dış girişe tahmin RAM durumu eklenir; tanh gizli katman ve lineer çıkış fiziksel RAM hızını verir. xNext=x+rate/Fs; beta=0 burada genel arayüz alanıdır, ağ durum bağımlı sızıntı/hareketi öğrenebilir. Ne sıfır girişte tutma ne de kararlılık zorunlu kılınır. Hız hedefli ridge warm-start yalnız başlangıçtır; W,b,v,bias birlikte base MATLAB fminsearch ile çok-adımlı konum MSE üzerinden optimize edilir. TRAIN penceresi başında ölçülen x0 kullanılır; pencere boyunca tahmin x geri beslenir. Tam validation/test yalnız uçuş başlangıç x0'ını kullanır. Birim test target-RAM çıkarıldığında aynı çıkarımı kontrol eder.

Toolbox'sız çıkarım: bb_forward DMLP için dış giriş ön işlemesini, bb_dmlp_step ağ/durum güncellemesini yapar. Durum 8 DDV gecikmesi + 1 RAM değeridir; sonlu sabit boyut. bb_rollout_rate ikinci çıktısı fiziksel hız, info.externalInputs one-step teşhisi için normalleştirilmiş dış girişlerdir. NullEquilibrium/tau DMLP için NaN; BetaCurve referans modele aitse ReferenceOnly=true. Closed tek-adım yolunda aynı durum güncellemesi kullanılır. C/S-Function üretilmedi.

FilterBank steadyStart metadata'sı eğitim/çıkarımda aynıdır; eski modellerde alan yoksa sıfır IIR başlangıcı. İlk örnekte DF-II first=p*u0; second=[1-b;-a3]*u0, çünkü bölümlerin DC kazancı 1'dir. İlk raw geçmiş u0 ile doldurulur. Chunk/Simulink adımlarında yalnız başlangıç/reset anında bu işlem uygulanmalı, her örnekte reset yapılmamalıdır. fbUniquePoles açıkken modeldeki poles listesi sıfırı içermez; çıkarım cfg yerine kaydedilmiş filterBank alanını kullanır. ddvAllSamples normalizasyonu yalnız TRAIN uygun satırlarından hesaplanır, uçuş başında filtre durumu sıfırlanır; başlangıç/guard/trim kapsamı eski adayla aynıdır.
