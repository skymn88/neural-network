# Aktif modelleme yaklaşımı

Amaç gerçek fiziksel denklemleri zorunlu tutmadan Current → DDV → RAM → açı zincirini uçuş verisinden tanımlamaktır. Kontrolcü döngülerinin modeli bu sürümde yoktur.

## DDV

Gecikmeli giriş MLP, doğrusal FIR ve sınırlı RBF Gaussian Process adayları validation DDV RMSE'siyle karşılaştırılır. RAM'de başarısız bulunan bir yöntemin DDV'de de başarısız olduğu varsayılmaz. MLP'de MATLAB trainscg/trainbfg kullanılır. Önceki Python Adam/L-BFGS deneyleri ile birebir aynı eğitim değildir.

## RAM

Aktif adaylar:

- **FIR_rich:** DDV giriş geçmişinden residual hız öğrenir; durumu integratörle ilerletir.
- **FIR_OE:** katsayıları doğrudan TRAIN konum trajektorisi hatasıyla öğrenir; fark/türev hedefi kullanmaz.
- **NL_OE:** normalize giriş geçmişinin `z`, `tanh(z)`, `abs(z)` bazlarıyla doğrusal olmayan sonlu hafıza modeli; katsayılar yine konum hatasından öğrenilir. NN veya NARX değildir.
- **HoldInitial:** başlangıç RAM konumunu sabit tutan karşılaştırma; nihai aday olarak seçilmez.

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
