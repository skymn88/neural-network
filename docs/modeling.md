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
