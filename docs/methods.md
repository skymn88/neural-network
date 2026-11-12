# Aday yöntemlerin durumu ve uygunluğu

Bu liste tüm yöntemlerin denenmiş olduğu anlamına gelmez. Gerçek uçuşlar bu ortamda yok; MATLAB sonuçları önceki kayıt/kullanıcı raporları, aşağıdaki yeni Python skorları sentetiktir.

| Yöntem | Projedeki durum | Bu problem için değerlendirme |
|---|---|---|
| NARX | MATLAB geçmişinde denendi; kullanıcı serbest-koşu sapması bildirdi | Uygun SysID ailesi; tekrar denenecekse teacher-forcing yerine serbest-koşu eğitimi/kararlılık gerekir |
| GRU | Py-0.7 tam-uçuş validation deneyi yapıldı | Küçük durum vektörü ve uzun geçmiş; bu turda en iyi yeni aday |
| LSTM | NumPy çıkarımı ve küçük eğitim/parity testi var; tam deney sırası gelmedi | Uygun; GRU'dan daha fazla durum/ağırlık, kazanımı henüz ölçülmedi |
| Elman RNN | Yeni küçük eğitim/parity testi var; tam deney yapılmadı | Uygun temel çizgi; uzun hafıza ve gradient sorunları mümkün |
| Jordan RNN | Özel Jordan mimarisi denenmedi; eski dinamik MLP'de çıktı geri beslemesi var | Benzerliği eşdeğer deney saymıyoruz; free-run hata birikimi önemli |
| ESN / Reservoir Computing | Py-0.7 tam validation deneyi yapıldı | Ucuz ridge readout; sabit reservoir durumu, NN kadar esnek olmayabilir |
| LSM | Denenmedi | Spiking olay temsili mevcut düzenli 64 Hz veride öncelikli değil |
| TCN | İki nedensel dilated conv katmanı ve parity/küçük eğitim testi var; tam deney yapılmadı | Uygun; geçmiş uzunluğu sonlu, uzun tutmaları tek başına temsil etmeyebilir |
| TDNN | Geçmiş MLP/lag yaklaşımları var; Py-0.7 açık TDNN tam deneyi yapıldı | Ucuz/taşınabilir; uzun geçmiş için gecikme seçimi önemli |
| Neural ODE | Tam sürekli-zaman çözücülü model denenmedi | Eski hız-integrasyon ağları buna benzer ama Neural ODE deneyi değildir; çözüm maliyeti ve gözlenebilirlik sorunu kalır |
| Neural CDE | Denenmedi | Düzensiz örneklemeye uygun; 64 Hz'e hizalı akışta ilk öncelik değil |
| Neural SDE | Denenmedi | Belirsizlik dağılımını modelleyebilir; deterministik tek çıktı hedefinde ek maliyet |
| PINN | Denenmedi | Bilinen fizik denklemleri/kayıpları olmadan avantajı belirsiz; mevcut saf black-box hedefiyle doğrudan örtüşmüyor |
| HNN / LNN | Denenmedi | Hamiltonian/Lagrangian yapı; valf sürtünmesi, sönüm ve hidrolik kayıplar nedeniyle güçlü ek varsayım gerektirir |
| Deep Koopman / DeepDMD | Bu adlarla denenmedi; önceki NLSS bunlarla eşdeğer değil | Alternatif düşük boyutlu dinamik temsil; hold/saturasyon için uygun latent yapı gerekir |
| AE-ROM / CAE / VAE dinamik modelleme | Denenmedi | Az sayıdaki sensör kanalı için boyut indirgeme gereksinimi sınırlı; latent durum öğrenme yine incelenebilir |
| GAN tabanlı SysID | Denenmedi | Veri üretimi/dağılım için olası; tek deterministik rollout doğruluğu için öncelikli değil |
| Hammerstein–Wiener sinir ağları | MATLAB HAM_OE/NL_WH benzeri yapısal adaylar var; bunlar standart NN-HW deneyi değil | Uygun yapı; önceki yapısal adaylar belirgin kazanç getirmedi |
| Volterra–NN hibrit | Tam hibrit denenmedi; geçmiş nonlinear baz/lag terimleri benzerlik taşıyor | Düşük dereceyle makul; özellik sayısı ve uzun hafıza pahalılaşır |
| RBF dinamik ağları | Yeni Gaussian RBF/readout küçük eğitim testi var; tam deney yapılmadı | Uygun hafif alternatif; yüksek boyutlu gecikmelerde kapsama sorunu |
| DeepSINDy | Denenmedi | Seyrek öğrenilmiş denklemler taşınabilir; gürültülü türev ve doğru baz seçimi zor |
| DeepONet / FNO | Denenmedi | Operatör/alan öğrenme avantajı burada belirsiz; birkaç skaler zaman serisi için büyük maliyet |
| Time-Series Transformer / PatchTST | Denenmedi | Daha çok veri/bağlam maliyeti; nedensel rollout ve sabit hafıza ayrıca tasarlanmalı |
| PIAE / PRO-DL | Denenmedi | Fizik/parametrik ROM hedefleri ek bilgiler gerektirir; mevcut gözlenen kanallarda öncelikli değil |
| Spatial U-Net / ConvLSTM-UNet / ConvGRU-UNet | Denenmedi | Uzaysal alan/görüntü verisi yok; doğrudan eşleşmiyor |
| PI-UNet / U-FNO / Graph U-Net / Attention U-Net | Denenmedi | Fizik alanı, uzaysal ızgara veya grafik yapısı verilmedi; şu aşamada gerekçesiz karmaşıklık |

## Py-0.7 protokolü ve sonuç

Aynı mixed-valve seed-621, aynı 7/2/2 split, donuk DDV/açı. En fazla 20 önceden tanımlı aday; validation RAM Cascade Fit >=90'da dur. İlk üç aday: TDNN 85.53, ESN 76.34, GRU **91.43**. Bu nedenle kalan 17 tam deney çalıştırılmadı; küçük unit-test eğitimi performans deneyi sayılmadı. Tek seed ve çok kez kullanılan validation; bağımsız başarı kanıtı değildir. TEST açılmadı.

GRU, 12 gizli durumlu bir katman ve skaler lineer readout ile öğrenilmiş referans hızına düzeltme verir. TRAIN'den öğrenilmiş null bandında hız sıfır tutulur. Bu bir **öğrenilmiş hız modeli + GRU düzeltmesi**; GRU tek başına veya tüm Cascade ortak eğitimi değildir. Girişler tahmin DDV, merkezlenmiş DDV, tahmin null merkezi, DDV farkı, referans hız; fizik katsayıları/Truth yok. Her aday kendi TRAIN ölçekleriyle çalışır. DDV TRAIN tahminleri mevcut ailede in-sample'dır; OOF olduğu iddia edilmez.

RAM Cascade Train/Val **69.06/91.43**, RAM Component **56.60/40.16**. Component kötüleşmesi ciddi dağılım/gürültü hassasiyeti gösterir. Angle Cascade 68.79/91.76. Varsayılan değiştirilmedi; yeni aday opt-in. Gerçek uçuş doğrulaması ve bağımsız sentetik audit hâlâ gerekli.
