# Hasat 2026 — Form cevapları (NEST)

Son başvuru: 21 Eylül 2026, 23:59. Form Türkçe doldurulmalı.
Portal: https://hasat.eventiqs.com/entrepreneur

---

## Girişim / proje adı

NEST

## Tek cümlelik tanım

Okuma sırasında kaydedilen non-invaziv EEG sinyalini metne çeviren, donanımdan
bağımsız açık yapay zeka altyapısı.

## Web sitesi

https://nest-eeg.org

## Sektör

Biyoteknoloji & Sağlık. Form birden fazla etiket alıyorsa Yapay Zeka'yı da işaretle.

## Aşama

Prototip / MVP. Kod tabanı, veri hattı ve modeller mevcut; doğrulanmış benchmark
henüz yok. "Erken gelir" seçilmemeli.

## Takım büyüklüğü

1

## Kuruluş tarihi

2026 (ilk commit 12 Şubat 2026). Şirket henüz kurulmadı.

## TEKMER / Teknopark

Bağlı değilim.

## Tübitak BİGG

Almadık.

## Yatırım

Almadık.

---

## Çözülen problem & çözümünüz

ALS, inme, locked-in sendromu ve serebral palsi nedeniyle konuşamayan ya da
yazamayan insanların bugün iki seçeneği var: yavaş ya da ameliyatlı. Göz takibiyle
harf harf seçim yapan iletişim cihazları dakikada yaklaşık 10 kelimede kalıyor. Beyne yerleştirilen
implantlar (Stanford ve UCSF, 2023) dakikada 62-78 kelimeye çıkıyor ama beyin
ameliyatı gerektiriyor. Bugünkü ticari ameliyatsız EEG arayüzleri (P300/SSVEP tabanlı) ise sabit bir
menüden seçim yaptırıyor; serbest cümle üretemiyor. Sabi (2026) açık sözlüğe
yöneliyor ama kendi kapalı donanımına bağlı ve bağımsız doğrulanmış sonucu yok. Dünyada yaklaşık 97 milyon kişi karmaşık
iletişim ihtiyacı taşıyor; Türkiye'de tahmini 8-10 bin ALS hastası var ve yılda
125-200 bin inme yaşanıyor.

NEST, okuma sırasında kaydedilen non-invaziv EEG sinyalini açık sözlüklü metne
çeviren bir yazılım katmanı. 105 kanallı EEG, kelime hizalı 840 boyutlu frekans
özniteliklerine dönüştürülüyor; 6 katmanlı bir Transformer encoder bu diziyi
işliyor, çapraz dikkatli bir BART decoder metni üretiyor. Donanım üretmiyoruz:
araştırma amaçlı kullanılan her EEG sistemiyle çalışan, açık kaynaklı (MIT) bir
altyapıyız. Bugünkü veri okuma sırasında kaydedilmiş EEG; hastanın kendi ürettiği
cümleyi çözmek yol haritasının ikinci fazındaki araştırma sorusu.

Alanın yayınlanmış sonuçları 2024'te sorgulandı: önceki modellerin skorları
teacher forcing ile şişmiş, gürültü girdisiyle aynı sonucu veriyordu. NEST bu
eleştiriden sonra kuruldu. Değerlendirmemiz serbest çözümleme (teacher forcing yok) ve
denekten bağımsız bölme (8 denekle eğit, görülmemiş 2 denekte test) kullanıyor.
Ölçmediğimiz hiçbir sayıyı yayınlamıyoruz; ilk benchmark, protokolüyle birlikte
raporlanacak.

---

## Traction / metrikler

Ürün araştırma altyapısı aşamasında; gelir ve yatırım yok. Elimizdeki
doğrulanabilir durum:

**Kod ve model.** 24.137 satır Python: ön işleme (band-pass, ICA), 6 model
mimarisi, eğitim ve değerlendirme betikleri. NEST v2 CTC modeli 45,3 milyon,
NEST v2 BART modeli 184 milyon (64,7 milyonu eğitilebilir) parametreyle
oluşturulup doğrulandı. Pruning, quantization, ONNX/TorchScript export ve
streaming modülleri yazıldı, henüz ölçülmedi.

**Veri.** ZuCo korpusu (66 GB, 12 denek, 105 kanal) için veri hattı kodlandı ve
gerçek veriyle çalıştırıldı; 12.071 cümle-denek çifti işleniyor. Ham veri şu an
diskte değil, GPU koşusu öncesi yeniden indirilecek. Alanın standart açık veri
seti; CC BY 4.0 lisanslı, ticari kullanıma açık.

**Eğitim.** İlk nesil model gerçek ZuCo verisinde 100 epoch eğitildi (12.071
örnek, 5,4 saat, M2 dizüstü); loss düzenli düştü. Kelime hata oranı (WER) henüz
ölçülmedi. v2 modelinin ilk tam eğitimi bulutta 8-12 saatlik, yaklaşık 20 dolarlık
bir A100 işi; sıradaki adım bu ve sonucu ne çıkarsa çıksın gürültü kontrolüyle
birlikte yayınlanacak.

**Dürüstlük.** Şubat 2026'da loss değerinden türetilmiş bir WER tahminini
yayından geri çektik. Değerlendirme kodumuz serbest çözümleme ve denekten
bağımsız protokol kullanıyor; gürültü kontrolü ilk raporun parçası olacak.

**Marka ve site.** nest-eeg.org yayında; kod github.com/wazder/nest adresinde
açık kaynak (MIT).

**Ekip ve akademik dayanak.** Tek kurucu. Proje İstanbul Ticaret Üniversitesi'nde
bir araştırma olarak başladı; etik kurul ve EEG kaydı için üniversite iş birliği
planlanıyor. Kurucunun ölçüm disiplini akademik bir yayınla kanıtlı: IEEE SIU 2026'da
yayımlanan çok dilli VLA çalışmasının ilk yazarı (ieeexplore.ieee.org/document/11636432).

---

## Hasat 2026 yarışmasından beklentileriniz nelerdir?

Üç şey. Birincisi hibe ve sürekli hesaplama erişimi. Tek bir eğitim ucuz, ama
hiperparametre taramaları, Türkçe korpus için model eğitimleri ve veri toplamanın
gerçek maliyetleri (EEG laboratuvar zamanı, katılımcı ödemeleri) için kaynak
gerekiyor. Bu bir yatırım turu başvurusu değil; tohum yatırımı ilk benchmark'tan
sonra. İkincisi klinik ekosistem
erişimi: İSEK ve Teknopark İstanbul sağlık ağı üzerinden nöroloji ve
rehabilitasyon bölümleriyle araştırma amaçlı bir pilot çalışma ve Türkçe EEG-okuma
korpusu için katılımcı erişimi. Üçüncüsü şirketleşme, patent ve mentorluk: Cube
kuluçka programı, üniversite spin-off yapısı ve tıbbi cihaz (MDR/TİTCK) yol haritası için
danışmanlık. Karşılığında ölçülebilir çıktı veriyoruz: benchmark raporu, açık
korpus, ilk laboratuvar lisansları.

---

## Dosyalar

- **Logo (kare 1:1, PNG):** `logo/nest-logo-1024.png` (1024x1024, 18 KB). Hazır.
- **Pitch deck (PDF):** `nest-hasat-2026.pdf` (13 slayt, 16:9). `./build.sh` ile
  yeniden üretilir.

## Göndermeden önce

1. Ekip slaytı tek kurucu olarak yazıldı; formda takım büyüklüğü 1.
2. ALS-MNH Derneği'nin güncel hasta tahminini (als.org.tr) kontrol et; 8-10 bin
   aralığı derneğin kendi tahmini, Sağlık Bakanlığı kaydı yok.
3. Kütahya Dumlupınar Üniversitesi'nin TÜBİTAK projesi Türkçe bir EEG korpusu
   yayımladıysa "ilk Türkçe korpus" ifadesini yumuşat.
4. Formdaki e-posta ve LinkedIn (linkedin.com/in/wazder) doğru.
5. v2 eğitimi için `docs/hasat-2026/gpu-runbook.md` hazır (~20 $, bir gün); acelesi yok.
6. Lisans MIT (LICENSE dosyası güncellendi).
7. IEEE Xplore 11636432 bağlantısını aç, başlık ve yazar sırasını doğrula.
8. Deck'te tarih taahhüdü ve kaynak tahmini yok; istersen ekle.
9. Şirket kurulunca marka tescili için TÜRKPATENT araması yaptır (9, 10, 42.
   sınıflar). "NEST" adı NEST Simulator ve Google Nest ile çakışıyor; yedek ad
   "NEST Neuro" (nestneuro.ai boşta görünüyor).
