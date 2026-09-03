# NEST — startup araştırma notları (3 Eylül 2026)

Deck, form ve site metinlerinin dayandığı bulgular. Her satırın kaynağı var;
"türetilmiş" yazanlar hesaplama, kaynak değil. Güven düzeyi: (Y) yüksek, (O) orta, (D) düşük.

## 1. Alan: EEG'den metne, dürüst durum

- Wang & Ji, AAAI 2022: ZuCo üzerinde BLEU-1 %40,1; değerlendirme teacher forcing ile. (Y) https://ojs.aaai.org/index.php/AAAI/article/view/20472
- DeWave, NeurIPS 2023: BLEU-1 %41,35; aynı protokol. (Y) https://arxiv.org/abs/2309.14030
- BELT, 2023: BLEU-1 %42,31. (Y) https://arxiv.org/abs/2309.12056
- Jo ve ark. 2024 "Are EEG-to-Text Models Working?": teacher forcing kaldırılınca gürültü girdisi gerçek EEG ile aynı skoru alıyor. DeWave'in yazarı bu eleştirinin ortak yazarı. (Y) https://arxiv.org/abs/2405.06459
- arXiv 2607.18749 (Temmuz 2026): BrainBART, BELT, DeWave için serbest çözümlemede WER %99-121, BLEU-1 7-16; COFETT benchmark önerisi. (O) https://arxiv.org/html/2607.18749v1
- "Escaping the BLEU Trap" (arXiv 2603.03312, 2026): 4 sınıflı anlamsal sınıflandırmada %50 (şans %25), içerik geri çağırma %2,7. (O) https://arxiv.org/abs/2603.03312
- Meta Brain2Qwerty (Şubat 2025): MEG karakter hata oranı %32 (en iyi %19), EEG %67; kapalı görev, gerçek zamanlı değil. (Y) https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/
- Brain2Qwerty v2 (Haziran 2026, BCBL ile): MEG kelime doğruluğu ortalama %61, en iyi %78. (Y) https://www.marktechpost.com/2026/06/30/meta-ai-releases-brain2qwerty-v2-a-non-invasive-meg-brain-to-text-pipeline-decoding-typed-sentences-at-61-word-accuracy/
- İnvaziv referans: Willett ve ark., Nature 2023: 62 kelime/dk, WER %9,1 (50 kelime) / %23,8 (125 bin kelime). Metzger ve ark., Nature 2023: 78 kelime/dk, medyan WER %25. (Y) https://www.nature.com/articles/s41586-023-06377-x · https://www.nature.com/articles/s41586-023-06443-4
- Neuralink için hakemli WER/kelime-dakika verisi yok; dolaşan "120 kelime/dk" iddiası kaynaksız, kullanma. (D)
- ZuCo 1.0: 12 denek, 105 kanal, 1.107 cümle, 21.629 kelime; lisans CC BY 4.0 (OSF API ile doğrulandı), ticari kullanıma açık, atıf zorunlu. (Y) https://api.osf.io/v2/nodes/q3zws/ · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6289117/
- Türkçe EEG-okuma korpusu yok. En yakın çalışma: Kütahya Dumlupınar Üniversitesi TÜBİTAK projesi (NeuroSilentia), Arapça alfabe hayali konuşma verisiyle; Türkçe korpus yayımlanmadı. (O) https://ibro.org/author-interview-ai-imagined-speech-decoding/

## 2. Rekabet

- İnvaziv: Neuralink 656,5 M $ Seri E (Haziran 2025, 9,6 milyar $ değerleme); Synchron 200 M $ Seri D (Kasım 2025); Precision 102 M $ ve ilk FDA onayı (2025); Paradromics; Science Corp 230 M $ (Mart 2026); Merge Labs 250 M $ seed (Ocak 2026, ultrason). (Y/O)
- Non-invaziv donanım: Emotiv, OpenBCI, Kernel, Muse, Neurable (35 M $ Seri A, Aralık 2025), BrainCo (272 M $, Ocak 2026), Cognixion (P300/SSVEP seçim, kapalı sözlük; FDA Breakthrough 2023, onay yok). Hiçbiri açık sözlüklü metin çözümlemiyor. (O)
- Non-invaziv metin: Sabi (Nisan 2026, Khosla; 70-100 bin sensörlü bere, kendi modeli, 2026 sonu erken erişim; bağımsız WER yok). En yakın rakip. (Y) https://insidebci.com/market/2026-04-16-sabi-emerges-from-stealth-with-a-thought-to-text-beanie-and-khosla-backing/
- Farklı modalite: AlterEgo (EMG, 35 M $), Wisear (EMG, Naqi tarafından satın alındı). (O)
- Türkiye: EEG-metin ticari girişim bulunamadı; Neurocess (Ankara, 2021, giyilebilir) tek nörotekno girişim. (D)
- BCI risk sermayesi: 2024 ~0,4 milyar $, 2025 ~1,0-1,08 milyar $ (implant payı %92,5); 2026 ilk 7 ay ~0,93-1 milyar $, non-invaziv payı ~%29. (O) https://newmarketpitch.com/blogs/news/brain-computer-interface-funding-trends
- Donanımdan bağımsız, üçüncü taraf EEG cihazlarına lisanslanan saf çözümleme yazılımı satan şirket bulunamadı. (O)

## 3. Pazar ve hasta havuzu

- BCI pazarı 2024/25: 0,24 milyar $ (Fortune BI), 0,26 (MarketsandMarkets), 2,44 (Grand View), 2,94 (Precedence; non-invaziv %82). Kapsam farkı; aralık olarak ver. (O)
- AAC cihazları: 2,32 milyar $ (2025), %11,3 CAGR (TBRC). (O)
- Karmaşık iletişim ihtiyacı: dünyada ~97 milyon kişi (Beukelman & Light 2020; WHO değil). (O)
- ALS: dünyada 222-250 bin (Neurology 2023); Türkiye 8-10 bin (ALS-MNH Derneği tahmini, bakanlık kaydı yok; bir kaynakta 6-8 bin). (Y/D) https://als.org.tr/als-mnh-hastaligi-nedir/
- İnme Türkiye: yılda 125.345 (GBD 2019, Topçuoğlu, Türk Nöroloji Dergisi 2022) ile ~200 bin (Türk Beyin Damar Hastalıkları Derneği, 2024). Aralık ver. (Y/D)
- Locked-in: prevalans verisi yok; Fransa/Norveç 0,75-0,94/100 bin → Türkiye ~640-800 (türetilmiş). (D)
- İnme sonrası afazi: başlangıçta %25-50; okuma-anlama bozulabilir, NEST için ALS ve locked-in daha uygun. (Y)

## 4. İş modeli, düzenleme, finansman

- Sıralama: araştırma aracı (Emotiv, g.tec, Neurable, Kernel, IDUN yolu) → klinik. Cognixion (~29 M $, Breakthrough 2023) hâlâ onaysız; Synchron 200 M $ ile pivotal deneme. (O)
- Araştırma yazılımı fiyat referansı: EmotivPRO 1.068-2.689 $/yıl. (Y) https://www.emotiv.com/pages/pro-licensing-table-v1
- Düzenleme: teşhis/tedavi iddiası yoksa FDA SaMD, EU MDR Rule 11 ve TİTCK kapsamı dışında; Türkiye 2021'den beri MDR ile uyumlu, CE = TİTCK tanıması. (Y)
- Geri ödeme: Medicare yalnız özel SGD donanımı; Türkiye SUT'ta AAC/BCI kalemi bulunamadı. Deck'te geri ödeme vaat etme. (O/D)
- TÜBİTAK 1512 kapandı (2024-1'den itibaren 1812'ye dönüştü). 1812 BiGG Yatırım: %3 hisse karşılığı 1.350.000 TL (2026; bazı sayfalarda eski 900.000 TL), öğrenci veya mezun, şirket ortaklığı olmayan başvurabilir; dilutif. 1812 Faz 3 (BİGG+): 8 M TL'ye kadar, %10 hisse, 2026/1 çağrısı 5 Ekim - 20 Kasım 2026. 1507: 3,5 M TL'ye kadar; 1501: 20 M TL'ye kadar (şirket şart). (Y/O) https://tubitak.gov.tr/en/funds/industrial/national-support-programs/1812-investment-based-entrepreneurship-support-program-bigg-investment
- Teknopark İstanbul Cube: 18 aya kadar ücretsiz ofis, mentorluk, lab; 4691 sayılı kanun: yazılım/Ar-Ge kazancı kurumlar vergisi istisnası 2028 sonuna kadar. (O/Y)
- EIC Accelerator: Türkiye tam üye ülke, 2,5 M €'ya kadar hibe + yatırım. (O)
- İstanbul Ticaret Üniversitesi'nin Teknopark İstanbul'un kurucu ortağı olduğu iddiası düşük güvenli; deck'te kullanılmadı. (D)

## 5. Hasat 2026

- Son başvuru 21 Eylül 2026 23:59; değerlendirme Ekim; finalist seçimi Kasım; final Aralık, Turgut Özal Etkinlik Merkezi. Havuz 100 M TL'ye kadar (yatırım, hibe, ofis, patent, şirketleşme, mentorluk). Şirket kurulmamış ekipler başvurabilir. Jüri ve rubrik yayımlanmadı. (Y) https://foundern.com/2026/08/hasat-2026/
- En yakın ölçüt: Cube Incubation'ın altı kriteri: tematik uyum, teknik/Ar-Ge kalitesi, pazar talebi, ticarileşme potansiyeli, ekip yetkinliği, finansal sürdürülebilirlik. (O) https://www.cubeincubation.com/sikca-sorulan-sorular
- Formda sektör olarak "Biyoteknoloji & Sağlık" ve mümkünse "Yapay Zeka" işaretle. Biocube (ıslak lab) uygun değil; İSEK (Acıbadem üyesi) klinik pilot için doğal ağ. (O)

## 6. KVKK ve veri

- EEG, KVKK m.6 kapsamında sağlık verisi (özel nitelikli). Özel şirket için tek kullanılabilir hukuki dayanak açık rıza: dar amaçlı, aydınlatma sonrası, ayrı opt-in, geri alınabilir. (Y) https://www.kvkk.gov.tr/Icerik/2037/Acik-Riza-Alirken-Dikkat-Edilecek-Hususlar
- Kurul kararı 2018/10: özel nitelikli veri için yazılı politika, personel eğitimi ve gizlilik taahhüdü, erişim matrisi, şifreli saklama ve iletim. (Y) https://www.kvkk.gov.tr/Icerik/4110/2018-10
- VERBİS: ana faaliyeti özel nitelikli veri olan şirketler kayıt yaptırır; 2025/1572 kararıyla 10 çalışan ve 10 M TL altı için istisna (teyit et). (O)
- Yurt dışı aktarım (m.9, 2024 reformu): Türk katılımcı EEG'sini yurt dışı GPU'da eğitmek için yeterlilik kararı veya standart sözleşme gerekir; ZuCo (kamuya açık, de-identified) için sorun yok. (O)

## 7. Marka

- "NEST": NEST Simulator (aynı alan, nest-simulator.org, github.com/nest), Google Nest (ABD marka ailesi, sınıf 9), NestJS ile çakışıyor. Türkiye'de engelleyici tescil bulunamadı ama TÜRKPATENT araması yapılamadı. Yedek: "NEST Neuro" (nestneuro.ai DNS kaydı yok, github.com/nestneuro boş; .com/.org park halinde). (Y/O)
- Tasarım DNA'sı (camellayer.com, icu.wazder.com): Inter, sıfır köşe yarıçapı, 20 px nokta deseni, büyük harf geniş aralıklı etiketler, sıkı ağır wordmark, semantik renk tokenları. Seçilen accent: Signal Violet #4338ca (açık) / #a99bf5 (koyu); kontrast 7,9:1 ve 8,2:1. (Y)

## 8. GPU maliyeti

- RunPod Secure Cloud A100 80 GB: 1,39-1,59 $/saat → 8-12 saat 11-19 $. Lambda 1x A100 40 GB 1,99 $/saat. Vast.ai 0,40-1,10 $/saat (kesintili). AWS/GCP tek GPU için 3-20 kat pahalı. (Y) https://www.runpod.io/pricing
- 5-10 koşuluk hiperparametre taraması ~75-190 $. (türetilmiş)

## 9. Depo gerçekleri (3 Eylül 2026, kod okunarak doğrulandı)

- 24.137 satır Python (src 14.440; scripts 5.782; tests 2.009); 44 commit, 12 Şubat - 7 Nisan 2026.
- NEST_CTC_v2 45.310.637; NEST_BART_v2 184.424.592 toplam / 64.697.232 eğitilebilir parametre (örneklenerek doğrulandı).
- eval_bart(): model.generate, beam 4; eval_ctc(): greedy; WER/CER Levenshtein. Teacher forcing yok.
- Tek tamamlanmış eğitim: ad hoc SimpleLSTMDecoder (~2,78 M), 100 epoch, 12.071 örnek, 5,4 saat (M2); WER hesaplanmadı. "26,1% WER" detailed_analysis.py'deki loss→WER sezgisel formülünden; ölçüm değil; f8d6675 ile kaldırıldı.
- Test paketi: 80 test fonksiyonu, import hatalarıyla çalışmıyor. pyproject'te paket metadata'sı yok; pip install -q -e . çalışmaz; NESTDecoder sınıfı yok.
- ZuCo verisi (66 GB) artık diskte değil; pickle'lar Wang & Ji dönüştürücüsüyle yeniden üretilmeli (bkz. gpu-runbook.md).
