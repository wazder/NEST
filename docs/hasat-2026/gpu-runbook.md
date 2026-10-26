# NEST v2 — ilk doğrulanmış benchmark için GPU runbook

Amaç: Hasat 2026 finalinden önce, v2 modelini bir kez tam eğitip serbest
çözümlemeyle üç girdi koşulunda ölçmek: gerçek EEG, gaze-only ve gürültü.
Sonuç ne çıkarsa çıksın deck'e o sayı yazılır; çıkmazsa "pending" kalır.

## 0. Karar

- Sağlayıcı: RunPod, A100 80 GB (PCIe veya SXM), 1,79 $/saat (runpod.io/pricing,
  8 Ekim 2026 güncellemesi; Eylül'de 1,39-1,59 $ idi). Daha ucuz seçenek:
  L40S 48 GB, 1,09 $/saat; model 48 GB'a rahat sığar, hızı ölçülmedi.
- Alternatif: Lambda 1x A100 40 GB, 1,99 $/saat (v2 40 GB'a sığıyorsa).
- Vast.ai daha ucuz ama kesintili; checkpoint kaydını açık tut.
- ZuCo verisi kamuya açık ve de-identified (CC BY 4.0); KVKK açısından
  Türkiye dışında işlemek sorun değil. Türkçe korpus toplandığında bu geçerli
  olmayacak; onu Türkiye'de tutulan makinede eğit.

### Süre ve maliyet (tahmin, ölçülmedi; Ekim 2026)

Eski tahmin "CTC + BART, 8-12 saat, 11-19 $" idi ve ölçülmemişti. Yeni plan
CTC'yi çıkarır (bölüm 3) ve BART'ı EEG ile gaze girdisiyle birlikte eğitir.

| Koşu | Amaç | Süre | Maliyet |
|---|---|---|---|
| BART, `--input-mode eeg gaze` | Üç koşul tek checkpoint'te (asıl ölçüm) | 8-14 saat | 14-25 $ |
| Değerlendirme (`eval_conditions.py` + shuffled kontrol) | Tablo ve p değerleri | 1 saatten az | 2 $'dan az |
| Opsiyonel: BART, yalnız EEG | Ortak eğitimin EEG satırını bozmadığını kontrol | 6-10 saat | 11-18 $ |
| Toplam | | 9-25 saat | 16-45 $ |

Maliyet sütunu A100 için 1,79 $/saat ile hesaplandı. `--heldout-sentences` ile
eğitim seti 8.483 yerine 6.770 örnek, doğrulama seti 2.202 yerine 216 örnektir
(bölüm 3). Bu yüzden süre büyük olasılıkla tablodan kısa olur.

- Ortak koşu tek modlu koşudan uzun sürer: her doğrulama iki kez çözümler
  (EEG ve gaze) ve her mod eğitim adımlarının yarısını alır.
- İlk iki epoch'un süresini ölç ve tabloyu yeniden hesapla:
  toplam süre = epoch süresi x epoch sayısı (erken durdurma, patience 20).

## 1. Veri (bir kez, yerelde veya pod üzerinde)

ZuCo 1.0 .mat dosyaları OSF'ten indirilir: https://osf.io/q3zws/. Yalnız her
task'ın "Matlab files" klasörü gerekir: 12 denek x 3 task = 36 dosya. OSF
metadata'sına göre boyutlar: task1-SR 11,0 GB, task2-NR 10,35 GB, task3-TSR
8,37 GB, toplam 29,7 GB. "Raw data" ve "Preprocessed" klasörleri gerekmez.
`scripts/download_zuco.py` bunu dener; OSF bazen manuel indirme ister.

Dosyaları `ZuCo_Dataset/ZuCo/<task>/Matlab_files/` altına koy ve NEST'in
dönüştürücüsüyle pickle'a çevir. Bu dönüştürücü Wang & Ji (AAAI 2022) pickle
düzenini korur ve her fixation'lı kelimeye gaze ölçülerini de yazar
(nFixations, FFD, GD, TRT, GPT). Wang & Ji'nin kendi dönüştürücüsü yalnız
nFixations'ı tutar; gaze-only koşulu onun pickle'larıyla çalışmaz.

```bash
for task in task1-SR task2-NR task3-TSR; do
  python scripts/zuco_mat_to_pickle.py --task $task \
      --mat-dir ZuCo_Dataset/ZuCo/$task/Matlab_files \
      --out-dir ZuCo_Dataset/ZuCo/$task/pickle
done
```

Doğrula:

```bash
python scripts/inspect_zuco_mat.py ZuCo_Dataset/ZuCo/task1-SR/Matlab_files/resultsZAB_SR.mat
python src/data/zuco_pickle_dataset.py ZuCo_Dataset/ZuCo
python -c "from src.data.zuco_pickle_dataset import ZuCoWordSequenceDataset as D; \
d = D('ZuCo_Dataset/ZuCo', input_modes=('eeg', 'gaze', 'noise')); print(len(d), d[0]['gaze'][:3])"
```

Mac'te ölçüldü (26 Ekim 2026): 36 dosya indi ve dönüştü. Pickle'lar toplam
3,1 GB (task başına 1,0-1,1 GB). 156.720 fixation'lı kelimenin hepsinde gaze
ölçüleri var ve hepsinde FFD <= GD <= TRT, GD <= GPT. Eksik cümleler:
task1-SR'da ZDN 101; task2-NR'da ZJS 50 ve ZPH 50; task3-TSR'da ZGW 46,
ZKB 48 ve ZPH 89.

Pickle'ları bir kez üretip sakla; pod her açıldığında .mat indirmek gerekmesin.

## 2. Pod kurulumu

Pod'u açarken: PyTorch şablonu, container disk en az 40 GB, TCP 22 açık ve
`PUBLIC_KEY` ortam değişkeninde Mac'teki `~/.ssh/id_ed25519.pub` içeriği.
Böylece Mac'ten SSH ile bağlanılır ve pickle'lar rsync ile gönderilir:

```bash
rsync -av --progress -e "ssh -p <port>" ZuCo_Dataset/ZuCo/ root@<ip>:/workspace/nest/ZuCo_Dataset/ZuCo/ \
    --include "*/" --include "*.pickle" --exclude "*"
```

Pod içinde:

```bash
cd /workspace && git clone https://github.com/wazder/nest.git && cd nest
git checkout feat/gaze-only-control   # main'e merge edilene kadar
python -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt
# pickle'ları ZuCo_Dataset/ZuCo/ altına kopyala (rsync veya bulut depodan)
python src/data/zuco_pickle_dataset.py ZuCo_Dataset/ZuCo
python scripts/train_nest_v2.py --quick-test --model bart --input-mode eeg gaze \
    --output-dir results/smoke_bart                          # 2 epoch, 50 örnek
python scripts/eval_conditions.py --quick-test --checkpoint results/smoke_bart/best_model.pt \
    --n-permutations 200 --output results/smoke_conditions.json
```

## 3. Eğitim

**CTC bu koşuda yok.** `NEST_CTC_v2` kelime başına bir zaman adımı üretir,
ama hedef karakter dizisidir. CTC'de zaman adımı sayısı hedef uzunluğundan
az olamaz: 12 kelimelik bir cümlede 12 adım ve 58 karakter var.
`zero_infinity=True` bu örneklerin loss'unu ve gradient'ini sıfırlar, bu
yüzden model öğrenmez. Düzeltme (çıktıyı kelime başına birkaç adıma açmak ya
da token hedefi) ayrı bir iş.

Asıl ölçüm: BART, EEG ve gaze girdisiyle tek checkpoint. Batch'ler sırayla
EEG ve gaze kullanır. Checkpoint iki modun ortalama doğrulama WER'ine göre
seçilir, böylece model seçimi hiçbir koşulu kayırmaz.

**Cümleler de ayrılır (`--heldout-sentences`).** ZuCo'da her denek aynı
cümleleri okur. Ölçüldü: yalnız denek bölmesinde 1.031 benzersiz test
cümlesinin 1.031'i eğitim deneklerinde de var. Model bu durumda WER'i cümleyi
tanıyarak düşürebilir; gaze-only bunu kelime süreleriyle (kelime uzunluğu)
özellikle kolay yapar. `--heldout-sentences` cümleleri metin hash'iyle 80/10/10
böler. Eğitim yalnız eğitim cümlelerini görür; doğrulama ve test görülmemiş
cümlelerdir. Gerçek veride ölçülen boyutlar:

| Küme | Örnek | Benzersiz cümle |
|---|---|---|
| Eğitim (8 denek x eğitim cümleleri) | 6.770 | 825 |
| Doğrulama (ZKH, ZKW x doğrulama cümleleri) | 216 | 105 |
| Test, görülmemiş (ZMG, ZPH x test cümleleri) | 205 | 108 |
| Test, görülmüş (ZMG, ZPH x eğitim cümleleri, teşhis) | 1.636 | 818 |

```bash
python scripts/train_nest_v2.py --model bart --input-mode eeg gaze --heldout-sentences \
    --epochs 200 --batch-size 16 --fp16 --d-model 768 --num-layers 6 \
    --tasks task1-SR task2-NR task3-TSR --output-dir results/nest_v2_bart_eeg_gaze
```

Opsiyonel: yalnız EEG ile eğitilmiş BART. Ortak checkpoint'in EEG satırıyla
karşılaştırılır.

```bash
python scripts/train_nest_v2.py --model bart --heldout-sentences --epochs 200 --batch-size 16 \
    --fp16 --d-model 768 --num-layers 6 --tasks task1-SR task2-NR task3-TSR \
    --output-dir results/nest_v2_bart
```

`nohup ... &` ile çalıştır. `results/*/results.json`, `history.json` ve
`best_model.pt` dosyalarını pod kapanmadan indir.

Denek bölmesi (8/2/2): `get_subject_split` sıralı son iki deneği test
(ZMG, ZPH), kalanların son ikisini doğrulama (ZKH, ZKW) yapar; kalan 8
denek eğitimdir. Bölme `config.json` içine yazılır ve `eval_conditions.py`
test deneklerini buna göre kontrol eder.

## 4. Değerlendirme (asıl iş)

Test denekleri ve test cümleleri eğitimde hiç görülmedi. Aynı checkpoint ve
aynı serbest çözümleme (teacher forcing yok; mevcut kodda greedy), üç girdiyle:

- `eeg`: gerçek kelime düzeyi EEG
- `gaze`: kelime düzeyi gaze ölçüleri (nFixations, FFD, GD, TRT, GPT), ayrı projection
- `noise`: test EEG'sinin özellik başına ortalama ve std'siyle Gauss gürültüsü
  (Jo ve ark. 2024)

Üç koşul aynı cümleleri, aynı dizi uzunluğunu ve aynı maskeyi kullanır.

```bash
python scripts/eval_conditions.py --checkpoint results/nest_v2_bart_eeg_gaze/best_model.pt \
    --data-dir ZuCo_Dataset/ZuCo --n-permutations 10000 --output results/conditions_bart.json
python scripts/eval_conditions.py --checkpoint results/nest_v2_bart_eeg_gaze/best_model.pt \
    --data-dir ZuCo_Dataset/ZuCo --sentences seen --n-permutations 10000 \
    --output results/conditions_bart_seen.json
```

İlk komut asıl sonuçtur: görülmemiş cümleler (`--sentences heldout`, varsayılan).
İkincisi teşhistir: aynı test deneklerinin eğitim cümleleri. İkisi arasındaki
fark, modelin cümleleri ne kadar ezberlediğini gösterir.

Script şunları verir:

- her koşul için WER ve CER;
- her koşul çifti ve metrik için paired permutation testi (işaret çevirme,
  iki yönlü). Birim benzersiz cümledir: test denekleri aynı cümleleri okur,
  bu yüzden bir cümlenin tüm kopyalarındaki hatalar toplanır;
- 6 testin tamamı (3 çift x 2 metrik) üzerinden Holm düzeltmeli p değerleri.

JSON tüm hipotezleri ve referansları da saklar.

Ek kontrol: EEG dizilerini cümleler arasında karıştıran shuffled kontrol.
Not: `eval_noise_control.py` cümle filtresi uygulamaz, test deneklerinin tüm
cümlelerinde ölçer.

```bash
python scripts/eval_noise_control.py --checkpoint results/nest_v2_bart_eeg_gaze/best_model.pt \
    --data-dir ZuCo_Dataset/ZuCo --model bart --noise shuffled \
    --output results/noise_control_bart_shuffled.json
```

Raporlanacak tablolar (deck slayt 7 ve site "Demo" sayfası):

| Koşul | WER | CER |
|---|---|---|
| Gerçek EEG, görülmemiş denekler | ... | ... |
| Gaze-only (aynı model) | ... | ... |
| Gürültü girdisi (aynı model) | ... | ... |

| Karşılaştırma | WER farkı | p (Holm) | CER farkı | p (Holm) |
|---|---|---|---|---|
| EEG - gürültü | ... | ... | ... | ... |
| EEG - gaze | ... | ... | ... | ... |
| Gaze - gürültü | ... | ... | ... | ... |

Okuma:

- EEG ile gürültü arasında fark yoksa model EEG'yi değil dil önbilgisini
  kullanıyor.
- EEG gürültüden iyi ama gaze'den iyi değilse EEG'nin katkısı bakış
  bilgisiyle (kelime uzunluğu, sıklık) açıklanabilir.
- "EEG bakışın ötesinde bilgi taşıyor" demek için EEG hem gürültüden hem
  gaze'den anlamlı iyi olmalı (Holm sonrası).

Hangisi çıkarsa çıksın yayınlanır.

## 5. Sonucun deck'e işlenmesi

- `docs/hasat-2026/deck.html` slayt 7 ("Durum"): "Doğrulanmış WER: Henüz yok"
  kartını ölçülen tablolarla (EEG, gaze-only, gürültü ve Holm p değerleri)
  değiştir; protokolü (serbest çözümleme, üç koşul, denek bağımsız) not
  olarak ekle.
- `website/index.html` ve `website/pages/demo.html` tablolarındaki "pending"
  hücrelerini doldur; `docs/STATUS.md` ve `docs/MODEL_CARD.md` güncelle.
- `./build.sh` ile PDF'i yeniden üret.
