# NEST v2 — ilk doğrulanmış benchmark için GPU runbook

Amaç: Hasat 2026 finalinden önce, v2 modelini bir kez tam eğitip serbest
çözümleme ve gürültü kontrolüyle ölçmek. Maliyet yaklaşık 20 dolar, süre bir
gün. Sonuç ne çıkarsa çıksın deck'e o sayı yazılır; çıkmazsa "pending" kalır.

## 0. Karar

- Sağlayıcı: RunPod Secure Cloud, A100 80 GB, 1,39-1,59 $/saat (Eylül 2026
  fiyatı; runpod.io/pricing üzerinden teyit et). 8-12 saat = 11-19 $.
- Alternatif: Lambda 1x A100 40 GB, 1,99 $/saat (v2 40 GB'a sığıyorsa).
- Vast.ai daha ucuz ama kesintili; checkpoint kaydını açık tut.
- ZuCo verisi kamuya açık ve de-identified (CC BY 4.0); KVKK açısından
  Türkiye dışında işlemek sorun değil. Türkçe korpus toplandığında bu geçerli
  olmayacak; onu Türkiye'de tutulan makinede eğit.

## 1. Veri (bir kez, yerelde veya pod üzerinde)

ZuCo 1.0 .mat dosyaları OSF'ten indirilir: https://osf.io/q3zws/ (task1-SR,
task2-NR, task3-TSR; toplam ~66 GB). Depodaki `scripts/download_zuco.py`
bunu dener; OSF bazen manuel indirme ister.

`src/data/zuco_pickle_dataset.py`, Wang & Ji (AAAI 2022) formatındaki
pickle dosyalarını bekler: `ZuCo_Dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle`
(task2-NR ve task3-TSR için aynı düzen). Bu pickle'lar .mat dosyalarından
Wang & Ji'nin açık kaynak dönüştürücüsüyle üretilir:

```bash
git clone https://github.com/MikeWangWZHL/EEG-To-Text.git
cd EEG-To-Text
# util/ altındaki construct_dataset_mat_to_pickle_v1.py betiğini
# .mat klasörünü gösterecek şekilde çalıştır (betik adını depoda teyit et).
python util/construct_dataset_mat_to_pickle_v1.py
```

Çıktıyı `ZuCo_Dataset/ZuCo/<task>/pickle/` altına taşı, sonra doğrula:

```bash
python src/data/zuco_pickle_dataset.py ZuCo_Dataset/ZuCo
```

Pickle'ları bir kez üretip sıkıştırılmış olarak (birkaç GB) sakla; pod her
açıldığında 66 GB .mat indirmek gerekmesin.

## 2. Pod kurulumu

```bash
git clone https://github.com/wazder/nest.git && cd nest
python -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt
# pickle'ları ZuCo_Dataset/ZuCo/ altına kopyala (rsync veya bulut depodan)
python src/data/zuco_pickle_dataset.py ZuCo_Dataset/ZuCo
python scripts/train_nest_v2.py --quick-test --model ctc   # 2 epoch, 50 örnek
```

## 3. Eğitim

Önce hızlı ve ucuz olan CTC, sonra BART:

```bash
python scripts/train_nest_v2.py --model ctc  --epochs 200 --output-dir results/nest_v2_ctc
python scripts/train_nest_v2.py --model bart --epochs 200 --batch-size 16 --fp16 \
    --d-model 768 --num-layers 6 --tasks task1-SR task2-NR task3-TSR \
    --output-dir results/nest_v2_bart
```

`nohup ... &` ile çalıştır, `results/*/results.json` ve `best_model.pt`
dosyalarını pod kapanmadan indir.

## 4. Değerlendirme (asıl iş)

Test denekleri eğitimde hiç görülmedi (varsayılan: sıralı son iki denek).
Serbest çözümleme (teacher forcing yok; mevcut kodda greedy) ile WER/CER:

```bash
python scripts/train_nest_v2.py --model bart --resume results/nest_v2_bart/best_model.pt --epochs 0
```

Gürültü kontrolü (Jo ve ark. 2024): aynı checkpoint, aynı çözümleme, girdi
olarak gerçek EEG yerine eşleşen istatistikli rastgele gürültü:

```bash
python scripts/eval_noise_control.py --checkpoint results/nest_v2_bart/best_model.pt \
    --data-dir ZuCo_Dataset/ZuCo --model bart --noise gaussian --output results/noise_control_bart.json
python scripts/eval_noise_control.py --checkpoint results/nest_v2_bart/best_model.pt \
    --data-dir ZuCo_Dataset/ZuCo --model bart --noise shuffled --output results/noise_control_bart_shuffled.json
```

Raporlanacak tablo (deck slayt 7 ve site "Demo" sayfasındaki tablo):

| Koşul | WER | CER |
|---|---|---|
| Gerçek EEG, görülmemiş denekler | ... | ... |
| Gürültü girdisi (aynı model) | ... | ... |
| Fark | ... | ... |

Yalnızca "Fark" satırı çözümleme kanıtıdır. Fark sıfıra yakınsa sonuç
"model dil önbilgisini kullanıyor, EEG'yi değil" diye raporlanır; bu da
yayınlanır.

## 5. Sonucun deck'e işlenmesi

- `docs/hasat-2026/deck.html` slayt 7 ("Durum"): "Doğrulanmış WER: Henüz yok"
  kartını ölçülen üç satırla değiştir; protokolü (serbest çözümleme, gürültü kontrolü,
  denek bağımsız) not olarak ekle.
- `website/index.html` ve `website/pages/demo.html` tablolarındaki "pending"
  hücrelerini doldur; `docs/STATUS.md` ve `docs/MODEL_CARD.md` güncelle.
- `./build.sh` ile PDF'i yeniden üret.
