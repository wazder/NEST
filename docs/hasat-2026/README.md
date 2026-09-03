# Hasat 2026 başvuru paketi (NEST)

| Dosya | Ne |
|---|---|
| `form-cevaplari.md` | Forma yapıştırılacak Türkçe cevaplar ve göndermeden önce kontrol listesi |
| `deck.html` + `deck-template.css` | 13 slaytlık pitch deck kaynağı (16:9, 13,333 × 7,5 inç) |
| `build.sh` | `deck.html` → `nest-hasat-2026.pdf` (headless Chrome), sayfa boyutu ve taşma kontrolü |
| `nest-hasat-2026.pdf` | Forma yüklenecek deck |
| `logo/nest-logo-1024.png` | Forma yüklenecek kare logo (1024×1024, 18 KB) |
| `logo/` | Mark, wordmark, lockup SVG'leri; 512 px açık/koyu PNG; OG görseli |
| `gpu-runbook.md` | Final öncesi v2 eğitimi ve gürültü kontrollü değerlendirme adımları (~20 $, 1 gün) |
| `arastirma-notlari.md` | Deck ve formun dayandığı kaynaklı araştırma notları |

Deck'i güncelledikten sonra:

```bash
cd docs/hasat-2026 && ./build.sh
```

Çıktı `pages=13 size=960x540pt bad_pages=[]` ve her slayt için `overflow: []` olmalı.
