from huggingface_hub import snapshot_download

print("İndirme başlıyor... Sadece ZuCo klasörü indirilecek.")

# Bu komut indirmeyi yarıda kesersen, tekrar çalıştırdığında kaldığı yerden devam eder.
snapshot_download(
    repo_id="NiallMcGuire12/ZuCo",
    repo_type="dataset",
    local_dir="./ZuCo_Dataset",
    allow_patterns=["ZuCo/*"], # SADECE EEG verisini indirir, ses dosyalarını (Audio-Book) indirmez.
    resume_download=True
)

print("İşlem tamamlandı.")