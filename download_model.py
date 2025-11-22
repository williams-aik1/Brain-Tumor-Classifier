import gdown

# Google Drive DIRECT download link
url = "https://drive.google.com/uc?id=1umY4PgZJLdFJeXiv93ZqmZKN1Wa3VNHT"

output = "brain_tumor_model.keras"

print("Downloading model...")
gdown.download(url, output, quiet=False, fuzzy=True)
print("Model downloaded successfully!")
