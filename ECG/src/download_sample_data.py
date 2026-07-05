import wfdb
import os

print("Downloading a few sample PTB-XL records for testing...")

# We will just download a few records from the first folder (00000)
records_to_download = ['00001_lr', '00002_lr', '00003_lr', '00004_lr', '00005_lr']
download_dir = os.path.join("..", "dataset", "ptb-xl", "records100", "00000")
os.makedirs(download_dir, exist_ok=True)

try:
    for record in records_to_download:
        print(f"Downloading {record}...")
        wfdb.dl_database('ptb-xl', download_dir, records=[f'records100/00000/{record}'])
    print("Download complete!")
except Exception as e:
    print(f"Error downloading: {e}")
