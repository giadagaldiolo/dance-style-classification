import pandas as pd
import os
import urllib.request
from tqdm import tqdm

TARGET = "gPO_sFM"  # Sostituisci con il target desiderato

os.makedirs("videos", exist_ok=True)
print("Cartella videos creata in:", os.path.abspath("videos"))

url = "https://aistdancedb.ongaaccel.jp/v1.0.0/data/video_raw/10M/raw_10M_all_video_url.csv"
df = pd.read_csv(url, header=None)

df = df[df[0].str.contains(TARGET)]
print(f"Trovati {len(df)} video per il target '{TARGET}'")

for video_url in tqdm(df[0].head(5)):  # LIMITA per test
    filename = video_url.split("/")[-1]
    path = os.path.join("videos", filename)
    print("Scarico:", video_url, "->", os.path.abspath(path))

    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(video_url, path)
            print("OK:", filename)
        except Exception as e:
            print("ERRORE:", video_url, "->", repr(e))
    else:
        print("Già presente, salto:", filename)