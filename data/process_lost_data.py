import pandas as pd

DATA_PATH = r"final data/spotify_songs_final_comments_merged.csv" 

df = pd.read_csv(DATA_PATH,encoding="utf-8-sig")

df['comments'] = df['comments'].fillna('This is a normal song for everyone.')

df['lyrics'] = df['lyrics'].fillna(df['track_name'])


df.to_csv(DATA_PATH,index=False,encoding='utf-8-sig')

print(df.isna().sum())