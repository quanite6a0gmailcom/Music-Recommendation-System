import pandas as pd

SOURCE_FILE = r'spotify_songs.csv/spotify_songs_processed.csv'
FINAL_FILE = r'spotify_songs.csv/spotify_songs_final.csv'

cols_to_use = ['track_artist','track_name','track_popularity','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','lyrics']

df = pd.read_csv(SOURCE_FILE,usecols=cols_to_use,encoding='utf-8-sig')

df.to_csv(FINAL_FILE,index=False,encoding='utf-8-sig')

print(f'Saved file in :{FINAL_FILE}')