import praw
import pandas as pd
from tabulate import tabulate

# Reemplaza con tus credenciales
reddit = praw.Reddit(client_id='NoDVu9joldZigcTMYKkFCw',
                     client_secret='O1Nes42qcod6ZuqMBrw3xp2Jqh2RjA',
                     user_agent='script_noticias_reddit(by u/MartinPueblaRivera)')

# Elegir subreddit (por ejemplo: WallStreetBets)
subreddit = reddit.subreddit("WallStreetBets")

# Buscar posts populares
posts = []
for post in subreddit.hot(limit=10):  # Cambia limit
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

# Crear DataFrame
df = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

