# extractors/twitter_extractor.py

import tweepy
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import TWITTER_CONFIG, TARGET

class TwitterExtractor:

    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=TWITTER_CONFIG["bearer_token"],
            wait_on_rate_limit=True  # espera automática si excede límite
        )
        self.handle = TARGET["twitter_handle"]

    # ─────────────────────────────────────────
    # 1. Obtener ID del usuario
    # ─────────────────────────────────────────
    def get_user_id(self):
        user = self.client.get_user(
            username=self.handle,
            user_fields=["id", "name", "public_metrics", "description", "created_at"]
        )
        info = user.data
        print(f"\n Usuario encontrado:")
        print(f"   Nombre   : {info.name}")
        print(f"   ID       : {info.id}")
        print(f"   Followers: {info.public_metrics['followers_count']:,}")
        print(f"   Tweets   : {info.public_metrics['tweet_count']:,}")
        return info.id

    # ─────────────────────────────────────────
    # 2. Extraer sus tweets propios
    # ─────────────────────────────────────────
    def get_tweets(self, user_id, max_tweets=500):
        print(f"\n Extrayendo tweets de @{self.handle}...")
        tweets_data = []

        paginator = tweepy.Paginator(
            self.client.get_users_tweets,
            id=user_id,
            tweet_fields=[
                "text", "created_at", "public_metrics",
                "entities", "lang", "conversation_id"
            ],
            exclude=["replies"],   # solo tweets originales y RTs
            max_results=100
        ).flatten(limit=max_tweets)

        for tweet in tqdm(paginator, total=max_tweets):
            tweets_data.append({
                "id"          : tweet.id,
                "texto"       : tweet.text,
                "fecha"       : tweet.created_at,
                "likes"       : tweet.public_metrics["like_count"],
                "retweets"    : tweet.public_metrics["retweet_count"],
                "replies"     : tweet.public_metrics["reply_count"],
                "quotes"      : tweet.public_metrics["quote_count"],
                "idioma"      : tweet.lang,
                "hashtags"    : self._extract_hashtags(tweet),
                "menciones"   : self._extract_mentions(tweet),
                "conv_id"     : tweet.conversation_id
            })

        df = pd.DataFrame(tweets_data)
        df["fecha"] = pd.to_datetime(df["fecha"])
        print(f"    {len(df)} tweets extraídos")
        return df

    # ─────────────────────────────────────────
    # 3. Extraer replies que recibe (lo que dice la gente)
    # ─────────────────────────────────────────
    def get_replies(self, max_replies=1000):
        print(f"\n Extrayendo replies hacia @{self.handle}...")
        replies_data = []

        query = f"@{self.handle} lang:es -is:retweet"

        paginator = tweepy.Paginator(
            self.client.search_recent_tweets,
            query=query,
            tweet_fields=[
                "text", "created_at", "public_metrics",
                "author_id", "lang"
            ],
            max_results=100
        ).flatten(limit=max_replies)

        for tweet in tqdm(paginator, total=max_replies):
            replies_data.append({
                "id"        : tweet.id,
                "texto"     : tweet.text,
                "fecha"     : tweet.created_at,
                "autor_id"  : tweet.author_id,
                "likes"     : tweet.public_metrics["like_count"],
                "retweets"  : tweet.public_metrics["retweet_count"],
                "idioma"    : tweet.lang
            })

        df = pd.DataFrame(replies_data)
        df["fecha"] = pd.to_datetime(df["fecha"])
        print(f"    {len(df)} replies extraídos")
        return df

    # ─────────────────────────────────────────
    # Helpers privados
    # ─────────────────────────────────────────
    def _extract_hashtags(self, tweet):
        if tweet.entities and "hashtags" in tweet.entities:
            return [h["tag"] for h in tweet.entities["hashtags"]]
        return []

    def _extract_mentions(self, tweet):
        if tweet.entities and "mentions" in tweet.entities:
            return [m["username"] for m in tweet.entities["mentions"]]
        return []

    # ─────────────────────────────────────────
    # 4. Guardar datos
    # ─────────────────────────────────────────
    def save(self, df, filename):
        path = f"data/raw/{filename}_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"     Guardado en: {path}")
        return path