# test_twitter.py
from extractors.twitter_extractor import TwitterExtractor

ext = TwitterExtractor()
user_id = ext.get_user_id()       # Verifica que encuentra la cuenta
tweets  = ext.get_tweets(user_id, max_tweets=50)   # Prueba con 50 primero
print(tweets.head())