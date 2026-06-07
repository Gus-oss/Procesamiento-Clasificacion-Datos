# test_facebook.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.facebook_extractor import FacebookExtractor

ext = FacebookExtractor()

df_posts, df_comments = ext.run_pipeline(
    page_url     = "https://www.facebook.com/SAMUELGARCIASEPULVEDA/",
    max_posts    = 3,
    max_comments = 50
)

print(f"\nPosts      : {len(df_posts)}")
print(f"Comentarios: {len(df_comments)}")

if not df_posts.empty:
    ext.save(df_posts, "posts_samuel_fb")

if not df_comments.empty:
    ext.save(df_comments, "comentarios_samuel_fb")
    print(df_comments[["profileName", "texto", "likesCount"]].head(10))