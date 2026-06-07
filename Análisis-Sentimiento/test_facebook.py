# test_facebook.py
from extractors.facebook_extractor import FacebookExtractor

ext = FacebookExtractor()

df_posts, df_comments = ext.run_pipeline(
    page_url     = "https://www.facebook.com/SAMUELGARCIASEPULVEDA/",
    max_posts    = 5,    # conservador para no gastar créditos
    max_comments = 50
)

print(f"\n Posts      : {len(df_posts)}")
print(f" Comentarios: {len(df_comments)}")

if not df_posts.empty:
    ext.save(df_posts, "posts_samuel_fb")

if not df_comments.empty:
    ext.save(df_comments, "comentarios_samuel_fb")
    print(df_comments.head(5))