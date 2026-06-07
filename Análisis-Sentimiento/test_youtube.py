# test_youtube.py
from extractors.youtube_extractor import YouTubeExtractor

ext = YouTubeExtractor()

# PASO 1 — Buscar el canal oficial
canales = ext.get_channel_id("Samuel García Nuevo Leon")

# PASO 2 — Buscar videos donde aparece
df_videos = ext.search_videos(
    query="Samuel García",
    max_results=20        # empezamos con 20
)
print(df_videos[["titulo", "canal", "fecha"]].to_string())

# PASO 3 — Obtener estadísticas
df_videos = ext.get_video_stats(df_videos)

# PASO 4 — Guardar videos
ext.save(df_videos, "videos_sg")

# PASO 5 — Extraer comentarios
df_comentarios = ext.get_all_comments(
    df_videos,
    max_comments_per_video=50   # 50 por video para empezar
)
print(f"\n Total comentarios: {len(df_comentarios)}")
print(df_comentarios[["autor", "texto", "likes"]].head(10))

# PASO 6 — Guardar comentarios
ext.save(df_comentarios, "comentarios_sg")