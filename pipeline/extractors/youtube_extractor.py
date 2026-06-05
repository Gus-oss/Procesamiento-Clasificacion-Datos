# extractors/youtube_extractor.py

from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import YOUTUBE_CONFIG, TARGET

class YouTubeExtractor:

    def __init__(self):
        self.youtube = build(
            "youtube", "v3",
            developerKey=YOUTUBE_CONFIG["api_key"]
        )
        self.nombre = TARGET["nombre"]

    # ─────────────────────────────────────────
    # 1. Buscar canal
    # ─────────────────────────────────────────
    def get_channel_id(self, query="Adrian de la Garza Nuevo Leon"):
        print(f"\n Buscando canal: {query}")
        request = self.youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=5
        )
        response = request.execute()

        canales = []
        for item in response["items"]:
            canales.append({
                "channel_id"  : item["snippet"]["channelId"],
                "nombre"      : item["snippet"]["title"],
                "descripcion" : item["snippet"]["description"]
            })
            print(f"    {item['snippet']['title']} → {item['snippet']['channelId']}")

        return canales

    # ─────────────────────────────────────────
    # 2. Buscar videos donde aparece o lo mencionan
    # ─────────────────────────────────────────
    def search_videos(self, query="Adrian de la Garza", max_results=50):
        print(f"\n Buscando videos: '{query}'")
        videos = []
        next_page_token = None

        while len(videos) < max_results:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                relevanceLanguage="es",
                regionCode="MX",
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                videos.append({
                    "video_id"    : item["id"]["videoId"],
                    "titulo"      : item["snippet"]["title"],
                    "canal"       : item["snippet"]["channelTitle"],
                    "fecha"       : item["snippet"]["publishedAt"],
                    "descripcion" : item["snippet"]["description"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        df = pd.DataFrame(videos)
        print(f"    {len(df)} videos encontrados")
        return df

    # ─────────────────────────────────────────
    # 3. Obtener estadísticas de cada video
    # ─────────────────────────────────────────
    def get_video_stats(self, df_videos):
        print(f"\n Obteniendo estadísticas de {len(df_videos)} videos...")
        stats = []

        # YouTube acepta hasta 50 IDs por llamada
        video_ids = df_videos["video_id"].tolist()
        chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]

        for chunk in chunks:
            request = self.youtube.videos().list(
                part="statistics,contentDetails",
                id=",".join(chunk)
            )
            response = request.execute()

            for item in response["items"]:
                s = item.get("statistics", {})
                stats.append({
                    "video_id"         : item["id"],
                    "vistas"           : int(s.get("viewCount", 0)),
                    "likes"            : int(s.get("likeCount", 0)),
                    "comentarios_total": int(s.get("commentCount", 0)),
                    "duracion"         : item["contentDetails"]["duration"]
                })

        df_stats = pd.DataFrame(stats)
        df_final = df_videos.merge(df_stats, on="video_id", how="left")
        print(f"     Estadísticas obtenidas")
        return df_final

    # ─────────────────────────────────────────
    # 4. Extraer comentarios de un video
    # ─────────────────────────────────────────
    def get_comments(self, video_id, titulo="", max_comments=200):
        comments = []
        next_page_token = None

        try:
            while len(comments) < max_comments:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    order="relevance",
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response["items"]:
                    c = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "video_id"  : video_id,
                        "titulo_video": titulo,
                        "autor"     : c["authorDisplayName"],
                        "texto"     : c["textDisplay"],
                        "likes"     : c["likeCount"],
                        "fecha"     : c["publishedAt"],
                        "respuestas": item["snippet"]["totalReplyCount"]
                    })

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

        except Exception as e:
            # Algunos videos tienen comentarios desactivados
            print(f"       Video {video_id}: comentarios desactivados")

        return comments

    # ─────────────────────────────────────────
    # 5. Extraer comentarios de múltiples videos
    # ─────────────────────────────────────────
    def get_all_comments(self, df_videos, max_comments_per_video=100):
        print(f"\n Extrayendo comentarios de {len(df_videos)} videos...")
        all_comments = []

        for _, row in tqdm(df_videos.iterrows(), total=len(df_videos)):
            comments = self.get_comments(
                video_id=row["video_id"],
                titulo=row["titulo"],
                max_comments=max_comments_per_video
            )
            all_comments.extend(comments)

        df = pd.DataFrame(all_comments)
        print(f"    {len(df)} comentarios extraídos en total")
        return df
    # ─────────────────────────────────────────
    # # 6. Guardar datos
    # # ─────────────────────────────────────────
    def save(self, df, filename):
     # Ruta absoluta basada en la ubicación del archivo extractor
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      output_dir = os.path.join(base_dir, "data", "raw")
      os.makedirs(output_dir, exist_ok=True)
    
      path = os.path.join(output_dir, f"{filename}_{datetime.now().strftime('%Y%m%d')}.csv")
      df.to_csv(path, index=False, encoding="utf-8-sig")
      print(f"   💾 Guardado en: {path}")
      return path