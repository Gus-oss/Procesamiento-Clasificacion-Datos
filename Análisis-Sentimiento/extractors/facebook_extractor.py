# extractors/facebook_extractor.py
import requests
import pandas as pd
import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.credentials import APIFY_CONFIG

class FacebookExtractor:

    def __init__(self):
        self.token    = APIFY_CONFIG["api_token"]
        self.base_url = "https://api.apify.com/v2"

    # ─────────────────────────────────────────
    # 1. Extraer posts de una página
    # ─────────────────────────────────────────
    def get_posts(self, page_url, max_posts=20):
        print(f"\n Extrayendo posts de: {page_url}")

        # Lanzar el actor de páginas
        actor_id = "apify~facebook-pages-scraper"
        run_url  = f"{self.base_url}/acts/{actor_id}/runs"

        payload = {
            "startUrls"     : [{"url": page_url}],
            "resultsLimit"  : max_posts
        }

        response = requests.post(
            run_url,
            params={"token": self.token},
            json=payload
        )
        run_data = response.json()
        run_id   = run_data["data"]["id"]
        print(f"    Run iniciado: {run_id}")

        # Esperar a que termine
        posts = self._wait_and_fetch(run_id)
        print(f"    {len(posts)} posts extraídos")
        return posts

    # ─────────────────────────────────────────
    # 2. Extraer comentarios de los posts
    # ─────────────────────────────────────────
    def get_comments(self, post_urls, max_comments=100):
        print(f"\n Extrayendo comentarios de {len(post_urls)} posts...")

        actor_id = "apify~facebook-comments-scraper"
        run_url  = f"{self.base_url}/acts/{actor_id}/runs"

        payload = {
            "startUrls"    : [{"url": url} for url in post_urls],
            "resultsLimit" : max_comments
        }

        response = requests.post(
            run_url,
            params={"token": self.token},
            json=payload
        )
        run_data = response.json()
        run_id   = run_data["data"]["id"]
        print(f"    Run iniciado: {run_id}")

        comments = self._wait_and_fetch(run_id)
        print(f"    {len(comments)} comentarios extraídos")
        return comments

    # ─────────────────────────────────────────
    # 3. Esperar resultado y descargar
    # ─────────────────────────────────────────
    def _wait_and_fetch(self, run_id, timeout=300):
        print(f"    Esperando resultado...")
        elapsed = 0

        while elapsed < timeout:
            status_url = f"{self.base_url}/actor-runs/{run_id}"
            status     = requests.get(
                status_url,
                params={"token": self.token}
            ).json()

            state = status["data"]["status"]
            print(f"   Estado: {state} ({elapsed}s)")

            if state == "SUCCEEDED":
                # Descargar dataset
                dataset_id  = status["data"]["defaultDatasetId"]
                dataset_url = f"{self.base_url}/datasets/{dataset_id}/items"
                items = requests.get(
                    dataset_url,
                    params={"token": self.token, "format": "json"}
                ).json()
                return items

            elif state in ["FAILED", "ABORTED", "TIMED-OUT"]:
                print(f"    Run falló: {state}")
                return []

            time.sleep(15)
            elapsed += 15

        print("    Timeout alcanzado")
        return []

    # ─────────────────────────────────────────
    # 4. Pipeline completo
    # ─────────────────────────────────────────
    def run_pipeline(self, page_url, max_posts=20, max_comments=100):

        # Paso 1 — obtener posts
        posts    = self.get_posts(page_url, max_posts)
        df_posts = pd.DataFrame(posts)

        # Extraer URLs de posts
        post_urls = []
        for post in posts:
            url = post.get("url") or post.get("postUrl")
            if url:
                post_urls.append(url)

        print(f"\n    {len(post_urls)} URLs de posts obtenidas")

        # Paso 2 — obtener comentarios
        comments    = self.get_comments(post_urls, max_comments)
        df_comments = pd.DataFrame(comments)

        # Normalizar columna de texto
        if "text" in df_comments.columns:
            df_comments = df_comments.rename(columns={"text": "texto"})
        elif "commentText" in df_comments.columns:
            df_comments = df_comments.rename(columns={"commentText": "texto"})

        return df_posts, df_comments

    # ─────────────────────────────────────────
    # 5. Guardar
    # ─────────────────────────────────────────
    def save(self, df, filename):
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(
            output_dir,
            f"{filename}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"    Guardado en: {path}")
        return path