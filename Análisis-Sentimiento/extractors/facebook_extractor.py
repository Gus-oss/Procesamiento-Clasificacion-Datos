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

        actor_id = "apify~facebook-pages-scraper"
        run_url  = f"{self.base_url}/acts/{actor_id}/runs"

        payload = {
            "startUrls"    : [{"url": page_url}],
            "resultsLimit" : max_posts
        }

        response = requests.post(
            run_url,
            params={"token": self.token},
            json=payload
        )

        if response.status_code not in [200, 201]:
            print(f"    Error al lanzar actor: {response.status_code}")
            print(f"   Detalle: {response.text[:300]}")
            return []

        run_data = response.json()

        if "data" not in run_data:
            print(f"    Respuesta inesperada: {run_data}")
            return []

        run_id = run_data["data"]["id"]
        print(f"    Run iniciado: {run_id}")

        posts = self._wait_and_fetch(run_id)
        print(f"    {len(posts)} posts extraídos")
        return posts

    # ─────────────────────────────────────────
    # 2. Extraer comentarios de los posts
    # ─────────────────────────────────────────
    def get_comments(self, post_urls, max_comments=100):
        print(f"\n Extrayendo comentarios de {len(post_urls)} posts...")

        if not post_urls:
            print("    No hay URLs de posts para procesar")
            return []

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

        if response.status_code not in [200, 201]:
            print(f"    Error al lanzar actor: {response.status_code}")
            print(f"   Detalle: {response.text[:300]}")
            return []

        run_data = response.json()

        if "data" not in run_data:
            print(f"    Respuesta inesperada: {run_data}")
            return []

        run_id = run_data["data"]["id"]
        print(f"    Run iniciado: {run_id}")

        comments = self._wait_and_fetch(run_id)
        print(f"    {len(comments)} comentarios extraídos")
        return comments

    # ─────────────────────────────────────────
    # 3. Esperar resultado y descargar
    # ─────────────────────────────────────────
    def _wait_and_fetch(self, run_id, timeout=600):
        print(f"    Esperando resultado...")
        elapsed = 0

        while elapsed < timeout:
            status_url = f"{self.base_url}/actor-runs/{run_id}"
            status     = requests.get(
                status_url,
                params={"token": self.token}
            ).json()

            if "data" not in status:
                print(f"    Respuesta inesperada: {status}")
                return []

            state = status["data"]["status"]
            print(f"   Estado: {state} ({elapsed}s)")

            if state == "SUCCEEDED":
                dataset_id  = status["data"]["defaultDatasetId"]
                dataset_url = f"{self.base_url}/datasets/{dataset_id}/items"
                items = requests.get(
                    dataset_url,
                    params={"token": self.token, "format": "json"}
                ).json()
                print(f"    {len(items)} items descargados")
                return items

            elif state in ["FAILED", "ABORTED", "TIMED-OUT"]:
                print(f"    Run falló con estado: {state}")
                log_url = f"{self.base_url}/actor-runs/{run_id}/log"
                log     = requests.get(log_url, params={"token": self.token})
                print(f"   Log: {log.text[:500]}")
                return []

            time.sleep(20)
            elapsed += 20

        print("    Timeout alcanzado")
        return []

    # ─────────────────────────────────────────
    # 4. Pipeline completo
    # ─────────────────────────────────────────
    def run_pipeline(self, page_url, max_posts=5, max_comments=50):

        # Paso 1 — obtener posts
        posts = self.get_posts(page_url, max_posts)

        if not posts:
            print(" No se obtuvieron posts. Verifica la URL o el actor.")
            return pd.DataFrame(), pd.DataFrame()

        df_posts = pd.DataFrame(posts)
        print(f"\n Columnas disponibles en posts: {df_posts.columns.tolist()}")
        print(f" Muestra del primer post:\n{df_posts.iloc[0].to_dict()}\n")

        # Extraer URLs de posts
        post_urls = []
        for post in posts:
            url = (post.get("url")     or
                   post.get("postUrl") or
                   post.get("link")    or
                   post.get("pageUrl"))
            if url and "facebook.com" in str(url):
                post_urls.append(url)

        print(f"    {len(post_urls)} URLs de posts obtenidas")

        if not post_urls:
            print(" No se encontraron URLs en el resultado")
            print(f"   Claves disponibles: {list(posts[0].keys()) if posts else 'vacío'}")
            return df_posts, pd.DataFrame()

        # Paso 2 — obtener comentarios
        comments    = self.get_comments(post_urls, max_comments)
        df_comments = pd.DataFrame(comments) if comments else pd.DataFrame()

        if not df_comments.empty:
            for col in ["text", "commentText", "message", "body"]:
                if col in df_comments.columns:
                    df_comments = df_comments.rename(columns={col: "texto"})
                    break
            print(f"\n Columnas en comentarios: {df_comments.columns.tolist()}")

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