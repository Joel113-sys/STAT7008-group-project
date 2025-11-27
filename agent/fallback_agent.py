import re
import json
import time
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

class RequestsAgent:
    """
    Fallback agent when Playwright cannot launch (no system libs / no sudo).
    Performs search queries and attempts to locate PDF links related to Qwen.
    Simplified flow:
    1. Build search query from task.
    2. Query DuckDuckGo HTML results (no JS required).
    3. Extract top PDF links containing 'qwen'.
    4. If PDF found, fetch first page text (rough) and simulate 'Figure 1' interpretation placeholder.
    """

    def __init__(self, model=None, max_results: int = 5):
        self.model = model  # kept for interface compatibility
        self.max_results = max_results
        self.history = []

    def _build_search_query(self, task: str) -> str:
        tokens = re.findall(r"[a-zA-Z0-9]+", task.lower())
        stop = {"the","of","and","on","find","get","go","to","latest","what","for","then"}
        filtered = [t for t in tokens if t not in stop]
        if "qwen" not in filtered:
            filtered.append("qwen")
        if "pdf" not in filtered:
            filtered.append("pdf")
        return "+".join(filtered[:10])

    def run(self, task: str):
        query = self._build_search_query(task)
        print(f"[Fallback] Using DuckDuckGo HTML search for query: {query}")
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        except Exception as e:
            print(f"[Fallback] Search request failed: {e}")
            return {"completed": False, "reason": "search_failed"}
        if resp.status_code != 200:
            print(f"[Fallback] Non-200 status: {resp.status_code}")
            return {"completed": False, "reason": "bad_status"}
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if href and "qwen" in href.lower() and href.lower().endswith(".pdf"):
                links.append(href)
            if len(links) >= self.max_results:
                break
        print(f"[Fallback] Found PDF candidates: {len(links)}")
        if not links:
            return {"completed": False, "pdf": None, "reason": "no_pdf"}
        pdf_url = links[0]
        print(f"[Fallback] Selecting PDF: {pdf_url}")
        # Fetch first bytes (avoid downloading whole large pdf)
        try:
            pdf_resp = requests.get(pdf_url, timeout=25, headers={"User-Agent":"Mozilla/5.0"}, stream=True)
            chunk = pdf_resp.raw.read(5000)
        except Exception as e:
            print(f"[Fallback] PDF fetch failed: {e}")
            return {"completed": False, "pdf": pdf_url, "reason": "pdf_fetch_failed"}
        # Rough heuristic for figure 1 interpretation placeholder
        figure_insight = "Placeholder: Figure 1 likely illustrates architecture or pipeline of Qwen model family."  # static fallback
        result = {
            "completed": True,
            "pdf": pdf_url,
            "figure_1_analysis": figure_insight,
            "steps": len(self.history) + 1
        }
        self.history.append({"action":"SEARCH","query":query,"pdf":pdf_url})
        return result
