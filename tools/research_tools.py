"""Shared research helpers for the RL-Trader research phase.

Every citation MUST be verified before it is written into a doc.
Use `verify_arxiv` / `verify_url` for that and record the returned evidence.

Env vars required (already present in this machine's shell env):
  PARALLEL_API_KEY      -> Parallel.ai Search + Task (deep research) API
  SUPERMEMORY_CC_API_KEY-> Supermemory v3 API
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

import httpx

PARALLEL_KEY = os.environ.get("PARALLEL_API_KEY", "")
SUPERMEMORY_KEY = os.environ.get("SUPERMEMORY_CC_API_KEY", "")
SM_CONTAINER_TAG = "repo_tradeai__6191323c4dd2d8a7"
PARALLEL_BASE = "https://api.parallel.ai"


# --------------------------------------------------------------------------
# Parallel.ai Search API  (cheap: $1-5 / 1000 requests)
# --------------------------------------------------------------------------
async def psearch(objective: str, queries: list[str], max_results: int = 10,
                  processor: str = "advanced", max_chars: int = 6000) -> list[dict]:
    """Natural-language web search. Returns [{url,title,publish_date,excerpts}]."""
    payload = {
        "objective": objective,
        "search_queries": queries,
        "mode": processor,
        "advanced_settings": {
            "max_results": max_results,
            "excerpt_settings": {"max_chars_per_result": max_chars},
        },
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{PARALLEL_BASE}/v1/search",
                         headers={"x-api-key": PARALLEL_KEY,
                                  "Content-Type": "application/json"},
                         json=payload)
        r.raise_for_status()
        return r.json().get("results", [])


# --------------------------------------------------------------------------
# Parallel.ai Task API  (deep research; pro=$0.10/run, ultra=$0.30/run)
# --------------------------------------------------------------------------
async def ptask_create(input_text: str, processor: str = "pro",
                       output_description: str | None = None) -> str:
    """Start a deep-research run. Returns run_id. Text schema => markdown report."""
    spec: dict[str, Any] = {"output_schema": {"type": "text"}}
    if output_description:
        spec["output_schema"] = {"type": "text",
                                 "description": output_description}
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{PARALLEL_BASE}/v1/tasks/runs",
                         headers={"x-api-key": PARALLEL_KEY,
                                  "Content-Type": "application/json"},
                         json={"input": input_text, "processor": processor,
                               "task_spec": spec})
        r.raise_for_status()
        return r.json()["run_id"]


async def ptask_status(run_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}",
                        headers={"x-api-key": PARALLEL_KEY})
        r.raise_for_status()
        return r.json()


async def ptask_result(run_id: str, api_timeout: int = 1800) -> dict:
    """Blocking long-poll for a finished run. Returns {"output": {...}}."""
    async with httpx.AsyncClient(timeout=api_timeout + 60) as c:
        r = await c.get(f"{PARALLEL_BASE}/v1/tasks/runs/{run_id}/result",
                        headers={"x-api-key": PARALLEL_KEY},
                        params={"api_timeout": api_timeout})
        r.raise_for_status()
        return r.json()


def ptask_text(result: dict) -> str:
    out = result.get("output", {})
    content = out.get("content", "")
    return content if isinstance(content, str) else json.dumps(content, indent=2)


def ptask_citations(result: dict) -> list[dict]:
    basis = result.get("output", {}).get("basis", []) or []
    cites = []
    for b in basis:
        for c in b.get("citations", []) or []:
            cites.append({"url": c.get("url"), "title": c.get("title"),
                          "excerpt": (c.get("excerpts") or [""])[0][:300]})
    return cites


# --------------------------------------------------------------------------
# Citation verification -- NEVER cite anything that fails these checks
# --------------------------------------------------------------------------
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


async def verify_arxiv(id_or_url: str, retries: int = 4) -> dict | None:
    """Confirm an arXiv paper exists. Returns {id,title,authors,published,updated,abs_url}."""
    m = _ARXIV_ID.search(id_or_url)
    if not m:
        return None
    aid = m.group(1)
    url = "https://export.arxiv.org/api/query"
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(url, params={"id_list": aid})
            if r.status_code == 200 and "<entry>" in r.text:
                def grab(tag: str) -> str:
                    mm = re.search(rf"<{tag}>(.*?)</{tag}>", r.text, re.S)
                    return re.sub(r"\s+", " ", mm.group(1)).strip() if mm else ""
                title = grab("title")
                if not title or title.lower().startswith("error"):
                    return None
                authors = re.findall(r"<name>(.*?)</name>", r.text)
                return {"id": aid, "title": title, "authors": authors[:12],
                        "published": grab("published"), "updated": grab("updated"),
                        "abs_url": f"https://arxiv.org/abs/{aid}"}
        except Exception:
            pass
        await asyncio.sleep(3 * (attempt + 1))
    # Fallback: the arXiv API is often rate-limited; the abs page is authoritative too.
    page = await verify_url(f"https://arxiv.org/abs/{aid}")
    if page["ok"] and page["title"] and "not found" not in page["title"].lower():
        return {"id": aid, "title": page["title"], "authors": [],
                "published": "", "updated": "",
                "abs_url": f"https://arxiv.org/abs/{aid}",
                "note": "verified via abs page (API rate-limited)"}
    return None


async def arxiv_search(query: str, max_results: int = 15,
                       start_year: int = 2022) -> list[dict]:
    """Search arXiv directly (authoritative existence check + dates)."""
    params = {"search_query": query, "start": 0, "max_results": max_results,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
        for attempt in range(4):
            r = await c.get("https://export.arxiv.org/api/query", params=params)
            if r.status_code == 200:
                break
            await asyncio.sleep(3 * (attempt + 1))
        else:
            return []
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    out = []
    for e in entries:
        def g(tag: str) -> str:
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return re.sub(r"\s+", " ", mm.group(1)).strip() if mm else ""
        link = re.search(r'<id>(.*?)</id>', e)
        pub = g("published")
        if pub and int(pub[:4]) < start_year:
            continue
        out.append({"title": g("title"), "published": pub, "updated": g("updated"),
                    "summary": g("summary")[:600],
                    "abs_url": link.group(1).strip() if link else "",
                    "authors": re.findall(r"<name>(.*?)</name>", e)[:10]})
    return out


async def verify_url(url: str) -> dict:
    """Confirm a URL is live. Returns {url,status,ok,title}."""
    try:
        async with httpx.AsyncClient(timeout=45, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 research-bot"}) as c:
            r = await c.get(url)
            title = ""
            mm = re.search(r"<title[^>]*>(.*?)</title>", r.text[:20000], re.S | re.I)
            if mm:
                title = re.sub(r"\s+", " ", mm.group(1)).strip()[:200]
            return {"url": url, "status": r.status_code, "ok": r.status_code < 400,
                    "title": title}
    except Exception as exc:
        return {"url": url, "status": None, "ok": False, "title": f"ERROR {exc}"}


async def gh_repo(full_name: str) -> dict:
    """GitHub repo metadata via the gh-authenticated REST API (no token needed for public)."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        h = {"Accept": "application/vnd.github+json"}
        tok = os.environ.get("GITHUB_TOKEN")
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        r = await c.get(f"https://api.github.com/repos/{full_name}", headers=h)
        if r.status_code != 200:
            return {"full_name": full_name, "error": r.status_code}
        d = r.json()
        lang = await c.get(f"https://api.github.com/repos/{full_name}/languages", headers=h)
        commits = await c.get(f"https://api.github.com/repos/{full_name}/commits",
                              headers=h, params={"per_page": 1})
        return {"full_name": d["full_name"], "url": d["html_url"],
                "description": d.get("description"), "stars": d["stargazers_count"],
                "forks": d["forks_count"], "open_issues": d["open_issues_count"],
                "created_at": d["created_at"], "pushed_at": d["pushed_at"],
                "archived": d["archived"], "license": (d.get("license") or {}).get("spdx_id"),
                "default_branch": d["default_branch"],
                "languages": lang.json() if lang.status_code == 200 else {},
                "last_commit_date": (commits.json()[0]["commit"]["committer"]["date"]
                                     if commits.status_code == 200 and commits.json() else None)}


# --------------------------------------------------------------------------
# Supermemory
# --------------------------------------------------------------------------
async def sm_add(content: str, container_tag: str = SM_CONTAINER_TAG,
                 metadata: dict | None = None, title: str | None = None) -> dict:
    body: dict[str, Any] = {"content": content, "containerTag": container_tag}
    if metadata:
        body["metadata"] = metadata
    if title:
        body.setdefault("metadata", {})["title"] = title
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post("https://api.supermemory.ai/v3/documents",
                         headers={"Authorization": f"Bearer {SUPERMEMORY_KEY}",
                                  "Content-Type": "application/json"},
                         json=body)
        return {"status": r.status_code, "body": r.text[:500]}


async def sm_search(q: str, container_tag: str = SM_CONTAINER_TAG, limit: int = 10) -> dict:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post("https://api.supermemory.ai/v3/search",
                         headers={"Authorization": f"Bearer {SUPERMEMORY_KEY}",
                                  "Content-Type": "application/json"},
                         json={"q": q, "containerTag": container_tag, "limit": limit})
        return {"status": r.status_code, "body": r.json() if r.status_code < 400 else r.text[:500]}
