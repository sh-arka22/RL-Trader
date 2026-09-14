# Research protocol (non-negotiable)

This repo's research must be reproducible and citation-clean.

## 1. No unverified citations

Every paper, dataset, API, or repo named in a doc must be verified **in the session that
wrote it**:

| Source type | Verification |
|---|---|
| arXiv paper | `rt.verify_arxiv(id_or_url)` — returns `None` if it does not exist → delete the citation |
| Any URL | `rt.verify_url(url)` — must return `ok: True` |
| GitHub repo | `rt.gh_repo("owner/name")` — records stars, last commit, archived flag, license |
| Dataset / API tier | fetch the live docs or pricing page; record the fetch date |

Each dossier ends with a **Verification log**: URL, check used, status, date.

Model memory is not evidence. Today is later than the training cutoff of any model used
here, so anything time-sensitive (paper recency, API pricing, dataset availability,
best-reported metrics) must come from a live fetch.

## 2. Metrics must carry their context

Never report a bare Sharpe or return. Always record: asset universe, test window,
transaction-cost assumption, and the baseline it was compared against. A number without
these is not evidence.

## 3. Tooling

```python
import sys; sys.path.insert(0, "tools")
import research_tools as rt

await rt.psearch(objective, ["query 1", "query 2"], max_results=10)   # Parallel.ai Search v1
run_id = await rt.ptask_create("deep research prompt", processor="pro")
text   = rt.ptask_text(await rt.ptask_result(run_id))                  # markdown report + citations
await rt.verify_arxiv("2406.08013")
await rt.gh_repo("tensortrade-org/tensortrade")
await rt.sm_add("milestone note", title="...")                         # Supermemory
```

Requires `PARALLEL_API_KEY` and `SUPERMEMORY_CC_API_KEY` in the environment.

## 4. Honest negatives

If evidence contradicts a project assumption, say so in the dossier. If a component is
infeasible (paid-only data, dead API, unmaintained library), state it plainly and propose
the nearest feasible substitute. Silently dropping a requirement is a protocol violation.
