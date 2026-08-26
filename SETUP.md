# Development setup

How to run RegAgent locally and contribute.

## Prerequisites
- **Git**
- **Python 3.11+**
- An editor (**VS Code** recommended)
- A **GitHub account** (ask to be added as a collaborator)

## Get it running
```bash
git clone https://github.com/Lagra2105/regagent.git
cd regagent
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run it — **no API key needed** (mock mode):
```bash
python demo.py                          # prints example grounded answers
uvicorn service.api:app --port 8000     # web UI
```
Open <http://localhost:8000/demo> and <http://localhost:8000/assess>.

With a real key (optional — for real answers instead of mock):
```bash
OPENAI_API_KEY=sk-... uvicorn service.api:app --port 8000
```

## Contributing (branch → Pull Request)
Never push to `main`. Always work on a branch and open a PR for review:
```bash
git checkout -b my-change
# ... make your changes ...
git add -A
git commit -m "Describe the change"
git push -u origin my-change
```
Then open a **Pull Request** on GitHub.

## Where things live
| Path | What it is |
|---|---|
| `regagent/store.py` | retrieval + embeddings (dense) |
| `regagent/sparse.py` | BM25 keyword search |
| `regagent/graph.py` | knowledge graph |
| `regagent/agent.py` | the pipeline (route → retrieve → answer → verify) |
| `regagent/assess.py` | startup profile → compliance roadmap |
| `regagent/ingest.py` | loads the regulation text into the corpus |
| `service/api.py` | API + web pages (`/`, `/demo`, `/assess`) |
| `eval/` | accuracy benchmark |
| `data/` | regulation text |

**Good areas to start contributing:** the web UI (`service/api.py`), the corpus (`data/`, `ingest.py`), the eval set (`eval/`). Avoid the retrieval / graph / agent core until you know the codebase well.
