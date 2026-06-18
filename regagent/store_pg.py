"""pgvector-backed store — Postgres + vector search in one component.

Same interface as the in-memory DocStore (add / search), so the agent doesn't
change — only where the vectors live. pgvector replaces both "a database" and
"a vector store": one piece of infra instead of two. Scales past in-memory and
survives restarts (ready for cloud deploy).

Needs Postgres with the `vector` extension and:  pip install "psycopg[binary]" pgvector
Point it at a DSN, e.g. PgVectorStore("postgresql://user:pass@host:5432/db").
"""
from __future__ import annotations

from .store import Chunk, embed


class PgVectorStore:
    def __init__(self, dsn: str, table: str = "chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        self._psycopg = psycopg
        self._register = register_vector
        self.dsn = dsn
        self.table = table
        self.chunks: list[Chunk] = []      # kept for graph/bm25 which need text
        self._dim: int | None = None

    def _conn(self):
        con = self._psycopg.connect(self.dsn)
        con.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._register(con)
        return con

    def _ensure_table(self, dim: int) -> None:
        with self._conn() as con:
            con.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.table} (
                       id        TEXT PRIMARY KEY,
                       source    TEXT,
                       text      TEXT,
                       embedding vector({dim})
                   )""")
            con.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_emb ON {self.table} "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)")
            con.commit()

    def add(self, chunks: list[Chunk]) -> None:
        embs = embed([c.text for c in chunks])
        self._dim = len(embs[0])
        self._ensure_table(self._dim)
        with self._conn() as con:
            for c, e in zip(chunks, embs):
                con.execute(
                    f"INSERT INTO {self.table} (id, source, text, embedding) "
                    f"VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE "
                    f"SET embedding = EXCLUDED.embedding",
                    (c.id, c.source, c.text, e))
            con.commit()
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        qe = embed([query])[0]
        with self._conn() as con:
            rows = con.execute(
                f"SELECT id, source, text, 1 - (embedding <=> %s::vector) AS sim "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (qe, qe, k)).fetchall()
        return [(Chunk(id=r[0], source=r[1], text=r[2]), float(r[3])) for r in rows]
