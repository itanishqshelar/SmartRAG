"""
Registry of uploaded documents: original files on disk, metadata in SQLite.
"""

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    content_type TEXT NOT NULL,
    document_type TEXT NOT NULL,
    size         INTEGER NOT NULL,
    chunks       INTEGER NOT NULL,
    created_at   TEXT NOT NULL
)
"""


@dataclass
class Document:
    id: str
    filename: str
    content_type: str
    document_type: str
    size: int
    chunks: int
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentStore:
    def __init__(self, data_directory: Path):
        self.files_dir = data_directory / "uploads"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = data_directory / "smartrag.db"
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def file_path(self, document_id: str, filename: str) -> Path:
        return self.files_dir / f"{document_id}{Path(filename).suffix.lower()}"

    def add(self, document_id: str, filename: str, content_type: str, document_type: str,
            size: int, chunks: int) -> Document:
        doc = Document(
            id=document_id,
            filename=filename,
            content_type=content_type,
            document_type=document_type,
            size=size,
            chunks=chunks,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO documents VALUES (:id, :filename, :content_type, :document_type, :size, :chunks, :created_at)",
                doc.to_dict(),
            )
        return doc

    def get(self, document_id: str) -> Optional[Document]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return Document(**row) if row else None

    def list(self) -> List[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [Document(**row) for row in rows]

    def delete(self, document_id: str) -> Optional[Document]:
        doc = self.get(document_id)
        if doc is None:
            return None
        self.file_path(doc.id, doc.filename).unlink(missing_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return doc
