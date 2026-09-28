"""Checkpointer SQLite local compatible con la interfaz de LangGraph.

Es una implementación propia sobre InMemorySaver. Persiste las tres estructuras
que LangGraph necesita para reconstruir el estado (checkpoints, writes y blobs),
sin instalar el paquete opcional langgraph-checkpoint-sqlite.
"""

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver


class SqliteSaver(InMemorySaver):
    """Checkpointer con persistencia transaccional en un archivo SQLite."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL, ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL, checkpoint BLOB NOT NULL,
                metadata_type TEXT NOT NULL, metadata BLOB NOT NULL,
                parent_id TEXT,
                PRIMARY KEY (thread_id, ns, checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL, ns TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL, idx INTEGER NOT NULL,
                channel TEXT NOT NULL, value_type TEXT NOT NULL, value BLOB NOT NULL,
                task_path TEXT NOT NULL,
                PRIMARY KEY (thread_id, ns, checkpoint_id, task_id, idx)
            );
            CREATE TABLE IF NOT EXISTS blobs (
                thread_id TEXT NOT NULL, ns TEXT NOT NULL, channel TEXT NOT NULL,
                version TEXT NOT NULL, value_type TEXT NOT NULL, value BLOB NOT NULL,
                PRIMARY KEY (thread_id, ns, channel, version)
            );
        """)
        self._load()

    def _load(self) -> None:
        for row in self.db.execute(
            "SELECT thread_id, ns, checkpoint_id, checkpoint_type, checkpoint, "
            "metadata_type, metadata, parent_id FROM checkpoints"
        ):
            thread, ns, checkpoint_id, cp_type, cp, md_type, md, parent = row
            self.storage[thread][ns][checkpoint_id] = (
                (cp_type, cp), (md_type, md), parent
            )
        for row in self.db.execute(
            "SELECT thread_id, ns, checkpoint_id, task_id, idx, channel, "
            "value_type, value, task_path FROM writes"
        ):
            thread, ns, checkpoint_id, task, idx, channel, kind, value, path = row
            self.writes[(thread, ns, checkpoint_id)][(task, idx)] = (
                task, channel, (kind, value), path
            )
        for row in self.db.execute(
            "SELECT thread_id, ns, channel, version, value_type, value FROM blobs"
        ):
            thread, ns, channel, version, kind, value = row
            self.blobs[(thread, ns, channel, json.loads(version))] = (kind, value)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self.lock:
            saved = super().put(config, checkpoint, metadata, new_versions)
            thread = saved["configurable"]["thread_id"]
            ns = saved["configurable"]["checkpoint_ns"]
            checkpoint_id = saved["configurable"]["checkpoint_id"]
            cp, md, parent = self.storage[thread][ns][checkpoint_id]
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                    (thread, ns, checkpoint_id, cp[0], cp[1], md[0], md[1], parent),
                )
                for channel, version in new_versions.items():
                    kind, value = self.blobs[(thread, ns, channel, version)]
                    self.db.execute(
                        "INSERT OR REPLACE INTO blobs VALUES (?,?,?,?,?,?)",
                        (thread, ns, channel, json.dumps(version), kind, value),
                    )
            return saved

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self.lock:
            super().put_writes(config, writes, task_id, task_path)
            fields = config["configurable"]
            thread = fields["thread_id"]
            ns = fields.get("checkpoint_ns", "")
            checkpoint_id = fields["checkpoint_id"]
            rows = self.writes[(thread, ns, checkpoint_id)]
            with self.db:
                for (task, idx), (_, channel, (kind, value), path) in rows.items():
                    self.db.execute(
                        "INSERT OR REPLACE INTO writes VALUES (?,?,?,?,?,?,?,?,?)",
                        (thread, ns, checkpoint_id, task, idx, channel, kind, value, path),
                    )

    def delete_thread(self, thread_id: str) -> None:
        with self.lock:
            with self.db:
                for table in ("writes", "blobs", "checkpoints"):
                    self.db.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                    )
            super().delete_thread(thread_id)

    def close(self) -> None:
        self.db.close()
