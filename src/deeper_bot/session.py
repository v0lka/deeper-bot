"""Session dataclass (state machine) and SessionStore (SQLite + in-memory cache with eviction)."""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic

import aiosqlite

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "[Summary of previous conversation]\n\n"


class SessionState(StrEnum):
    """Possible states of a research session."""

    IDLE = "idle"
    RESEARCHING = "researching"
    AWAITING_ANSWER = "awaiting_answer"


@dataclass
class Session:
    """Per-chat session holding conversation history, state, and metadata."""

    chat_id: int
    state: SessionState = SessionState.IDLE
    messages: list[dict] = field(default_factory=list)
    research_start_idx: int = 0
    _pending_future: asyncio.Future | None = field(default=None, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    todo_list: str | None = field(default=None, repr=False)
    _status_announced: bool = field(default=False, repr=False)
    _initialized: bool = field(default=False, repr=False)
    last_accessed: float = field(default_factory=monotonic, repr=False)

    def clear_status(self) -> None:
        """Clear the TODO list and reset status announcement flag."""
        self.todo_list = None
        self._status_announced = False

    def set_awaiting_answer(self, future: asyncio.Future) -> None:
        """Store the future and transition to AWAITING_ANSWER state."""
        self._pending_future = future
        self.state = SessionState.AWAITING_ANSWER

    def resolve_answer(self, text: str) -> None:
        """Resolve the pending future with the user's answer and transition to RESEARCHING."""
        if self._pending_future and not self._pending_future.done():
            self._pending_future.set_result(text)
        self._pending_future = None
        self.state = SessionState.RESEARCHING

    def has_pending_question(self) -> bool:
        """Return True if the session is waiting for a user answer."""
        return (
            self.state == SessionState.AWAITING_ANSWER
            and self._pending_future is not None
            and not self._pending_future.done()
        )

    def cancel_pending(self) -> None:
        """Cancel the pending future and clear it."""
        if self._pending_future and not self._pending_future.done():
            self._pending_future.cancel()
        self._pending_future = None

    def timeout_pending(self) -> None:
        """Clear the pending future and transition back to RESEARCHING on timeout."""
        self._pending_future = None
        self.state = SessionState.RESEARCHING


class SessionStore:
    """SQLite-backed session storage with in-memory LRU cache and eviction."""

    _MAX_CACHED = 1000
    _EVICTION_TTL = 3600  # seconds

    def __init__(self, database_path: str) -> None:
        """Initialize with the path to the SQLite database file."""
        self._db_path = database_path
        self._sessions: dict[int, Session] = {}
        self._db: aiosqlite.Connection | None = None
        self._get_locks: dict[int, asyncio.Lock] = {}

    def _get_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self._get_locks:
            self._get_locks[chat_id] = asyncio.Lock()
        return self._get_locks[chat_id]

    def _evict_stale(self) -> None:
        """Remove idle, unlocked sessions that haven't been accessed recently."""
        now = monotonic()
        to_evict: list[int] = []
        for cid, session in self._sessions.items():
            if session.state != SessionState.IDLE:
                continue
            if session.lock.locked():
                continue
            if now - session.last_accessed > self._EVICTION_TTL:
                to_evict.append(cid)

        for cid in to_evict:
            self._sessions.pop(cid, None)
            self._get_locks.pop(cid, None)

        # If still over capacity, evict oldest idle/unlocked sessions
        if len(self._sessions) > self._MAX_CACHED:
            candidates = [
                (s.last_accessed, cid)
                for cid, s in self._sessions.items()
                if s.state == SessionState.IDLE and not s.lock.locked()
            ]
            candidates.sort()
            excess = len(self._sessions) - self._MAX_CACHED
            for _, cid in candidates[:excess]:
                self._sessions.pop(cid, None)
                self._get_locks.pop(cid, None)

    async def init(self) -> None:
        """Create the database and schema, and reset non-idle sessions to idle."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                chat_id INTEGER PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'idle',
                messages TEXT NOT NULL DEFAULT '[]',
                research_start_idx INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                todo_list TEXT,
                initialized INTEGER NOT NULL DEFAULT 0,
                status_announced INTEGER NOT NULL DEFAULT 0
            )
        """)
        for alter_stmt in (
            "ALTER TABLE sessions ADD COLUMN todo_list TEXT",
            "ALTER TABLE sessions ADD COLUMN initialized INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN status_announced INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                await self._db.execute(alter_stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    continue
                raise
        await self._db.execute("UPDATE sessions SET state = 'idle' WHERE state != 'idle'")
        await self._db.commit()
        logger.info("Session store initialized: %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def get_or_create(self, chat_id: int) -> Session:
        """Retrieve an existing session from cache or DB, or create a new one."""
        self._evict_stale()
        async with self._get_lock(chat_id):
            if chat_id in self._sessions:
                self._sessions[chat_id].last_accessed = monotonic()
                return self._sessions[chat_id]

            if self._db is None:
                raise RuntimeError("SessionStore not initialized — call init() first")
            async with self._db.execute(
                "SELECT state, messages, research_start_idx, todo_list, initialized, status_announced FROM sessions WHERE chat_id = ?",  # noqa: E501
                (chat_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                session = Session(
                    chat_id=chat_id,
                    state=SessionState(row[0]),
                    messages=json.loads(row[1]),
                    research_start_idx=row[2],
                    todo_list=row[3],
                    _initialized=bool(row[4]),
                    _status_announced=bool(row[5]),
                )
            else:
                session = Session(chat_id=chat_id)

            self._sessions[chat_id] = session
            return session

    async def save(self, session: Session) -> None:
        """Persist the session to the database and update cache."""
        if self._db is None:
            raise RuntimeError("SessionStore not initialized — call init() first")
        session.last_accessed = monotonic()
        messages_json = json.dumps(session.messages, ensure_ascii=False)
        await self._db.execute(
            """
            INSERT INTO sessions (chat_id, state, messages, research_start_idx, updated_at, todo_list, initialized, status_announced)
            VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                state = excluded.state,
                messages = excluded.messages,
                research_start_idx = excluded.research_start_idx,
                updated_at = excluded.updated_at,
                todo_list = excluded.todo_list,
                initialized = excluded.initialized,
                status_announced = excluded.status_announced
            """,  # noqa: E501
            (
                session.chat_id,
                session.state.value,
                messages_json,
                session.research_start_idx,
                session.todo_list,
                1 if session._initialized else 0,
                1 if session._status_announced else 0,
            ),
        )
        await self._db.commit()

    async def delete(self, chat_id: int) -> None:
        """Remove the session from cache and database."""
        if self._db is None:
            raise RuntimeError("SessionStore not initialized — call init() first")
        self._sessions.pop(chat_id, None)
        self._get_locks.pop(chat_id, None)
        await self._db.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
        await self._db.commit()
