import asyncio

import pytest

from deeper_bot.session import Session, SessionState, SessionStore

# ---------------------------------------------------------------------------
# Session state machine tests
# ---------------------------------------------------------------------------


class TestSessionStateMachine:
    def test_initial_state(self):
        s = Session(chat_id=1)
        assert s.state == SessionState.IDLE
        assert s.messages == []
        assert s.research_start_idx == 0
        assert s.allowed_domains == set()
        assert s.language_code is None
        assert s.todo_list is None
        assert s.status_announced is False

    async def test_set_awaiting_answer(self):
        s = Session(chat_id=1)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        s.set_awaiting_answer(future)
        assert s.state == SessionState.AWAITING_ANSWER
        assert s.has_pending_question()

    async def test_resolve_answer(self):
        s = Session(chat_id=1)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        s.set_awaiting_answer(future)
        s.resolve_answer("hello")
        assert s.state == SessionState.RESEARCHING
        assert future.result() == "hello"
        assert not s.has_pending_question()

    async def test_resolve_answer_on_done_future(self):
        s = Session(chat_id=1)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result("already done")
        s._pending_future = future
        s.state = SessionState.AWAITING_ANSWER
        # Should not raise even though future is already done
        s.resolve_answer("new answer")
        assert s.state == SessionState.RESEARCHING

    async def test_cancel_pending(self):
        s = Session(chat_id=1)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        s.set_awaiting_answer(future)
        s.cancel_pending()
        assert s._pending_future is None
        assert future.cancelled()

    def test_cancel_pending_when_none(self):
        s = Session(chat_id=1)
        # Should not raise
        s.cancel_pending()

    def test_has_pending_question_false_when_idle(self):
        s = Session(chat_id=1)
        assert not s.has_pending_question()

    async def test_has_pending_question_false_when_future_done(self):
        s = Session(chat_id=1)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result("done")
        s._pending_future = future
        s.state = SessionState.AWAITING_ANSWER
        assert not s.has_pending_question()

    def test_clear_status_resets_fields(self):
        s = Session(chat_id=1)
        s.todo_list = "- [ ] Step 1"
        s.status_announced = True
        s.clear_status()
        assert s.todo_list is None
        assert s.status_announced is False

    def test_clear_status_when_already_none(self):
        s = Session(chat_id=1)
        s.clear_status()
        assert s.todo_list is None
        assert s.status_announced is False


# ---------------------------------------------------------------------------
# SessionStore tests
# ---------------------------------------------------------------------------


class TestSessionStore:
    @pytest.fixture
    async def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = SessionStore(db_path)
        await s.init()
        yield s
        await s.close()

    async def test_init_creates_db(self, store):
        # Should be able to get_or_create without error
        session = await store.get_or_create(100)
        assert session.chat_id == 100
        assert session.state == SessionState.IDLE

    async def test_wal_mode_enabled(self, store):
        async with store._db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        assert row[0] == "wal"

    async def test_get_or_create_returns_same_object(self, store):
        s1 = await store.get_or_create(100)
        s2 = await store.get_or_create(100)
        assert s1 is s2

    async def test_save_and_reload(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store1 = SessionStore(db_path)
        await store1.init()

        session = await store1.get_or_create(42)
        session.messages = [{"role": "system", "content": "hello"}]
        session.research_start_idx = 1
        session.state = SessionState.RESEARCHING
        session.todo_list = "- [ ] Step 1"
        session.initialized = True
        session.status_announced = True
        session.language_code = "ru"
        session.allowed_domains = {"example.com", "python.org"}
        await store1.save(session)
        await store1.close()

        # Reload from a fresh store
        store2 = SessionStore(db_path)
        await store2.init()
        # init resets non-idle states
        reloaded = await store2.get_or_create(42)
        assert reloaded.chat_id == 42
        assert reloaded.state == SessionState.IDLE  # reset by init
        assert reloaded.messages == [{"role": "system", "content": "hello"}]
        assert reloaded.research_start_idx == 1
        assert reloaded.todo_list == "- [ ] Step 1"
        assert reloaded.initialized is True
        assert reloaded.status_announced is True
        assert reloaded.language_code == "ru"
        assert reloaded.allowed_domains == {"example.com", "python.org"}
        await store2.close()

    async def test_delete(self, store):
        session = await store.get_or_create(100)
        await store.save(session)
        await store.delete(100)
        # Creating again should give a fresh session
        new_session = await store.get_or_create(100)
        assert new_session.messages == []
        assert new_session is not session

    async def test_runtime_error_before_init(self, tmp_path):
        store = SessionStore(str(tmp_path / "test.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.get_or_create(1)

    async def test_runtime_error_save_before_init(self, tmp_path):
        store = SessionStore(str(tmp_path / "test.db"))
        s = Session(chat_id=1)
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.save(s)

    async def test_runtime_error_delete_before_init(self, tmp_path):
        store = SessionStore(str(tmp_path / "test.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.delete(1)

    async def test_concurrent_get_or_create(self, store):
        """Two concurrent get_or_create for the same chat_id should return the same Session."""
        results = await asyncio.gather(
            store.get_or_create(999),
            store.get_or_create(999),
        )
        assert results[0] is results[1]

    async def test_migration_adds_new_columns(self, tmp_path):
        """Old DB without new columns should migrate gracefully on init()."""
        db_path = str(tmp_path / "legacy.db")
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                chat_id INTEGER PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'idle',
                messages TEXT NOT NULL DEFAULT '[]',
                research_start_idx INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "INSERT INTO sessions (chat_id, state, messages, research_start_idx) VALUES (?, ?, ?, ?)",
            (7, "idle", '[{"role":"user","content":"hi"}]', 1),
        )
        conn.commit()
        conn.close()

        store = SessionStore(db_path)
        await store.init()
        reloaded = await store.get_or_create(7)
        assert reloaded.messages == [{"role": "user", "content": "hi"}]
        assert reloaded.research_start_idx == 1
        assert reloaded.todo_list is None
        assert reloaded.initialized is False
        assert reloaded.status_announced is False
        assert reloaded.language_code is None
        assert reloaded.allowed_domains == set()

        reloaded.todo_list = "plan"
        reloaded.initialized = True
        await store.save(reloaded)
        await store.close()

        store2 = SessionStore(db_path)
        await store2.init()
        again = await store2.get_or_create(7)
        assert again.todo_list == "plan"
        assert again.initialized is True
        await store2.close()


# ---------------------------------------------------------------------------
# Eviction tests
# ---------------------------------------------------------------------------


class TestSessionEviction:
    @pytest.fixture
    async def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        s = SessionStore(db_path)
        await s.init()
        yield s
        await s.close()

    async def test_stale_idle_session_evicted(self, store):
        """An idle session past TTL should be evicted from memory."""
        session = await store.get_or_create(100)
        await store.save(session)
        # Simulate staleness
        session.last_accessed = 0.0
        store._evict_stale()
        assert 100 not in store._sessions
        assert 100 not in store._get_locks

    async def test_stale_session_reloaded_from_db(self, store):
        """After eviction, get_or_create should reload session from DB."""
        session = await store.get_or_create(100)
        session.messages = [{"role": "system", "content": "test"}]
        session.research_start_idx = 1
        await store.save(session)
        # Force eviction
        session.last_accessed = 0.0
        store._evict_stale()
        assert 100 not in store._sessions

        reloaded = await store.get_or_create(100)
        assert reloaded.chat_id == 100
        assert reloaded.messages == [{"role": "system", "content": "test"}]
        assert reloaded.research_start_idx == 1

    async def test_locked_session_not_evicted(self, store):
        """A session with a held lock should not be evicted even if stale."""
        session = await store.get_or_create(100)
        session.last_accessed = 0.0
        await session.lock.acquire()
        try:
            store._evict_stale()
            assert 100 in store._sessions
        finally:
            session.lock.release()

    async def test_researching_session_not_evicted(self, store):
        """A RESEARCHING session should not be evicted even if stale."""
        session = await store.get_or_create(100)
        session.state = SessionState.RESEARCHING
        session.last_accessed = 0.0
        store._evict_stale()
        assert 100 in store._sessions

    async def test_capacity_eviction(self, store):
        """When over _MAX_CACHED, oldest idle sessions should be evicted."""
        store._MAX_CACHED = 3
        for i in range(5):
            s = await store.get_or_create(i)
            s.last_accessed = float(i)
            await store.save(s)

        store._evict_stale()
        assert len(store._sessions) <= 3
        # Newest sessions (2, 3, 4) should survive
        assert 3 in store._sessions
        assert 4 in store._sessions

    async def test_get_or_create_updates_last_accessed(self, store):
        """Accessing a session should refresh its last_accessed timestamp."""
        session = await store.get_or_create(100)
        old_ts = session.last_accessed
        await asyncio.sleep(0.01)
        await store.get_or_create(100)
        assert session.last_accessed > old_ts

    async def test_save_updates_last_accessed(self, store):
        """Saving a session should refresh its last_accessed timestamp."""
        session = await store.get_or_create(100)
        old_ts = session.last_accessed
        await asyncio.sleep(0.01)
        await store.save(session)
        assert session.last_accessed > old_ts
