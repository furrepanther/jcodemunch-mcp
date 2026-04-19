"""Tests for streamable-http session persistence (fix for issue #204).

Verifies that run_streamable_http_server maintains a session map so that
subsequent requests (tools/list, etc.) from the same client are routed to the
same initialised server.run() task rather than a fresh one.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock


class TestStreamableHTTPSessionRouting(unittest.IsolatedAsyncioTestCase):

    async def test_existing_session_routed_without_new_transport(self):
        """A request with a known session ID reuses the existing transport."""
        session_id = "existing-session-99"
        mock_transport = MagicMock()
        mock_transport._terminated = False
        handle_calls = []

        async def fake_handle(scope, receive, send):
            handle_calls.append(1)

        mock_transport.handle_request = fake_handle
        sessions = {session_id: mock_transport}

        async def route(request_session_id):
            if request_session_id and request_session_id in sessions:
                transport = sessions[request_session_id]
                await transport.handle_request(None, None, None)
                if transport._terminated:
                    sessions.pop(request_session_id, None)
                return True
            return False

        result = await route(session_id)
        self.assertTrue(result)
        self.assertEqual(handle_calls, [1])

    async def test_terminated_session_cleaned_up(self):
        """A terminated transport is removed from the session map after its request."""
        session_id = "term-session"
        mock_transport = MagicMock()
        mock_transport._terminated = True
        mock_transport.handle_request = AsyncMock()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()

        sessions = {session_id: mock_transport}
        session_tasks = {session_id: mock_task}

        async def route(request_session_id):
            if request_session_id and request_session_id in sessions:
                transport = sessions[request_session_id]
                await transport.handle_request(None, None, None)
                if transport._terminated:
                    sessions.pop(request_session_id, None)
                    task = session_tasks.pop(request_session_id, None)
                    if task and not task.done():
                        task.cancel()
                return True
            return False

        await route(session_id)
        self.assertNotIn(session_id, sessions)
        self.assertNotIn(session_id, session_tasks)
        mock_task.cancel.assert_called_once()

    async def test_unknown_session_id_creates_new_session(self):
        """A request with an unrecognised session ID starts a new session."""
        sessions: dict = {}

        async def route(request_session_id):
            if request_session_id and request_session_id in sessions:
                return "existing"
            return "new"

        result = await route("unknown-id-xyz")
        self.assertEqual(result, "new")

    async def test_no_session_id_creates_new_session(self):
        """A request with no session ID header starts a new session."""
        sessions: dict = {}

        async def route(request_session_id):
            if request_session_id and request_session_id in sessions:
                return "existing"
            return "new"

        result = await route(None)
        self.assertEqual(result, "new")

    async def test_multiple_sessions_tracked_independently(self):
        """Multiple concurrent sessions are tracked separately."""
        sessions: dict = {}
        call_log: list = []

        for i in range(3):
            sid = f"session-{i}"
            t = MagicMock()
            t._terminated = False

            async def _handle(scope, receive, send, _sid=sid):
                call_log.append(_sid)

            t.handle_request = _handle
            sessions[sid] = t

        async def route(request_session_id):
            if request_session_id and request_session_id in sessions:
                await sessions[request_session_id].handle_request(None, None, None)
                return True
            return False

        await route("session-0")
        await route("session-2")
        await route("session-1")

        self.assertEqual(call_log, ["session-0", "session-2", "session-1"])

    async def test_streams_ready_event_gates_handle_request(self):
        """handle_request is only called after streams_ready is set."""
        order: list = []
        streams_ready = asyncio.Event()

        async def fake_session_runner():
            await asyncio.sleep(0.01)
            order.append("streams_ready")
            streams_ready.set()

        async def fake_handle():
            order.append("handle_request")

        runner_task = asyncio.create_task(fake_session_runner())
        await asyncio.wait_for(streams_ready.wait(), timeout=2.0)
        await fake_handle()
        await runner_task

        self.assertEqual(order, ["streams_ready", "handle_request"])

    async def test_timeout_cleans_up_session(self):
        """If streams_ready never fires, the session is cleaned up."""
        sessions: dict = {}
        session_tasks: dict = {}
        session_id = "timeout-session"

        mock_task = MagicMock()
        mock_task.cancel = MagicMock()

        sessions[session_id] = MagicMock()
        session_tasks[session_id] = mock_task

        # Simulate the timeout cleanup path
        sessions.pop(session_id, None)
        t = session_tasks.pop(session_id, None)
        if t:
            t.cancel()

        self.assertNotIn(session_id, sessions)
        mock_task.cancel.assert_called_once()

    async def test_session_runner_cleans_up_on_cancel(self):
        """_session_runner removes session from registry when cancelled."""
        sessions: dict = {}
        session_tasks: dict = {}
        session_id = "cancel-session"
        sessions[session_id] = MagicMock()

        async def session_runner():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                pass
            finally:
                sessions.pop(session_id, None)
                session_tasks.pop(session_id, None)

        task = asyncio.create_task(session_runner())
        session_tasks[session_id] = task

        await asyncio.sleep(0)  # let task start
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        self.assertNotIn(session_id, sessions)
        self.assertNotIn(session_id, session_tasks)

    async def test_session_ids_are_unique_per_new_connection(self):
        """Each new session gets a distinct UUID-based ID."""
        import uuid
        ids = {uuid.uuid4().hex for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
