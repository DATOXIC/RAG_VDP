"""Test cho finco.llm: retry, backoff, Retry-After, lỗi cố định, Usage, template.

Chạy từ thư mục gốc repo (chỉ dùng thư viện chuẩn + httpx có sẵn cùng openai):
    python -m unittest discover -s tests -v

Không gọi mạng: client LLM được thay bằng đối tượng giả, time.sleep bị chặn.
"""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from finco import llm  # noqa: E402
from finco.types import Usage  # noqa: E402

REQUEST = httpx.Request("POST", "https://example.test/v1/chat/completions")


def http_error(cls, status, headers=None):
    resp = httpx.Response(status, request=REQUEST, headers=headers or {})
    return cls("lỗi giả lập", response=resp, body=None)


def ok_response(content="Thought: xong", prompt=100, completion=20, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def fake_client(*outcomes):
    """Client giả: mỗi lần create() trả/ném lần lượt từng phần tử của outcomes."""
    client = MagicMock()
    client.chat.completions.create.side_effect = list(outcomes)
    return client


MSGS = [{"role": "user", "content": "xin chào"}]


class CallLlmBase(unittest.TestCase):
    def setUp(self):
        sleep = patch("finco.llm.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        # jitter cố định để kiểm tra thời gian chờ chính xác
        jitter = patch("finco.llm.random.uniform", return_value=0.2)
        jitter.start()
        self.addCleanup(jitter.stop)

    def call(self, client, **kw):
        with patch("finco.llm.get_client", return_value=client):
            with redirect_stderr(io.StringIO()) as err:
                try:
                    return llm.call_llm(MSGS, **kw), err.getvalue()
                finally:
                    self.stderr = err.getvalue()


class TestSuccess(CallLlmBase):
    def test_returns_content_and_usage(self):
        (content, usage), _ = self.call(fake_client(ok_response("Answer: ok", 120, 30)))
        self.assertEqual(content, "Answer: ok")
        self.assertEqual(usage, Usage(calls=1, prompt_tokens=120, completion_tokens=30, max_prompt=120))

    def test_stop_forwarded_only_when_given(self):
        c1 = fake_client(ok_response())
        self.call(c1, stop=["Observation:"])
        self.assertEqual(c1.chat.completions.create.call_args.kwargs["stop"], ["Observation:"])

        c2 = fake_client(ok_response())
        self.call(c2)
        self.assertNotIn("stop", c2.chat.completions.create.call_args.kwargs)

    def test_missing_usage_counts_zero(self):
        resp = ok_response()
        resp.usage = None
        (_, usage), _ = self.call(fake_client(resp))
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (0, 0))


class TestRetry(CallLlmBase):
    def test_transient_error_then_success(self):
        client = fake_client(http_error(RateLimitError, 429), ok_response())
        (content, usage), _ = self.call(client)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(self.sleep.call_count, 1)
        self.assertEqual(usage.calls, 1)  # lần thử thất bại không được đếm

    def test_all_transient_kinds_are_retried(self):
        errors = [
            http_error(InternalServerError, 503),
            APITimeoutError(request=REQUEST),
            APIConnectionError(request=REQUEST),
        ]
        client = fake_client(*errors, ok_response())
        self.call(client)
        self.assertEqual(client.chat.completions.create.call_count, 4)

    def test_backoff_doubles_with_jitter(self):
        client = fake_client(*[http_error(RateLimitError, 429)] * 3, ok_response())
        self.call(client, initial_delay=1.0, max_delay=20.0)
        waits = [c.args[0] for c in self.sleep.call_args_list]
        self.assertEqual(waits, [1.2, 2.2, 4.2])

    def test_backoff_capped_by_max_delay(self):
        client = fake_client(*[http_error(RateLimitError, 429)] * 3, ok_response())
        self.call(client, initial_delay=10.0, max_delay=15.0)
        waits = [c.args[0] for c in self.sleep.call_args_list]
        self.assertEqual(waits, [10.2, 15.2, 15.2])

    def test_retry_after_header_is_used(self):
        client = fake_client(http_error(RateLimitError, 429, {"retry-after": "7"}), ok_response())
        self.call(client, initial_delay=1.0)
        self.assertEqual(self.sleep.call_args.args[0], 7.2)

    def test_retry_after_is_capped(self):
        client = fake_client(http_error(RateLimitError, 429, {"retry-after": "9999"}), ok_response())
        self.call(client)
        self.assertEqual(self.sleep.call_args.args[0], llm.MAX_RETRY_AFTER + 0.2)

    def test_invalid_retry_after_falls_back_to_backoff(self):
        client = fake_client(http_error(RateLimitError, 429, {"retry-after": "abc"}), ok_response())
        self.call(client, initial_delay=2.0)
        self.assertEqual(self.sleep.call_args.args[0], 2.2)

    def test_gives_up_after_max_retries(self):
        client = fake_client(*[http_error(RateLimitError, 429)] * 3)
        with self.assertRaises(RuntimeError) as cm:
            self.call(client, max_retries=2)
        self.assertEqual(client.chat.completions.create.call_count, 3)  # 1 lần đầu + 2 retry
        self.assertIn("429", str(cm.exception))


class TestPermanentErrors(CallLlmBase):
    def test_auth_error_not_retried(self):
        client = fake_client(http_error(AuthenticationError, 401), ok_response())
        with self.assertRaises(AuthenticationError):
            self.call(client)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.sleep.assert_not_called()
        self.assertIn("HTTP 401", self.stderr)  # mã HTTP được ghi vào log

class TestBadResponses(CallLlmBase):
    def test_no_choices(self):
        resp = SimpleNamespace(choices=[], usage=None)
        with self.assertRaises(llm.EmptyResponseError):
            self.call(fake_client(resp))

    def test_empty_content_is_error_not_silent(self):
        with self.assertRaises(llm.EmptyResponseError):
            self.call(fake_client(ok_response(content="")))

    def test_empty_content_cut_by_length_has_hint(self):
        with self.assertRaises(llm.EmptyResponseError) as cm:
            self.call(fake_client(ok_response(content="", finish="length")))
        self.assertIn("max_tokens", str(cm.exception))

    def test_empty_response_is_not_retried(self):
        client = fake_client(ok_response(content=""), ok_response())
        with self.assertRaises(llm.EmptyResponseError):
            self.call(client)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_truncated_but_nonempty_is_returned_with_warning(self):
        (content, _), _ = self.call(fake_client(ok_response("Thought: dở", finish="length")))
        self.assertEqual(content, "Thought: dở")
        self.assertIn("bị cắt", self.stderr)


class TestClient(unittest.TestCase):
    def test_sdk_internal_retries_disabled(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "k"}, clear=False), patch.object(llm, "_CLIENT", None):
            client = llm.get_client()
        self.assertEqual(client.max_retries, 0)

    def test_missing_key_raises(self):
        env = {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}
        with patch.dict("os.environ", env), patch.object(llm, "_CLIENT", None):
            with self.assertRaises(ValueError):
                llm.get_client()


class TestMergeUsage(unittest.TestCase):
    def test_sums_and_takes_max_prompt(self):
        a = Usage(1, 1200, 80, 1200)
        b = Usage(1, 1500, 60, 1500)
        c = Usage(1, 1900, 120, 1900)
        self.assertEqual(llm.merge_usage(a, b, c), Usage(3, 4600, 260, 1900))

    def test_does_not_mutate_inputs(self):
        a, b = Usage(1, 10, 1, 10), Usage(1, 20, 2, 20)
        llm.merge_usage(a, b)
        self.assertEqual(a, Usage(1, 10, 1, 10))
        self.assertEqual(b, Usage(1, 20, 2, 20))

    def test_empty(self):
        self.assertEqual(llm.merge_usage(), Usage(0, 0, 0, 0))


def tool_response(calls=(), content=None, prompt=50, completion=5, finish="tool_calls"):
    """Phản hồi giả có tool_calls; calls là danh sách (tên, chuỗi_arguments)."""
    tcs = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=args))
        for i, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tcs or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]


class CallLlmToolsBase(unittest.TestCase):
    def setUp(self):
        for target, kw in (("finco.llm.time.sleep", {}), ("finco.llm.random.uniform", {"return_value": 0.2})):
            p = patch(target, **kw)
            p.start()
            self.addCleanup(p.stop)

    def call(self, client, **kw):
        with patch("finco.llm.get_client", return_value=client):
            with redirect_stderr(io.StringIO()):
                return llm.call_llm_tools(MSGS, TOOLS, **kw)


class TestCallLlmTools(CallLlmToolsBase):
    def test_returns_tool_calls_and_usage(self):
        client = fake_client(tool_response([("get_application", '{"app_id": "APP-001"}')], prompt=80, completion=12))
        turn = self.call(client)
        self.assertEqual(turn.content, "")
        self.assertEqual(
            turn.tool_calls,
            [llm.ToolCall(id="c0", name="get_application", arguments='{"app_id": "APP-001"}')],
        )
        self.assertEqual(turn.usage, Usage(calls=1, prompt_tokens=80, completion_tokens=12, max_prompt=80))

    def test_text_only_answer(self):
        turn = self.call(fake_client(tool_response(content=" Xong. ", finish="stop")))
        self.assertEqual(turn.content, "Xong.")
        self.assertEqual(turn.tool_calls, [])

    def test_parallel_calls_keep_order(self):
        turn = self.call(fake_client(tool_response([("a", "{}"), ("b", "{}")])))
        self.assertEqual([tc.name for tc in turn.tool_calls], ["a", "b"])

    def test_request_carries_tools_timeout_and_choice(self):
        client = fake_client(tool_response(content="ok", finish="stop"))
        self.call(client, model="m-test", timeout=7.0)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tools"], TOOLS)
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertEqual(kwargs["timeout"], 7.0)
        self.assertEqual(kwargs["model"], "m-test")

    def test_shares_retry_policy_with_call_llm(self):
        client = fake_client(
            http_error(RateLimitError, 429, {"retry-after": "3"}),
            tool_response(content="ok", finish="stop"),
        )
        turn = self.call(client)
        self.assertEqual(turn.content, "ok")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertAlmostEqual(llm.time.sleep.call_args.args[0], 3.2)  # Retry-After + jitter 0.2

    def test_tool_use_failed_becomes_malformed_error_without_retry(self):
        err = http_error(BadRequestError, 400)
        err.code = "tool_use_failed"
        client = fake_client(err, tool_response(content="ok", finish="stop"))
        with self.assertRaises(llm.MalformedToolCallError):
            self.call(client)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_other_bad_request_is_not_swallowed(self):
        err = http_error(BadRequestError, 400)
        err.code = "context_length_exceeded"
        with self.assertRaises(BadRequestError):
            self.call(fake_client(err))

    def test_missing_usage_counts_zero(self):
        resp = tool_response(content="ok", finish="stop")
        resp.usage = None
        turn = self.call(fake_client(resp))
        self.assertEqual(turn.usage, Usage(calls=1, prompt_tokens=0, completion_tokens=0, max_prompt=0))


if __name__ == "__main__":
    unittest.main()
