"""Critic có thể dùng endpoint/model riêng (Gemini) mà không đổi client của Actor."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from finco import llm  # noqa: E402

CLEAN = {"FINCO_CRITIC_PROVIDER": "", "FINCO_CRITIC_BASE_URL": "", "FINCO_CRITIC_API_KEY": "",
         "FINCO_CRITIC_MODEL": "", "GEMINI_API_KEY": "", "GROQ_API_KEY": "groq-key"}


class TestCriticProfile(unittest.TestCase):
    def env(self, **kw):
        return patch.dict("os.environ", {**CLEAN, **kw})

    def test_no_override_means_shared_endpoint(self):
        with self.env():
            self.assertIsNone(llm.critic_endpoint())
            self.assertIsNone(llm.critic_default_model())

    def test_gemini_provider(self):
        with self.env(FINCO_CRITIC_PROVIDER="gemini", GEMINI_API_KEY="g-key"):
            key, base = llm.critic_endpoint()
            self.assertEqual(key, "g-key")
            self.assertIn("generativelanguage.googleapis.com", base)
            self.assertEqual(llm.critic_default_model(), "gemini-3.5-flash")

    def test_gemini_without_key_fails_clearly(self):
        with self.env(FINCO_CRITIC_PROVIDER="gemini"):
            with self.assertRaises(ValueError):
                llm.critic_endpoint()

    def test_explicit_model_overrides_default(self):
        with self.env(FINCO_CRITIC_PROVIDER="gemini", GEMINI_API_KEY="k", FINCO_CRITIC_MODEL="gemini-x"):
            self.assertEqual(llm.critic_default_model(), "gemini-x")

    def test_generic_endpoint(self):
        with self.env(FINCO_CRITIC_BASE_URL="http://h/v1", FINCO_CRITIC_API_KEY="k"):
            self.assertEqual(llm.critic_endpoint(), ("k", "http://h/v1"))

    def test_clients_are_separate(self):
        with self.env(FINCO_CRITIC_PROVIDER="gemini", GEMINI_API_KEY="g-key"), \
                patch.object(llm, "_CLIENT", None), patch.object(llm, "_CRITIC_CLIENT", None):
            actor, critic = llm.get_client(), llm.get_client("critic")
            self.assertIsNot(actor, critic)
            self.assertIn("generativelanguage", str(critic.base_url))
            self.assertNotIn("generativelanguage", str(actor.base_url))

    def test_critic_profile_falls_back_to_actor_client(self):
        with self.env(), patch.object(llm, "_CLIENT", None), patch.object(llm, "_CRITIC_CLIENT", None):
            self.assertIs(llm.get_client("critic"), llm.get_client())


class TestThoughtSignature(unittest.TestCase):
    """Gemini 3 đòi gửi lại extra_content (thought_signature) của tool_call ở lượt kế tiếp."""

    SIG = {"google": {"thought_signature": "abc123"}}

    def _raw_call(self, **kw):
        from types import SimpleNamespace
        fn = SimpleNamespace(name="get_application", arguments='{"app_id": "APP-001"}')
        return SimpleNamespace(id="c1", function=fn, **kw)

    def test_extra_content_attribute_is_captured(self):
        self.assertEqual(llm._extra_content(self._raw_call(extra_content=self.SIG)), self.SIG)

    def test_extra_content_from_model_extra_is_captured(self):
        self.assertEqual(llm._extra_content(self._raw_call(model_extra={"extra_content": self.SIG})), self.SIG)

    def test_absent_is_none(self):
        self.assertIsNone(llm._extra_content(self._raw_call()))

    def test_signature_is_echoed_back_in_assistant_message(self):
        tc = llm.ToolCall(id="c1", name="get_application", arguments='{"app_id": "APP-001"}', extra_content=self.SIG)
        msg = llm.ChatTurn(content="", tool_calls=[tc]).to_assistant_message()
        self.assertEqual(msg["tool_calls"][0]["extra_content"], self.SIG)

    def test_no_extra_key_when_provider_gives_none(self):
        tc = llm.ToolCall(id="c1", name="x", arguments="{}")
        msg = llm.ChatTurn(content="", tool_calls=[tc]).to_assistant_message()
        self.assertNotIn("extra_content", msg["tool_calls"][0])


if __name__ == "__main__":
    unittest.main()
