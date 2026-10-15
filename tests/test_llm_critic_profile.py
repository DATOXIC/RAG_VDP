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


if __name__ == "__main__":
    unittest.main()
