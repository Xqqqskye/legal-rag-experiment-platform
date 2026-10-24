"""Offline regressions for chat persistence, route controls and batch metrics."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import main
from server.chat_store import ChatStore
from server.qwen37_retrieval import retrieve_evidence, rrf_candidates


class ChatRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ChatStore(Path(self.temp.name) / "chat.sqlite3")
        self.store.initialize()
        self.session = self.store.create_session("regression")
        self.patches = [
            patch.object(main, "chat_store", self.store),
            patch.object(main.runtime, "embedder", object()),
            patch.object(main.runtime, "client", object()),
            patch.object(main.runtime, "error", ""),
        ]
        for item in self.patches:
            item.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_single_character_can_be_clarified_and_blank_is_rejected(self):
        result = {"intent": "ambiguous", "analysis": {"intent": "ambiguous", "needs_context": True}, "answer": "请补充上下文"}
        url = f"/api/chat/sessions/{self.session['id']}/messages"
        with patch.object(main, "_run_query", return_value=result):
            response = self.client.post(url, json={"message": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["session"]["messages"]), 2)
        self.assertEqual(self.client.post(url, json={"message": "  "}).status_code, 422)

    def test_failed_generation_does_not_commit_a_partial_turn(self):
        with patch.object(main, "_run_query", side_effect=RuntimeError("upstream unavailable")):
            response = self.client.post(f"/api/chat/sessions/{self.session['id']}/messages", json={"message": "公司欠薪"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.store.get_session(self.session["id"])["messages"], [])

    def test_deletion_removes_messages_and_memory(self):
        self.store.append_turn(self.session["id"], "问题", "回答", {}, {"facts": {"amount": 100}})
        self.assertEqual(self.client.delete(f"/api/chat/sessions/{self.session['id']}").status_code, 200)
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_case_option_reaches_pipeline(self):
        with patch.object(main, "_run_query", return_value={"analysis": {}, "answer": "回答"}) as run:
            self.client.post(f"/api/chat/sessions/{self.session['id']}/messages", json={"message": "公司欠薪", "include_cases": False})
        self.assertFalse(run.call_args.args[0].include_cases)


class MemoryAndRetrievalTests(unittest.TestCase):
    def test_switch_keeps_legacy_case_and_can_switch_back(self):
        legacy = {"entities": {"关系": "房东"}, "legal_issues": ["押金"]}
        newer = main._merge_case_memory(legacy, {"intent": "legal", "case_action": "new_case", "entities": {"关系": "公司"}})
        self.assertEqual(len(newer["cases"]), 2)
        restored = main._merge_case_memory(newer, {"intent": "legal", "case_action": "switch", "target_case_id": "case-1"})
        self.assertEqual(restored["entities"]["关系"], "房东")
        self.assertEqual(newer["active_case_id"], "case-2")  # input not mutated

    def test_chitchat_does_not_overwrite_case(self):
        original = main._merge_case_memory({}, {"intent": "legal", "entities": {"关系": "房东"}})
        changed = main._merge_case_memory(original, {"intent": "non_legal", "case_action": "new_case", "entities": {"关系": "朋友"}})
        self.assertEqual(changed["cases"], original["cases"])

    def test_dense_only_skips_all_lexical_work_and_disabled_cases(self):
        class Index:
            def search(self, vector, k):
                return [{"_id": "law-1", "_source": {"corpus": "law"}}]
        model = SimpleNamespace(encode=lambda *args, **kwargs: [1.0], law_index=Index())
        with patch("server.qwen37_retrieval.retrieve", side_effect=AssertionError("BM25 must not run")):
            law_bm25, law_dense, case_bm25, case_dense, _ = retrieve_evidence(None, model, "query", include_bm25=False, include_cases=False)
        self.assertTrue(law_dense)
        self.assertEqual((law_bm25, case_bm25, case_dense), ([], [], []))

    def test_case_and_law_lexical_scores_are_fused_by_rank(self):
        law = [{"_id": "law", "_source": {"chunk_id": "law"}}]
        case = [{"_id": "case", "_source": {"chunk_id": "case", "corpus": "case"}}]
        result = rrf_candidates(law, [], case_bm25=case)
        self.assertEqual(result[0]["rrf"], result[1]["rrf"])

    def test_canonical_legal_labels_are_included_in_batch_citation_metric(self):
        run = {"benchmark": {"expected_intent": "legal_consultation", "citation_valid": True}, "result": {}}
        case = {"case_id": "x", "question": "问题", "category": "法律", "result": {"runs": [run]}}
        self.assertEqual(main._aggregate_pipeline([case], 0)["citation_valid_rate"], 1.0)

    def test_case_toggle_is_a_reported_ab_variable(self):
        payload = main.CompareRequest(question="你好", pipeline_a=main.ExperimentPipelineConfig(include_cases=False), pipeline_b=main.ExperimentPipelineConfig(include_cases=True))
        with patch.object(main, "_run_query", return_value={"intent": "non_legal", "timings_sec": {"total": 1}}):
            comparison = main._run_comparison(payload)
        self.assertEqual(comparison["comparison"]["changed_fields"], ["include_cases"])
        self.assertEqual(comparison["comparison"]["changed_labels"], ["案例检索"])


if __name__ == "__main__":
    unittest.main()
