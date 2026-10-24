import unittest

try:
    from server.main import _canonical_intent, _merge_case_memory
    MAIN_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # The minimal CI requirements omit FastAPI.
    _canonical_intent = None
    _merge_case_memory = None
    MAIN_IMPORT_ERROR = exc

try:
    from server.qwen37_retrieval import LocalBM25Index
    RETRIEVAL_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    LocalBM25Index = None
    RETRIEVAL_IMPORT_ERROR = exc


class MultiTurnRetrievalTests(unittest.TestCase):
    @unittest.skipIf(RETRIEVAL_IMPORT_ERROR is not None, "retrieval dependencies are not installed")
    def test_case_bm25_uses_case_corpus(self):
        index = LocalBM25Index(
            [
                {"chunk_id": "case-1", "title": "劳动争议案例", "text": "公司拖欠工资"},
                {"chunk_id": "case-2", "title": "租赁纠纷案例", "text": "房东拒绝退还押金"},
            ],
            corpus="case",
        )
        hit = index.search("押金", 1)[0]
        self.assertEqual(hit["_source"]["corpus"], "case")
        self.assertEqual(hit["_source"]["chunk_id"], "case-2")

    @unittest.skipIf(MAIN_IMPORT_ERROR is not None, "server runtime dependencies are not installed")
    def test_memory_tracks_cases_and_user_corrections(self):
        memory = _merge_case_memory(
            {},
            {
                "intent": "legal_consultation",
                "entities": {"关系": "男朋友", "行为": "殴打"},
                "case_action": "continue",
                "is_complete": True,
                "needs_context": False,
            },
        )
        memory = _merge_case_memory(
            memory,
            {
                "intent": "legal_consultation",
                "entities": {"关系": "同事"},
                "corrections": {"关系": "同事"},
                "case_action": "correction",
            },
        )
        case = memory["cases"][memory["active_case_id"]]
        self.assertEqual(case["facts"]["关系"], "同事")
        self.assertTrue(memory["conflicts"])
        self.assertEqual(memory["schema_version"], 2)

    @unittest.skipIf(MAIN_IMPORT_ERROR is not None, "server runtime dependencies are not installed")
    def test_intent_aliases(self):
        self.assertEqual(_canonical_intent("法律咨询"), "legal_consultation")
        self.assertEqual(_canonical_intent("harmful_request"), "harmful_request")
        self.assertEqual(_canonical_intent("信息不足"), "ambiguous")


if __name__ == "__main__":
    unittest.main()
