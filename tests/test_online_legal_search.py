from __future__ import annotations

import unittest

from server.online_legal_search import extract_law_names, extract_web_search_sources, is_official_legal_url


class OfficialLegalSourceTests(unittest.TestCase):
    def test_extracts_formal_law_names_without_keyword_guessing(self) -> None:
        names = extract_law_names(
            "核验《中华人民共和国反家庭暴力法》",
            "另外可能适用《民法典》。不要把普通的赔偿关键词当成法律名称。",
        )
        self.assertEqual(names, ["中华人民共和国反家庭暴力法", "民法典"])

    def test_official_domain_allowlist_is_suffix_safe(self) -> None:
        self.assertTrue(is_official_legal_url("https://www.gov.cn/zhengce/content/test.htm"))
        self.assertTrue(is_official_legal_url("https://www.npc.gov.cn/npc/c2/c30834/test.shtml"))
        self.assertTrue(is_official_legal_url("https://gongbao.court.gov.cn/details/test.html"))
        self.assertFalse(is_official_legal_url("https://gov.cn.evil.example/fake"))
        self.assertFalse(is_official_legal_url("https://example.com/?next=https://www.gov.cn"))
        self.assertFalse(is_official_legal_url("file:///etc/passwd"))

    def test_extracts_search_sources_and_message_citations(self) -> None:
        payload = {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "query": "site:gov.cn 民法典",
                        "sources": [
                            {"url": "https://www.gov.cn/xinwen/test.htm", "title": "民法典"},
                            {"url": "https://example.com/repost", "title": "转载"},
                        ],
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {"type": "url_citation", "url": "https://www.spp.gov.cn/test", "title": "解释"}
                            ],
                        }
                    ],
                },
            ]
        }
        sources, queries, request_count = extract_web_search_sources(payload)
        self.assertEqual(request_count, 1)
        self.assertEqual(queries, ["site:gov.cn 民法典"])
        self.assertEqual(len(sources), 3)
        self.assertTrue(sources[0]["official_domain"])
        self.assertFalse(sources[1]["official_domain"])
        self.assertTrue(sources[2]["official_domain"])


if __name__ == "__main__":
    unittest.main()
