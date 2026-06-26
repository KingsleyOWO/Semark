import unittest

from app.eval.retrieval import (
    RetrievalCandidate,
    query_terms_for_scoring,
    rank_candidates,
    score_candidate,
)


class RetrievalEvalTest(unittest.TestCase):
    def test_score_candidate_counts_query_terms(self):
        score = score_candidate("property form approval", "property increase approval workflow")
        self.assertEqual(score, 2)

    def test_query_terms_include_cjk_phrases(self):
        terms = query_terms_for_scoring("財產增加單 簽核流程 approval")
        self.assertIn("財產增加單", terms)
        self.assertIn("簽核流程", terms)
        self.assertIn("approval", terms)

    def test_query_terms_expand_visual_diagram_terms(self):
        terms = query_terms_for_scoring("組織圖 架構")
        self.assertIn("flowchart", terms)
        self.assertIn("hierarchical", terms)
        self.assertIn("structure", terms)

    def test_rank_candidates_prefers_matching_asset(self):
        candidates = [
            RetrievalCandidate(
                candidate_id="c1",
                candidate_type="chunk",
                text="unrelated text",
                source_path="chunks.jsonl",
            ),
            RetrievalCandidate(
                candidate_id="form1",
                candidate_type="form_asset",
                text="property form approval workflow",
                source_path="assets_index.jsonl",
            ),
        ]

        ranked = rank_candidates("property approval", candidates, top_k=1)

        self.assertEqual(ranked[0].candidate_id, "form1")
        self.assertEqual(ranked[0].score, 2)

    def test_rank_candidates_boosts_visual_asset_aliases(self):
        candidates = [
            RetrievalCandidate(
                candidate_id="c1",
                candidate_type="chunk",
                text="董事會 院長",
                source_path="chunks.jsonl",
            ),
            RetrievalCandidate(
                candidate_id="fig1",
                candidate_type="figure_asset",
                text="flowchart hierarchical accounting system",
                source_path="assets_index.jsonl",
            ),
        ]

        ranked = rank_candidates("組織圖 架構 董事會 院長", candidates, top_k=2)

        self.assertEqual(ranked[0].candidate_id, "fig1")
        self.assertEqual(ranked[0].candidate_type, "figure_asset")


if __name__ == "__main__":
    unittest.main()
