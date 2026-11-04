from datetime import date

from evidence.rank import BM25, classify_design, rank, recency, tokenize


def test_tokenize_drops_stopwords():
    assert tokenize("The patients were compared with placebo") == ["placebo"]


def test_bm25_prefers_matching_document():
    docs = [tokenize("tranexamic acid transfusion knee"), tokenize("aspirin headache")]
    bm = BM25(docs)
    q = tokenize("tranexamic acid")
    assert bm.score(q, 0) > bm.score(q, 1) == 0


def test_design_classification(articles):
    assert classify_design(articles[0]) == ("systematic_review", 1.0)
    assert classify_design(articles[1])[0] == "rct"
    assert classify_design(articles[2])[0] == "case_control"  # retrospective studies MeSH
    assert classify_design(articles[3])[0] == "case_report"


def test_design_from_title_when_types_are_thin(articles):
    a = articles[2]
    a.mesh_terms = []
    a.publication_types = []
    assert classify_design(a)[0] == "cohort"  # 'registry' and 'cohort' in the title


def test_recency_decays():
    today = date(2026, 1, 1)
    assert recency(2026, today) == 1.0
    assert 0.49 < recency(2018, today) < 0.51
    assert recency(None, today) == 0.3


def test_rank_puts_systematic_review_first_and_case_report_last(articles):
    top = rank(articles[:4], "tranexamic acid knee replacement transfusion", top_k=4, today=date(2026, 1, 1))
    assert top[0].pmid == "30000001"
    assert top[1].pmid == "30000002"
    assert top[1].score > top[2].score > top[3].score
    assert all(0 <= a.score <= 1 for a in top)


def test_rank_respects_top_k(articles):
    assert len(rank(articles, "tranexamic", top_k=2)) == 2


def test_rank_empty():
    assert rank([], "x", top_k=5) == []
