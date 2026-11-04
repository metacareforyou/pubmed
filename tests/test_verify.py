from evidence.models import Finding, Quote, Synthesis
from evidence.rank import rank
from evidence.verify import numbers_in, quote_in, verify, verify_finding, weakest_cap


def ranked(articles):
    return rank(articles[:4], "tranexamic acid", top_k=4)


def test_quote_matching_is_case_and_punctuation_tolerant():
    record = "Tranexamic acid reduced the risk of transfusion (relative risk 0.45, 95% CI 0.38 to 0.53)."
    assert quote_in("tranexamic acid reduced the risk of transfusion", record)
    assert quote_in("Relative risk 0.45, 95% CI 0.38 to 0.53", record)
    assert not quote_in("tranexamic acid increased the risk of transfusion", record)
    assert not quote_in("short", record)


def test_numbers_extracted_from_statement():
    assert numbers_in("RR 0.45 (95% CI 0.38 to 0.53) in 42 trials; p<0.001.") == ["0.45", "95", "0.38", "0.53", "42", "0.001"]


def test_verified_finding_survives(articles):
    by = {a.pmid: a for a in ranked(articles)}
    f = Finding("Tranexamic acid reduced transfusion (relative risk 0.45, 95% CI 0.38 to 0.53).", ["30000001"],
                [Quote("30000001", "Tranexamic acid reduced the risk of transfusion (relative risk 0.45, 95% CI 0.38 to 0.53)")])
    assert verify_finding(f, by).verified


def test_finding_with_unretrieved_pmid_is_rejected(articles):
    by = {a.pmid: a for a in ranked(articles)}
    f = Finding("x", ["99999999"], [Quote("99999999", "anything at all here")])
    assert not verify_finding(f, by).verified
    assert "not retrieved" in f.rejected_reason


def test_finding_with_invented_number_is_rejected(articles):
    by = {a.pmid: a for a in ranked(articles)}
    f = Finding("Tranexamic acid cut transfusion by 80%.", ["30000001"], [Quote("30000001", "Tranexamic acid reduced the risk of transfusion")])
    assert not verify_finding(f, by).verified
    assert "80" in f.rejected_reason


def test_finding_without_quote_for_each_pmid_is_rejected(articles):
    by = {a.pmid: a for a in ranked(articles)}
    f = Finding("Both agree.", ["30000001", "30000002"], [Quote("30000001", "Tranexamic acid reduced the risk of transfusion")])
    assert not verify_finding(f, by).verified
    assert "30000002" in f.rejected_reason


def test_quote_attributed_to_wrong_pmid_is_rejected(articles):
    by = {a.pmid: a for a in ranked(articles)}
    f = Finding("x", ["30000001"], [Quote("30000002", "transfusion occurred in 4% of the intravenous group")])
    assert not verify_finding(f, by).verified


def test_weakest_cap(articles):
    top = ranked(articles)
    by = {a.pmid: a for a in top}
    assert weakest_cap([by["30000001"]]) == ("high", "")
    cap, reason = weakest_cap([by["30000001"], by["30000003"]])
    assert cap == "low" and "30000003" in reason
    assert weakest_cap([by["30000004"]])[0] == "very_low"


def test_verify_caps_certainty_by_design(articles):
    top = ranked(articles)
    s = Synthesis("bl", [Finding("Transfusion was 11% with tranexamic acid and 23% without.", ["30000003"],
                                [Quote("30000003", "transfusion rate was 11% with tranexamic acid and 23% without")])],
                  "high", "r", stated_certainty="high")
    out = verify(s, top)
    assert out.findings[0].verified
    assert out.certainty == "low" and out.certainty_cap == "low"


def test_verify_keeps_lower_stated_certainty(articles):
    top = ranked(articles)
    s = Synthesis("bl", [Finding("RR 0.45.", ["30000001"], [Quote("30000001", "Tranexamic acid reduced the risk of transfusion (relative risk 0.45")])],
                  "moderate", "r", stated_certainty="moderate")
    assert verify(s, top).certainty == "moderate"


def test_verify_marks_insufficient_when_nothing_survives(articles):
    top = ranked(articles)
    s = Synthesis("bl", [Finding("x", ["1"], [])], "high", "r", stated_certainty="high")
    out = verify(s, top)
    assert out.insufficient and out.certainty == "insufficient"
