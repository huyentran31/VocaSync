"""Smoke test extract_vocab with a mocked AI call (no key / network needed)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy")):
    sys.path.insert(0, p)

import config
import extract_vocab as ev


def test_no_key_is_system_error():
    config.AI_API_KEY = ""
    try:
        ev.extract_vocab("hello")
        assert False, "should have raised SystemError_"
    except ev.SystemError_:
        pass


def test_parse_dedup_and_skip_bad():
    config.AI_API_KEY = "fake"
    fenced = (
        "```json\n"
        '[{"term":"reduce","sentence":"We must reduce emissions.","tag":"Word"},'
        ' {"term":"REDUCE","sentence":"dup line","tag":"Word"},'
        ' {"bad":"item"},'
        ' {"term":"run into","sentence":"I ran into him.","surface":"ran into","tag":"Phrasal Verb"}]'
        "\n```"
    )
    ev.call_ai = lambda prompt, sysp, model=None: fenced
    out = ev.extract_vocab(["We must reduce emissions.", "I ran into him."])
    terms = [c["term"].lower() for c in out]
    assert terms == ["reduce", "run into"], terms          # dup + bad dropped
    assert out[1]["surface"] == "ran into"
    print("extract_vocab:", terms)


def test_self_correct_drops_ungrounded():
    """Python self-correct: a term that does NOT appear in the transcript is flagged
    and dropped (the fix call here keeps returning the same hallucination)."""
    config.AI_API_KEY = "fake"
    # 'reduce' IS in the transcript; 'unicorn' is NOT (hallucinated) -> must be dropped.
    payload = (
        '[{"term":"reduce","sentence":"We must reduce emissions.","surface":"reduce","tag":"Word"},'
        ' {"term":"unicorn","sentence":"made up line","surface":"unicorn","tag":"Word"}]'
    )
    ev.call_ai = lambda prompt, sysp, model=None: payload
    out = ev.extract_vocab(["We must reduce emissions."])
    terms = [c["term"].lower() for c in out]
    assert terms == ["reduce"], terms     # ungrounded 'unicorn' removed
    print("self-correct:", terms)


def test_grounded_rejects_translation_mismatch():
    """S15 T1: term matched via `surface` must share a content lemma with the term.
    Blocks the 'AI translated a foreign line into an English idiom' bug."""
    text = ev._norm_match("Bah, jouez-moi, cherie. We are getting a divorce. I figured it out.")
    # (a) foreign-word surface, English term, NO shared lemma -> ungrounded
    bad = {"term": "do someone a favor", "surface": "jouez-moi",
           "sentence": "Bah, jouez-moi, cherie."}
    assert ev._grounded(bad, text) is False
    # (b) inflected surface sharing a lemma with the term -> grounded
    ok1 = {"term": "get a divorce", "surface": "getting a divorce",
           "sentence": "We are getting a divorce."}
    assert ev._grounded(ok1, text) is True
    ok2 = {"term": "figure out", "surface": "figured it out",
           "sentence": "I figured it out."}
    assert ev._grounded(ok2, text) is True
    # (c) term itself appears verbatim -> self-grounded even without surface
    ok3 = {"term": "divorce", "surface": "", "sentence": "We are getting a divorce."}
    assert ev._grounded(ok3, text) is True
    print("grounded translation-mismatch OK")


if __name__ == "__main__":
    test_no_key_is_system_error()
    test_parse_dedup_and_skip_bad()
    test_self_correct_drops_ungrounded()
    test_grounded_rejects_translation_mismatch()
    print("OK")
