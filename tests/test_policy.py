"""
Deterministic test for the Policy Server (Day 5 zero-trust / structural gating).

Verifies that registry.call_tool is intercepted by policy.py BEFORE a tool runs:
  • role 'agent' (default)  -> full toolset allowed
  • role 'viewer'           -> only read/deterministic tools; write tools BLOCKED
  • disabled policy          -> permissive (nothing breaks)

Offline, no key, no network. Run: python tests/test_policy.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "legacy"), os.path.join(ROOT, "agent")):
    sys.path.insert(0, p)

from policy import PolicyService, PolicyViolation
import registry


def test_agent_role_allows_everything():
    p = PolicyService(role="agent", env="demo")
    for t in ("recall", "wordnet_lookup", "conceptnet_lookup", "enrich", "make_anki", "build_render_graph"):
        assert p.is_tool_allowed(t), f"agent should be allowed {t}"
    print("agent role: full toolset allowed OK")


def test_viewer_role_blocks_write_tools():
    p = PolicyService(role="viewer", env="demo")
    # read/deterministic -> allowed
    assert p.is_tool_allowed("recall")
    assert p.is_tool_allowed("wordnet_lookup")
    assert p.is_tool_allowed("conceptnet_lookup")
    # write / AI tools -> blocked for a read-only viewer
    assert not p.is_tool_allowed("enrich")
    assert not p.is_tool_allowed("make_anki")
    assert not p.is_tool_allowed("ingest_transcript")
    print("viewer role: write tools blocked OK")


def test_disabled_policy_is_permissive():
    p = PolicyService(policy={"enabled": False})
    assert p.is_tool_allowed("make_anki")        # no policy -> allow all
    print("disabled policy: permissive OK")


def test_call_tool_enforces_policy():
    """The gate must fire INSIDE registry.call_tool, not just in PolicyService."""
    # Force a viewer role for this process, rebuild the singleton, and expect a block.
    os.environ["CAPSTONE_ROLE"] = "viewer"
    import policy as policy_mod
    policy_mod._DEFAULT = None                    # reset cached singleton to pick up the env

    blocked = False
    try:
        registry.call_tool("make_anki", {"units": []})
    except PolicyViolation:
        blocked = True
    finally:
        del os.environ["CAPSTONE_ROLE"]
        policy_mod._DEFAULT = None                # restore default for any later test

    assert blocked, "call_tool should have raised PolicyViolation for viewer->make_anki"
    # And the default (agent) role lets a read tool through:
    out = registry.call_tool("recall", {"lemma": "nothing"})
    assert isinstance(out, dict)
    print("call_tool enforces policy OK (viewer blocked, agent allowed)")


def test_mask_secrets_redacts_values():
    """mask_secrets must hide the secret VALUE inside strings, not just the keyword (S14 T2)."""
    from _common import mask_secrets
    fake = "AIzaFAKE1234567890abcdefghijklmn"
    masked = mask_secrets(f"https://x/y?key={fake}")
    assert fake not in masked, f"key value leaked: {masked}"
    masked2 = mask_secrets({"error": f"HTTPError for url .../gen?key={fake}"})
    assert fake not in str(masked2)
    masked3 = mask_secrets(f"Authorization: Bearer sk_live_{fake}")
    assert fake not in masked3
    # a bare Google-shaped key with no keyword prefix is still caught
    assert fake not in mask_secrets(f"oops {fake} in text")
    print("mask_secrets redacts values OK")


if __name__ == "__main__":
    test_agent_role_allows_everything()
    test_viewer_role_blocks_write_tools()
    test_disabled_policy_is_permissive()
    test_call_tool_enforces_policy()
    test_mask_secrets_redacts_values()
    print("OK")
