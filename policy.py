"""
policy.py — Policy Server (Day 5: Zero-Trust Development / structural gating).

Autonomous agents are probabilistic; hard-coding constraints into a system prompt is
brittle (contexts overflow, prompts get injected). So tool permissions are enforced
OUTSIDE the model, at the execution boundary: every `registry.call_tool` passes through
`PolicyService.check(tool)` before the tool can touch disk or the network.

This is the STRUCTURAL gate (the "traffic lights" from the Day-5 paper): fast, binary,
role/environment based. It does NOT call an LLM. The complementary SEMANTIC gate
(an LLM judging tool ARGUMENTS for e.g. PII) is described in the writeup as Vision.

Rules live in `specs/config/execution_policy.yaml` (`policy:` block) — tune without
touching code. Safe defaults: if the file/section is missing, the policy is permissive
(allow all) so the harness never breaks. Pick the role/env via env vars:
    CAPSTONE_ROLE = agent | viewer        (default: policy.default_role)
    CAPSTONE_ENV  = demo | ...            (default: policy.default_env)
"""

from __future__ import annotations

import os

_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "specs", "config", "execution_policy.yaml")


class PolicyViolation(Exception):
    """Raised when the structural policy blocks a tool call (zero-trust)."""


def _load_policy() -> dict:
    """Read the `policy:` block from the YAML. Permissive default if absent/unpar. """
    try:
        import yaml  # pyyaml; optional — fall back to permissive if missing
        with open(_POLICY_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("policy", {}) or {}
    except Exception:
        return {}


class PolicyService:
    """Structural tool-access gate. One instance is enough; cheap to construct."""

    def __init__(self, role: str | None = None, env: str | None = None,
                 policy: dict | None = None):
        self.policy = policy if policy is not None else _load_policy()
        self.enabled = bool(self.policy.get("enabled", False))
        self.role = role or os.environ.get("CAPSTONE_ROLE") or self.policy.get("default_role", "agent")
        self.env = env or os.environ.get("CAPSTONE_ENV") or self.policy.get("default_env", "demo")

    def is_tool_allowed(self, tool_name: str) -> bool:
        if not self.enabled:
            return True                                   # policy off -> permissive
        # 1) Environment block list (e.g. send_email blocked on localhost)
        env_cfg = (self.policy.get("environments", {}) or {}).get(self.env, {}) or {}
        if tool_name in (env_cfg.get("blocked_tools", []) or []):
            return False
        # 2) Role allow list ('*' = full toolset)
        role_cfg = (self.policy.get("roles", {}) or {}).get(self.role, {}) or {}
        allowed = role_cfg.get("allowed_tools", ["*"]) or ["*"]
        return "*" in allowed or tool_name in allowed

    def check(self, tool_name: str) -> None:
        """Raise PolicyViolation if the call is not permitted (intercept point)."""
        if not self.is_tool_allowed(tool_name):
            raise PolicyViolation(
                f"Policy denied tool '{tool_name}' for role='{self.role}', env='{self.env}'. "
                f"Adjust specs/config/execution_policy.yaml or CAPSTONE_ROLE/CAPSTONE_ENV."
            )


# Process-wide singleton (reads env vars at first use; cheap).
_DEFAULT: PolicyService | None = None


def default_policy() -> PolicyService:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyService()
    return _DEFAULT


if __name__ == "__main__":
    p = PolicyService()
    print(f"role={p.role} env={p.env} enabled={p.enabled}")
    for t in ("recall", "wordnet_lookup", "conceptnet_lookup", "enrich", "make_anki"):
        print(f"  {t:20} allowed={p.is_tool_allowed(t)}")
