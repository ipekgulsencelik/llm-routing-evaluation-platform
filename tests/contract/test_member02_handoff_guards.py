"""Member 02 delivery gates: architecture direction and offline fail-closed baseline.

These are static/manifest checks, not an approval of live Groq, financial quality,
privacy controls, costs, or an enterprise security certification.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "llm_routing_eval"
_FORBIDDEN_SHARED = (
    "llm_routing_eval.application",
    "llm_routing_eval.cli",
    "llm_routing_eval.demo",
    "llm_routing_eval.domains",
    "llm_routing_eval.evaluation",
    "llm_routing_eval.privacy",
    "llm_routing_eval.reporting",
)


def _absolute_imports(path: Path) -> tuple[str, ...]:
    """Audit imports without importing production modules or running any IO."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level <= 1, f"Cross-package relative import in {path}"
            if node.level == 0 and node.module is not None:
                modules.append(node.module)
    return tuple(modules)


def test_router_layer_never_imports_application_provider_or_evaluation_code() -> None:
    """Core-facing strategies must remain substitutable, not a hidden gateway."""
    forbidden = (
        *_FORBIDDEN_SHARED,
        "llm_routing_eval.adapters",
        "httpx",
        "requests",
        "urllib",
        "socket",
    )
    for path in sorted((PACKAGE / "routing").glob("*.py")):
        for module in _absolute_imports(path):
            assert not any(
                module == prefix or module.startswith(prefix + ".") for prefix in forbidden
            ), f"Member02 router dependency violation in {path.name}: {module}"


def test_provider_adapters_never_import_application_demo_or_member03_member04() -> None:
    """Provider implementations depend inward on shared Core contracts only."""
    for path in sorted((PACKAGE / "adapters" / "providers").glob("*.py")):
        for module in _absolute_imports(path):
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _FORBIDDEN_SHARED
            ), f"Member02 provider dependency violation in {path.name}: {module}"


def test_current_release_configuration_is_explicitly_offline_and_unpriced() -> None:
    """Update these release guards only after cross-member approval evidence."""
    providers = yaml.safe_load((ROOT / "config" / "providers.yaml").read_text(encoding="utf-8"))
    execution = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text(encoding="utf-8"))

    assert providers["providers"]["groq"]["enabled"] is False
    assert providers["providers"]["nvidia_build"]["enabled"] is False
    assert execution["execution"]["mode"] == "offline"
    assert execution["execution"]["live_enabled"] is False
    assert pricing["status"] == "unverified"
    assert pricing["rates"] == {}
