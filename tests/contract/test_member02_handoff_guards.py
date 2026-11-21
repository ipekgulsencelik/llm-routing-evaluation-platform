"""Member 02 import-direction and offline-release regression guards.

Static checks limit accidental dependency inversions; they are not a sandbox,
proof of absence of dynamic execution, or production security authorization.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Literal

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "llm_routing_eval"
_LOCAL_ROOT = "llm_routing_eval"
_ROUTER_NAMESPACE = "llm_routing_eval.routing"
_PROVIDER_NAMESPACE = "llm_routing_eval.adapters.providers"
_CORE_NAMESPACE = "llm_routing_eval.core"
_ROUTER_NETWORK_MODULES = frozenset(
    {"httpx", "requests", "urllib", "urllib3", "aiohttp", "socket", "http", "groq", "openai"}
)


def _member_module(path: Path) -> str:
    relative = path.relative_to(PACKAGE)
    return ".".join((_LOCAL_ROOT, *relative.with_suffix("").parts))


def _dependencies(tree: ast.AST, *, package: str) -> tuple[str, ...]:
    """Resolve direct imports, namespace imports, and relative imports correctly."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            else:
                base = node.module or ""
            if base:
                # A root namespace import is valid only through permitted
                # concrete members (e.g. "from llm_routing_eval import core").
                # Wildcard root imports remain rejected.
                if base != _LOCAL_ROOT or any(alias.name == "*" for alias in node.names):
                    modules.append(base)
                modules.extend(base + "." + alias.name for alias in node.names if alias.name != "*")
    return tuple(modules)


def _known_dynamic_imports(tree: ast.AST) -> bool:
    """Reject direct importlib / __import__ entrypoints, including common aliases."""
    module_names = {"importlib"}
    function_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(
                alias.asname or "importlib" for alias in node.names if alias.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            function_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in function_names:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id in module_names
        ):
            return True
    return False


def _is_namespace(name: str, namespace: str) -> bool:
    return name == namespace or name.startswith(namespace + ".")


def _check_source(source: str, *, package: str, layer: Literal["router", "provider"]) -> None:
    syntax = ast.parse(source)
    assert not _known_dynamic_imports(syntax), f"{layer}: DYNAMIC_IMPORT_REQUIRES_REVIEW"
    allowed = (
        (_CORE_NAMESPACE, _ROUTER_NAMESPACE)
        if layer == "router"
        else (_CORE_NAMESPACE, _PROVIDER_NAMESPACE)
    )
    for module in _dependencies(syntax, package=package):
        if module == _LOCAL_ROOT or module.startswith(_LOCAL_ROOT + "."):
            assert any(_is_namespace(module, prefix) for prefix in allowed), (
                f"{layer}: FORBIDDEN_PROJECT_IMPORT: {module}"
            )
        if layer == "router":
            assert module.split(".", maxsplit=1)[0] not in _ROUTER_NETWORK_MODULES, (
                f"{layer}: NETWORK_IMPLEMENTATION_IMPORT: {module}"
            )


def _check_package(path: Path, *, layer: Literal["router", "provider"]) -> None:
    assert path.is_dir(), f"Missing Member 02 package {path}"
    files = sorted(path.glob("*.py"))
    assert files, f"No Member 02 modules found under {path}"
    for module_path in files:
        _check_source(
            module_path.read_text(encoding="utf-8"),
            package=_member_module(module_path).rsplit(".", maxsplit=1)[0],
            layer=layer,
        )


def test_router_layer_depends_only_on_core_and_routing_implementations() -> None:
    _check_package(PACKAGE / "routing", layer="router")


def test_provider_layer_depends_only_on_core_and_provider_implementations() -> None:
    _check_package(PACKAGE / "adapters" / "providers", layer="provider")


@pytest.mark.parametrize(
    ("layer", "package", "source"),
    [
        ("router", _ROUTER_NAMESPACE, "import llm_routing_eval.adapters.providers"),
        ("router", _ROUTER_NAMESPACE, "from llm_routing_eval import adapters"),
        ("router", _ROUTER_NAMESPACE, "from ..application import orchestrator"),
        ("router", _ROUTER_NAMESPACE, "import httpx"),
        ("router", _ROUTER_NAMESPACE, "from urllib.request import Request"),
        ("provider", _PROVIDER_NAMESPACE, "from llm_routing_eval import routing"),
        ("provider", _PROVIDER_NAMESPACE, "from .. import embeddings"),
        ("provider", _PROVIDER_NAMESPACE, "from llm_routing_eval.privacy import adapter"),
        ("provider", _PROVIDER_NAMESPACE, "import llm_routing_eval.application"),
        ("provider", _PROVIDER_NAMESPACE, "import importlib as il\nil.import_module(name)"),
        (
            "router",
            _ROUTER_NAMESPACE,
            "from importlib import import_module as loader\nloader(name)",
        ),
        ("provider", _PROVIDER_NAMESPACE, "__import__(name)"),
    ],
)
def test_known_cross_layer_import_bypasses_are_rejected(
    layer: Literal["router", "provider"], package: str, source: str
) -> None:
    with pytest.raises(AssertionError):
        _check_source(source, package=package, layer=layer)


@pytest.mark.parametrize(
    ("layer", "package", "source"),
    [
        ("router", _ROUTER_NAMESPACE, "from .strategies import ModelPair"),
        ("router", _ROUTER_NAMESPACE, "from llm_routing_eval.core import RouterPort"),
        ("provider", _PROVIDER_NAMESPACE, "from .transport import GroqTransport"),
        ("provider", _PROVIDER_NAMESPACE, "from llm_routing_eval import core"),
    ],
)
def test_allowed_internal_imports_still_pass(
    layer: Literal["router", "provider"], package: str, source: str
) -> None:
    _check_source(source, package=package, layer=layer)


def test_current_release_configuration_is_explicitly_offline_and_unpriced() -> None:
    """Configuration check is defense-in-depth, not a complete runtime policy."""
    providers = yaml.safe_load((ROOT / "config" / "providers.yaml").read_text(encoding="utf-8"))
    execution = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text(encoding="utf-8"))

    assert providers["providers"]["groq"]["enabled"] is False
    assert providers["providers"]["nvidia_build"]["enabled"] is False
    assert execution["execution"]["mode"] == "offline"
    assert execution["execution"]["live_enabled"] is False
    assert pricing["status"] == "unverified"
    assert pricing["rates"] == {}
