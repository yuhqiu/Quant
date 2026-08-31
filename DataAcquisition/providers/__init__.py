"""Provider registry: adding a vendor means registering one class here."""

from __future__ import annotations

from typing import Any

from .base import FetchRequest, FetchResult, MarketDataProvider
from .yahoo import YahooProvider

_REGISTRY: dict[str, type[MarketDataProvider]] = {}


def register_provider(provider: type[MarketDataProvider]) -> type[MarketDataProvider]:
    _REGISTRY[provider.name] = provider
    return provider


def provider_names() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, **kwargs: Any) -> MarketDataProvider:
    try:
        provider = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown provider {name!r}; choose from {provider_names()}") from exc
    return provider(**kwargs)


register_provider(YahooProvider)

__all__ = [
    "FetchRequest",
    "FetchResult",
    "MarketDataProvider",
    "YahooProvider",
    "get_provider",
    "provider_names",
    "register_provider",
]
