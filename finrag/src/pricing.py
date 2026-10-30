"""Validate external token rates and resolve explicit provider/model aliases."""
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def load_prices(path: Path) -> dict[tuple[str, str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("currency") != "USD" or data.get("rate_unit") != "per_million_tokens" or not isinstance(data.get("providers"), dict):
        raise ValueError("Pricing configuration needs USD rates per million tokens and a providers object.")
    prices = {}
    for provider, models in data["providers"].items():
        if not provider.strip() or not isinstance(models, dict):
            raise ValueError("Each pricing provider must contain a model object.")
        for model, values in models.items():
            if not isinstance(values, dict):
                raise ValueError("Each priced model must contain a rate object.")
            rates = {"model": model}
            for field in ("input", "cached_input", "output"):
                raw = values.get(field)
                try:
                    rate = Decimal(str(raw)) if raw is not None and not isinstance(raw, bool) else None
                except InvalidOperation as error:
                    raise ValueError(f"Invalid {provider}/{model} {field} price.") from error
                if raw is not None and (rate is None or not rate.is_finite() or rate < 0):
                    raise ValueError(f"Invalid {provider}/{model} {field} price.")
                rates[field] = rate
            aliases = values.get("aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
                raise ValueError("Pricing aliases must be nonempty model names.")
            for alias in (model, *aliases):
                key = (provider.strip(), alias.strip())
                if not key[1] or key in prices:
                    raise ValueError(f"Duplicate or empty pricing alias: {provider}/{alias}.")
                prices[key] = rates
    return prices
