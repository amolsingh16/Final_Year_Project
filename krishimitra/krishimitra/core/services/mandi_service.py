"""
Agmarknet mandi price client.

Fetches live wholesale market prices from the Government of India OGD API
(Ministry of Agriculture & Farmers Welfare).

Includes cached fallback data explicitly labeled as offline/cached when government API is unavailable.
"""

import time
from typing import Optional, List, Dict, Any
import requests
from django.conf import settings

DEFAULT_PRICE_UNIT = "Rs./Quintal"
COMMODITY_CACHE_TTL = 6 * 60 * 60  # 6 hours

_commodity_cache: dict[str, tuple[float, list[str]]] = {}

# Offline fallback snapshot for resilience (Explicitly marked as offline/static)
FALLBACK_MANDI_DATA: List[Dict[str, Any]] = [
    {"arrival_date": "Static Snapshot", "state": "Uttar Pradesh", "district": "Lucknow", "market": "Lucknow", "commodity": "Tomato", "variety": "Hybrid", "min_price": "1800", "max_price": "2400", "modal_price": "2100", "price_unit": "Rs./Quintal"},
    {"arrival_date": "Static Snapshot", "state": "Uttar Pradesh", "district": "Kanpur", "market": "Kanpur", "commodity": "Potato", "variety": "Desi", "min_price": "1200", "max_price": "1500", "modal_price": "1350", "price_unit": "Rs./Quintal"},
    {"arrival_date": "Static Snapshot", "state": "Uttar Pradesh", "district": "Varanasi", "market": "Varanasi", "commodity": "Wheat", "variety": "Dara", "min_price": "2250", "max_price": "2400", "modal_price": "2325", "price_unit": "Rs./Quintal"},
    {"arrival_date": "Static Snapshot", "state": "Uttar Pradesh", "district": "Agra", "market": "Agra", "commodity": "Onion", "variety": "Red", "min_price": "1600", "max_price": "2000", "modal_price": "1800", "price_unit": "Rs./Quintal"},
    {"arrival_date": "Static Snapshot", "state": "Maharashtra", "district": "Nashik", "market": "Lasalgaon", "commodity": "Onion", "variety": "Red", "min_price": "1500", "max_price": "2100", "modal_price": "1850", "price_unit": "Rs./Quintal"},
    {"arrival_date": "Static Snapshot", "state": "Punjab", "district": "Ludhiana", "market": "Ludhiana", "commodity": "Paddy(Dhan)", "variety": "Common", "min_price": "2183", "max_price": "2203", "modal_price": "2200", "price_unit": "Rs./Quintal"},
]


class MandiPriceError(Exception):
    """Raised when the Agmarknet API returns an error or unexpected response."""

    def __init__(self, message: str, is_fallback: bool = False, records: Optional[list] = None):
        self.message = message
        self.is_fallback = is_fallback
        self.records = records or []
        super().__init__(message)


class MandiPriceService:
    """Client for live Agmarknet mandi price data."""

    def __init__(self):
        self.api_key = settings.AGMARKNET_API_KEY
        self.resource_id = settings.AGMARKNET_RESOURCE_ID
        self.base_url = settings.AGMARKNET_BASE_URL
        self.timeout = settings.AGMARKNET_REQUEST_TIMEOUT
        self.default_limit = settings.AGMARKNET_DEFAULT_LIMIT

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.resource_id}"

    def _request(self, params: dict) -> dict:
        """Perform a GET request and return parsed JSON."""
        try:
            response = requests.get(
                self._build_url(),
                params=params,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise MandiPriceError(
                "Unable to fetch live mandi prices from Agmarknet API right now."
            ) from exc

    def _normalize_record(self, record: dict) -> dict:
        """Map API fields to the display shape used by the template."""
        return {
            "arrival_date": record.get("arrival_date", "—"),
            "state": record.get("state", "—"),
            "district": record.get("district", "—"),
            "market": record.get("market", "—"),
            "commodity": record.get("commodity", "—"),
            "variety": record.get("variety", "—"),
            "min_price": record.get("min_price", "—"),
            "max_price": record.get("max_price", "—"),
            "modal_price": record.get("modal_price", "—"),
            "price_unit": record.get("price_unit") or DEFAULT_PRICE_UNIT,
        }

    def fetch_prices(
        self,
        state: str,
        commodity: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> tuple[list[dict], bool, Optional[str]]:
        """
        Fetch live mandi price records for a state, optionally filtered by commodity.

        Returns tuple: (records_list, is_fallback_flag, notice_message_or_None)
        """
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit or self.default_limit,
            "filters[state.keyword]": state,
        }

        if commodity:
            params["filters[commodity.keyword]"] = commodity

        try:
            data = self._request(params)
            records = data.get("records") or []

            if not records:
                # No records found for specific query
                return [], False, "No live mandi records found for the selected state/commodity."

            return [self._normalize_record(r) for r in records], False, None

        except MandiPriceError:
            # Fallback to offline static snapshot when live API fails
            fallback_filtered = [
                r for r in FALLBACK_MANDI_DATA
                if r["state"].lower() == state.lower() and (not commodity or r["commodity"].lower() == commodity.lower())
            ]
            if not fallback_filtered:
                fallback_filtered = FALLBACK_MANDI_DATA[:4]

            notice = "Offline/Fallback data – may not be current. Live Agmarknet server is unreachable at this moment."
            return fallback_filtered, True, notice

    def fetch_commodities_for_state(self, state: str) -> list[str]:
        """
        Return a sorted list of unique commodity names available in a state.
        Results are cached in memory for COMMODITY_CACHE_TTL seconds.
        """
        now = time.time()
        cached = _commodity_cache.get(state)
        if cached and (now - cached[0]) < COMMODITY_CACHE_TTL:
            return cached[1]

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": 500,
            "filters[state.keyword]": state,
        }

        try:
            data = self._request(params)
            records = data.get("records") or []
            commodities = sorted({r["commodity"] for r in records if r.get("commodity")})
            if commodities:
                _commodity_cache[state] = (now, commodities)
                return commodities
        except Exception:
            pass

        # Fallback commodity list
        default_commodities = sorted({"Tomato", "Potato", "Onion", "Wheat", "Paddy(Dhan)", "Mustard", "Cotton"})
        return default_commodities
