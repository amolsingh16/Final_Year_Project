"""
Live Weather Advisory Service using Open-Meteo API for Indian agricultural regions.
"""

import requests
from typing import Dict, Any, Optional

# Pre-mapped coordinates for common Indian agricultural hubs
INDIAN_CITIES_COORDS = {
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
}


class WeatherServiceError(Exception):
    """Raised when weather API cannot be reached or returns invalid data."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def fetch_weather_for_city(city_name: str = "Lucknow") -> Dict[str, Any]:
    """
    Fetches real-time weather and 7-day forecast from Open-Meteo API.
    Raises WeatherServiceError if network request fails or city not found.
    """
    city_info = INDIAN_CITIES_COORDS.get(city_name)
    
    if not city_info:
        # Try Open-Meteo Geocoding API for unlisted locations
        try:
            geo_res = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city_name, "count": 1, "format": "json"},
                timeout=8
            )
            geo_data = geo_res.json()
            if geo_data.get("results"):
                res = geo_data["results"][0]
                city_info = {
                    "lat": res["latitude"],
                    "lon": res["longitude"],
                    "state": res.get("admin1", "India")
                }
        except Exception:
            pass

    if not city_info:
        raise WeatherServiceError(f"Could not locate '{city_name}'. Please select a valid city or district.")

    lat, lon = city_info["lat"], city_info["lon"]

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
            f"&hourly=relative_humidity_2m,precipitation_probability"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
            f"&timezone=Asia%2FKolkata"
        )
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        current = data.get("current_weather", {})
        daily = data.get("daily", {})

        temp = current.get("temperature", "—")
        windspeed = current.get("windspeed", "—")
        weathercode = current.get("weathercode", 0)

        # Weather code mapping
        weather_desc = _map_weather_code(weathercode)

        # Hourly humidity average
        hourly_hum = data.get("hourly", {}).get("relative_humidity_2m", [])
        humidity = round(sum(hourly_hum[:24]) / len(hourly_hum[:24])) if hourly_hum else "—"

        # Today max/min & rain probability
        max_temp = daily.get("temperature_2m_max", ["—"])[0]
        min_temp = daily.get("temperature_2m_min", ["—"])[0]
        rain_prob = daily.get("precipitation_probability_max", [0])[0]
        precip_sum = daily.get("precipitation_sum", [0])[0]

        # Agriculture Advice based on live metrics
        advisory = _generate_agri_advisory(temp, rain_prob, windspeed, humidity)

        # Build 7-day forecast array
        forecast = []
        time_list = daily.get("time", [])
        max_list = daily.get("temperature_2m_max", [])
        min_list = daily.get("temperature_2m_min", [])
        rain_list = daily.get("precipitation_probability_max", [])

        for i in range(min(7, len(time_list))):
            forecast.append({
                "date": time_list[i],
                "max_temp": max_list[i] if i < len(max_list) else "—",
                "min_temp": min_list[i] if i < len(min_list) else "—",
                "rain_prob": rain_list[i] if i < len(rain_list) else 0
            })

        return {
            "city": city_name,
            "state": city_info.get("state", "India"),
            "temperature": temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "humidity": humidity,
            "windspeed": windspeed,
            "rain_prob": rain_prob,
            "precip_sum": precip_sum,
            "weather_desc": weather_desc,
            "advisory": advisory,
            "forecast": forecast
        }

    except Exception as exc:
        raise WeatherServiceError("Unable to fetch live weather data at this moment. Please check your internet connection or try again later.") from exc


def _map_weather_code(code: int) -> str:
    if code == 0:
        return "Clear Sky ☀️"
    elif code in [1, 2, 3]:
        return "Partly Cloudy ⛅"
    elif code in [45, 48]:
        return "Foggy 🌫️"
    elif code in [51, 53, 55, 61, 63, 65]:
        return "Rain / Drizzle 🌧️"
    elif code in [80, 81, 82]:
        return "Heavy Showers 🌧️"
    elif code in [95, 96, 99]:
        return "Thunderstorm 🌩️"
    return "Overcast ☁️"


def _generate_agri_advisory(temp: Any, rain_prob: int, windspeed: Any, humidity: Any) -> list:
    advisories = []
    
    try:
        r_prob = int(rain_prob)
    except (ValueError, TypeError):
        r_prob = 0

    try:
        w_speed = float(windspeed)
    except (ValueError, TypeError):
        w_speed = 0.0

    if r_prob > 60:
        advisories.append("🌧️ **High Rain Alert**: High chance of rainfall today. Postpone pesticide spraying and irrigation.")
        advisories.append("🌾 **Drainage**: Ensure proper field drainage to prevent waterlogging in low-lying crop roots.")
    else:
        advisories.append("☀️ **Irrigation**: Weather is suitable for routine field irrigation if required.")
        advisories.append("🧪 **Spraying Window**: Favorable weather conditions for foliar nutrient & pesticide application.")

    if w_speed > 20:
        advisories.append("💨 **High Winds**: Avoid pesticide spraying as wind velocity may cause spray drift.")
    
    return advisories
