"""Weather module — solar geometry, hourly disaggregation, CO2."""

from dssat.weather.weather import WeatherModule
from dssat.weather.solar import day_length, solar_params, generate_hourly

__all__ = ["WeatherModule", "day_length", "solar_params", "generate_hourly"]
