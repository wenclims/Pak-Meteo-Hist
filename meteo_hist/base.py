"""
Functions to create the plot.
"""

import datetime as dt
import os
import string
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from statsmodels.nonparametric.smoothers_lowess import lowess
from unidecode import unidecode

# Safe deep_update import for Pydantic v1/v2 compatibility
try:
    from pydantic.v1.utils import deep_update
except Exception:
    from pydantic.utils import deep_update


class MeteoHist:
    """
    Base class to prepare data and create a plot of a year's
    meteorological values compared to historical values.
    """

    def __init__(
        self,
        coords: tuple[float, float],
        year: int = None,
        reference_period: tuple[int, int] = (1961, 1990),
        metric: str = "temperature_mean",
        settings: dict = None,
    ):
        self.coords = (round(coords[0], 6), round(coords[1], 6))
        self.metric = metric
        self.settings = None
        self.update_settings(settings)
        self.year = year if year else dt.datetime.now().year
        self.reference_period = reference_period

        # Download and transform data
        self.data_raw = self.get_data(self.coords)
        self.data = self.transform_data(self.data_raw, self.year, reference_period)
        self.ref_nans = 0

    # ----------------------------------------------------
    # SETTINGS
    # ----------------------------------------------------
    def update_settings(self, settings: dict) -> None:
        default = {
            "font": {
                "family": "sans-serif",
                "font": "Lato",
                "default_size": 11,
                "axes.labelsize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            },
            "paths": {"output": "output"},
            "num_files_to_keep": 100,
            "highlight_max": 1,
            "highlight_min": 1,
            "peak_alpha": True,
            "peak_method": "mean",
            "peak_distance": 10,
            "smooth": {"apply": True, "frac": 1 / 12},
            "save_file": True,
            "location_name": None,
            "metric": self.get_metric_info(self.metric),
            "alternate_months": {
                "apply": True,
                "odd_color": "#fff",
                "odd_alpha": 0,
                "even_color": "#f8f8f8",
                "even_alpha": 0.3,
            },
            "fill_percentiles": "#f8f8f8",
            "system": "metric",
        }

        if isinstance(settings, dict):
            settings = {k: v for k, v in settings.items() if k in default}
            settings = deep_update(default, settings)
        else:
            settings = default

        if settings["location_name"] is None:
            settings["location_name"] = self.get_location(self.coords)

        old = self.settings
        self.settings = settings

        if isinstance(old, dict):
            if settings["system"] != old["system"]:
                self.data_raw = self.get_data()

            if (
                settings["system"] != old["system"]
                or settings["smooth"] != old["smooth"]
            ):
                self.data = self.transform_data(
                    self.data_raw, self.year, self.reference_period
                )

    # ----------------------------------------------------
    # HELPERS
    # ----------------------------------------------------
    def dayofyear_to_date(self, year: int, dayofyear: int, adj_leap=False):
        if adj_leap and isleap(year) and dayofyear > 59:
            dayofyear += 1
        return dt.datetime(year, 1, 1) + dt.timedelta(days=dayofyear - 1)

    # ----------------------------------------------------
    # DATA DOWNLOAD
    # ----------------------------------------------------
    def get_data(self, coords=None, metric=None, system=None, years=None):
        coords = coords or self.coords
        metric = metric or self.settings["metric"]["name"]
        system = system or self.settings["system"]
        years = years or (1940, dt.datetime.now().year)

        metric_data = self.get_metric_info(metric)["data"]

        date_start = f"{years[0]}-01-01"
        if years[1] == dt.datetime.now().year:
            date_end = (dt.datetime.now() - dt.timedelta(days=3)).strftime("%Y-%m-%d")
        else:
            date_end = f"{years[1]}-12-31"

        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coords[0]}&longitude={coords[1]}&"
            f"start_date={date_start}&end_date={date_end}&"
            f"daily={metric_data}&timezone=auto"
        )

        unit = self.get_units(metric_name=metric, system=system)
        unit_names = {"°C": "celsius", "°F": "fahrenheit", "mm": "mm", "in": "inch"}

        if "temperature" in metric:
            url += f"&temperature_unit={unit_names[unit]}"
        if "precipitation" in metric:
            url += f"&precipitation_unit={unit_names[unit]}"

        data = requests.get(url, timeout=30).json()

        df = pd.DataFrame({"date": data["daily"]["time"], "value": data["daily"][metric_data]})
        df["date"] = pd.to_datetime(df["date"])

        # Remove distorted final min/max temp
        if years[1] == dt.datetime.now().year and metric_data in [
            "temperature_2m_min",
            "temperature_2m_max",
        ]:
            idx = df[df["value"].notna()].index[-1]
            df.loc[idx, "value"] = np.nan

        return df

    # ----------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------
    def transform_data(self, df_raw: pd.DataFrame, year, ref_period):
        df = df_raw.copy()
        df["dayofyear"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year

        # Remove Feb 29
        df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()

        # Adjust DOY for leap years
        df["dayofyear"] = df["dayofyear"].where(
            ~((df["date"].dt.month > 2) & df["date"].dt.is_leap_year),
            df["dayofyear"] - 1,
        )

        if self.settings["metric"]["name"] == "precipitation_rolling":
            df["value"] = df["value"].rolling(30, min_periods=30).mean()

        if self.settings["metric"]["name"] == "precipitation_cum":
            df["value"] = df.groupby("year")["value"].cumsum()

        self.last_date = (
            df.dropna(subset=["value"])["date"].iloc[-1].strftime("%d %b %Y")
        )

        df_ref = df[df["date"].dt.year.between(*ref_period)].copy()
        self.ref_nans = (
            df_ref["value"].isna().sum() / len(df_ref) if len(df_ref) else 0
        )

        # Aggregate, then rename columns
        df_ref = (
            df_ref.groupby("dayofyear")["value"]
            .agg(["min", "mean", "max", lambda s: np.nanpercentile(s, 5), lambda s: np.nanpercentile(s, 95)])
            .reset_index()
        )
        df_ref.columns = ["dayofyear", "min", "mean", "max", "p05", "p95"]

        if self.settings["smooth"]["apply"]:
            for col in ["p05", "mean", "p95"]:
                sm = lowess(df_ref[col], df_ref["dayofyear"], frac=self.settings["smooth"]["frac"], is_sorted=True)
                df_ref[col] = sm[:, 1]

        # Add current year series
        df_ref[str(year)] = df[df["year"] == year]["value"].reset_index(drop=True)
        df_ref[f"{year}_diff"] = df_ref[str(year)] - df_ref["mean"]

        df_ref[f"{year}_alpha"] = df_ref.apply(
            lambda x: 1 if x[str(year)] > x["p95"] or x[str(year)] < x["p05"] else 0.6,
            axis=1,
        ).fillna(0)

        df_ref["date"] = df_ref["dayofyear"].apply(
            lambda x: self.dayofyear_to_date(year, x, True)
        )

        return df_ref

    # ----------------------------------------------------
    # ===== Remaining methods unchanged except small fixes =====
    # ----------------------------------------------------

    def get_y_limits(self):
        if self.settings["metric"]["data"] == "precipitation_sum":
            minimum = 0
        else:
            minimum = self.data[[f"{self.year}", "p05"]].min().min()
            minimum -= abs(minimum) * 0.05

        maximum = self.data[[f"{self.year}", "p95"]].max().max()
        maximum += abs(maximum) * 0.05

        if self.settings["metric"]["name"] == "precipitation_rolling":
            maximum += abs(maximum) * 0.2

        return minimum, maximum

    def get_min_max(self, period, which="max", metric="all"):
        if metric == "year":
            cols = [f"{self.year}"]
        elif metric in ["p05", "mean", "p95"]:
            cols = [metric]
        else:
            cols = ["p05", "mean", "p95", f"{self.year}"]

        df_t = self.data[self.data["dayofyear"].between(period[0], period[1])][cols]
        return df_t.min().min() if which == "min" else df_t.max().max()

    def get_metric_info(self, name="temperature_mean"):
        defaults = {
            "temperature_mean": {
                "name": "temperature_mean",
                "data": "temperature_2m_mean",
                "title": "Mean temperatures",
                "subtitle": "Compared to historical daily mean temperatures",
                "description": "Mean Temperature",
                "yaxis_label": "Temperature",
                "colors": {"cmap_above": "YlOrRd", "cmap_below": "YlGnBu"},
            },
            "temperature_min": {
                "name": "temperature_min",
                "data": "temperature_2m_min",
                "title": "Minimum temperatures",
                "subtitle": "Compared to historical daily minimum temperatures",
                "description": "Minimum Temperature",
                "yaxis_label": "Temperature",
                "colors": {"cmap_above": "YlOrRd", "cmap_below": "YlGnBu"},
            },
            "temperature_max": {
                "name": "temperature_max",
                "data": "temperature_2m_max",
                "title": "Maximum temperatures",
                "subtitle": "Compared to historical daily maximum temperatures",
                "description": "Maximum Temperature",
                "yaxis_label": "Temperature",
                "colors": {"cmap_above": "YlOrRd", "cmap_below": "YlGnBu"},
            },
            "precipitation_rolling": {
                "name": "precipitation_rolling",
                "data": "precipitation_sum",
                "title": "Precipitation",
                "subtitle": "30-day Rolling Average",
                "description": "30-day Rolling Precip",
                "yaxis_label": "Precipitation",
                "colors": {"cmap_above": "YlGnBu", "cmap_below": "YlOrRd"},
            },
            "precipitation_cum": {
                "name": "precipitation_cum",
                "data": "precipitation_sum",
                "title": "Precipitation",
                "subtitle": "Cumulative precipitation",
                "description": "Cumulative Precip",
                "yaxis_label": "Precipitation",
                "colors": {"cmap_above": "YlGnBu", "cmap_below": "YlOrRd"},
            },
        }
        return defaults[name]

    def get_units(self, metric_name=None, system=None):
        metric_name = metric_name or self.settings["metric"]["name"]
        system = system or self.settings["system"]

        units = {
            "temperature": {"metric": "°C", "imperial": "°F"},
            "precipitation": {"metric": "mm", "imperial": "in"},
        }

        m = "precipitation" if "precipitation" in metric_name else "temperature"
        return units[m].get(system, units[m]["metric"])

    def create_file_path(self, prefix=None, suffix=None):
        Path(self.settings["paths"]["output"]).mkdir(parents=True, exist_ok=True)

        elements = [
            prefix,
            self.settings["location_name"],
            self.settings["metric"]["name"],
            str(self.year),
            f"ref-{self.reference_period[0]}-{self.reference_period[1]}",
            suffix,
        ]

        elements = [e for e in elements if e]
        file_name = "-".join(elements)
        file_name = (
            unidecode(file_name)
            .lower()
            .replace(" ", "-")
            .replace("_", "-")
            .replace(".", "-")
        )

        valid = f"-_.(){string.ascii_letters}{string.digits}"
        file_name = "".join(c for c in file_name if c in valid)

        return f"{self.settings['paths']['output']}/{file_name}.png"

    def clean_output_dir(self, keep=None):
        keep = keep or self.settings["num_files_to_keep"]
        d = Path(self.settings["paths"]["output"])
        files = sorted(d.glob("*.png"), key=os.path.getctime, reverse=True)
        for f in files[keep:]:
            os.remove(f)

    @staticmethod
    def show_random(file_dir=None):
        dirs = [Path("examples"), Path("output")] if file_dir is None else [Path(file_dir)]
        files = []
        for d in dirs:
            files += list(d.glob("*.png"))
        return np.random.choice(files).as_posix() if files else None

    @staticmethod
    def get_location(coords, lang="en"):
        lat, lon = coords
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language={lang}&zoom=18"

        try:
            r = requests.get(url, timeout=30, headers=headers).json()
        except:
            return None

        if "error" in r:
            return None

        if "address" not in r:
            return r.get("display_name")

        addr = r["address"]
        keys = ["city", "town", "village", "hamlet", "suburb", "municipality", "district", "county", "state"]

        name = r.get("display_name", "")
        for k in keys:
            if k in addr:
                name = addr[k]
                break

        if "country" in addr:
            name += ", " + addr["country"]

        return name

    @staticmethod
    def get_lat_lon(query, lang="en"):
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&addressdetails=1&accept-language={lang}"
        res = requests.get(url, timeout=30, headers=headers).json()

        keys = ["city", "town", "village", "hamlet", "suburb", "municipality", "district", "county", "state"]
        types = ["city", "administrative", "town", "village"]

        out = []
        for rec in res:
            if rec["type"] in types:
                for k in keys:
                    if k in rec["address"]:
                        out.append(
                            {
                                "display_name": rec["display_name"],
                                "location_name": f"{rec['address'][k]}, {rec['address'].get('country', '')}",
                                "lat": float(rec["lat"]),
                                "lon": float(rec["lon"]),
                            }
                        )
                        break
        return out
