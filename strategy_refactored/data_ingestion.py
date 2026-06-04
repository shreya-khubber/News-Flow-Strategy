import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import BacktestConfig


@dataclass
class PriceData:
    raw:        pd.DataFrame   # original prices df
    open_wide:  pd.DataFrame   # date x ticker
    close_wide: pd.DataFrame   # date x ticker
    dates_arr:  np.ndarray     # sorted unique trading day timestamps


@dataclass
class NewsData:
    chunks: pd.DataFrame       # one row per chunk, with signal_date assigned


class DataLoader:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def load_prices(self) -> PriceData:
        prices = pd.read_csv(
            self.config.data_dir / "prices.csv",
            parse_dates=["date"]
        )
        dates_arr   = prices["date"].drop_duplicates().sort_values().values
        open_wide   = prices.pivot(index="date", columns="ticker", values="open").sort_index()
        close_wide  = prices.pivot(index="date", columns="ticker", values="close").sort_index()
        return PriceData(
            raw=prices,
            open_wide=open_wide,
            close_wide=close_wide,
            dates_arr=dates_arr,
        )

    def load_news(self) -> NewsData:
        chunks = pd.read_csv(self.config.data_dir / "news_chunks.csv")

        # Parse timestamps and convert to ET
        chunks["ts"]    = pd.to_datetime(chunks["ts"], format="mixed", utc=True)
        chunks["ts_et"] = chunks["ts"].dt.tz_convert("America/New_York")

        # Assign signal_date: articles at/after cutoff roll to next calendar day
        chunks["signal_date"] = chunks["ts_et"].dt.normalize().dt.tz_localize(None)
        after_close = chunks["ts_et"].dt.hour >= self.config.news_cutoff_hour
        chunks.loc[after_close, "signal_date"] += pd.Timedelta(days=1)

        chunks["tickers"] = chunks["tickers"].apply(json.loads)
        return NewsData(chunks=chunks)
