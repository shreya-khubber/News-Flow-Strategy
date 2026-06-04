import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import BacktestConfig
from .data_ingestion import NewsData, PriceData


@dataclass
class SignalData:
    signal:     pd.DataFrame   # date x ticker combined signal
    zscore_u:   pd.DataFrame   # date x ticker z-score (universe-aligned)
    direction_u: pd.DataFrame  # date x ticker momentum direction (universe-aligned)
    universe:   list           # tickers common to both news and price data


class SignalBuilder:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def build_coverage(self, news: NewsData) -> pd.DataFrame:
        """Weighted daily coverage per (signal_date, ticker)."""
        ct = news.chunks[news.chunks["tickers"].map(len) > 0].copy()
        ct = ct.explode("tickers").rename(columns={"tickers": "ticker"})

        total_chunks = (
            ct.groupby("article_id")["chunk_index"]
            .nunique()
            .rename("total_chunks")
        )
        art = (
            ct.groupby(["article_id", "story_id", "signal_date", "ticker"])["chunk_index"]
            .nunique()
            .rename("chunks_mentioning")
            .reset_index()
        )
        art = art.merge(total_chunks, on="article_id")
        # depth-weighted: fraction of article chunks mentioning this ticker
        art["ticker_weight"] = art["chunks_mentioning"] / art["total_chunks"]

        # Novelty: first day a story appears gets full weight, repeats get 0.5
        story_days = (
            art.drop_duplicates(["story_id", "signal_date"])[["story_id", "signal_date"]]
            .sort_values(["story_id", "signal_date"])
            .copy()
        )
        story_days["day_count"] = story_days.groupby("story_id").cumcount() + 1
        art = art.merge(story_days[["story_id", "signal_date", "day_count"]], on=["story_id", "signal_date"])
        art["novelty_weight"] = 0.5 + 0.5 * (art["day_count"] == 1).astype(float)
        art["w"] = art["ticker_weight"] * art["novelty_weight"]

        daily = (
            art.groupby(["signal_date", "ticker"])
            .agg(weighted_cov=("w", "sum"))
            .reset_index()
        )
        return daily

    def build_zscore(self, daily_coverage: pd.DataFrame) -> pd.DataFrame:
        """Rolling cross-sectional z-score of coverage."""
        cov_wide = (
            daily_coverage
            .pivot(index="signal_date", columns="ticker", values="weighted_cov")
            .sort_index()
            .fillna(0.0)
        )
        roll_mean = cov_wide.rolling(self.config.lookback, min_periods=self.config.min_periods).mean()
        roll_std  = cov_wide.rolling(self.config.lookback, min_periods=self.config.min_periods).std()
        zscore = (cov_wide - roll_mean) / (roll_std + 1e-8)
        zscore = zscore.clip(-self.config.signal_clip, self.config.signal_clip)
        return zscore

    def build_direction(self, prices: PriceData) -> pd.DataFrame:
        """5-day midprice momentum direction."""
        midprice  = (prices.open_wide + prices.close_wide) / 2
        direction = (midprice / midprice.rolling(self.config.mid_lookback, min_periods=2).mean()) - 1
        return direction

    def build(self, news: NewsData, prices: PriceData) -> SignalData:
        """Full signal pipeline: coverage -> z-score x direction."""
        daily_coverage = self.build_coverage(news)
        zscore         = self.build_zscore(daily_coverage)
        direction      = self.build_direction(prices)

        universe     = sorted(set(prices.close_wide.columns) & set(zscore.columns))
        common_dates = zscore.index.intersection(direction.index)

        zscore_u    = zscore.loc[common_dates, universe]
        direction_u = direction.loc[common_dates, universe]
        signal      = np.sign(direction_u) * zscore_u

        return SignalData(
            signal=signal,
            zscore_u=zscore_u,
            direction_u=direction_u,
            universe=universe,
        )
