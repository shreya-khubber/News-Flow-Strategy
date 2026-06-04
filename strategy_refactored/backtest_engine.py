import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from .config import BacktestConfig
from .data_ingestion import PriceData
from .signal_builder import SignalData


@dataclass
class BacktestResult:
    portfolio:          pd.Series           # daily L/S returns indexed by signal_date
    dates_used:         list
    leg_sizes:          list
    ranked_count:       list
    long_count:         list
    short_count:        list
    long_leg_rets:      list
    short_leg_rets:     list
    long_tickers:       list
    short_tickers:      list
    ic_signal:          list
    ic_ret:             list
    portfolio_rows:     list                # per-ticker per-day detail
    signal_to_ret:      dict               # signal_date -> trade_date mapping


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def _first_td_after(date, sorted_td_arr: np.ndarray) -> pd.Timestamp:
        """First trading day strictly after date (binary search, handles non-trading dates)."""
        idx = np.searchsorted(sorted_td_arr, np.datetime64(date), side="right")
        return pd.Timestamp(sorted_td_arr[idx]) if idx < len(sorted_td_arr) else None

    def _build_forward_returns(self, signal_data: SignalData, prices: PriceData) -> dict:
        """Map every signal_date to its forward open-to-close return Series."""
        same_day_ret = (prices.close_wide / prices.open_wide - 1).reindex(columns=signal_data.universe)
        common_dates = signal_data.signal.index

        signal_to_ret = {
            d: self._first_td_after(d, prices.dates_arr)
            for d in common_dates
        }
        signal_to_ret = {
            d: r for d, r in signal_to_ret.items()
            if r is not None and r in same_day_ret.index
        }

        valid_dates = sorted(signal_to_ret.keys())
        fwd_ret = pd.DataFrame(
            {d: same_day_ret.loc[signal_to_ret[d]] for d in valid_dates}
        ).T
        fwd_ret.index = pd.DatetimeIndex(valid_dates)
        fwd_ret = fwd_ret.sort_index()

        # Hard assertion: no look-ahead
        for d in valid_dates:
            ret_date = signal_to_ret[d]
            assert ret_date > d, f"LOOK-AHEAD: return {ret_date} not after signal {d}"
            assert ret_date == self._first_td_after(d, prices.dates_arr), \
                f"ALIGNMENT: {ret_date} is not first trading day after {d}"

        gaps = [(signal_to_ret[d] - d).days for d in valid_dates]
        print(f"Look-ahead assertion PASSED. Calendar gap min={min(gaps)}d max={max(gaps)}d")

        return fwd_ret, signal_to_ret

    def run(self, signal_data: SignalData, prices: PriceData) -> BacktestResult:
        fwd_ret, signal_to_ret = self._build_forward_returns(signal_data, prices)

        port_rets, dates_used, leg_sizes       = [], [], []
        ranked_count, long_count, short_count  = [], [], []
        long_leg_rets, short_leg_rets          = [], []
        long_tickers, short_tickers            = [], []
        ic_signal, ic_ret                      = [], []
        portfolio_rows                         = []

        for day in pd.DatetimeIndex(prices.dates_arr):
            if day not in signal_data.signal.index or day not in fwd_ret.index:
                continue

            sig = signal_data.signal.loc[day].dropna()
            ret = fwd_ret.loc[day].dropna()
            sig = sig[signal_data.zscore_u.loc[day].reindex(sig.index).notna()]
            sig = sig[sig != 0]

            valid = sig.index.intersection(ret.index)
            if len(valid) < self.config.min_universe:
                continue

            sig, ret = sig[valid], ret[valid]
            n = len(sig)
            q = int(n * self.config.quintile)

            ranked = sig.rank(method="first")
            longs  = ranked > (n - q)
            shorts = ranked <= q

            trade_date = signal_to_ret[day]

            for ticker in sig.index[longs | shorts]:
                portfolio_rows.append({
                    "signal_date":  day.date(),
                    "trade_date":   trade_date.date(),
                    "ticker":       ticker,
                    "position":     "Long" if longs[ticker] else "Short",
                    "zscore":       round(signal_data.zscore_u.loc[day, ticker], 4),
                    "direction":    round(signal_data.direction_u.loc[day, ticker], 6),
                    "signal":       round(sig[ticker], 4),
                    "rank":         int(ranked[ticker]),
                    "return_pct":   round(ret[ticker] * 100, 4),
                })

            port_rets.append(ret[longs].mean() - ret[shorts].mean())
            dates_used.append(day)
            leg_sizes.append(q)
            ranked_count.append(n)
            long_count.append(int(longs.sum()))
            short_count.append(int(shorts.sum()))
            long_leg_rets.append(ret[longs].mean())
            short_leg_rets.append(ret[shorts].mean())
            long_tickers.append(list(sig.index[longs]))
            short_tickers.append(list(sig.index[shorts]))
            ic_signal.append(sig.copy())
            ic_ret.append(ret.copy())

            print(f"  {day.date()}  ranked={n:>4}  long={longs.sum():>3}  short={shorts.sum():>3}")

        portfolio = pd.Series(port_rets, index=pd.DatetimeIndex(dates_used)).dropna()

        return BacktestResult(
            portfolio=portfolio,
            dates_used=dates_used,
            leg_sizes=leg_sizes,
            ranked_count=ranked_count,
            long_count=long_count,
            short_count=short_count,
            long_leg_rets=long_leg_rets,
            short_leg_rets=short_leg_rets,
            long_tickers=long_tickers,
            short_tickers=short_tickers,
            ic_signal=ic_signal,
            ic_ret=ic_ret,
            portfolio_rows=portfolio_rows,
            signal_to_ret=signal_to_ret,
        )
