import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import quantstats as qs
from dataclasses import dataclass
from .config import BacktestConfig
from .backtest_engine import BacktestResult


@dataclass
class Metrics:
    sharpe:    float
    ann_ret:   float
    ann_vol:   float
    total_ret: float
    max_dd:    float
    calmar:    float
    hit_rate:  float
    avg_leg:   float

    def print_summary(self, port: pd.Series):
        print("\n" + "=" * 52)
        print("  PERFORMANCE SUMMARY")
        print("=" * 52)
        print(f"  Backtest period : {port.index[0].date()} to {port.index[-1].date()}")
        print(f"  Days traded     : {len(port)}")
        print(f"  Avg leg size    : {self.avg_leg:.0f} tickers (each side)")
        print(f"  {'-'*45}")
        print(f"  Sharpe (ann.)   : {self.sharpe:+.3f}")
        print(f"  Ann. Return     : {self.ann_ret*100:+.2f}%")
        print(f"  Ann. Volatility : {self.ann_vol*100:.2f}%")
        print(f"  Total Return    : {self.total_ret*100:+.2f}%")
        print(f"  Max Drawdown    : {self.max_dd*100:.2f}%")
        print(f"  Calmar Ratio    : {self.calmar:.3f}")
        print(f"  Hit Rate        : {self.hit_rate*100:.1f}%")
        print("=" * 52)


class PerformanceReporter:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def compute_metrics(self, result: BacktestResult) -> Metrics:
        port = result.portfolio
        return Metrics(
            sharpe    = qs.stats.sharpe(port, periods=252),
            ann_ret   = qs.stats.cagr(port),
            ann_vol   = qs.stats.volatility(port, periods=252),
            total_ret = qs.stats.comp(port),
            max_dd    = qs.stats.max_drawdown(port),
            calmar    = qs.stats.calmar(port),
            hit_rate  = qs.stats.win_rate(port),
            avg_leg   = float(np.mean(result.leg_sizes)),
        )

    def plot(self, result: BacktestResult, metrics: Metrics):
        port     = result.portfolio
        cum_ret  = (1 + port).cumprod()
        drawdown = cum_ret / cum_ret.cummax() - 1

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(
            f"News Flow L/S  |  Sharpe={metrics.sharpe:.3f}  |  "
            f"Ann.Return={metrics.ann_ret*100:+.1f}%  |  MaxDD={metrics.max_dd*100:.1f}%",
            fontsize=12, fontweight="bold"
        )

        ax = axes[0, 0]
        cum_ret.plot(ax=ax, color="steelblue", linewidth=1.8)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        ax.fill_between(cum_ret.index, cum_ret, 1.0, where=cum_ret >= 1.0, alpha=0.15, color="steelblue")
        ax.fill_between(cum_ret.index, cum_ret, 1.0, where=cum_ret < 1.0,  alpha=0.15, color="red")
        ax.set_title("Cumulative L/S Return")
        ax.set_ylabel("Portfolio value ($1 start)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        ax = axes[0, 1]
        (drawdown * 100).plot(ax=ax, color="crimson", linewidth=1.2)
        ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color="crimson")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        ax = axes[1, 0]
        (port * 100).plot(ax=ax, color="steelblue", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
        port.rolling(10).std().mul(np.sqrt(252) * 100).plot(
            ax=ax, color="orange", linewidth=1.2, label="Rolling vol (10d ann, %)", alpha=0.9
        )
        ax.set_title("Daily L/S Returns")
        ax.set_ylabel("Return (%)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        ax = axes[1, 1]
        (port * 100).hist(ax=ax, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="grey", linestyle="--", linewidth=1.0)
        ax.axvline(port.mean() * 100, color="orange", linewidth=1.5,
                   label=f"Mean = {port.mean()*100:.3f}%")
        ax.set_title("Return Distribution")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out = self.config.report_dir / "performance_charts.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {out}")
        plt.show()

        # Portfolio count chart
        dates_idx   = pd.DatetimeIndex(result.dates_used)
        total_pos   = [l + s for l, s in zip(result.long_count, result.short_count)]
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(dates_idx, total_pos, color="steelblue", alpha=0.4, label="Total positions", width=0.8)
        ax.plot(dates_idx, result.long_count,  color="green",  linewidth=1.8, marker="o", markersize=4, label="Long")
        ax.plot(dates_idx, result.short_count, color="crimson", linewidth=1.8, marker="o", markersize=4, linestyle="--", label="Short")
        ax.plot(dates_idx, result.ranked_count, color="orange", linewidth=1.2, linestyle=":", label="Ranked")
        ax.axhline(np.mean(total_pos), color="grey", linestyle="--", linewidth=0.8,
                   label=f"Avg total = {np.mean(total_pos):.0f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Tickers")
        ax.set_title("Daily Portfolio Size (dollar neutral)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.tight_layout()
        out2 = self.config.report_dir / "portfolio_count_chart.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Count chart saved: {out2}")
        plt.show()

    def save(self, result: BacktestResult, metrics: Metrics):
        port = result.portfolio

        metrics_df = pd.DataFrame(
            list({
                "Backtest Period":  f"{port.index[0].date()} to {port.index[-1].date()}",
                "Days Traded":      len(port),
                "Avg Leg Size":     f"{metrics.avg_leg:.0f}",
                "Sharpe (ann.)":    f"{metrics.sharpe:+.3f}",
                "Ann. Return":      f"{metrics.ann_ret*100:+.2f}%",
                "Ann. Volatility":  f"{metrics.ann_vol*100:.2f}%",
                "Total Return":     f"{metrics.total_ret*100:+.2f}%",
                "Max Drawdown":     f"{metrics.max_dd*100:.2f}%",
                "Calmar Ratio":     f"{metrics.calmar:.3f}",
                "Hit Rate":         f"{metrics.hit_rate*100:.1f}%",
            }.items()),
            columns=["Metric", "Value"]
        )

        total_pos = [l + s for l, s in zip(result.long_count, result.short_count)]
        count_df = pd.DataFrame({
            "date":             [d.date() for d in result.dates_used],
            "ranked_universe":  result.ranked_count,
            "long_positions":   result.long_count,
            "short_positions":  result.short_count,
            "total_positions":  total_pos,
            "daily_return_pct": (port * 100).values,
        })

        port_ts = pd.DataFrame(result.portfolio_rows)

        with pd.ExcelWriter(self.config.report_dir / "performance_metrics.xlsx", engine="openpyxl") as w:
            metrics_df.to_excel(w, index=False, sheet_name="Metrics")
            count_df.to_excel(w,   index=False, sheet_name="Daily Portfolio Count")
        print(f"Metrics saved: {self.config.report_dir / 'performance_metrics.xlsx'}")

        port_ts.to_excel(self.config.report_dir / "portfolio_timeseries.xlsx", index=False,
                         sheet_name="Portfolio TimeSeries")
        print(f"Portfolio time series saved: {self.config.report_dir / 'portfolio_timeseries.xlsx'} ({len(port_ts):,} rows)")
