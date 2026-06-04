import warnings
warnings.filterwarnings("ignore")

from strategy_refactored import (
    BacktestConfig,
    DataLoader,
    SignalBuilder,
    BacktestEngine,
    PerformanceReporter,
    Diagnostics,
)

# ── Configure ─────────────────────────────────────────────────────────────────
# To hypertune: change any parameter here and re-run.
# To run the pre-market-only variant: set market_open_hour=9, market_open_minute=30
from pathlib import Path

config = BacktestConfig(
    report_dir        = Path(r"C:\Users\SHREYA\Desktop\Infer Edge Assignment Final\Backtest_Report_Temp"),
    lookback          = 10,
    min_periods       = 5,
    signal_clip       = 3.0,
    quintile          = 0.20,
    min_universe      = 40,
    mid_lookback      = 5,
    news_cutoff_hour  = 16,
    market_open_hour  = None,   # None = use all news (original strategy)
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("Loading data...")
loader  = DataLoader(config)
prices  = loader.load_prices()
news    = loader.load_news()

print("Building signal...")
builder = SignalBuilder(config)
signal  = builder.build(news, prices)
print(f"Universe: {len(signal.universe)} tickers")

print("Running backtest...")
engine  = BacktestEngine(config)
result  = engine.run(signal, prices)

# ── Performance ───────────────────────────────────────────────────────────────
reporter = PerformanceReporter(config)
metrics  = reporter.compute_metrics(result)
metrics.print_summary(result.portfolio)
reporter.plot(result, metrics)
reporter.save(result, metrics)

# ── Diagnostics ───────────────────────────────────────────────────────────────
diag = Diagnostics(config)
diag.run_and_save(
    result     = result,
    metrics    = metrics,
    prices_raw = prices.raw,
    universe   = signal.universe,
    close_wide = prices.close_wide,
)

print(f"\nAll outputs saved to: {config.report_dir}")
