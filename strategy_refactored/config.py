from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BacktestConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir:   Path = field(default_factory=lambda: Path(r"C:\Users\SHREYA\Desktop\Infer Edge Assignment\Take Home Data"))
    report_dir: Path = field(default_factory=lambda: Path(r"C:\Users\SHREYA\Desktop\Infer Edge Assignment Final\Backtest_Report"))

    # ── News timing ───────────────────────────────────────────────────────────
    news_cutoff_hour: int = 16       # articles at/after this hour ET roll to next calendar day

    # ── Signal construction ───────────────────────────────────────────────────
    lookback:     int   = 10         # rolling window for z-score normalisation
    min_periods:  int   = 5          # min observations before z-score is trusted
    signal_clip:  float = 3.0        # z-score winsorisation cap
    mid_lookback: int   = 5          # days for midprice momentum (one trading week)

    # ── Portfolio construction ────────────────────────────────────────────────
    quintile:     float = 0.20       # top/bottom fraction selected long/short
    min_universe: int   = 40         # min tickers with valid signal to trade a day

    def __post_init__(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)
