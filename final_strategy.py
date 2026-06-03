# News Flow Long/Short Signal -- Final Strategy
# Signal: sign(5-day midprice momentum) x news coverage z-score
# Timing: signal on day D -> trade D+1 open-to-close (look-ahead-free)

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import quantstats as qs
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(r"C:\Users\SHREYA\Desktop\Infer Edge Assignment\Take Home Data")
REPORT_DIR   = Path(r"C:\Users\SHREYA\Desktop\Infer Edge Assignment\Backtest_Report")
REPORT_DIR.mkdir(exist_ok=True)
LOOKBACK     = 10    # rolling window for z-score
MIN_PERIODS  = 5     # min days before z-score is trusted
SIGNAL_CLIP  = 3.0   # z-score cap
QUINTILE     = 0.20  # top/bottom 20% long/short
MIN_UNIVERSE = 40    # min tickers with signal required to trade a day
MID_LOOKBACK = 5     # one trading week 


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
chunks = pd.read_csv(DATA_DIR / "news_chunks.csv")
prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])

price_dates_arr = prices["date"].drop_duplicates().sort_values().values


# ── 2. Parse timestamps -> signal dates ──────────────────────────────────────
# News after 4 PM ET is pushed to next calendar day so it is never used same-day
chunks["ts"]    = pd.to_datetime(chunks["ts"], format="mixed", utc=True)
chunks["ts_et"] = chunks["ts"].dt.tz_convert("America/New_York")
chunks["signal_date"] = chunks["ts_et"].dt.normalize().dt.tz_localize(None)
after_close = chunks["ts_et"].dt.hour >= 16
chunks.loc[after_close, "signal_date"] += pd.Timedelta(days=1)
chunks["tickers"] = chunks["tickers"].apply(json.loads)


# ── 3. Chunk-frequency weight per (article, ticker) ──────────────────────────
ct = chunks[chunks["tickers"].map(len) > 0].copy()
ct = ct.explode("tickers").rename(columns={"tickers": "ticker"})
total_chunks = ct.groupby("article_id")["chunk_index"].nunique().rename("total_chunks")
art = (
    ct.groupby(["article_id", "story_id", "signal_date", "ticker"])["chunk_index"]
    .nunique().rename("chunks_mentioning").reset_index()
)
art = art.merge(total_chunks, on="article_id")
# fraction of article chunks mentioning this ticker -- rewards tickers discussed in depth,
# not just tagged once. 1/n would treat a 1-chunk mention equally to a 10-chunk deep-dive.
art["ticker_weight"] = art["chunks_mentioning"] / art["total_chunks"]


# ── 4. Novelty weight (no look-ahead: cumcount on sorted dates) ───────────────
# day_count=1 -> first day story appears -> novelty_weight=1.0
# day_count>1 -> repeat coverage -> novelty_weight=0.5
story_days = (
    art.drop_duplicates(["story_id", "signal_date"])
    [["story_id", "signal_date"]]
    .sort_values(["story_id", "signal_date"])
    .copy()
)
story_days["day_count"] = story_days.groupby("story_id").cumcount() + 1
art = art.merge(story_days[["story_id", "signal_date", "day_count"]],
                on=["story_id", "signal_date"])
art["novelty_weight"] = 0.5 + 0.5 * (art["day_count"] == 1).astype(float)
art["w"]              = art["ticker_weight"] * art["novelty_weight"]


# ── 5. Daily aggregation ──────────────────────────────────────────────────────
daily = (
    art.groupby(["signal_date", "ticker"])
    .agg(weighted_cov=("w", "sum"))
    .reset_index()
)


# ── 6. Rolling z-score (normalises structural volume differences across tickers)
cov_wide  = (
    daily.pivot(index="signal_date", columns="ticker", values="weighted_cov")
    .sort_index().fillna(0.0)
)
roll_mean = cov_wide.rolling(LOOKBACK, min_periods=MIN_PERIODS).mean()
roll_std  = cov_wide.rolling(LOOKBACK, min_periods=MIN_PERIODS).std()
zscore    = (cov_wide - roll_mean) / (roll_std + 1e-8)
zscore    = zscore.clip(-SIGNAL_CLIP, SIGNAL_CLIP)


# ── 7. Direction: 5-day midprice momentum ────────────────────────────────────
open_wide  = prices.pivot(index="date", columns="ticker", values="open").sort_index()
close_wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
midprice   = (open_wide + close_wide) / 2
direction  = (midprice / midprice.rolling(MID_LOOKBACK, min_periods=2).mean()) - 1


# ── 8. Signal: sign(direction) x zscore ──────────────────────────────────────
# direction picks the side (long/short); zscore provides conviction magnitude.
# This separates the two roles cleanly and avoids scale mismatch between
# raw % direction and bounded z-score.
universe     = list(set(close_wide.columns) & set(zscore.columns))
common_dates = zscore.index.intersection(direction.index)
zscore_u     = zscore.loc[common_dates, universe]
direction_u  = direction.loc[common_dates, universe]
signal       = np.sign(direction_u) * zscore_u


# ── 9. Forward returns (explicit, look-ahead-free) ───────────────────────────
# fwd_ret[D] = open-to-close return of the FIRST trading day strictly after D.
# signal_date can be a non-trading day (e.g. Friday after-4PM rolls to Saturday);
# mapping to the next real session is correct -- news is known before that open.
def first_td_after(date, sorted_td_arr):
    # binary search for first trading day strictly after date -- works for non-trading dates too
    idx = np.searchsorted(sorted_td_arr, np.datetime64(date), side='right')  # side='right' skips date itself
    return pd.Timestamp(sorted_td_arr[idx]) if idx < len(sorted_td_arr) else None  # None if past last trading day

same_day_ret   = (close_wide / open_wide - 1).reindex(columns=universe)  # intraday return for every ticker on every date
signal_to_ret  = {d: first_td_after(d, price_dates_arr)
                  for d in common_dates}  # map each signal date -> next real trading session
signal_to_ret  = {d: r for d, r in signal_to_ret.items()
                  if r is not None and r in same_day_ret.index}  # drop signal dates with no tradeable next session

valid_signal_dates = sorted(signal_to_ret.keys())
fwd_ret = pd.DataFrame(
    {d: same_day_ret.loc[signal_to_ret[d]] for d in valid_signal_dates}
).T
fwd_ret.index = pd.DatetimeIndex(valid_signal_dates)
fwd_ret = fwd_ret.sort_index()

# Look-ahead assertion: return date must be the first trading day strictly after signal date
for d in valid_signal_dates:
    ret_date = signal_to_ret[d]
    assert ret_date > d, f"LOOK-AHEAD: return {ret_date} not after signal {d}"
    assert ret_date == first_td_after(d, price_dates_arr), \
        f"ALIGNMENT: {ret_date} is not the first trading day after {d}"
gaps = [(signal_to_ret[d] - d).days for d in valid_signal_dates]
print(f"Look-ahead assertion PASSED. Calendar gap min={min(gaps)}d max={max(gaps)}d "
      f"(>1 expected for weekend/holiday rolls).")


# ── 10. Backtest ──────────────────────────────────────────────────────────────
print(f"\nUniverse: {len(universe)} tickers")

port_rets, dates_used, leg_sizes = [], [], []
ranked_count, long_count, short_count = [], [], []
portfolio_rows = []  # per-ticker rows for portfolio time series export
# extended diagnostics accumulators
long_leg_rets, short_leg_rets = [], []   # raw per-leg daily returns
long_tickers_by_day, short_tickers_by_day = [], []  # tickers held each day
ic_signal_by_day, ic_ret_by_day = [], []  # full valid cross-sections for IC

for day in pd.DatetimeIndex(price_dates_arr):
    if day not in signal.index or day not in fwd_ret.index:
        continue

    sig = signal.loc[day].dropna()
    ret = fwd_ret.loc[day].dropna()
    sig = sig[zscore_u.loc[day].reindex(sig.index).notna()]
    sig = sig[sig != 0]

    valid = sig.index.intersection(ret.index)
    # Below MIN_UNIVERSE the quintile degenerates into a handful of concentrated
    # positions where idiosyncratic noise drowns the signal.
    if len(valid) < MIN_UNIVERSE:
        continue

    sig, ret = sig[valid], ret[valid]
    n = len(sig)
    q = int(n * QUINTILE)

    ranked = sig.rank(method="first")
    longs  = ranked > (n - q)
    shorts = ranked <= q

    trade_date = signal_to_ret[day]  # D+1 -- actual day positions were held

    # Collect per-ticker data for every position (long and short)
    for ticker in sig.index[longs | shorts]:
        portfolio_rows.append({
            "signal_date"   : day.date(),
            "trade_date"    : trade_date.date(),
            "ticker"        : ticker,
            "position"      : "Long" if longs[ticker] else "Short",
            "zscore"        : round(zscore_u.loc[day, ticker], 4),
            "direction"     : round(direction_u.loc[day, ticker], 6),
            "signal"        : round(sig[ticker], 4),
            "rank"          : int(ranked[ticker]),
            "return_pct"    : round(ret[ticker] * 100, 4),
        })

    port_rets.append(ret[longs].mean() - ret[shorts].mean())
    dates_used.append(day)
    leg_sizes.append(q)
    ranked_count.append(n)
    long_count.append(int(longs.sum()))
    short_count.append(int(shorts.sum()))
    # extended diagnostics -- 4 new lines, no effect on port
    long_leg_rets.append(ret[longs].mean())
    short_leg_rets.append(ret[shorts].mean())
    long_tickers_by_day.append(list(sig.index[longs]))
    short_tickers_by_day.append(list(sig.index[shorts]))
    ic_signal_by_day.append(sig.copy())   # full valid cross-section for IC
    ic_ret_by_day.append(ret.copy())      # aligned to same tickers as sig
    print(f"  {day.date()}  ranked={n:>4}  long={longs.sum():>3}  short={shorts.sum():>3}")

port = pd.Series(port_rets, index=pd.DatetimeIndex(dates_used)).dropna()


# ── 11. Performance metrics (quantstats) ─────────────────────────────────────
cum_ret  = (1 + port).cumprod()
drawdown = cum_ret / cum_ret.cummax() - 1
avg_leg  = np.mean(leg_sizes)

sharpe    = qs.stats.sharpe(port, periods=252)
ann_ret   = qs.stats.cagr(port)
ann_vol   = qs.stats.volatility(port, periods=252)
total_ret = qs.stats.comp(port)
max_dd    = qs.stats.max_drawdown(port)
calmar    = qs.stats.calmar(port)
hit_rate  = qs.stats.win_rate(port)

print("\n" + "=" * 52)
print("  PERFORMANCE SUMMARY  (Final Strategy)")
print("=" * 52)
print(f"  Backtest period : {port.index[0].date()} to {port.index[-1].date()}")
print(f"  Days traded     : {len(port)}")
print(f"  Avg leg size    : {avg_leg:.0f} tickers (each side)")
print(f"  {'-'*45}")
print(f"  Sharpe (ann.)   : {sharpe:+.3f}")
print(f"  Ann. Return     : {ann_ret*100:+.2f}%")
print(f"  Ann. Volatility : {ann_vol*100:.2f}%")
print(f"  Total Return    : {total_ret*100:+.2f}%")
print(f"  Max Drawdown    : {max_dd*100:.2f}%")
print(f"  Calmar Ratio    : {calmar:.3f}")
print(f"  Hit Rate        : {hit_rate*100:.1f}%")
print("=" * 52)


# ── 11b. Extended Diagnostics ────────────────────────────────────────────────
print("\n" + "=" * 52)
print("  EXTENDED DIAGNOSTICS")
print("=" * 52)

long_port  = pd.Series(long_leg_rets,  index=pd.DatetimeIndex(dates_used))
short_port = pd.Series(short_leg_rets, index=pd.DatetimeIndex(dates_used))

# --- Metric 1: Long-leg vs Short-leg attribution ------------------------------
# Convention: long_port = raw avg return of long tickers (positive = gained)
#             short_port = raw avg return of short tickers (positive = they gained, hurts us)
#             port = long_port - short_port  (matches the loop exactly)
try:
    print("\n[1] Long / Short leg attribution")
    for label, leg in [("Long leg (raw)", long_port), ("Short leg (raw, negative contribution = good)", short_port)]:
        comp   = qs.stats.comp(leg)
        sharpe_leg = qs.stats.sharpe(leg, periods=252)
        mean_d = leg.mean()
        print(f"    {label}")
        print(f"      Total compounded return : {comp*100:+.2f}%")
        print(f"      Annualized Sharpe       : {sharpe_leg:+.3f}")
        print(f"      Mean daily return       : {mean_d*100:+.4f}%")
    print(f"    Edge: long_mean - short_mean = {(long_port.mean()-short_port.mean())*100:+.4f}% per day")
except Exception as e:
    print(f"    [Metric 1 failed: {e}]")

# --- Metric 2: Sharpe statistical significance --------------------------------
try:
    print("\n[2] Sharpe statistical significance")
    N = len(port)
    t_stat, p_val = scipy_stats.ttest_1samp(port, 0)
    # Lo (2002) approximation: SE of annualized Sharpe under i.i.d. assumption
    SE_SR = np.sqrt((1 + 0.5 * sharpe**2) / N)
    print(f"    N (trading days)            : {N}")
    print(f"    t-stat (mean != 0)          : {t_stat:.3f}  (p={p_val:.3f})")
    print(f"    Annualized Sharpe +/- SE    : {sharpe:+.3f} +/- {SE_SR:.3f}  (Lo 2002 approximation)")
    print(f"    95% CI approx               : [{sharpe-1.96*SE_SR:+.3f}, {sharpe+1.96*SE_SR:+.3f}]")
    print(f"    CAVEAT: With N={N} observations the Sharpe estimate has wide confidence intervals;")
    print(f"            it is not statistically distinguishable from a much lower true value.")
except Exception as e:
    print(f"    [Metric 2 failed: {e}]")

# --- Metric 3: Information Coefficient (IC) -----------------------------------
# IC = Spearman rank correlation between signal and next-day return, computed
# cross-sectionally each day. Measures predictive power independently of
# quintile bucketing -- a positive mean IC means signal rank predicts return rank.
try:
    print("\n[3] Information Coefficient (IC)")
    ic_values = []
    for i in range(len(dates_used)):
        s = ic_signal_by_day[i]
        r = ic_ret_by_day[i]
        # ic_signal_by_day and ic_ret_by_day were stored from the same valid set
        # inside the loop, so they are already aligned on identical tickers and date.
        common = s.index.intersection(r.index)
        s, r = s.loc[common].dropna(), r.loc[common].dropna()
        common2 = s.index.intersection(r.index)
        if len(common2) < 10:
            continue
        corr, _ = scipy_stats.spearmanr(s.loc[common2].values, r.loc[common2].values)
        ic_values.append(corr)
    ic_series = pd.Series(ic_values)
    ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan
    print(f"    Days with IC computed       : {len(ic_series)}")
    print(f"    Mean IC                     : {ic_series.mean():+.4f}")
    print(f"    Std IC                      : {ic_series.std():.4f}")
    print(f"    IC Information Ratio (IR)   : {ir:+.3f}")
    print(f"    Fraction of days IC > 0     : {(ic_series > 0).mean()*100:.1f}%")
except Exception as e:
    print(f"    [Metric 3 failed: {e}]")

# --- Metric 4: Return concentration ------------------------------------------
try:
    print("\n[4] Return concentration")
    total_pnl = port.sum()
    top5_sum  = port.nlargest(5).sum()
    if abs(total_pnl) > 1e-8:
        top5_frac = top5_sum / total_pnl * 100
        print(f"    Top 5 days sum              : {top5_sum*100:+.3f}%")
        print(f"    Total daily P&L sum         : {total_pnl*100:+.3f}%")
        print(f"    Top 5 days = {top5_frac:.1f}% of total daily P&L")
    else:
        print(f"    Top 5 days sum              : {top5_sum*100:+.3f}%  (total near zero -- reporting raw)")
    print(f"    Top 5 dates: {list(port.nlargest(5).index.date)}")
except Exception as e:
    print(f"    [Metric 4 failed: {e}]")

# --- Metric 5: Lag-1 autocorrelation -----------------------------------------
try:
    print("\n[5] Return autocorrelation (lag-1)")
    ac1 = port.autocorr(lag=1)
    print(f"    Lag-1 autocorrelation       : {ac1:+.4f}")
    if ac1 > 0.05:
        print(f"    Note: positive autocorrelation -- sqrt(252) annualization OVERSTATES true Sharpe")
    elif ac1 < -0.05:
        print(f"    Note: negative autocorrelation -- sqrt(252) annualization UNDERSTATES true Sharpe")
    else:
        print(f"    Note: autocorrelation near zero -- sqrt(252) annualization is approximately valid")
except Exception as e:
    print(f"    [Metric 5 failed: {e}]")

# --- Metric 6: Liquidity profile (volume-based) ------------------------------
try:
    print("\n[6] Liquidity profile (dollar volume proxy)")
    print("    Using dollar volume = volume x close price (close available in prices)")
    print("    CAVEAT: Borrow cost / shortability NOT assessed -- requires securities-lending")
    print("            data not available here; this is a tradability proxy from volume only.")
    vol_wide = prices.pivot(index="date", columns="ticker", values="volume").reindex(columns=universe)
    dv_wide  = vol_wide * close_wide.reindex(columns=universe)   # dollar volume

    leg_dvols = []
    bottom_q_flags = []
    for i, day in enumerate(pd.DatetimeIndex(dates_used)):
        if day not in dv_wide.index:
            continue
        day_dv = dv_wide.loc[day].dropna()
        univ_q25 = day_dv.quantile(0.25)  # bottom quartile threshold for that day's universe
        held = long_tickers_by_day[i] + short_tickers_by_day[i]
        for t in held:
            if t in day_dv.index:
                dv = day_dv[t]
                leg_dvols.append(dv)
                bottom_q_flags.append(1 if dv <= univ_q25 else 0)

    leg_dvols = pd.Series(leg_dvols)
    print(f"    Median daily dollar volume (leg names) : ${leg_dvols.median():,.0f}")
    print(f"    25th pct daily dollar volume           : ${leg_dvols.quantile(0.25):,.0f}")
    frac_bottom = np.mean(bottom_q_flags) * 100
    print(f"    Fraction of leg-days in bottom quartile: {frac_bottom:.1f}%")
except Exception as e:
    print(f"    [Metric 6 failed: {e}]")

print("\n" + "=" * 52)

# ── Save extended diagnostics to Backtest_Report ─────────────────────────────
try:
    # Metric 1: leg attribution
    leg_attr = pd.DataFrame({
        "Metric": [
            "Long leg -- total compounded return",
            "Long leg -- annualized Sharpe",
            "Long leg -- mean daily return",
            "Short leg -- total compounded return (raw)",
            "Short leg -- annualized Sharpe (raw)",
            "Short leg -- mean daily return (raw)",
            "Edge (long mean - short mean, per day)",
        ],
        "Value": [
            f"{qs.stats.comp(long_port)*100:+.2f}%",
            f"{qs.stats.sharpe(long_port, periods=252):+.3f}",
            f"{long_port.mean()*100:+.4f}%",
            f"{qs.stats.comp(short_port)*100:+.2f}%",
            f"{qs.stats.sharpe(short_port, periods=252):+.3f}",
            f"{short_port.mean()*100:+.4f}%",
            f"{(long_port.mean()-short_port.mean())*100:+.4f}%",
        ],
        "Note": [
            "", "", "",
            "positive = short leg stocks went up (hurts strategy)",
            "", "",
            "matches daily port return on average",
        ]
    })

    # Metric 2: significance
    N = len(port)
    t_stat, p_val = scipy_stats.ttest_1samp(port, 0)
    SE_SR = np.sqrt((1 + 0.5 * sharpe**2) / N)
    sig_df = pd.DataFrame({
        "Metric": ["N (trading days)", "t-statistic", "p-value", "Annualized Sharpe",
                   "SE of Sharpe (Lo 2002)", "95% CI lower", "95% CI upper", "Caveat"],
        "Value": [N, f"{t_stat:.3f}", f"{p_val:.3f}", f"{sharpe:+.3f}",
                  f"{SE_SR:.3f}", f"{sharpe-1.96*SE_SR:+.3f}", f"{sharpe+1.96*SE_SR:+.3f}",
                  f"N={N} is too small for statistical significance at conventional confidence levels"]
    })

    # Metric 3: IC
    ic_df = pd.DataFrame({
        "Metric": ["Days with IC", "Mean IC", "Std IC", "IC IR (mean/std)", "Fraction IC > 0"],
        "Value": [len(ic_series), f"{ic_series.mean():+.4f}", f"{ic_series.std():.4f}",
                  f"{ir:+.3f}", f"{(ic_series > 0).mean()*100:.1f}%"]
    })
    ic_daily = pd.DataFrame({
        "trade_date": [d.date() for d in pd.DatetimeIndex(dates_used)[:len(ic_series)]],
        "IC": ic_series.values
    })

    # Metric 4: concentration
    total_pnl = port.sum()
    top5 = port.nlargest(5)
    conc_df = pd.DataFrame({
        "Metric": ["Top 5 days sum", "Total daily P&L sum",
                   "Top 5 as % of total P&L", "Top 5 dates"],
        "Value": [f"{top5.sum()*100:+.3f}%", f"{total_pnl*100:+.3f}%",
                  f"{top5.sum()/total_pnl*100:.1f}%" if abs(total_pnl) > 1e-8 else "N/A",
                  str(list(top5.index.date))]
    })

    # Metric 5: autocorrelation
    ac1 = port.autocorr(lag=1)
    ac_note = ("overstates Sharpe" if ac1 > 0.05 else
               "understates Sharpe" if ac1 < -0.05 else "annualization approx valid")
    auto_df = pd.DataFrame({
        "Metric": ["Lag-1 autocorrelation", "Implication for sqrt(252) annualization"],
        "Value": [f"{ac1:+.4f}", ac_note]
    })

    # Metric 6: liquidity
    liq_df = pd.DataFrame({
        "Metric": ["Median daily dollar volume (leg names)",
                   "25th pct daily dollar volume",
                   "Fraction of leg-days in bottom universe quartile",
                   "Caveat"],
        "Value": [f"${leg_dvols.median():,.0f}",
                  f"${leg_dvols.quantile(0.25):,.0f}",
                  f"{np.mean(bottom_q_flags)*100:.1f}%",
                  "Borrow cost / shortability NOT assessed -- volume proxy only"]
    })

    with pd.ExcelWriter(REPORT_DIR / "extended_diagnostics.xlsx", engine="openpyxl") as writer:
        leg_attr.to_excel(writer, index=False, sheet_name="1_Leg_Attribution")
        sig_df.to_excel(writer,   index=False, sheet_name="2_Significance")
        ic_df.to_excel(writer,    index=False, sheet_name="3_IC_Summary")
        ic_daily.to_excel(writer, index=False, sheet_name="3_IC_Daily")
        conc_df.to_excel(writer,  index=False, sheet_name="4_Concentration")
        auto_df.to_excel(writer,  index=False, sheet_name="5_Autocorrelation")
        liq_df.to_excel(writer,   index=False, sheet_name="6_Liquidity")
    print(f"Extended diagnostics saved: {REPORT_DIR / 'extended_diagnostics.xlsx'}")
except Exception as e:
    print(f"Extended diagnostics save failed: {e}")


# ── 12. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    f"News Flow L/S -- Final Strategy  |  "
    f"Sharpe={sharpe:.3f}  |  Ann.Return={ann_ret*100:+.1f}%  |  MaxDD={max_dd*100:.1f}%",
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
plt.savefig(REPORT_DIR / "performance_charts.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {REPORT_DIR / 'performance_charts.png'}")
plt.show()

# ── Portfolio count chart ─────────────────────────────────────────────────────
dates_idx = pd.DatetimeIndex(dates_used)
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(dates_idx, [l + s for l, s in zip(long_count, short_count)],
       color="steelblue", alpha=0.4, label="Total positions", width=0.8)
ax.plot(dates_idx, long_count,  color="green",  linewidth=1.8, marker="o", markersize=4, label="Long")
ax.plot(dates_idx, short_count, color="crimson", linewidth=1.8, marker="o", markersize=4, linestyle="--", label="Short")
ax.plot(dates_idx, ranked_count, color="orange", linewidth=1.2, linestyle=":", label="Ranked (non-zero signal)")
ax.axhline(np.mean([l + s for l, s in zip(long_count, short_count)]),
           color="grey", linestyle="--", linewidth=0.8,
           label=f"Avg total = {np.mean([l+s for l,s in zip(long_count,short_count)]):.0f}")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Tickers")
ax.set_title("Final Strategy -- Daily Portfolio Size (dollar neutral)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.tight_layout()
plt.savefig(REPORT_DIR / "portfolio_count_chart.png", dpi=150, bbox_inches="tight")
print(f"Count chart saved: {REPORT_DIR / 'portfolio_count_chart.png'}")
plt.show()


# ── 13. Save all outputs to Backtest_Report/ ─────────────────────────────────

# 1. Portfolio time series (per-ticker, per-day)
port_ts = pd.DataFrame(portfolio_rows)
port_ts.to_excel(REPORT_DIR / "portfolio_timeseries.xlsx", index=False, sheet_name="Portfolio TimeSeries")
print(f"Portfolio time series saved: {REPORT_DIR / 'portfolio_timeseries.xlsx'}  ({len(port_ts):,} rows)")

# 2. Performance metrics + daily portfolio count
metrics = {
    "Backtest Period"  : f"{port.index[0].date()} to {port.index[-1].date()}",
    "Days Traded"      : len(port),
    "Avg Leg Size"     : f"{avg_leg:.0f}",
    "Sharpe (ann.)"    : f"{sharpe:+.3f}",
    "Ann. Return"      : f"{ann_ret*100:+.2f}%",
    "Ann. Volatility"  : f"{ann_vol*100:.2f}%",
    "Total Return"     : f"{total_ret*100:+.2f}%",
    "Max Drawdown"     : f"{max_dd*100:.2f}%",
    "Calmar Ratio"     : f"{calmar:.3f}",
    "Hit Rate"         : f"{hit_rate*100:.1f}%",
}
metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

count_df = pd.DataFrame({
    "date"            : [d.date() for d in dates_used],
    "ranked_universe" : ranked_count,
    "long_positions"  : long_count,
    "short_positions" : short_count,
    "total_positions" : [l + s for l, s in zip(long_count, short_count)],
    "daily_return_pct": (port * 100).values,
})

with pd.ExcelWriter(REPORT_DIR / "performance_metrics.xlsx", engine="openpyxl") as writer:
    metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
    count_df.to_excel(writer, index=False, sheet_name="Daily Portfolio Count")
print(f"Metrics saved: {REPORT_DIR / 'performance_metrics.xlsx'}")

# 3. Date audit -- classify every price date with a trade reason
audit_rows = []
for day in pd.DatetimeIndex(price_dates_arr):
    row = {"date": day.date(), "traded": "", "reason": ""}
    if day not in signal.index:
        row["traded"] = "No"
        row["reason"] = "No news on this date -- signal absent from news data"
    elif day not in fwd_ret.index:
        row["traded"] = "No"
        row["reason"] = "Last date in dataset -- no next-day return available"
    else:
        sig_d = signal.loc[day].dropna()
        ret_d = fwd_ret.loc[day].dropna()
        sig_d = sig_d[zscore_u.loc[day].reindex(sig_d.index).notna()]
        sig_d = sig_d[sig_d != 0]
        n_val = len(sig_d.index.intersection(ret_d.index))
        if n_val >= MIN_UNIVERSE:
            row["traded"] = "Yes"
            row["reason"] = f"Traded -- {n_val} tickers with valid signal and return"
        else:
            row["traded"] = "No"
            row["reason"] = f"Only {n_val} tickers with valid signal -- below MIN_UNIVERSE={MIN_UNIVERSE}"
    audit_rows.append(row)

audit_df = pd.DataFrame(audit_rows)
audit_df.to_excel(REPORT_DIR / "date_audit.xlsx", index=False, sheet_name="Date Audit")
print(f"Date audit saved: {REPORT_DIR / 'date_audit.xlsx'}")
print(f"\nAll outputs saved to: {REPORT_DIR}")
