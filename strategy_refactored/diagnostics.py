import numpy as np
import pandas as pd
import quantstats as qs
from scipy import stats as scipy_stats
from .config import BacktestConfig
from .backtest_engine import BacktestResult
from .performance import Metrics


class Diagnostics:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def _leg_attribution(self, result: BacktestResult, metrics: Metrics):
        long_port  = pd.Series(result.long_leg_rets,  index=pd.DatetimeIndex(result.dates_used))
        short_port = pd.Series(result.short_leg_rets, index=pd.DatetimeIndex(result.dates_used))
        print("\n[1] Long / Short leg attribution")
        for label, leg in [("Long leg (raw)", long_port),
                            ("Short leg (raw, negative contribution = good)", short_port)]:
            print(f"    {label}")
            print(f"      Total compounded return : {qs.stats.comp(leg)*100:+.2f}%")
            print(f"      Annualized Sharpe       : {qs.stats.sharpe(leg, periods=252):+.3f}")
            print(f"      Mean daily return       : {leg.mean()*100:+.4f}%")
        print(f"    Edge: long_mean - short_mean = {(long_port.mean()-short_port.mean())*100:+.4f}% per day")
        return long_port, short_port

    def _significance(self, result: BacktestResult, metrics: Metrics):
        port   = result.portfolio
        N      = len(port)
        t_stat, p_val = scipy_stats.ttest_1samp(port, 0)
        SE_SR  = np.sqrt((1 + 0.5 * metrics.sharpe**2) / N)
        print("\n[2] Sharpe statistical significance")
        print(f"    N (trading days)            : {N}")
        print(f"    t-stat (mean != 0)          : {t_stat:.3f}  (p={p_val:.3f})")
        print(f"    Annualized Sharpe +/- SE    : {metrics.sharpe:+.3f} +/- {SE_SR:.3f}  (Lo 2002)")
        print(f"    95% CI approx               : [{metrics.sharpe-1.96*SE_SR:+.3f}, {metrics.sharpe+1.96*SE_SR:+.3f}]")
        print(f"    CAVEAT: N={N} is too small for statistical significance at conventional levels.")
        return t_stat, p_val, SE_SR

    def _ic(self, result: BacktestResult):
        ic_values = []
        for s, r in zip(result.ic_signal, result.ic_ret):
            common = s.index.intersection(r.index)
            s2, r2 = s.loc[common].dropna(), r.loc[common].dropna()
            common2 = s2.index.intersection(r2.index)
            if len(common2) < 10:
                continue
            corr, _ = scipy_stats.spearmanr(s2.loc[common2].values, r2.loc[common2].values)
            ic_values.append(corr)
        ic_series = pd.Series(ic_values)
        ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan
        print("\n[3] Information Coefficient (IC)")
        print(f"    Days with IC computed       : {len(ic_series)}")
        print(f"    Mean IC                     : {ic_series.mean():+.4f}")
        print(f"    Std IC                      : {ic_series.std():.4f}")
        print(f"    IC Information Ratio (IR)   : {ir:+.3f}")
        print(f"    Fraction of days IC > 0     : {(ic_series > 0).mean()*100:.1f}%")
        return ic_series, ir

    def _concentration(self, result: BacktestResult):
        port      = result.portfolio
        total_pnl = port.sum()
        top5      = port.nlargest(5)
        print("\n[4] Return concentration")
        print(f"    Top 5 days sum              : {top5.sum()*100:+.3f}%")
        print(f"    Total daily P&L sum         : {total_pnl*100:+.3f}%")
        if abs(total_pnl) > 1e-8:
            print(f"    Top 5 days = {top5.sum()/total_pnl*100:.1f}% of total daily P&L")
        print(f"    Top 5 dates: {list(top5.index.date)}")
        return top5, total_pnl

    def _autocorrelation(self, result: BacktestResult):
        ac1 = result.portfolio.autocorr(lag=1)
        note = ("overstates Sharpe" if ac1 > 0.05 else
                "understates Sharpe" if ac1 < -0.05 else "annualization approx valid")
        print("\n[5] Return autocorrelation (lag-1)")
        print(f"    Lag-1 autocorrelation       : {ac1:+.4f}")
        print(f"    Note: {note}")
        return ac1, note

    def _liquidity(self, result: BacktestResult, prices_raw, universe, close_wide):
        print("\n[6] Liquidity profile (dollar volume proxy)")
        vol_wide = prices_raw.pivot(index="date", columns="ticker", values="volume").reindex(columns=universe)
        dv_wide  = vol_wide * close_wide.reindex(columns=universe)

        leg_dvols, bottom_q_flags = [], []
        for i, day in enumerate(pd.DatetimeIndex(result.dates_used)):
            if day not in dv_wide.index:
                continue
            day_dv   = dv_wide.loc[day].dropna()
            univ_q25 = day_dv.quantile(0.25)
            for t in result.long_tickers[i] + result.short_tickers[i]:
                if t in day_dv.index:
                    dv = day_dv[t]
                    leg_dvols.append(dv)
                    bottom_q_flags.append(1 if dv <= univ_q25 else 0)

        leg_dvols = pd.Series(leg_dvols)
        print(f"    Median daily dollar volume  : ${leg_dvols.median():,.0f}")
        print(f"    25th pct daily dollar volume: ${leg_dvols.quantile(0.25):,.0f}")
        print(f"    Fraction in bottom quartile : {np.mean(bottom_q_flags)*100:.1f}%")
        print(f"    CAVEAT: Borrow cost not assessed -- volume proxy only.")
        return leg_dvols, bottom_q_flags

    def run_and_save(self, result: BacktestResult, metrics: Metrics, prices_raw, universe, close_wide):
        print("\n" + "=" * 52)
        print("  EXTENDED DIAGNOSTICS")
        print("=" * 52)

        try:
            long_port, short_port = self._leg_attribution(result, metrics)
        except Exception as e:
            print(f"    [Metric 1 failed: {e}]")
            long_port = short_port = pd.Series(dtype=float)

        try:
            t_stat, p_val, SE_SR = self._significance(result, metrics)
        except Exception as e:
            print(f"    [Metric 2 failed: {e}]")
            t_stat = p_val = SE_SR = np.nan

        try:
            ic_series, ir = self._ic(result)
        except Exception as e:
            print(f"    [Metric 3 failed: {e}]")
            ic_series = pd.Series(dtype=float)
            ir = np.nan

        try:
            top5, total_pnl = self._concentration(result)
        except Exception as e:
            print(f"    [Metric 4 failed: {e}]")
            top5 = pd.Series(dtype=float)
            total_pnl = np.nan

        try:
            ac1, ac_note = self._autocorrelation(result)
        except Exception as e:
            print(f"    [Metric 5 failed: {e}]")
            ac1, ac_note = np.nan, ""

        try:
            leg_dvols, bottom_q_flags = self._liquidity(result, prices_raw, universe, close_wide)
        except Exception as e:
            print(f"    [Metric 6 failed: {e}]")
            leg_dvols = pd.Series(dtype=float)
            bottom_q_flags = []

        print("\n" + "=" * 52)

        # Save to Excel
        try:
            N = len(result.portfolio)
            leg_attr = pd.DataFrame({
                "Metric": [
                    "Long leg -- total compounded return", "Long leg -- annualized Sharpe", "Long leg -- mean daily return",
                    "Short leg -- total compounded return (raw)", "Short leg -- annualized Sharpe (raw)", "Short leg -- mean daily return (raw)",
                    "Edge (long mean - short mean, per day)",
                ],
                "Value": [
                    f"{qs.stats.comp(long_port)*100:+.2f}%", f"{qs.stats.sharpe(long_port, periods=252):+.3f}",
                    f"{long_port.mean()*100:+.4f}%", f"{qs.stats.comp(short_port)*100:+.2f}%",
                    f"{qs.stats.sharpe(short_port, periods=252):+.3f}", f"{short_port.mean()*100:+.4f}%",
                    f"{(long_port.mean()-short_port.mean())*100:+.4f}%",
                ],
                "Note": ["", "", "", "positive = short leg stocks went up (hurts strategy)", "", "", ""],
            })
            sig_df = pd.DataFrame({
                "Metric": ["N (trading days)", "t-statistic", "p-value", "Annualized Sharpe",
                           "SE of Sharpe (Lo 2002)", "95% CI lower", "95% CI upper", "Caveat"],
                "Value": [N, f"{t_stat:.3f}", f"{p_val:.3f}", f"{metrics.sharpe:+.3f}",
                          f"{SE_SR:.3f}", f"{metrics.sharpe-1.96*SE_SR:+.3f}", f"{metrics.sharpe+1.96*SE_SR:+.3f}",
                          f"N={N} is too small for statistical significance at conventional confidence levels"],
            })
            ic_df = pd.DataFrame({
                "Metric": ["Days with IC", "Mean IC", "Std IC", "IC IR (mean/std)", "Fraction IC > 0"],
                "Value": [len(ic_series), f"{ic_series.mean():+.4f}", f"{ic_series.std():.4f}",
                          f"{ir:+.3f}", f"{(ic_series > 0).mean()*100:.1f}%"],
            })
            ic_daily = pd.DataFrame({
                "trade_date": [d.date() for d in pd.DatetimeIndex(result.dates_used)[:len(ic_series)]],
                "IC": ic_series.values,
            })
            conc_df = pd.DataFrame({
                "Metric": ["Top 5 days sum", "Total daily P&L sum", "Top 5 as % of total P&L", "Top 5 dates"],
                "Value": [f"{top5.sum()*100:+.3f}%", f"{total_pnl*100:+.3f}%",
                          f"{top5.sum()/total_pnl*100:.1f}%" if abs(total_pnl) > 1e-8 else "N/A",
                          str(list(top5.index.date))],
            })
            auto_df = pd.DataFrame({
                "Metric": ["Lag-1 autocorrelation", "Implication for sqrt(252) annualization"],
                "Value": [f"{ac1:+.4f}", ac_note],
            })
            liq_df = pd.DataFrame({
                "Metric": ["Median daily dollar volume", "25th pct daily dollar volume",
                           "Fraction of leg-days in bottom universe quartile", "Caveat"],
                "Value": [f"${leg_dvols.median():,.0f}", f"${leg_dvols.quantile(0.25):,.0f}",
                          f"{np.mean(bottom_q_flags)*100:.1f}%",
                          "Borrow cost / shortability NOT assessed -- volume proxy only"],
            })

            out = self.config.report_dir / "extended_diagnostics.xlsx"
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                leg_attr.to_excel(writer, index=False, sheet_name="1_Leg_Attribution")
                sig_df.to_excel(writer,   index=False, sheet_name="2_Significance")
                ic_df.to_excel(writer,    index=False, sheet_name="3_IC_Summary")
                ic_daily.to_excel(writer, index=False, sheet_name="3_IC_Daily")
                conc_df.to_excel(writer,  index=False, sheet_name="4_Concentration")
                auto_df.to_excel(writer,  index=False, sheet_name="5_Autocorrelation")
                liq_df.to_excel(writer,   index=False, sheet_name="6_Liquidity")
            print(f"Extended diagnostics saved: {out}")
        except Exception as e:
            print(f"Extended diagnostics save failed: {e}")
