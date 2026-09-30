"""쏠림(crowding) → 반대 방향 가설 검증.

가설:
    펀딩비가 극단 + OI 급증  =  한쪽으로 레버리지가 쏠린 상태
    → 이후 N시간 동안 쏠린 반대 방향으로 가격이 움직인다 (털어내기).

    long_crowded  : 펀딩비 상위 q  & OI 증가율 상위 oi_q  → 숏 진입(-1) 가정
    short_crowded : 펀딩비 하위 q  & OI 증가율 상위 oi_q  → 롱 진입(+1) 가정

검증:
    1) 이벤트 이후 수익률(수수료 차감) vs 같은 구간·같은 방향의 무작위 진입
    2) Welch t-test + 무작위 표본 추출 검정(one-sided) + 부트스트랩 95% CI
    3) 매크로 국면(bull/bear) 별, 기간(IS/OOS) 별로 쪼개서 결과가 유지되는지

미래 참조 방지:
    - 백분위는 과거 window 만으로 계산 (rolling rank)
    - 국면은 전일 종가 기준 200일선
    - 진입은 신호가 확정된 봉의 종가

사용 예:
    python backtest.py --data data/bybit_BTCUSDT_1h.csv
    python backtest.py --synthetic            # 파이프라인 점검용 가짜 데이터
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ 데이터
def load_csv(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    # 결측 봉이 있어도 shift(-h) 가 정확히 h시간 뒤를 가리키도록 시간축 고정
    return df.asfreq("1h")


def make_synthetic(years=5, seed=7, effect=0.004):
    """쏠림 효과를 일부러 심어 둔 가짜 데이터. 파이프라인이 효과를 잡아내는지 확인용."""
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 24)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    # 매크로: 1년 단위로 상승/하락 추세가 번갈아 나온다
    macro = np.where((np.arange(n) // (24 * 365)) % 2 == 0, 4e-5, -3e-5)

    ret = np.zeros(n)
    funding = np.zeros(n)
    oi = np.zeros(n)
    f, o, press = 1e-4, 1e9, 0.0
    for i in range(n):
        if i % 8 == 0 and i >= 24:
            # 펀딩비는 최근 24시간 상승을 쫓아간다 (추격 매수 = 쏠림)
            chase = ret[i - 24:i].sum()
            f = 0.85 * f + 0.15 * (1e-4 + chase * 0.02) + rng.normal(0, 3e-5)
        funding[i] = f
        z = (f - 1e-4) / 2e-4
        # 쏠림이 클수록 OI 가 늘어나고, 반대 방향 압력이 쌓인다
        o *= 1 + 0.002 * abs(z) + rng.normal(0, 0.004)
        oi[i] = o
        press = 0.96 * press + 0.04 * (-effect * np.tanh(z) * min(abs(z) / 2, 1.5))
        ret[i] = macro[i] + press / 24 + rng.normal(0, 0.007)
    close = 30000 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"close": close, "funding": funding, "oi": oi}, index=idx.rename("time"))


# ---------------------------------------------------------------- 피처/국면
def add_features(df, window_h, oi_lookback_h):
    minp = window_h // 2
    df["funding_rank"] = df["funding"].rolling(window_h, min_periods=minp).rank(pct=True)
    df["oi_chg"] = df["oi"].pct_change(oi_lookback_h, fill_method=None)
    df["oi_rank"] = df["oi_chg"].rolling(window_h, min_periods=minp).rank(pct=True)
    return df


def add_regime(df, sma_days, macro_csv=None):
    if macro_csv:
        # 사용자 제공 국면: date,regime (regime 은 bull/bear 등 자유 라벨)
        # 해당 날짜에 이미 알 수 있었던 값이어야 한다.
        m = pd.read_csv(macro_csv, parse_dates=["date"])
        m["date"] = pd.to_datetime(m["date"], utc=True)
        reg = m.set_index("date")["regime"].sort_index()
    else:
        daily = df["close"].resample("1D").last()
        sma = daily.rolling(sma_days).mean()
        reg = pd.Series(np.where(daily > sma, "bull", "bear"), index=daily.index)
        reg = reg.where(sma.notna()).shift(1)  # 전일 종가 기준 → 미래 참조 없음
    df["regime"] = reg.reindex(df.index, method="ffill")
    return df


# ------------------------------------------------------------------- 이벤트
def detect_events(df, q, oi_q, cooldown_h):
    long_c = df["funding_rank"] >= 1 - q
    short_c = df["funding_rank"] <= q
    if oi_q > 0:
        surge = df["oi_rank"] >= 1 - oi_q
        long_c &= surge
        short_c &= surge

    rows = []
    gap = pd.Timedelta(hours=cooldown_h)
    for name, mask, direction in (("long_crowded", long_c, -1), ("short_crowded", short_c, 1)):
        last = None
        for t in df.index[mask.fillna(False).to_numpy()]:
            # 쏠림이 계속되는 동안 매 시간 신호가 뜨는 것을 막는다 (표본 중복 제거)
            if last is None or t - last >= gap:
                rows.append((t, name, direction))
                last = t
    return pd.DataFrame(rows, columns=["time", "signal", "direction"]).set_index("time").sort_index()


# --------------------------------------------------------------------- 통계
def summarize(ev, base, n_perm, rng):
    ev, base = ev.dropna().to_numpy(), base.dropna().to_numpy()
    n = len(ev)
    if n < 2 or len(base) < 2:
        return dict(n=n)
    mean = ev.mean()
    boot = rng.choice(ev, size=(n_perm, n)).mean(axis=1)
    rand = rng.choice(base, size=(n_perm, n)).mean(axis=1)
    return dict(
        n=n,
        mean=mean,
        median=np.median(ev),
        win=(ev > 0).mean(),
        base_mean=base.mean(),
        excess=mean - base.mean(),
        ci_lo=np.percentile(boot, 2.5),
        ci_hi=np.percentile(boot, 97.5),
        p_t=stats.ttest_ind(ev, base, equal_var=False).pvalue,
        # 무작위 진입 n 번의 평균이 이벤트 평균 이상일 확률 (단측)
        p_rand=(rand >= mean).mean(),
    )


def run(df, events, horizons, cost, split_date, n_perm, seed):
    rng = np.random.default_rng(seed)
    split = pd.Timestamp(split_date, tz="UTC")
    df["period"] = np.where(df.index < split, "IS", "OOS")
    events = events.join(df[["regime", "period"]])

    fwd = {h: df["close"].shift(-h) / df["close"] - 1 for h in horizons}
    periods = {"ALL": df.index, "IS": df.index[df.index < split], "OOS": df.index[df.index >= split]}
    regimes = ["ALL"] + sorted(df["regime"].dropna().unique())

    out = []
    for pname, pidx in periods.items():
        for reg in regimes:
            base_idx = pidx if reg == "ALL" else pidx[df.loc[pidx, "regime"].to_numpy() == reg]
            for sig, direction in (("long_crowded", -1), ("short_crowded", 1)):
                e = events[events["signal"] == sig]
                e_idx = e.index[e.index.isin(base_idx)]
                for h in horizons:
                    ev_ret = direction * fwd[h].reindex(e_idx) - cost
                    base_ret = direction * fwd[h].reindex(base_idx) - cost
                    r = summarize(ev_ret, base_ret, n_perm, rng)
                    out.append(dict(period=pname, regime=reg, signal=sig, h=h, **r))
    return pd.DataFrame(out), events


def print_table(res):
    t = res.copy()
    for c in ("mean", "median", "base_mean", "excess", "ci_lo", "ci_hi"):
        if c in t:
            t[c] = (t[c] * 1e4).round(1)  # bp
    if "win" in t:
        t["win"] = (t["win"] * 100).round(1)
    for c in ("p_t", "p_rand"):
        if c in t:
            t[c] = t[c].round(3)
    t["flag"] = ""
    if "p_rand" in t:
        t.loc[(t["p_rand"] < 0.05) & (t["n"] >= 30), "flag"] = "*"
    t.loc[t["n"] < 30, "flag"] = "n<30"
    cols = ["period", "regime", "signal", "h", "n", "mean", "median", "win",
            "base_mean", "excess", "ci_lo", "ci_hi", "p_t", "p_rand", "flag"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(t[[c for c in cols if c in t]].to_string(index=False))
    print("\n단위: mean/median/base_mean/excess/ci = bp(0.01%), 수수료 차감 후 | win = %")
    print("excess = 이벤트 평균 - 같은 구간·같은 방향 무작위 진입 평균")
    print("* = p_rand<0.05 & n>=30.  행이 많으므로 우연히 * 가 몇 개 나오는 것은 정상 (다중검정).")
    print("핵심 확인: IS 에서 찾은 효과가 OOS 에서도 같은 부호·비슷한 크기로 유지되는가?")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="fetch_data.py 가 만든 CSV")
    src.add_argument("--synthetic", action="store_true", help="효과를 심어 둔 가짜 데이터로 실행")
    ap.add_argument("--synthetic-effect", type=float, default=0.004,
                    help="가짜 데이터에 심을 쏠림 효과 크기. 0 이면 효과 없음(검정이 헛것을 잡는지 확인)")
    ap.add_argument("--q", type=float, default=0.10, help="펀딩비 극단 기준 (상/하위 비율)")
    ap.add_argument("--oi-q", type=float, default=0.20, help="OI 증가율 상위 비율. 0 이면 OI 조건 끔")
    ap.add_argument("--window-days", type=int, default=90, help="백분위 계산용 과거 구간")
    ap.add_argument("--oi-lookback", type=int, default=24, help="OI 증가율 계산 시간(h)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24])
    ap.add_argument("--cooldown", type=int, default=24, help="같은 신호 재발생 최소 간격(h)")
    ap.add_argument("--cost", type=float, default=0.0012,
                    help="왕복 비용(수수료+슬리피지). 기본 0.12%%")
    ap.add_argument("--split-date", default="2024-01-01", help="IS/OOS 분할 날짜")
    ap.add_argument("--regime-sma", type=int, default=200, help="국면 판단 이동평균(일)")
    ap.add_argument("--macro-csv", help="직접 만든 국면 CSV (date,regime)")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    if a.synthetic:
        df = make_synthetic(effect=a.synthetic_effect)
        name = "synthetic"
        if a.split_date == ap.get_default("split_date"):
            a.split_date = "2023-01-01"
    else:
        df = load_csv(a.data)
        name = Path(a.data).stem

    if a.oi_q > 0 and df["oi"].notna().mean() < 0.5:
        print("경고: OI 데이터가 절반 이상 비어 있어 OI 조건을 끕니다 (--oi-q 0).")
        a.oi_q = 0

    df = add_features(df, a.window_days * 24, a.oi_lookback)
    df = add_regime(df, a.regime_sma, a.macro_csv)
    events = detect_events(df, a.q, a.oi_q, a.cooldown)

    print(f"데이터: {name}  {df.index.min()} ~ {df.index.max()}  ({len(df)} 시간)")
    print(f"조건: 펀딩 상/하위 {a.q:.0%}, OI증가 상위 {a.oi_q:.0%}, cooldown {a.cooldown}h, "
          f"비용 {a.cost:.2%}, 분할 {a.split_date}")
    print(f"이벤트: {events['signal'].value_counts().to_dict()}\n")

    res, events = run(df, events, a.horizons, a.cost, a.split_date, a.n_perm, a.seed)
    print_table(res)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{name}_summary.csv", index=False)
    events.to_csv(out / f"{name}_events.csv")
    print(f"\n저장: {out}/{name}_summary.csv, {out}/{name}_events.csv")


if __name__ == "__main__":
    main()
