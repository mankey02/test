"""무기한 선물 1시간봉 + 펀딩비 + 미결제약정(OI) 수집기.

사용 예:
    python fetch_data.py --exchange bybit --symbol BTCUSDT --start 2021-01-01
    python fetch_data.py --exchange binance --symbol ETHUSDT --start 2022-01-01

출력: data/{exchange}_{symbol}_1h.csv
    time     : 봉 마감 시각(UTC). 이 시점에 close/funding/oi 가 모두 "이미 알려진" 값이다.
    close    : 종가
    funding  : time 시점까지 마지막으로 정산된 펀딩비
    oi       : time 시점까지 마지막으로 기록된 미결제약정

주의:
    - Binance 의 OI 히스토리 API 는 최근 30일만 제공한다. 장기 검증은 Bybit 를 권장.
    - 예측 펀딩비(실시간 값)는 과거 기록이 없으므로 "정산된 펀딩비"만 사용한다.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

HOUR_MS = 3_600_000
_session = requests.Session()


def _get(url, params, retries=5):
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=15)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def _to_ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def _paginate_backward(fetch_page, start, end, ts_of):
    """최신 → 과거 순으로 end 를 당겨가며 페이지를 모은다 (Bybit 방식)."""
    rows, cur_end = [], end
    while cur_end > start:
        page = fetch_page(start, cur_end)
        if not page:
            break
        rows.extend(page)
        oldest = min(ts_of(x) for x in page)
        if oldest <= start:
            break
        cur_end = oldest - 1
        time.sleep(0.1)
    return rows


def _paginate_forward(fetch_page, start, end, ts_of, step):
    """과거 → 최신 순으로 start 를 밀어가며 페이지를 모은다 (Binance 방식)."""
    rows, cur = [], start
    while cur < end:
        page = fetch_page(cur, end)
        if not page:
            break
        rows.extend(page)
        cur = max(ts_of(x) for x in page) + step
        time.sleep(0.1)
    return rows


# ---------------------------------------------------------------- Bybit (v5)
BYBIT = "https://api.bybit.com"


def _bybit(path, params):
    data = _get(BYBIT + path, params)
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit {data.get('retCode')}: {data.get('retMsg')}")
    return data["result"]["list"]


def bybit_klines(symbol, start, end):
    rows = _paginate_backward(
        lambda s, e: _bybit("/v5/market/kline", dict(
            category="linear", symbol=symbol, interval="60", start=s, end=e, limit=1000)),
        start, end, lambda x: int(x[0]))
    return pd.DataFrame({"open_ms": [int(r[0]) for r in rows],
                         "close": [float(r[4]) for r in rows]})


def bybit_funding(symbol, start, end):
    rows = _paginate_backward(
        lambda s, e: _bybit("/v5/market/funding/history", dict(
            category="linear", symbol=symbol, startTime=s, endTime=e, limit=200)),
        start, end, lambda x: int(x["fundingRateTimestamp"]))
    return pd.DataFrame({"ms": [int(r["fundingRateTimestamp"]) for r in rows],
                         "funding": [float(r["fundingRate"]) for r in rows]})


def bybit_oi(symbol, start, end):
    rows = _paginate_backward(
        lambda s, e: _bybit("/v5/market/open-interest", dict(
            category="linear", symbol=symbol, intervalTime="1h",
            startTime=s, endTime=e, limit=200)),
        start, end, lambda x: int(x["timestamp"]))
    return pd.DataFrame({"ms": [int(r["timestamp"]) for r in rows],
                         "oi": [float(r["openInterest"]) for r in rows]})


# ------------------------------------------------------------ Binance (USDⓈ-M)
BINANCE = "https://fapi.binance.com"


def binance_klines(symbol, start, end):
    rows = _paginate_forward(
        lambda s, e: _get(BINANCE + "/fapi/v1/klines", dict(
            symbol=symbol, interval="1h", startTime=s, endTime=e, limit=1500)),
        start, end, lambda x: int(x[0]), HOUR_MS)
    return pd.DataFrame({"open_ms": [int(r[0]) for r in rows],
                         "close": [float(r[4]) for r in rows]})


def binance_funding(symbol, start, end):
    rows = _paginate_forward(
        lambda s, e: _get(BINANCE + "/fapi/v1/fundingRate", dict(
            symbol=symbol, startTime=s, endTime=e, limit=1000)),
        start, end, lambda x: int(x["fundingTime"]), 1)
    return pd.DataFrame({"ms": [int(r["fundingTime"]) for r in rows],
                         "funding": [float(r["fundingRate"]) for r in rows]})


def binance_oi(symbol, start, end):
    # 최근 30일만 제공됨
    start = max(start, end - 29 * 24 * HOUR_MS)
    rows = _paginate_forward(
        lambda s, e: _get(BINANCE + "/futures/data/openInterestHist", dict(
            symbol=symbol, period="1h", startTime=s, endTime=e, limit=500)),
        start, end, lambda x: int(x["timestamp"]), HOUR_MS)
    return pd.DataFrame({"ms": [int(r["timestamp"]) for r in rows],
                         "oi": [float(r["sumOpenInterest"]) for r in rows]})


SOURCES = {
    "bybit": (bybit_klines, bybit_funding, bybit_oi),
    "binance": (binance_klines, binance_funding, binance_oi),
}


def _asof(left, right, col):
    """left.time 시점까지 알려진 right 의 마지막 값을 붙인다 (미래 참조 없음)."""
    if right.empty:
        left[col] = float("nan")
        return left
    right = right.drop_duplicates("ms").sort_values("ms")
    right["time"] = pd.to_datetime(right["ms"], unit="ms", utc=True)
    return pd.merge_asof(left, right[["time", col]], on="time", direction="backward")


def build(exchange, symbol, start, end):
    get_k, get_f, get_oi = SOURCES[exchange]
    print(f"[{exchange} {symbol}] 1시간봉 수집...")
    k = get_k(symbol, start, end).drop_duplicates("open_ms").sort_values("open_ms")
    # 봉 마감 시각 기준으로 인덱싱: 이 시각에 close 가 확정된다
    k["time"] = pd.to_datetime(k["open_ms"] + HOUR_MS, unit="ms", utc=True)
    k = k[["time", "close"]].reset_index(drop=True)
    print(f"  {len(k)} 개 봉")

    print("펀딩비 수집...")
    f = get_f(symbol, start, end)
    print(f"  {len(f)} 건")
    print("미결제약정 수집...")
    oi = get_oi(symbol, start, end)
    print(f"  {len(oi)} 건")

    df = _asof(k, f, "funding")
    df = _asof(df, oi, "oi")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", choices=SOURCES, default="bybit")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None, help="기본값: 현재")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    start = _to_ms(a.start)
    end = _to_ms(a.end) if a.end else int(time.time() * 1000)
    df = build(a.exchange, a.symbol, start, end)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.exchange}_{a.symbol}_1h.csv"
    df.to_csv(path, index=False)
    print(f"저장: {path}  ({df['time'].min()} ~ {df['time'].max()})")
    print(f"OI 결측 비율: {df['oi'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
