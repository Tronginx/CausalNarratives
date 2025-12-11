import requests
import pandas as pd

FMP_API_KEY = "rftAyYOSfpAxpVAVp9Zf0a7diXUivX1c"
FMP_STABLE_EOD = "https://financialmodelingprep.com/stable/historical-price-eod/full"

def _fmp_hist_stable(symbol: str, start_date: str, end_date: str ) -> pd.DataFrame:
    params = {"symbol": symbol, "apikey": FMP_API_KEY}
    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date

    r = requests.get(FMP_STABLE_EOD, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    # 🚀 新增类型判断
    if isinstance(js, dict):
        hist = js.get("historical", [])
    elif isinstance(js, list):
        hist = js
    else:
        raise ValueError(f"Unexpected JSON type: {type(js)}")

    if not hist:
        raise ValueError(f"No data returned for {symbol}.")

    df = pd.DataFrame(hist)
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    df = df[["date", price_col]].rename(columns={price_col: "px"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def get_forward_returns(symbol: str, start_date: str, horizons=(1,5,10,30)) -> pd.Series:
    max_h = max(horizons) if horizons else 30
    to_date = (pd.to_datetime(start_date) + pd.tseries.offsets.BDay(max_h + 10)).date().isoformat()
    df = _fmp_hist_stable(symbol, start_date, to_date)

    start_dt = pd.to_datetime(start_date)
    idx = df.index[df["date"] >= start_dt]
    if len(idx) == 0:
        raise ValueError("Start date is after last available trading day.")
    i = int(idx[0])
    px0 = float(df.at[i, "px"])

    out = {}
    for h in horizons:
        j = i + int(h)
        out[h] = (float(df.at[j, "px"]) / px0 - 1.0) if j < len(df) else float("nan")

    return pd.Series(out, name=f"{symbol}@{df.at[i,'date'].date()}")

if __name__ == "__main__":
    s = get_forward_returns("AAPL", "2024-01-15", horizons=(1,5,10,30))
    print(s.apply(lambda x: None if pd.isna(x) else round(x, 4)))
