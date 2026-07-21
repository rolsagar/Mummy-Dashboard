"""
Mummy Stocks — RSI + MACD Dashboard
=====================================
A Streamlit dashboard that mirrors the reference design:
 - "Fully Bullish" and "Weak / Oversold" highlight cards
 - A detailed table (Daily / Weekly / Monthly rows per stock) showing
   RSI (Wilder's, 14) with a colored bar, MACD(12,26,9) position,
   % of 52-week high, and volume.

Data source: yfinance (free, delayed ~15 min for NSE symbols).

Run with:
    streamlit run app.py
"""

import datetime as dt
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from stock_utils import IST, RSI_PERIOD, fetch_daily, macd_lines, resample_ohlc, rsi_wilder

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Mummy Stocks — RSI + MACD Dashboard",
    page_icon="📈",
    layout="wide",
)

BULLISH_RSI = 60
WEAK_RSI = 40

# Default watchlist -> (NSE ticker, Display name)
DEFAULT_WATCHLIST = [
    ("ASIANPAINT.NS", "ASIAN PAINT"),
    ("BDL.NS", "BDL"),
    ("BEL.NS", "BEL"),
    ("EXICOM.NS", "EXICOM"),
    ("GOLDBEES.NS", "GOLDBEES"),
    ("HAL.NS", "HAL"),
    ("HAPPSTMNDS.NS", "HAPPSTMNDS"),
    ("HDFCBANK.NS", "HDFCBANK"),
    ("HINDOILEXP.NS", "HINDOILEXP"),
    ("HINDZINC.NS", "HINDZINC"),
    ("IRCON.NS", "IRCON"),
    ("LGEINDIA.NS", "LGEINDIA"),
    ("LICI.NS", "LICI"),
    ("NIPPOBATRY.NS", "NIPPOBATRY"),
    ("OIL.NS", "OIL"),
    ("ONGC.NS", "ONGC"),
    ("PARADEEP.NS", "PARADEEP"),
    ("RANEHOLDIN.NS", "RANEHOLDIN"),
    ("SAIL.NS", "SAIL"),
    ("TATSILV.NS", "TATSILV"),
    ("TCS.NS", "TCS"),
    ("TMCV.NS", "TMCV"),
    ("TMPV.NS", "TMPV"),
    ("WIPRO.NS", "WIPRO"),
]

# NOTE: rsi_wilder, macd_lines, fetch_daily, and resample_ohlc now live in
# stock_utils.py (imported above) so this dashboard and the chart page in
# pages/1_Chart_View.py always use identical indicator math.


def analyze_ticker(ticker: str, display_name: str) -> dict | None:
    daily = fetch_daily(ticker)
    if daily.empty or len(daily) < 60:
        return None

    weekly = resample_ohlc(daily, "W-FRI")
    monthly = resample_ohlc(daily, "ME")

    frames = {"Daily": daily, "Weekly": weekly, "Monthly": monthly}
    tf_data = {}
    for tf_name, frame in frames.items():
        if len(frame) < RSI_PERIOD + 2:
            tf_data[tf_name] = {"rsi": np.nan, "macd_above": None}
            continue
        rsi = rsi_wilder(frame["Close"]).iloc[-1]
        macd_line, signal_line = macd_lines(frame["Close"])
        macd_above = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
        tf_data[tf_name] = {"rsi": round(float(rsi), 2), "macd_above": macd_above}

    current_price = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else current_price
    day_change_pct = round(((current_price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
    day_change_abs = round(current_price - prev_close, 2)
    lookback = daily.tail(252)  # ~1 trading year
    high_52w = float(lookback["High"].max())
    low_52w = float(lookback["Low"].min())
    pct_of_high = round((current_price / high_52w) * 100, 0) if high_52w else np.nan

    latest_volume = float(daily["Volume"].iloc[-1])
    avg_volume_20 = float(daily["Volume"].tail(20).mean())
    vol_ratio = round((latest_volume / avg_volume_20) * 100, 0) if avg_volume_20 else np.nan

    return {
        "ticker": ticker,
        "name": display_name,
        "price": current_price,
        "day_change_pct": day_change_pct,
        "day_change_abs": day_change_abs,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_of_high": pct_of_high,
        "tf": tf_data,
        "latest_volume": latest_volume,
        "avg_volume_20": avg_volume_20,
        "vol_ratio": vol_ratio,
    }


def classify(stock: dict) -> str:
    """Returns 'bullish', 'watch', 'weak', or 'neutral'."""
    tf = stock["tf"]
    if any(tf[t]["rsi"] is None or np.isnan(tf[t]["rsi"]) for t in tf):
        return "neutral"

    bull_flags = [tf[t]["rsi"] > BULLISH_RSI and tf[t]["macd_above"] for t in ["Daily", "Weekly", "Monthly"]]

    if all(bull_flags):
        return "bullish"

    wm_bull = (
        tf["Weekly"]["rsi"] > BULLISH_RSI
        and tf["Weekly"]["macd_above"]
        and tf["Monthly"]["rsi"] > BULLISH_RSI
        and tf["Monthly"]["macd_above"]
    )
    if wm_bull:
        return "watch"

    if tf["Daily"]["rsi"] < WEAK_RSI and not tf["Daily"]["macd_above"]:
        return "weak"

    return "neutral"


# --------------------------------------------------------------------------
# HTML / CSS RENDERING  (built to visually match the reference image)
# --------------------------------------------------------------------------

CSS = """
<style>
.dash-wrap { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

.section-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 22px;
}
.section-head { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
.section-title { font-size: 17px; font-weight: 700; color:#111827; }
.section-pill { font-size: 12px; font-weight:600; padding: 4px 12px; border-radius: 14px; color:white; }
.pill-bull { background:#16a34a; }
.pill-weak { background:#dc2626; }

.card-row { display:flex; gap:16px; flex-wrap:wrap; }
.card { flex:1; min-width: 220px; border-radius: 12px; padding: 14px 16px; }
.card-bull { background:#eafaf0; border:1px solid #bfead0; }
.card-weak { background:#fdecec; border:1px solid #f5c6c6; }

.card-name { font-size: 15px; font-weight:700; color:#111827; }
.card-sub { font-size: 12px; color:#6b7280; margin-bottom:8px; }
.card-metric { display:flex; justify-content:space-between; font-size:13px; color:#374151; padding:2px 0; }
.card-metric b { color:#111827; }

.badge { display:inline-block; margin-top:10px; font-size:12px; font-weight:700; padding:5px 10px; border-radius: 8px; }
.badge-strong { background:#16a34a; color:white; }
.badge-watch { background:#f59e0b; color:white; }
.badge-weak { background:#dc2626; color:white; }

table.dash-table { width:100%; min-width:760px; table-layout:fixed; border-collapse: collapse; font-size: 13px; background:white; border-radius: 10px; overflow:hidden;}
table.dash-table thead th {
    background:#12121f; color:#ffffff; text-transform:uppercase; font-size:11px;
    letter-spacing: 0.03em; padding: 10px 12px; text-align:left; font-weight:600;
}
table.dash-table td { padding: 9px 8px; border-bottom: 1px solid #f0f1f3; vertical-align: middle; }
table.dash-table tr.block-start td { border-top: 2px solid #d1d5db; }

.stock-name { font-weight:700; color:#111827; font-size:13px; }
.chart-link { display:inline-block; margin-top:6px; font-size:11px; font-weight:600; color:#2563eb; text-decoration:none; }
.chart-link:hover { text-decoration:underline; }
.price-cur { font-weight:700; color:#111827; font-size:14px; }
.price-sub { font-size:11px; color:#6b7280; }
.price-sub b { color:#111827; }
.price-lo { color:#dc2626 !important; font-weight:600; }
.price-chg { font-size:12px; font-weight:700; margin-top:2px; }
.price-chg-up { color:#16a34a; }
.price-chg-down { color:#dc2626; }

.pct-wrap { display:flex; flex-direction:column; align-items:flex-start; gap:4px; }
.pct-num { font-weight:700; font-size:14px; }
.pct-bar-bg { width:68px; height:6px; background:#e5e7eb; border-radius:4px; overflow:hidden; }
.pct-bar-fill { height:100%; border-radius:4px; }
.pct-caption { font-size:10px; color:#9ca3af; }

.rsi-cell { display:flex; align-items:center; gap:8px; }
.rsi-num { font-weight:700; font-size:13px; width:34px; }
.rsi-bar-bg { flex:1; max-width:80px; height:6px; background:#e5e7eb; border-radius:4px; overflow:hidden; }
.rsi-bar-fill { height:100%; border-radius:4px; }

.macd-pill { display:inline-block; font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px; }
.macd-above { background:#e8f8ee; color:#16a34a; }
.macd-below { background:#fdecec; color:#dc2626; }

.vol-cell { font-size:12px; color:#374151; }
.vol-up { color:#16a34a; font-weight:600; }
.vol-down { color:#dc2626; font-weight:600; }

.day-chg-num { font-size:16px; font-weight:700; }
.day-chg-sub { font-size:12px; font-weight:600; margin-top:2px; }
.day-chg-label { font-size:10px; color:#9ca3af; margin-top:2px; }

.timeframe-lbl { font-weight:600; color:#374151; }
.no-data { color:#9ca3af; font-style: italic; padding: 14px; }
</style>
"""


def render_html(html: str) -> None:
    """
    Render raw HTML via st.markdown safely.

    Streamlit's markdown parser follows standard Markdown rules, where any
    line indented by 4+ spaces is treated as a literal code block. Since our
    HTML builder functions use indentation for readability, we strip leading
    whitespace from every line (and collapse blank lines) before handing the
    string to st.markdown, so it always renders as real HTML instead of text.
    """
    cleaned = "\n".join(line.strip() for line in html.splitlines())
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    st.markdown(cleaned, unsafe_allow_html=True)


def color_for_rsi(rsi: float) -> str:
    if rsi is None or np.isnan(rsi):
        return "#9ca3af"
    if rsi >= BULLISH_RSI:
        return "#16a34a"
    if rsi < WEAK_RSI:
        return "#dc2626"
    return "#2563eb"


def color_for_pct(pct: float) -> str:
    if pct is None or np.isnan(pct):
        return "#9ca3af"
    if pct >= 85:
        return "#16a34a"
    if pct >= 65:
        return "#f59e0b"
    return "#dc2626"


def macd_pill(above) -> str:
    if above is None:
        return '<span class="macd-pill" style="background:#f3f4f6;color:#9ca3af;">N/A</span>'
    if above:
        return '<span class="macd-pill macd-above">&#8593; Above</span>'
    return '<span class="macd-pill macd-below">&#8595; Below</span>'


def build_card(stock: dict, kind: str) -> str:
    tf = stock["tf"]
    rows = ""
    for label in ["Daily", "Weekly", "Monthly"]:
        rsi = tf[label]["rsi"]
        rows += f'<div class="card-metric"><span>{label} RSI</span><b>{rsi:.2f}</b></div>'
    macd_all_above = all(tf[t]["macd_above"] for t in tf)
    macd_wm_above = tf["Weekly"]["macd_above"] and tf["Monthly"]["macd_above"]

    if kind == "bullish":
        if all(tf[t]["rsi"] > BULLISH_RSI and tf[t]["macd_above"] for t in ["Daily", "Weekly", "Monthly"]):
            macd_txt, badge = "Above all 3", '<span class="badge badge-strong">&#9650; Strong Buy Signal</span>'
        else:
            macd_txt, badge = ("Above W+M" if macd_wm_above else "Mixed"), '<span class="badge badge-watch">&#9651; Watch</span>'
        card_class = "card-bull"
    else:
        macd_txt = "Below"
        badge = '<span class="badge badge-weak">&#9660; Weak</span>'
        card_class = "card-weak"

    pct = stock["pct_of_high"]
    return f"""
    <div class="card {card_class}">
        <div class="card-name">{stock['name']}</div>
        <div class="card-sub">&#8377;{stock['price']:,.0f} &nbsp;|&nbsp; {pct:.0f}% of 52W High &nbsp;|&nbsp; <span class="{'price-chg-up' if stock['day_change_pct']>=0 else 'price-chg-down'}">{'&#9650;' if stock['day_change_pct']>=0 else '&#9660;'} {abs(stock['day_change_pct']):.2f}%</span></div>
        {rows}
        <div class="card-metric"><span>MACD</span><b>{macd_txt}</b></div>
        {badge}
    </div>
    """


def build_highlight_sections(stocks: list[dict]) -> str:
    bullish = [s for s in stocks if classify(s) in ("bullish", "watch")]
    weak = [s for s in stocks if classify(s) == "weak"]

    bullish_sorted = sorted(bullish, key=lambda s: classify(s) != "bullish")  # fully bullish first
    html = ""

    if bullish_sorted:
        cards = "".join(build_card(s, "bullish") for s in bullish_sorted[:6])
        html += f"""
        <div class="section-box">
            <div class="section-head">
                <span class="section-title">&#9650; Fully Bullish</span>
                <span class="section-pill pill-bull">RSI &gt;60 + MACD Above Signal on ALL 3 Timeframes</span>
            </div>
            <div class="card-row">{cards}</div>
        </div>"""
    if weak:
        cards = "".join(build_card(s, "weak") for s in weak[:6])
        html += f"""
        <div class="section-box">
            <div class="section-head">
                <span class="section-title">&#9660; Weak / Oversold</span>
                <span class="section-pill pill-weak">RSI &lt;40 on Daily + MACD Below Signal</span>
            </div>
            <div class="card-row">{cards}</div>
        </div>"""
    if not bullish_sorted and not weak:
        html += '<div class="section-box"><div class="no-data">No stocks currently meet the Bullish or Weak criteria for this watchlist.</div></div>'
    return html


def build_table(stocks: list[dict]) -> str:
    rows_html = ""
    for stock in stocks:
        pct = stock["pct_of_high"]
        pct_color = color_for_pct(pct)
        price_block = f"""
        <td rowspan="3">
            <div class="stock-name">{stock['name']}</div>
            <a class="chart-link" href="/Chart_View?ticker={quote(stock['ticker'])}&name={quote(stock['name'])}" target="_blank">&#128202; View Chart &#8599;</a>
        </td>
        <td rowspan="3">
            <div class="price-cur">Current {stock['price']:,.0f}</div>
            <div class="price-sub">52W Hi <b>{stock['high_52w']:,.0f}</b></div>
            <div class="price-sub">52W Lo <span class="price-lo">{stock['low_52w']:,.0f}</span></div>
            <div class="vol-cell {'vol-up' if stock['vol_ratio']>=100 else 'vol-down'}">Vol: {stock['vol_ratio']:.0f}% of 20D avg</div>
        </td>
        <td rowspan="3">
            <div class="pct-wrap">
                <div class="pct-num" style="color:{pct_color};">{pct:.0f}%</div>
                <div class="pct-bar-bg"><div class="pct-bar-fill" style="width:{min(pct,100)}%; background:{pct_color};"></div></div>
                <div class="pct-caption">of 52W High</div>
            </div>
        </td>
        <td rowspan="3">
            <div class="day-chg-num {'price-chg-up' if stock['day_change_pct']>=0 else 'price-chg-down'}">{'&#9650;' if stock['day_change_pct']>=0 else '&#9660;'} {abs(stock['day_change_pct']):.2f}%</div>
            <div class="day-chg-sub {'price-chg-up' if stock['day_change_abs']>=0 else 'price-chg-down'}">{'+' if stock['day_change_abs']>=0 else ''}{stock['day_change_abs']:,.2f}</div>
            <div class="day-chg-label">vs prev day close</div>
        </td>
        """
        first = True
        for label in ["Daily", "Weekly", "Monthly"]:
            rsi = stock["tf"][label]["rsi"]
            above = stock["tf"][label]["macd_above"]
            rsi_color = color_for_rsi(rsi)
            row_cls = "block-start" if first else ""
            rsi_bar_pct = 0 if (rsi is None or np.isnan(rsi)) else min(max(rsi, 0), 100)
            row = f'<tr class="{row_cls}">'
            if first:
                row += price_block
            row += f"""
                <td><span class="timeframe-lbl">{label}</span></td>
                <td>
                    <div class="rsi-cell">
                        <span class="rsi-num" style="color:{rsi_color};">{rsi:.2f}</span>
                        <div class="rsi-bar-bg"><div class="rsi-bar-fill" style="width:{rsi_bar_pct}%; background:{rsi_color};"></div></div>
                    </div>
                </td>
                <td>{macd_pill(above)}</td>
            </tr>"""
            rows_html += row
            first = False

    return f"""
    <div class="section-box" style="padding:0; overflow-x:auto; -webkit-overflow-scrolling:touch;">
    <table class="dash-table">
        <colgroup>
            <col style="width:98px;">
            <col style="width:112px;">
            <col style="width:90px;">
            <col style="width:90px;">
            <col style="width:68px;">
            <col style="width:150px;">
            <col style="width:120px;">
        </colgroup>
        <thead>
            <tr>
                <th>Stock</th>
                <th>Price (&#8377;)</th>
                <th>% of 52W High</th>
                <th>Day Change</th>
                <th>Timeframe</th>
                <th>RSI</th>
                <th>MACD Position</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """


# --------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------


def main():
    render_html(CSS)

    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption("Data source: Yahoo Finance (yfinance), free tier — typically ~15 min delayed for NSE.")

        tickers_text = st.text_area(
            "Watchlist (NSE ticker : Display name — one per line)",
            value="\n".join(f"{t} : {n}" for t, n in DEFAULT_WATCHLIST),
            height=220,
        )
        watchlist = []
        for line in tickers_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                t, n = line.split(":", 1)
                watchlist.append((t.strip(), n.strip()))
            else:
                watchlist.append((line, line.replace(".NS", "")))

        if st.button("🔄 Force refresh (clear cache)"):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Last loaded: {dt.datetime.now(IST).strftime('%d %b %Y, %H:%M:%S')} IST")

    fetch_timestamp = dt.datetime.now(IST).strftime("%d %b %Y, %H:%M:%S") + " IST"

    render_html(
        f"""
        <div class="dash-wrap">
        <h1 style="margin-bottom:2px;">Mummy Stocks — RSI + MACD Dashboard</h1>
        <p style="color:#6b7280; margin-top:0;">RSI (14, Wilder's EWM) &middot; MACD (12,26,9) &middot; Daily / Weekly / Monthly &middot; NSE</p>
        <p style="color:#111827; margin-top:0; font-size:13px; font-weight:600;">&#128337; Data pulled: {fetch_timestamp}</p>
        </div>
        """
    )

    stocks = []
    progress = st.progress(0.0, text="Fetching live data...")
    failed = []
    for i, (ticker, name) in enumerate(watchlist):
        try:
            result = analyze_ticker(ticker, name)
            if result:
                stocks.append(result)
            else:
                failed.append(ticker)
        except Exception:
            failed.append(ticker)
        progress.progress((i + 1) / max(len(watchlist), 1), text=f"Fetching {name}...")
    progress.empty()

    if failed:
        st.warning(f"Could not fetch data for: {', '.join(failed)}. Check the ticker symbols (must end in .NS for NSE).")

    if not stocks:
        st.error("No data available. Please check your internet connection and ticker symbols.")
        return

    render_html(build_highlight_sections(stocks))

    if "sort_col" not in st.session_state:
        st.session_state.sort_col = "pct_of_high"
        st.session_state.sort_dir = "desc"

    def toggle_sort(col: str):
        if st.session_state.sort_col == col:
            st.session_state.sort_dir = "asc" if st.session_state.sort_dir == "desc" else "desc"
        else:
            st.session_state.sort_col = col
            st.session_state.sort_dir = "desc"

    def arrow(col: str) -> str:
        if st.session_state.sort_col != col:
            return "⇅"
        return "▲" if st.session_state.sort_dir == "asc" else "▼"

    sort_cols = st.columns([2, 2, 2, 2, 1.3, 1.5, 1.5])
    with sort_cols[0]:
        st.button(f"Sort: Stock {arrow('name')}", key="sort_btn_stock", on_click=toggle_sort, args=("name",), use_container_width=True)
    with sort_cols[2]:
        st.button(f"Sort: % of 52W High {arrow('pct_of_high')}", key="sort_btn_pct", on_click=toggle_sort, args=("pct_of_high",), use_container_width=True)
    with sort_cols[3]:
        st.button(f"Sort: Day Change {arrow('day_change')}", key="sort_btn_daychange", on_click=toggle_sort, args=("day_change",), use_container_width=True)

    reverse = st.session_state.sort_dir == "desc"
    sort_col = st.session_state.sort_col
    if sort_col == "name":
        table_stocks = sorted(stocks, key=lambda s: s["name"].lower(), reverse=reverse)
    elif sort_col == "day_change":
        table_stocks = sorted(stocks, key=lambda s: s["day_change_pct"], reverse=reverse)
    else:
        table_stocks = sorted(
            stocks,
            key=lambda s: s["pct_of_high"] if not np.isnan(s["pct_of_high"]) else -1,
            reverse=reverse,
        )
    render_html(build_table(table_stocks))

    st.caption(
        "Data refreshes automatically every 5 minutes, or click 'Force refresh' in the sidebar for the latest values. "
        "This dashboard is for informational purposes only and is not investment advice."
    )


if __name__ == "__main__":
    main()
