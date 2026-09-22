import os
import sqlite3
import datetime
import io
import urllib.request
import urllib.parse
import ssl
import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import google.generativeai as genai

# ==========================================
# ページ初期設定
# ==========================================
st.set_page_config(
    page_title="株式売買管理システム - Streamlit & Gemini AI",
    page_icon="📈",
    layout="wide"
)

DB_PATH = str(Path(__file__).resolve().parent / "portfolio_data.db")

# ==========================================
# データベース初期化
# ==========================================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            trade_type TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            fee REAL DEFAULT 0,
            trade_date TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS current_prices (
            code TEXT PRIMARY KEY,
            current_price REAL NOT NULL,
            dividend_yield REAL DEFAULT 0.0,
            target_strategy TEXT DEFAULT '月5万スイング'
        )
    """)
    conn.commit()
    conn.close()

init_database()

# ==========================================
# ロジック・データ取得関数
# ==========================================
def fetch_latest_stock_news(stock_code, stock_name):
    """ Google ニュースから該当銘柄の直近ニュースの見出しを取得 """
    query = f"{stock_name} {stock_code} 株"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=ssl_context, timeout=5) as resp:
            xml_data = resp.read()

        soup = BeautifulSoup(xml_data, 'xml')
        items = soup.find_all('item')[:5]

        news_list = [item.title.text for item in items if item.title]
        return news_list
    except Exception as e:
        st.warning(f"ニュース取得エラー ({stock_code}): {e}")
        return []


def analyze_news_with_gemini(stock_name, news_headlines, api_key):
    """ 利用可能なGeminiモデルを自動検出してニュース分析を実施 """
    if not news_headlines:
        return {
            "sentiment_score": 0,
            "category": "ニュースなし",
            "reason": "直近のニュース見出しが取得できませんでした。"
        }

    if not api_key:
        return {
            "sentiment_score": 0,
            "category": "APIキー未設定",
            "reason": "Gemini APIキーが設定されていないため、ニュース感情分析をスキップしました。"
        }

    news_text = "\n".join([f"- {h}" for h in news_headlines])

    prompt = f"""
以下は「{stock_name}」に関する直近の最新ニュースの見出しです。
これらを分析し、株価に対する定性的な影響（好材料、不祥事、悪材料など）を評価してください。

ニュース見出し:
{news_text}

以下のJSONフォーマットのみで結果を出力してください。他の文章は一切含めないでください。
{{
    "sentiment_score": -100から100の数値（悪材料/不祥事はマイナス、好材料はプラス、中立は0）,
    "category": "絶好材料" または "好材料" または "中立" または "悪材料" または "重篤な悪材料・不祥事",
    "reason": "判定の理由を日本語で1~2文で簡潔に"
}}
"""

    genai.configure(api_key=api_key)

    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                available_models.append(name)
        
        flash_models = [m for m in available_models if "flash" in m]
        candidate_models = flash_models + [m for m in available_models if m not in flash_models]

    except Exception:
        candidate_models = ["gemini-1.5-flash-8b", "gemini-1.5-flash", "gemini-2.0-flash-exp"]

    last_error = ""

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)
            return data

        except Exception as e:
            last_error = str(e)
            continue

    return {
        "sentiment_score": 0,
        "category": "解析エラー",
        "reason": f"通信エラー: {last_error[:60]}"
    }

def render_gemini_diagnosis_section(api_key_input: str):
    """ Gemini AI ニュース統合リアルタイム個別診断を描画・実行する関数 """
    st.markdown("---")
    st.subheader("🤖 Gemini AI ニュース統合リアルタイム個別診断")
    
    col_diag_1, col_diag_2 = st.columns([1, 2])
    with col_diag_1:
        diag_code = st.text_input("銘柄コードを入力 (例: 7203)", key="diag_code_input")
        btn_run_diag = st.button("🤖 Gemini AI 診断を実行")

    if btn_run_diag and diag_code:
        with st.spinner(f"銘柄 {diag_code} のデータ取得＆Gemini AI ニュース解析中..."):
            tse_stocks = get_tse_stock_list()
            info = tse_stocks.get(diag_code, {})
            default_name = info.get("name", "") if isinstance(info, dict) else str(info)

            data = fetch_stock_full_data(diag_code, default_name=default_name)
            if not data:
                st.error(f"銘柄コード {diag_code} の株価データを取得できませんでした。")
            else:
                stock_name = data["name"]
                price = data["price"]

                news_headlines = fetch_latest_stock_news(diag_code, stock_name)
                ai_res = analyze_news_with_gemini(stock_name, news_headlines, api_key_input)

                f_growth = data["earnings_growth"] >= 10.0
                f_roe = data["roe"] >= 8.0
                f_per = 0.0 < data["per"] <= 15.0
                f_mcap = data["market_cap"] >= 100.0
                fundamentals_pass = f_growth and f_roe and f_per and f_mcap

                t_rsi = data["rsi14"] <= 40.0
                t_dev = data["deviation_rate"] <= -10.0
                technical_pass = t_rsi and t_dev

                sentiment_score = ai_res.get("sentiment_score", 0)
                ai_category = ai_res.get("category", "分類不能")
                ai_reason = ai_res.get("reason", "判定結果を取得できませんでした。")

                base_score = 50
                if fundamentals_pass: base_score += 20
                if technical_pass: base_score += 20

                total_score = base_score + (sentiment_score * 0.3)

                if sentiment_score <= -50:
                    signal = "⚠️ 危険・買わないこと (悪材料・不祥事検知)"
                    eval_msg = "最新ニュースに重大な悪材料や不祥事が検知されました。買わないことを推奨します。"
                elif total_score >= 80:
                    signal = "★ 絶好の買い場 (Sランク)"
                    eval_msg = "業績・テクニカル指標に加えて、最新ニュースも非常に好材料を示しています！"
                elif total_score >= 60:
                    signal = "〇 買い検討可能 (Aランク)"
                    eval_msg = "全体的に好条件が揃っています。"
                elif total_score >= 40:
                    signal = "△ 様子見推奨 (Bランク)"
                    eval_msg = "業績またはタイミングにやや難があるか、材料が限定的です。"
                else:
                    signal = "× 買い見送り (Cランク)"
                    eval_msg = "設定した投資基準を満たしていません。"

                suggested_stop_loss = min(price * 0.95, data["sma25"] * 0.98)

                st.success(f"**{stock_name} ({diag_code}) の診断結果**")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("総合判定", signal)
                m2.metric("総合スコア", f"{total_score:.0f} 点")
                m3.metric("現在株価", f"¥{price:,.1f}")

                st.info(f"**判定要約:** {eval_msg}")

                st.markdown(f"""
                ### 🤖 Gemini AI ニュース感情分析
                - **ニュース分類**: {ai_category} (感情スコア: {sentiment_score:+d})
                - **AI分析理由**: {ai_reason}

                ---
                ### 📊 ファンダメンタルズ判定 ({'合格' if fundamentals_pass else '要確認'})
                - **営業利益変化率**: {data['earnings_growth']:+.1f}%
                - **ROE**: {data['roe']:.1f}% | **PER**: {data['per']:.1f}倍

                ---
                ### 📈 テクニカル / 株価目安
                - **RSI (14日)**: {data['rsi14']:.1f}% | **25日乖離率**: {data['deviation_rate']:+.1f}%
                - **推奨損切ライン**: ¥{suggested_stop_loss:,.1f} 付近
                """)

@st.cache_data(ttl=86400)
def get_tse_stock_list():
    """ JPX公式の最新上場銘柄一覧（Excel）を取得（1日キャッシュ） """
    base_page = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        ssl_context = ssl._create_unverified_context()
        page_req = urllib.request.Request(base_page, headers=headers)
        with urllib.request.urlopen(page_req, context=ssl_context) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        match = re.search(r'href="([^"]*data_j\.(?:xls|xlsx))"', html)
        if match:
            target_path = match.group(1)
            excel_url = target_path if target_path.startswith("http") else "https://www.jpx.co.jp" + target_path
        else:
            excel_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

        file_req = urllib.request.Request(excel_url, headers=headers)
        with urllib.request.urlopen(file_req, context=ssl_context) as response:
            excel_data = response.read()

        try:
            df = pd.read_excel(io.BytesIO(excel_data))
        except Exception:
            df = pd.read_excel(io.BytesIO(excel_data), engine='openpyxl')

        code_col = [c for c in df.columns if "コード" in str(c)][0]
        name_col = [c for c in df.columns if "銘柄名" in str(c)][0]
        market_col = [c for c in df.columns if "市場" in str(c)][0]
    
        tse_dict = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            if len(code) == 4 and code.isdigit():
                tse_dict[code] = {
                    "name": str(row[name_col]).strip(),
                    "market": str(row[market_col]).strip()
                }
            
        if tse_dict:
            return tse_dict

    except Exception as e:
        st.write(f"JPX自動取得スキップ (フォールバックデータを使用): {e}")

    return {
        "7203": {"name": "トヨタ自動車", "market": "プライム"},
        "8306": {"name": "三菱UFJフィナンシャルG", "market": "プライム"},
        "9432": {"name": "NTT", "market": "プライム"},
        "4755": {"name": "楽天グループ", "market": "プライム"},
        "4592": {"name": "サンバイオ", "market": "グロース"},
        "2160": {"name": "ジーエヌアイグループ", "market": "グロース"},
        "7014": {"name": "名村造船所", "market": "スタンダード"}
    }

def fetch_stock_full_data(code, default_name=""):
    """ 株価データを安全に取得 """
    try:
        clean_code = str(code).replace(".T", "").replace(".t", "").strip()
        symbol = f"{clean_code}.T" if clean_code.isdigit() else clean_code

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d")
        
        if hist.empty or len(hist) < 1:
            return None

        latest_price = float(hist['Close'].iloc[-1])
        latest_volume = float(hist['Volume'].iloc[-1])
        trading_value = latest_price * latest_volume

        sma_window = min(len(hist), 25)
        sma25 = float(hist['Close'].rolling(window=sma_window).mean().iloc[-1])
        deviation_rate = ((latest_price - sma25) / sma25) * 100 if sma25 > 0 else 0.0

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=min(len(hist), 14)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=min(len(hist), 14)).mean()
        
        last_loss = loss.iloc[-1] if len(loss) > 0 else 0
        if last_loss == 0 or pd.isna(last_loss):
            rsi14 = 50.0
        else:
            rs = gain / loss
            rsi14 = float((100 - (100 / (1 + rs))).iloc[-1])

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        stock_name = default_name or info.get('longName') or info.get('shortName') or f"銘柄 {clean_code}"

        raw_div = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0.0
        raw_val = float(raw_div)
        if raw_val > 50.0:
            div_yield = raw_val / 100.0
        elif 0.0 < raw_val < 0.2:
            div_yield = raw_val * 100.0
        else:
            div_yield = raw_val
        div_yield = round(div_yield, 2)

        per = float(info.get('trailingPE') or info.get('forwardPE') or 0.0)
        roe = float(info.get('returnOnEquity') or 0.0) * 100
        operating_margin = float(info.get('operatingMargins') or 0.0) * 100
        payout_ratio = float(info.get('payoutRatio') or 0.0) * 100
        earnings_growth = float(info.get('earningsGrowth') or 0.0) * 100
        market_cap = float(info.get('marketCap') or 0.0) / 100_000_000

        total_assets = info.get('totalAssets', 0)
        total_equity = info.get('totalStockholderEquity', 0)
        equity_ratio = (total_equity / total_assets * 100) if total_assets and total_assets > 0 else 50.0

        return {
            "code": clean_code,
            "name": stock_name,
            "price": latest_price,
            "trading_value": trading_value,
            "sma25": sma25,
            "rsi14": rsi14,
            "deviation_rate": deviation_rate,
            "per": per,
            "roe": roe,
            "earnings_growth": earnings_growth,
            "div_yield": div_yield,
            "equity_ratio": equity_ratio,
            "operating_margin": operating_margin,
            "payout_ratio": payout_ratio,
            "market_cap": market_cap
        }
    except Exception:
        return None

def run_screening(target_market):
    """ 
    指定市場のスクリーニングを実行し、
    ①スイング：5件、②配当・長期保有（グロース時は高成長・長期）：5件 の計10件のDataFrameを返す
    """
    tse_stocks = get_tse_stock_list()
    filtered_stocks = {k: v for k, v in tse_stocks.items() if target_market in v.get("market", "")}

    swing_candidates = []
    income_candidates = []

    progress_bar = st.progress(0)
    items_list = list(filtered_stocks.items())[:30]  # 分析対象（上位30件）
    total_count = len(items_list)

    for idx, (code, info) in enumerate(items_list):
        data = fetch_stock_full_data(code, default_name=info.get("name", ""))
        if data:
            price = data["price"]

            # --- ① スイング評価ロジック ---
            stop_loss_swing = min(price * 0.95, data["sma25"] * 0.98)
            c1_val = (data["trading_value"] >= 300_000_000)
            c1_sma = (price > data["sma25"])
            c1_rsi = (40 <= data["rsi14"] <= 60)
            c1_growth = (data["earnings_growth"] >= 10.0)
            c1_per = (10.0 <= data["per"] <= 30.0)

            ai_macro_score_swing = 1.5 if (c1_growth and c1_sma) else 0.0
            swing_score = sum([c1_val, c1_sma, c1_rsi, c1_growth, c1_per]) + ai_macro_score_swing

            signal1 = "◎ 上昇トレンド" if c1_sma else "△ 様子見"
            advice1 = f"目標:¥{price*1.04:,.0f} (RSI:{data['rsi14']:.1f}%, PER:{data['per']:.1f}倍)"

            swing_candidates.append({
                "score": swing_score,
                "銘柄コード": code,
                "銘柄名": data["name"],
                "推奨目的": f"①スイング({target_market})",
                "現在株価": f"¥{price:,.1f}",
                "予想配当利回り": f"{data['div_yield']:.2f}%" if data["div_yield"] > 0 else "---",
                "買い時判定": signal1,
                "損切目安": f"¥{stop_loss_swing:,.1f}",
                "根拠/指標": advice1
            })

            # --- ② 配当・長期保有評価ロジック ---
            stop_loss_income = price * 0.93
            c2_yield = (2.5 <= data["div_yield"] <= 5.0) if target_market != "グロース" else (data["earnings_growth"] >= 15.0)
            c2_equity = (data["equity_ratio"] >= 40.0)
            c2_margin = (data["operating_margin"] >= 8.0)
            c2_payout = (20.0 <= data["payout_ratio"] <= 60.0)

            ai_macro_score_income = 2.0 if (c2_yield and c2_equity and c2_margin) else 0.0
            income_score = sum([c2_yield, c2_equity, c2_margin, c2_payout]) + ai_macro_score_income

            strat_label = "②高成長・長期" if target_market == "グロース" else "②配当・長期保有"
            signal2 = "★ 優良財務" if ai_macro_score_income > 0 else ("〇 安定保有" if c2_yield else "△ 保留")
            advice2 = f"利回り:{data['div_yield']:.2f}% (自己資本:{data['equity_ratio']:.0f}%, 営業益率:{data['operating_margin']:.1f}%)"

            income_candidates.append({
                "score": income_score,
                "銘柄コード": code,
                "銘柄名": data["name"],
                "推奨目的": strat_label,
                "現在株価": f"¥{price:,.1f}",
                "予想配当利回り": f"{data['div_yield']:.2f}%" if data["div_yield"] > 0 else "---",
                "買い時判定": signal2,
                "損切目安": f"¥{stop_loss_income:,.1f}",
                "根拠/指標": advice2
            })

        if total_count > 0:
            progress_bar.progress((idx + 1) / total_count)

    progress_bar.empty()

    # それぞれスコア順に降順ソートし、上位5件ずつ（計10件）を抽出
    swing_candidates.sort(key=lambda x: x["score"], reverse=True)
    income_candidates.sort(key=lambda x: x["score"], reverse=True)

    selected_swing = swing_candidates[:5]
    selected_income = income_candidates[:5]

    # 表示用から 'score' キーを除外して結合
    results = []
    for item in selected_swing + selected_income:
        item_copy = item.copy()
        del item_copy["score"]
        results.append(item_copy)

    return pd.DataFrame(results)

def refresh_holdings_current_prices():
    """ 保有銘柄の最新株価を取得して DB (current_prices) を更新 """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code, name FROM transactions")
    codes = cursor.fetchall()
    conn.close()

    if not codes:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for raw_code, name in codes:
        clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
        data = fetch_stock_full_data(clean_code, default_name=name)
        if data and "price" in data:
            price = data["price"]
            div = data.get("div_yield", 0.0)
            cursor.execute("""
                INSERT INTO current_prices (code, current_price, dividend_yield)
                VALUES (?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET current_price = excluded.current_price, dividend_yield = excluded.dividend_yield
            """, (clean_code, price, div))
    conn.commit()
    conn.close()


def calculate_portfolio():
    """ DBから取引履歴と最新株価を取得し、保有資産・損益状況を計算 """
    conn = sqlite3.connect(DB_PATH)
    df_trans = pd.read_sql_query("SELECT * FROM transactions ORDER BY trade_date ASC, id ASC", conn)
    df_prices = pd.read_sql_query("SELECT code, current_price FROM current_prices", conn)
    conn.close()

    price_dict = dict(zip(df_prices['code'], df_prices['current_price'])) if not df_prices.empty else {}

    if df_trans.empty:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    portfolio = {}
    realized_pl = 0.0

    for _, row in df_trans.iterrows():
        code = str(row['code']).replace(".T", "").strip()
        trade_type = str(row['trade_type']).strip()
        shares = float(row['shares'])
        price = float(row['price'])
        fee = float(row['fee'])

        if code not in portfolio:
            portfolio[code] = {"name": row['name'], "shares": 0.0, "total_cost": 0.0, "last_price": price}

        portfolio[code]["last_price"] = price

        if any(k in trade_type for k in ["買", "現買", "買付"]):
            portfolio[code]["shares"] += shares
            portfolio[code]["total_cost"] += (shares * price) + fee
        elif any(k in trade_type for k in ["売", "現売", "売却"]):
            if portfolio[code]["shares"] > 0:
                avg_cost = portfolio[code]["total_cost"] / portfolio[code]["shares"]
                cost_basis = avg_cost * shares
                realized_pl += ((shares * price) - fee) - cost_basis
                portfolio[code]["shares"] -= shares
                portfolio[code]["total_cost"] -= cost_basis

    rows = []
    total_market_val = 0.0
    total_cost_val = 0.0

    for code, p in portfolio.items():
        if p["shares"] <= 0:
            continue
        avg_price = p["total_cost"] / p["shares"]
        c_price = price_dict.get(code, p["last_price"])
        market_val = p["shares"] * c_price
        pl = market_val - p["total_cost"]
        ret_rate = (pl / p["total_cost"] * 100) if p["total_cost"] > 0 else 0.0

        total_market_val += market_val
        total_cost_val += p["total_cost"]

        rows.append({
            "銘柄コード": code,
            "銘柄名": p["name"],
            "保有株数": f"{int(p['shares']):,} 株",
            "平均取得単価": f"¥{avg_price:,.1f}",
            "現在値": f"¥{c_price:,.1f}",
            "評価額": f"¥{market_val:,.0f}",
            "評価損益": f"¥{pl:+,.0f}",
            "損益率": f"{ret_rate:+.2f}%"
        })

    unrealized_pl = total_market_val - total_cost_val
    return pd.DataFrame(rows), total_market_val, unrealized_pl, realized_pl


def update_transaction(t_id, trade_date, code, name, trade_type, shares, price, fee):
    """ 指定されたIDの取引履歴を更新 (③用) """
    clean_code = str(code).replace(".T", "").replace(".t", "").strip()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE transactions 
        SET trade_date = ?, code = ?, name = ?, trade_type = ?, shares = ?, price = ?, fee = ?
        WHERE id = ?
    """, (trade_date, clean_code, name, trade_type, shares, price, fee, t_id))
    conn.commit()
    conn.close()

def render_tab1_content(api_key_input: str):
    """ TAB 1: スクリーニング ＆ Gemini AI 診断を表示・実行する関数 """
    st.subheader("🎯 スクリーニング ＆ Gemini AI 診断")

    st.markdown("### 🔍 スクリーニング条件の設定")
    
    # --- 市場選択 ＆ 条件入力 ---
    target_market = st.selectbox("対象市場", ["すべて", "プライム", "スタンダード", "グロース"], index=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_growth = st.number_input("最小増益率 (%)", value=10.0, step=1.0)
        min_roe = st.number_input("最小 ROE (%)", value=8.0, step=0.5)
    with col2:
        max_per = st.number_input("上限 PER (倍)", value=15.0, step=0.5)
        min_mcap = st.number_input("最小時価総額 (億円)", value=100.0, step=10.0)
    with col3:
        max_rsi = st.number_input("上限 RSI (14日)", value=40.0, step=1.0)
        max_dev = st.number_input("上限 25日乖離率 (%)", value=-10.0, step=0.5)

    # --- スクリーニング実行処理 ---
    if st.button("🔎 スクリーニング実行", key="btn_run_screening"):
        with st.spinner("東証上場銘柄のデータ取得・スクリーニングを実行中..."):
            tse_stocks = get_tse_stock_list()
            results = []

            for code, info in tse_stocks.items():
                name = info.get("name", "") if isinstance(info, dict) else str(info)
                market = info.get("market", "") if isinstance(info, dict) else ""

                # 市場フィルタ判定
                if target_market != "すべて" and target_market not in market:
                    continue

                data = fetch_stock_full_data(code, default_name=name)
                if not data:
                    continue

                # 条件判定
                f_growth = data["earnings_growth"] >= min_growth
                f_roe = data["roe"] >= min_roe
                f_per = 0.0 < data["per"] <= max_per
                f_mcap = data["market_cap"] >= min_mcap
                t_rsi = data["rsi14"] <= max_rsi
                t_dev = data["deviation_rate"] <= max_dev

                if f_growth and f_roe and f_per and f_mcap and t_rsi and t_dev:
                    results.append({
                        "コード": code,
                        "銘柄名": data["name"],
                        "市場": market,
                        "現在株価": f"¥{data['price']:,.1f}",
                        "増益率(%)": f"{data['earnings_growth']:+.1f}%",
                        "ROE(%)": f"{data['roe']:.1f}%",
                        "PER(倍)": f"{data['per']:.1f}",
                        "時価総額(億円)": f"{data['market_cap']:,.0f}",
                        "RSI(14日)": f"{data['rsi14']:.1f}%",
                        "25日乖離率(%)": f"{data['deviation_rate']:+.1f}%"
                    })

            if results:
                st.success(f"スクリーニング結果: {len(results)} 件の銘柄が検出されました！")
                import pandas as pd
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.warning("設定条件に合致する銘柄は見つかりませんでした。条件を緩和してお試しください。")

    # --- 🤖 Gemini AI リアルタイム統合診断の呼び出し ---
    render_gemini_diagnosis_section(api_key_input)

# ==========================================
# サイドバー設定
# ==========================================
st.sidebar.title("⚙️ システム設定")
api_key_input = st.sidebar.text_input(
    "Gemini API キー",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password"
)

st.title("📈 株式売買管理 ＆ 東証・Gemini AI リアルタイム診断")

# タブ作成
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 おすすめ購入株・買い時診断",
    "📊 保有銘柄一覧・資産状況",
    "📜 売買取引履歴",
    "📝 新規登録・SBI証券CSV取込"
])

# ==========================================
# TAB 1: おすすめ購入株・買い時診断
# ==========================================
with tab1:
    st.subheader("🎯 市場別スクリーニング・おすすめ診断")
    
    col_m, col_btn = st.columns([2, 3])
    with col_m:
        market_choice = st.radio("対象市場を選択", ["プライム", "スタンダード", "グロース"], horizontal=True)
    with col_btn:
        st.write("") # レイアウト調整
        run_screening = st.button("🔍 選択した市場でスクリーニング実行", type="primary")

    if run_screening:
        tse_stocks = get_tse_stock_list()
        filtered_stocks = {
            code: info.get("name", "") if isinstance(info, dict) else str(info)
            for code, info in tse_stocks.items()
            if market_choice in (info.get("market", "") if isinstance(info, dict) else "")
        }

        # 先頭25件程度に制限して高速取得
        target_items = list(filtered_stocks.items())[:25]
        
        swing_candidates = []
        income_candidates = []

        progress_text = f"【{market_choice}市場】銘柄データを取得・診断中..."
        my_bar = st.progress(0, text=progress_text)

        for idx, (code, name) in enumerate(target_items, start=1):
            my_bar.progress(idx / len(target_items), text=f"【{market_choice}】取得中 ({idx}/{len(target_items)}): {code} {name}")
            data = fetch_stock_full_data(code, default_name=name)
            if not data:
                continue

            price = data["price"]
            stop_loss_swing = min(price * 0.95, data["sma25"] * 0.98)
            stop_loss_income = price * 0.93

            c1_val = (data["trading_value"] >= 300_000_000)
            c1_sma = (price > data["sma25"])
            c1_rsi = (40 <= data["rsi14"] <= 60)
            c1_growth = (data["earnings_growth"] >= 10.0)
            c1_per = (10.0 <= data["per"] <= 30.0)

            ai_macro_score_swing = 1.5 if (c1_growth and c1_sma) else 0.0
            swing_score = sum([c1_val, c1_sma, c1_rsi, c1_growth, c1_per]) + ai_macro_score_swing

            signal1 = "◎ 上昇トレンド" if c1_sma else "△ 様子見"
            advice1 = f"目標:¥{price*1.04:,.0f} (RSI:{data['rsi14']:.1f}%, PER:{data['per']:.1f}倍)"
            swing_candidates.append((swing_score, code, data["name"], f"①スイング({market_choice})", price, data["div_yield"], signal1, f"¥{stop_loss_swing:,.1f}", advice1))

            c2_yield = (2.5 <= data["div_yield"] <= 5.0) if market_choice != "グロース" else (data["earnings_growth"] >= 15.0)
            c2_equity = (data["equity_ratio"] >= 40.0)
            c2_margin = (data["operating_margin"] >= 8.0)
            c2_payout = (20.0 <= data["payout_ratio"] <= 60.0)

            ai_macro_score_income = 2.0 if (c2_yield and c2_equity and c2_margin) else 0.0
            income_score = sum([c2_yield, c2_equity, c2_margin, c2_payout]) + ai_macro_score_income

            strat_label = "②高成長・長期" if market_choice == "グロース" else "②配当・長期保有"
            signal2 = "★ 優良財務" if ai_macro_score_income > 0 else ("〇 安定保有" if c2_yield else "△ 保留")
            advice2 = f"利回り:{data['div_yield']:.2f}% (自己資本:{data['equity_ratio']:.0f}%, 営業益率:{data['operating_margin']:.1f}%)"
            income_candidates.append((income_score, code, data["name"], strat_label, price, data["div_yield"], signal2, f"¥{stop_loss_income:,.1f}", advice2))

        my_bar.empty()

        swing_candidates.sort(key=lambda x: x[0], reverse=True)
        income_candidates.sort(key=lambda x: x[0], reverse=True)
        selected_items = swing_candidates[:5] + income_candidates[:5]

        res_data = []
        for item in selected_items:
            _, code, s_name, strategy, price, div, signal, stop_loss_str, advice = item
            res_data.append({
                "銘柄コード": code,
                "銘柄名": s_name,
                "推奨目的": strategy,
                "現在株価": f"¥{price:,.1f}",
                "予想配当利回り": f"{div:.2f}%" if div > 0 else "---",
                "買い時判定": signal,
                "損切ライン(目安)": stop_loss_str,
                "判定理由 / スクリーニング根拠": advice
            })

        st.session_state["screening_results"] = pd.DataFrame(res_data)

    if "screening_results" in st.session_state and not st.session_state["screening_results"].empty:
        st.dataframe(st.session_state["screening_results"], use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Gemini AI ニュース統合リアルタイム個別診断")
    
    col_diag_1, col_diag_2 = st.columns([1, 2])
    with col_diag_1:
        diag_code = st.text_input("銘柄コードを入力 (例: 7203)", key="diag_code_input")
        btn_run_diag = st.button("🤖 Gemini AI 診断を実行")

    if btn_run_diag and diag_code:
        with st.spinner(f"銘柄 {diag_code} のデータ取得＆Gemini AI ニュース解析中..."):
            tse_stocks = get_tse_stock_list()
            info = tse_stocks.get(diag_code, {})
            default_name = info.get("name", "") if isinstance(info, dict) else str(info)

            data = fetch_stock_full_data(diag_code, default_name=default_name)
            if not data:
                st.error(f"銘柄コード {diag_code} の株価データを取得できませんでした。")
            else:
                stock_name = data["name"]
                price = data["price"]

                news_headlines = fetch_latest_stock_news(diag_code, stock_name)
                ai_res = analyze_news_with_gemini(stock_name, news_headlines, api_key_input)

                f_growth = data["earnings_growth"] >= 10.0
                f_roe = data["roe"] >= 8.0
                f_per = 0.0 < data["per"] <= 15.0
                f_mcap = data["market_cap"] >= 100.0
                fundamentals_pass = f_growth and f_roe and f_per and f_mcap

                t_rsi = data["rsi14"] <= 40.0
                t_dev = data["deviation_rate"] <= -10.0
                technical_pass = t_rsi and t_dev

                sentiment_score = ai_res.get("sentiment_score", 0)
                ai_category = ai_res.get("category", "分類不能")
                ai_reason = ai_res.get("reason", "判定結果を取得できませんでした。")

                base_score = 50
                if fundamentals_pass: base_score += 20
                if technical_pass: base_score += 20

                total_score = base_score + (sentiment_score * 0.3)

                if sentiment_score <= -50:
                    signal = "⚠️ 危険・買わないこと (悪材料・不祥事検知)"
                    eval_msg = "最新ニュースに重大な悪材料や不祥事が検知されました。買わないことを推奨します。"
                    badge_color = "red"
                elif total_score >= 80:
                    signal = "★ 絶好の買い場 (Sランク)"
                    eval_msg = "業績・テクニカル指標に加えて、最新ニュースも非常に好材料を示しています！"
                    badge_color = "green"
                elif total_score >= 60:
                    signal = "〇 買い検討可能 (Aランク)"
                    eval_msg = "全体的に好条件が揃っています。"
                    badge_color = "green"
                elif total_score >= 40:
                    signal = "△ 様子見推奨 (Bランク)"
                    eval_msg = "業績またはタイミングにやや難があるか、材料が限定的です。"
                    badge_color = "orange"
                else:
                    signal = "× 買い見送り (Cランク)"
                    eval_msg = "設定した投資基準を満たしていません。"
                    badge_color = "red"

                suggested_stop_loss = min(price * 0.95, data["sma25"] * 0.98)

                st.success(f"**{stock_name} ({diag_code}) の診断結果**")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("総合判定", signal)
                m2.metric("総合スコア", f"{total_score:.0f} 点")
                m3.metric("現在株価", f"¥{price:,.1f}")

                st.info(f"**判定要約:** {eval_msg}")

                st.markdown(f"""
                ### 🤖 Gemini AI ニュース感情分析
                - **ニュース分類**: {ai_category} (感情スコア: {sentiment_score:+d})
                - **AI分析理由**: {ai_reason}

                ---
                ### 📊 ファンダメンタルズ判定 ({'合格' if fundamentals_pass else '要確認'})
                - **営業利益変化率**: {data['earnings_growth']:+.1f}%
                - **ROE**: {data['roe']:.1f}% | **PER**: {data['per']:.1f}倍

                ---
                ### 📈 テクニカル / 株価目安
                - **RSI (14日)**: {data['rsi14']:.1f}% | **25日乖離率**: {data['deviation_rate']:+.1f}%
                - **推奨損切ライン**: ¥{suggested_stop_loss:,.1f} 付近
                """)

# ==========================================
# TAB 2: 保有銘柄一覧・資産状況
# ==========================================
with tab2:
    st.subheader("📊 保有銘柄一覧・資産状況")

    def load_portfolio_data(fetched_prices=None):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code, name, trade_type, shares, price, fee FROM transactions ORDER BY trade_date ASC, id ASC")
            rows = cursor.fetchall()

        portfolio = {}
        realized_pl = 0.0

        for raw_code, name, t_type, shares, price, fee in rows:
            clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
            
            if clean_code not in portfolio:
                portfolio[clean_code] = {
                    "raw_code": raw_code,
                    "name": name,
                    "shares": 0.0,
                    "total_cost": 0.0,
                    "last_price": price
                }

            portfolio[clean_code]["last_price"] = price
            t_str = str(t_type).strip()

            if any(k in t_str for k in ["買", "現買", "買付"]):
                buy_cost = (shares * price) + fee
                portfolio[clean_code]["shares"] += shares
                portfolio[clean_code]["total_cost"] += buy_cost

            elif any(k in t_str for k in ["売", "現売", "売却"]):
                current_shares = portfolio[clean_code]["shares"]
                if current_shares > 0:
                    avg_p = portfolio[clean_code]["total_cost"] / current_shares
                    sell_proceeds = (shares * price) - fee
                    cost_basis = avg_p * shares
                    realized_pl += (sell_proceeds - cost_basis)

                    portfolio[clean_code]["shares"] -= shares
                    portfolio[clean_code]["total_cost"] -= cost_basis
                    if portfolio[clean_code]["shares"] <= 0:
                        portfolio[clean_code]["shares"] = 0.0
                        portfolio[clean_code]["total_cost"] = 0.0

        table_rows = []
        total_market_val = 0.0
        total_cost_val = 0.0

        for clean_code, data in portfolio.items():
            shares = data["shares"]
            if shares <= 0:
                continue

            avg_price = data["total_cost"] / shares if shares > 0 else 0.0
            raw_code = data["raw_code"]

            c_price = None
            if fetched_prices:
                c_price = (
                    fetched_prices.get(clean_code) or 
                    fetched_prices.get(f"{clean_code}.T") or 
                    fetched_prices.get(str(raw_code).strip())
                )
            
            if c_price is None or c_price <= 0:
                c_price = data["last_price"]

            cost_val = data["total_cost"]
            market_val = shares * c_price
            pl = market_val - cost_val
            ret_rate = (pl / cost_val * 100) if cost_val > 0 else 0.0

            total_market_val += market_val
            total_cost_val += cost_val

            table_rows.append({
                "銘柄コード": clean_code,
                "銘柄名": data["name"],
                "保有株数": int(shares) if shares.is_integer() else round(shares, 2),
                "平均取得単価": avg_price,
                "現在値": c_price,
                "評価額": market_val,
                "評価損益": pl,
                "損益率 (%)": ret_rate
            })

        total_unrealized_pl = total_market_val - total_cost_val
        total_unrealized_rate = (total_unrealized_pl / total_cost_val * 100) if total_cost_val > 0 else 0.0

        return table_rows, total_market_val, total_unrealized_pl, total_unrealized_rate, realized_pl

    # 株価取得用フラグ
    if "fetched_holdings_prices" not in st.session_state:
        st.session_state["fetched_holdings_prices"] = None

    if st.button("🔄 保有株の最新価格を一括更新"):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT code, name FROM transactions")
            codes = cursor.fetchall()

        if codes:
            updated_prices = {}
            p_bar = st.progress(0, text="保有銘柄の最新価格を取得中...")
            for idx, (raw_code, name) in enumerate(codes, start=1):
                clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
                p_bar.progress(idx / len(codes), text=f"保有株取得中 ({idx}/{len(codes)}): {clean_code}")
                
                data = fetch_stock_full_data(clean_code, default_name=name)
                if data and "price" in data:
                    price = data["price"]
                    updated_prices[clean_code] = price
            p_bar.empty()
            st.session_state["fetched_holdings_prices"] = updated_prices
            st.success("保有銘柄の株価を最新に更新しました！")

    rows, total_mval, total_upl, total_urate, realized_pl = load_portfolio_data(
        st.session_state.get("fetched_holdings_prices")
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("総評価額", f"¥{total_mval:,.0f}")
    c2.metric("評価損益", f"¥{total_upl:+,.0f}", f"{total_urate:+.2f}%")
    c3.metric("確定損益", f"¥{realized_pl:+,.0f}")

    if rows:
        df_portfolio = pd.DataFrame(rows)
        st.dataframe(
            df_portfolio.style.format({
                "平均取得単価": "¥{:,.1f}",
                "現在値": "¥{:,.1f}",
                "評価額": "¥{:,.0f}",
                "評価損益": "¥{:+,.0f}",
                "損益率 (%)": "{:+.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.info("保有している株式データがありません。")

# ==========================================
# TAB 3: 📜 売買取引履歴
# ==========================================
with tab3:
    st.subheader("📜 売買取引履歴")

    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query("SELECT id, trade_date, code, name, trade_type, shares, price, fee FROM transactions ORDER BY trade_date DESC, id DESC", conn)
    conn.close()

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ 履歴の操作")
        
        tab3_op1, tab3_op2 = st.tabs(["✏️ 選択した履歴を編集", "🗑️ 履歴の削除"])

        # ③ 選択した履歴の編集機能
        with tab3_op1:
            edit_id = st.number_input("編集する履歴IDを入力", min_value=1, step=1, key="edit_id_input")
            target_row = df_history[df_history["id"] == edit_id]

            if not target_row.empty:
                row_data = target_row.iloc[0]
                with st.form("edit_transaction_form"):
                    st.write(f"**履歴ID {edit_id} の編集**")
                    try:
                        default_date = datetime.datetime.strptime(str(row_data["trade_date"]), "%Y-%m-%d").date()
                    except Exception:
                        default_date = datetime.date.today()

                    e_date = st.date_input("取引日", default_date)
                    e_code = st.text_input("銘柄コード", value=str(row_data["code"]))
                    e_name = st.text_input("銘柄名", value=str(row_data["name"]))
                    
                    type_idx = 0 if row_data["trade_type"] in ["買付", "現買", "買"] else 1
                    e_type = st.selectbox("取引区分", ["買付", "売却"], index=type_idx)
                    
                    e_shares = st.number_input("株数", min_value=1, value=int(row_data["shares"]))
                    e_price = st.number_input("取引単価 (円)", min_value=0.0, value=float(row_data["price"]))
                    e_fee = st.number_input("手数料 (円)", min_value=0.0, value=float(row_data["fee"]))

                    btn_update = st.form_submit_button("💾 変更を保存する", type="primary")

                    if btn_update:
                        update_transaction(
                            edit_id,
                            e_date.strftime("%Y-%m-%d"),
                            e_code,
                            e_name,
                            e_type,
                            e_shares,
                            e_price,
                            e_fee
                        )
                        st.success(f"ID: {edit_id} のデータを更新しました！")
                        st.rerun()
            else:
                st.info("編集対象の履歴IDを指定してください。")

        # 履歴の削除エリア
        with tab3_op2:
            col_del1, col_del2 = st.columns([2, 1])
            with col_del1:
                target_id = st.number_input("削除する履歴IDを指定", min_value=1, step=1, key="delete_id_input")
                if st.button("🗑️ 指定した履歴を削除"):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM transactions WHERE id = ?", (target_id,))
                    conn.commit()
                    conn.close()
                    st.success(f"ID: {target_id} の履歴を削除しました。")
                    st.rerun()

            with col_del2:
                st.write("⚠️ 全データクリア")
                if st.button("⚠️ 取引履歴を全削除", type="primary"):
                    st.session_state["confirm_delete_all"] = True

            if st.session_state.get("confirm_delete_all", False):
                st.warning("本当に全ての取引履歴を削除しますか？この操作は元に戻せません。")
                c_yes, c_no = st.columns(2)
                if c_yes.button("はい（全削除を実行）"):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM transactions")
                    conn.commit()
                    conn.close()
                    st.session_state["confirm_delete_all"] = False
                    st.success("全ての取引履歴を削除しました。")
                    st.rerun()
                if c_no.button("キャンセル"):
                    st.session_state["confirm_delete_all"] = False
                    st.rerun()
    else:
        st.info("取引履歴はありません。")

# ==========================================
# TAB 4: 📝 新規登録・SBI証券CSV取込
# ==========================================
with tab4:
    st.subheader("🏦 SBI証券 CSVファイル一括読み込み")
    uploaded_file = st.file_uploader("SBI証券からダウンロードしたCSVファイルを選択してください", type=["csv"])

    if uploaded_file is not None:
        if st.button("📁 CSVファイルを取り込む"):
            try:
                # 文字コード判定・読み込み
                bytes_data = uploaded_file.getvalue()
                lines = []
                for enc in ["cp932", "utf-8", "shift_jis"]:
                    try:
                        lines = bytes_data.decode(enc).splitlines()
                        if lines:
                            break
                    except Exception:
                        continue

                imported_count = 0
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                for line in lines:
                    row = [col.strip() for col in line.split(",")]
                    if not row or len(row) < 10:
                        continue

                    trade_date_raw = row[0].replace('"', '')
                    stock_name = row[1].replace('"', '')
                    stock_code = row[2].replace('"', '')
                    trade_type_raw = row[4].replace('"', '')

                    if "買" in trade_type_raw:
                        trade_type = "買付"
                    elif "売" in trade_type_raw:
                        trade_type = "売却"
                    else:
                        continue

                    try:
                        shares = int(float(row[8].replace(",", "").replace('"', '')))
                        price = float(row[9].replace(",", "").replace('"', ''))
                        fee_str = row[10].replace(",", "").replace('"', '')
                        fee = float(fee_str) if fee_str and fee_str != "--" else 0.0
                        trade_date = trade_date_raw.replace("/", "-")

                        cursor.execute("""
                            INSERT INTO transactions (code, name, trade_type, shares, price, fee, trade_date)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (stock_code, stock_name, trade_type, shares, price, fee, trade_date))
                        imported_count += 1
                    except ValueError:
                        continue

                conn.commit()
                conn.close()

                if imported_count > 0:
                    st.success(f"{imported_count} 件の取引履歴を正常に取り込みました！")
                else:
                    st.warning("取り込み対象のデータが見つかりませんでした。フォーマットを確認してください。")

            except Exception as e:
                st.error(f"CSV読み込み中にエラーが発生しました: {e}")

    st.markdown("---")
    st.subheader("📝 個別取引の手動登録")
    
    with st.form("manual_add_form", clear_on_submit=True):
        f_date = st.date_input("取引日", datetime.date.today())
        f_code = st.text_input("銘柄コード", placeholder="例: 7203")
        f_name = st.text_input("銘柄名", placeholder="例: トヨタ自動車")
        f_type = st.selectbox("取引区分", ["買付", "売却"])
        f_shares = st.number_input("株数", min_value=1, step=100, value=100)
        f_price = st.number_input("取引単価 (円)", min_value=0.0, step=1.0)
        f_fee = st.number_input("手数料 (円)", min_value=0.0, step=1.0, value=0.0)

        submitted = st.form_submit_button("取引を登録する", type="primary")

        if submitted:
            if not (f_code and f_name and f_shares > 0 and f_price > 0):
                st.error("入力内容に不備があります。必須項目を正しく入力してください。")
            else:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO transactions (code, name, trade_type, shares, price, fee, trade_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (f_code, f_name, f_type, f_shares, f_price, f_fee, f_date.strftime("%Y-%m-%d")))
                conn.commit()
                conn.close()
                st.success(f"{f_name} ({f_code}) の取引データを登録しました。")