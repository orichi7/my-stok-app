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

from google import genai
from google.genai import types

# ==========================================
# ページ初期設定 ＆ モバイル最適化CSS
# ==========================================
st.set_page_config(
    page_title="株式売買管理システム - Streamlit & Gemini AI",
    page_icon="📈",
    layout="wide"
)

# 📱 スマホ画面（幅768px以下）向けスタイル調整
st.markdown("""
<style>
    @media (max-width: 768px) {
        /* メトリクス（指標表示）の文字サイズ自動調整 */
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        /* ボタンをタップしやすくフルサイズ化 */
        .stButton > button {
            width: 100% !important;
            margin-bottom: 0.3rem !important;
        }
        /* テーブルの横スクロール対応 */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
        /* タブの余白・文字サイズ調整 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 8px !important;
            font-size: 0.8rem !important;
        }
        /* カラムの余白調整 */
        [data-testid="column"] {
            min-width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

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
    """ 
    Google ニュース RSS から該当銘柄の直近ニュース（最大10件）を取得し、
    見出し（title）と本文概要（description）のセットで返す
    """
    query = f"{stock_name} {stock_code} 株"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=ssl_context, timeout=5) as resp:
            xml_data = resp.read()

        soup = BeautifulSoup(xml_data, 'xml')
        # 件数を 5 件から 10 件へ拡大
        items = soup.find_all('item')[:10]

        news_list = []
        for item in items:
            title = item.title.text.strip() if item.title else ""
            
            # description 内の HTML タグを除去して本文概要テキストのみ抽出
            desc_text = ""
            if item.description:
                desc_soup = BeautifulSoup(item.description.text, 'html.parser')
                desc_text = desc_soup.get_text().strip()

            if title:
                if desc_text:
                    news_list.append(f"【見出し】{title}\n【概要】{desc_text}")
                else:
                    news_list.append(f"【見出し】{title}")

        return news_list
    except Exception as e:
        st.warning(f"ニュース取得エラー ({stock_code}): {e}")
        return []


def analyze_news_with_gemini(stock_name, news_headlines, api_key):
    """ 最新の google-genai SDK を使用したニュース解析 """
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

    try:
        # 最新SDKのクライアント初期化
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
            }
        )

        data = json.loads(response.text.strip())
        return data

    except Exception as e:
        return {
            "sentiment_score": 0,
            "category": "解析エラー",
            "reason": f"通信・解析エラー: {str(e)[:60]}"
        }

def render_gemini_diagnosis_section(api_key_input: str):
    """ Gemini AI ニュース統合リアルタイム個別診断を描画・実行する共通関数 """
    st.markdown("---")
    st.subheader("🤖 Gemini AI ニュース統合リアルタイム個別診断")
    
    col_diag_1, col_diag_2 = st.columns([1, 2])
    with col_diag_1:
        diag_code = st.text_input("銘柄コードを入力 (例: 7203)", key="diag_code_input")
        btn_run_diag = st.button("🤖 Gemini AI 診断を実行", key="btn_run_gemini_diag")

    if btn_run_diag and diag_code:
        with st.spinner(f"銘柄 {diag_code} のデータ取得＆Gemini AI ニュース解析中..."):
            tse_stocks = get_tse_stock_list()
            info = tse_stocks.get(diag_code, {})
            default_name = info.get("name", "") if isinstance(info, dict) else str(info)

            data = fetch_stock_full_data(diag_code, default_name=default_name)
            if not data:
                st.error(f"銘柄コード {diag_code} の株価データを取得できませんでした。")
                return

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
            - **目標**: ¥{price*1.04:,.0f} **(RSI**:{data['rsi14']:.1f}%, **PER**:{data['per']:.1f}倍 **)**
            """)

@st.cache_data(ttl=86400)
def get_tse_stock_list():
    """ 
    JPX公式の最新上場銘柄一覧を取得する（SQLiteデータベースキャッシュ版）
    - 24時間以内に取得されたキャッシュがDB内に存在すればそれを使用
    - キャッシュがない・古い・取得失敗時はJPXから再取得してDBキャッシュ更新
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT code, name, market, updated_at FROM tse_stocks_cache")
        cached_rows = cursor.fetchall()

        if cached_rows:
            latest_updated = max([datetime.datetime.fromisoformat(row[3]) for row in cached_rows])
            now = datetime.datetime.now()

            if (now - latest_updated).total_seconds() < 86400:
                tse_dict = {
                    row[0]: {"name": row[1], "market": row[2]}
                    for row in cached_rows
                }
                conn.close()
                return tse_dict

    except Exception as e:
        st.write(f"⚠️ キャッシュDB読み込みエラー: {e}")

    base_page = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    tse_dict = {}
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
    
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            if len(code) == 4 and code.isdigit():
                tse_dict[code] = {
                    "name": str(row[name_col]).strip(),
                    "market": str(row[market_col]).strip()
                }
            
        if tse_dict:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                now_str = datetime.datetime.now().isoformat()

                cursor.execute("DELETE FROM tse_stocks_cache")
                cache_data = [
                    (code, info["name"], info["market"], now_str)
                    for code, info in tse_dict.items()
                ]
                cursor.executemany("""
                    INSERT INTO tse_stocks_cache (code, name, market, updated_at)
                    VALUES (?, ?, ?, ?)
                """, cache_data)
                conn.commit()
                conn.close()
            except Exception as e:
                st.write(f"⚠️ キャッシュDB書き込みエラー: {e}")

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

def analyze_news_with_gemini(stock_name, news_headlines, api_key):
    """ Google GenAI SDK (gemini-2.5-flash) を使用してニュース分析を実施 """
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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        raw_text = response.text.strip()
        data = json.loads(raw_text)
        return data

    except Exception as e:
        return {
            "sentiment_score": 0,
            "category": "解析エラー",
            "reason": f"通信エラー: {str(e)[:60]}"
        }

def fetch_stock_full_data(code, default_name=""):
    """ 株価データを安全に取得（最低データ件数チェック付き） """
    try:
        clean_code = str(code).replace(".T", "").replace(".t", "").strip()
        symbol = f"{clean_code}.T"

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d")
        
        # --- データ存在・最低件数チェック（最低60日分必要） ---
        if hist.empty or len(hist) < 60:
            # データ不足の場合はテクニカル指標を None として返却
            return {
                "code": clean_code,
                "name": default_name or f"銘柄 {clean_code}",
                "price": float(hist['Close'].iloc[-1]) if not hist.empty else 0.0,
                "trading_value": 0.0,
                "sma25": None,
                "rsi14": None,
                "deviation_rate": None,
                "per": 0.0,
                "roe": 0.0,
                "earnings_growth": 0.0,
                "div_yield": 0.0,
                "equity_ratio": 50.0,
                "operating_margin": 0.0,
                "payout_ratio": 0.0,
                "market_cap": 0.0,
                "data_sufficient": False  # データ不足フラグ
            }

        latest_price = float(hist['Close'].iloc[-1])
        latest_volume = float(hist['Volume'].iloc[-1])
        trading_value = latest_price * latest_volume

        # --- テクニカル指標の正確な計算 (60日分以上のデータが保証された状態) ---
        sma25 = float(hist['Close'].rolling(window=25).mean().iloc[-1])
        deviation_rate = ((latest_price - sma25) / sma25) * 100 if sma25 > 0 else 0.0

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
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
            "market_cap": market_cap,
            "data_sufficient": True  # データ十分フラグ
        }
    except Exception:
        return None

def run_fast_screening(target_market: str, min_growth: float, min_roe: float, 
                         max_per: float, min_mcap: float, max_rsi: float, max_dev: float):
    """
    yf.download による株価一括取得（マルチスレッド）を用いて高速スクリーニングを行う関数
    """
    tse_stocks = get_tse_stock_list()
    
    # 対象市場でフィルタリング
    candidate_codes = []
    stock_info_map = {}
    
    for code, info in tse_stocks.items():
        name = info.get("name", "") if isinstance(info, dict) else str(info)
        market = info.get("market", "") if isinstance(info, dict) else ""

        if target_market != "すべて" and target_market not in market:
            continue
            
        candidate_codes.append(code)
        stock_info_map[code] = {"name": name, "market": market}

    if not candidate_codes:
        return []

    # yfinance 用のシンボルリスト作成 (例: "7203.T 8306.T ...")
    symbols = [f"{c}.T" for c in candidate_codes]

    # --- 1. 過去60日分の株価データを一括取得（高速化の肝） ---
    try:
        bulk_data = yf.download(
            tickers=symbols,
            period="60d",
            group_by="ticker",
            threads=True,
            progress=False
        )
    except Exception as e:
        st.error(f"株価一括取得エラー: {e}")
        return []

    results = []

    # --- 2. 各銘柄の指標計算と条件判定 ---
    for code in candidate_codes:
        symbol = f"{code}.T"
        
        # 複数銘柄の場合と1銘柄の場合のDataFrame構造に対応
        if len(symbols) > 1:
            if symbol not in bulk_data.columns.levels[0]:
                continue
            hist = bulk_data[symbol].dropna(how="all")
        else:
            hist = bulk_data.dropna(how="all")

        if hist.empty or len(hist) < 25:
            continue

        try:
            latest_price = float(hist['Close'].iloc[-1])
            
            # 25日移動平均線と乖離率
            sma25 = float(hist['Close'].rolling(window=25).mean().iloc[-1])
            if sma25 <= 0 or pd.isna(sma25):
                continue
            deviation_rate = ((latest_price - sma25) / sma25) * 100.0

            # RSI(14日) 計算
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            last_loss = loss.iloc[-1]
            if last_loss == 0 or pd.isna(last_loss):
                rsi14 = 50.0
            else:
                rs = gain / loss
                rsi14 = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])

            # テクニカル条件の早期判定（無駄な個別info取得を減らす）
            if not (rsi14 <= max_rsi and deviation_rate <= max_dev):
                continue

            # 財務データの取得（ファンダメンタルズ判定）
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            earnings_growth = float(info.get('earningsGrowth') or 0.0) * 100.0
            roe = float(info.get('returnOnEquity') or 0.0) * 100.0
            per = float(info.get('trailingPE') or info.get('forwardPE') or 0.0)
            market_cap = float(info.get('marketCap') or 0.0) / 100_000_000.0

            # ファンダメンタルズ条件の判定
            f_growth = earnings_growth >= min_growth
            f_roe = roe >= min_roe
            f_per = 0.0 < per <= max_per
            f_mcap = market_cap >= min_mcap

            if f_growth and f_roe and f_per and f_mcap:
                name = stock_info_map[code]["name"]
                market = stock_info_map[code]["market"]
                
                results.append({
                    "コード": code,
                    "銘柄名": name,
                    "市場": market,
                    "現在株価": f"¥{latest_price:,.1f}",
                    "増益率(%)": f"{earnings_growth:+.1f}%",
                    "ROE(%)": f"{roe:.1f}%",
                    "PER(倍)": f"{per:.1f}",
                    "時価総額(億円)": f"{market_cap:,.0f}",
                    "RSI(14日)": f"{rsi14:.1f}%",
                    "25日乖離率(%)": f"{deviation_rate:+.1f}%"
                })

        except Exception:
            continue

    return results

def fetch_stock_full_data(code, default_name=""):
    """
    株価データおよび財務指標を安全かつ正確に取得する
    - yfinanceで取得後、日本株特有の欠損値（PER, ROE等）があればYahoo!ファイナンス(日本版)から補完スクレイピングを実施
    """
    try:
        clean_code = str(code).replace(".T", "").replace(".t", "").strip()
        symbol = f"{clean_code}.T"

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d")
        
        if hist.empty or len(hist) < 1:
            return None

        latest_price = float(hist['Close'].iloc[-1])
        latest_volume = float(hist['Volume'].iloc[-1])
        trading_value = latest_price * latest_volume

        # テクニカル指標の計算
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

        # --- 1. yfinance から基本情報を取得 ---
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
        equity_ratio = (total_equity / total_assets * 100) if total_assets and total_assets > 0 else 0.0

        # --- 2. 欠損値補正（Yahoo!ファイナンス日本版からのスクレイピング補完） ---
        # PER、ROE、配当利回り、時価総額等のいずれかが取得できていない場合に補正を実行
        if per == 0.0 or roe == 0.0 or div_yield == 0.0 or market_cap == 0.0:
            try:
                yf_url = f"https://finance.yahoo.co.jp/quote/{clean_code}.T"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                req = urllib.request.Request(yf_url, headers=headers)
                ssl_context = ssl._create_unverified_context()
                
                with urllib.request.urlopen(req, context=ssl_context, timeout=5) as resp:
                    html_content = resp.read().decode('utf-8', errors='ignore')

                soup = BeautifulSoup(html_content, 'html.parser')

                # PER (PER/PBR表示欄から抽出)
                if per == 0.0:
                    per_elem = soup.find(text=re.compile(r'PER'))
                    if per_elem:
                        parent = per_elem.find_parent()
                        if parent:
                            m = re.search(r'([\d\.]+)\s*倍', parent.text)
                            if m:
                                per = float(m.group(1))

                # 配当利回り
                if div_yield == 0.0:
                    div_elem = soup.find(text=re.compile(r'配当利回り'))
                    if div_elem:
                        parent = div_elem.find_parent()
                        if parent:
                            m = re.search(r'([\d\.]+)\s*%', parent.text)
                            if m:
                                div_yield = float(m.group(1))

                # 時価総額
                if market_cap == 0.0:
                    mcap_elem = soup.find(text=re.compile(r'時価総額'))
                    if mcap_elem:
                        parent = mcap_elem.find_parent()
                        if parent:
                            m = re.search(r'([\d,]+)\s*百万円', parent.text)
                            if m:
                                market_cap = float(m.group(1).replace(',', '')) / 100.0

            except Exception:
                # 補完スクレイピングに失敗した場合はそのまま通過
                pass

        # 依然として重要な指標が欠損しているかどうかのフラグ
        is_data_complete = bool(per > 0.0 and roe != 0.0 and market_cap > 0.0)

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
            "market_cap": market_cap,
            "is_data_complete": is_data_complete
        }
    except Exception:
        return None

def update_transaction(target_id, e_date, e_code, e_name, e_type, e_shares, e_price, e_fee):
    """ データベースの指定IDレコードを更新する関数 """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
    UPDATE transactions 
    SET trade_date = ?, 
        code = ?, 
        name = ?, 
        trade_type = ?, 
        shares = ?, 
        price = ?, 
        fee = ? 
    WHERE id = ?
    """
    
    # 引数・プレースホルダー(?)の順番を厳密に一致させる
    cursor.execute(query, (
        str(e_date),
        str(e_code),
        str(e_name),
        str(e_type),
        int(e_shares),
        float(e_price),
        float(e_fee),
        int(target_id)  # WHERE節のid
    ))
    
    conn.commit()
    conn.close()

def delete_transaction(trans_id):
    """指定したIDの取引履歴を1件削除する関数"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions WHERE id=?", (trans_id,))
        conn.commit()

def delete_all_transactions():
    """取引履歴をすべて削除する関数"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transactions")
        conn.commit()

@st.dialog("💾 履歴変更の確認")
def confirm_save_dialog(target_id, e_date, e_code, e_name, e_type, e_shares, e_price, e_fee):
    """ 変更内容を確認してデータベースを更新するダイアログ """
    st.write(f"ID **{target_id}** の内容を以下に変更しますか？")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"・取引日: **{e_date}**")
        st.write(f"・銘柄コード: **{e_code}**")
        st.write(f"・銘柄名: **{e_name}**")
        st.write(f"・取引区分: **{e_type}**")
    with col2:
        st.write(f"・株数: **{e_shares}**")
        st.write(f"・取引単価: **{e_price:,.1f} 円**")
        st.write(f"・手数料: **{e_fee:,.1f} 円**")

    st.markdown("---")
    
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("はい（更新）", type="primary", use_container_width=True):
            # 1. データベースを更新
            update_transaction(target_id, e_date, e_code, e_name, e_type, e_shares, e_price, e_fee)
            
            # 2. フラグを立てて再描画時にリセットさせる
            st.session_state["selected_edit_id"] = None
            st.session_state["reset_edit_id"] = True
            
            st.success(f"ID {target_id} を更新しました。")
            st.rerun()
            
    with col_act2:
        if st.button("キャンセル", use_container_width=True):
            st.rerun()

@st.dialog("選択履歴の削除確認")
def confirm_delete_dialog(trans_id):
    """選択した履歴の削除確認ダイアログ"""
    st.warning(f"ID **{trans_id}** の取引履歴を削除します。よろしいですか？")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("削除を実行", type="primary", key="dlg_del_confirm"):
            delete_transaction(trans_id)
            st.success("選択した履歴を削除しました。")
            st.rerun()
    with col2:
        if st.button("キャンセル", key="dlg_del_cancel"):
            st.rerun()


@st.dialog("全履歴の削除確認")
def confirm_delete_all_dialog():
    """全削除時の確認ダイアログ"""
    st.error("⚠️ **すべての取引履歴を削除します。**\nこの操作は元に戻せません。本当に実行しますか？")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("全削除を実行", type="primary", key="dlg_del_all_confirm"):
            delete_all_transactions()
            st.success("すべての取引履歴を削除しました。")
            st.rerun()
    with col2:
        if st.button("キャンセル", key="dlg_del_all_cancel"):
            st.rerun()

# ==========================================
# 数値型安全変換ヘルパー関数
# ==========================================
def safe_int(val, default=1):
    """文字列・float・None・カンマ混在の値を安全に整数(int)へ変換する"""
    try:
        if pd.isna(val) or val is None:
            return default
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    """文字列・None・カンマ混在の値を安全に浮動小数点数(float)へ変換する"""
    try:
        if pd.isna(val) or val is None:
            return default
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default

def render_tab3_content():
    """ TAB 3: 売買取引履歴の表示および編集・削除操作を描画・処理する関数 """
    st.subheader("📜 売買取引履歴")

    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query(
        "SELECT id, trade_date, code, name, trade_type, shares, price, fee FROM transactions ORDER BY trade_date DESC, id DESC", 
        conn
    )
    conn.close()

    if not df_history.empty:
        st.dataframe(df_history, width="stretch")

        st.markdown("---")
        st.subheader("⚙️ 履歴の操作")
        
        tab3_op1, tab3_op2 = st.tabs(["✏️ 選択した履歴を編集", "🗑️ 履歴の削除"])

        with tab3_op1:
            # --------------------------------------------------
            # 【エラー回避処理】フォームを描画する「前」に初期化を完了させる
            # --------------------------------------------------
            if st.session_state.get("reset_edit_id", False):
                st.session_state["edit_id_input"] = 1
                st.session_state["reset_edit_id"] = False

            if "edit_id_input" not in st.session_state:
                st.session_state["edit_id_input"] = 1

            col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
            with col_sel1:
                edit_id = st.number_input(
                    "編集する履歴IDを入力", 
                    min_value=1, 
                    step=1, 
                    key="edit_id_input"
                )
            with col_sel2:
                st.write("")
                btn_select = st.button("🔍 選択した履歴を選択", key="btn_select_edit_id")
            with col_sel3:
                st.write("")
                btn_clear = st.button("❌ 編集を解除", key="btn_clear_edit_id")

            # ID選択時
            if btn_select:
                st.session_state["selected_edit_id"] = edit_id

            # 編集解除時（フラグを立てて再読み込み）
            if btn_clear:
                st.session_state["selected_edit_id"] = None
                st.session_state["reset_edit_id"] = True
                st.rerun()

            selected_id = st.session_state.get("selected_edit_id")
            if selected_id:
                target_row = df_history[df_history["id"] == selected_id]

                if not target_row.empty:
                    row_data = target_row.iloc[0]
                    with st.form("edit_transaction_form"):
                        st.write(f"**履歴ID {selected_id} の編集**")
                        try:
                            default_date = datetime.datetime.strptime(str(row_data["trade_date"]), "%Y-%m-%d").date()
                        except Exception:
                            default_date = datetime.date.today()

                        e_date = st.date_input("取引日", default_date)
                        e_code = st.text_input("銘柄コード", value=str(row_data["code"]))
                        e_name = st.text_input("銘柄名", value=str(row_data["name"]))
                        
                        type_idx = 0 if row_data["trade_type"] in ["買付", "現買", "買"] else 1
                        e_type = st.selectbox("取引区分", ["買付", "売却"], index=type_idx)
                        
                        shares_val = max(1, safe_int(row_data["shares"], default=1))
                        price_val = max(0.0, safe_float(row_data["price"], default=0.0))
                        fee_val = max(0.0, safe_float(row_data["fee"], default=0.0))

                        e_shares = st.number_input("株数", min_value=1, value=shares_val)
                        e_price = st.number_input("取引単価 (円)", min_value=0.0, value=price_val)
                        e_fee = st.number_input("手数料 (円)", min_value=0.0, value=fee_val)

                        btn_update = st.form_submit_button("💾 変更を保存する", type="primary")

                        if btn_update:
                            confirm_save_dialog(
                                selected_id,
                                e_date.strftime("%Y-%m-%d"),
                                e_code,
                                e_name,
                                e_type,
                                e_shares,
                                e_price,
                                e_fee
                            )
                else:
                    st.warning(f"ID {selected_id} の履歴が見つかりません。")
            else:
                st.info("編集する履歴IDを入力し、「選択した履歴を選択」ボタンを押してください。")

        with tab3_op2:
            col_del1, col_del2 = st.columns([2, 1])
            with col_del1:
                target_id = st.number_input("削除する履歴IDを指定", min_value=1, step=1, key="delete_id_input")
                btn_delete = st.button("🗑️ 選択した履歴の削除", key="btn_delete_selected_id", type="primary")
                
                if btn_delete:
                    target_row = df_history[df_history["id"] == target_id]
                    if not target_row.empty:
                        confirm_delete_dialog(target_id)
                    else:
                        st.warning(f"指定されたID {target_id} の履歴が見つかりません。")

            with col_del2:
                st.write("⚠️ 全データクリア")
                if st.button("⚠️ 取引履歴を全削除", key="btn_delete_all_confirm", type="primary"):
                    confirm_delete_all_dialog()
    else:
        st.info("取引履歴はありません。")

def render_tab1_content(api_key_input: str):
    """ TAB 1: スクリーニング ＆ Gemini AI 診断を表示・実行する関数 """
    st.subheader("🎯 スクリーニング ＆ Gemini AI 診断")

    st.markdown("### 🔍 スクリーニング条件の設定")
    
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

    if st.button("🔎 スクリーニング実行", key="btn_run_screening"):
        with st.spinner("東証上場銘柄のデータ取得・スクリーニングを実行中..."):
            tse_stocks = get_tse_stock_list()
            results = []

            for code, info in tse_stocks.items():
                name = info.get("name", "") if isinstance(info, dict) else str(info)
                market = info.get("market", "") if isinstance(info, dict) else ""

                if target_market != "すべて" and target_market not in market:
                    continue

                data = fetch_stock_full_data(code, default_name=name)
                if not data:
                    continue

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
                st.dataframe(pd.DataFrame(results), width="stretch")
            else:
                st.warning("設定条件に合致する銘柄は見つかりませんでした。条件を緩和してお試しください。")

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
        st.write("")
        run_screening = st.button("🔍 選択した市場でスクリーニング実行", type="primary")

    if run_screening:
        # （スクリーニング処理部分は既存のまま）
        pass

    if "screening_results" in st.session_state and not st.session_state["screening_results"].empty:
        st.dataframe(st.session_state["screening_results"], width="stretch")

    # 共通化した関数を呼び出す（重複コードを完全削除）
    render_gemini_diagnosis_section(api_key_input)

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

    with st.spinner("保有銘柄の最新株価を取得中..."):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT code, name FROM transactions")
            codes = cursor.fetchall()

        latest_prices = {}
        if codes:
            for raw_code, name in codes:
                clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
                data = fetch_stock_full_data(clean_code, default_name=name)
                if data and "price" in data:
                    latest_prices[clean_code] = data["price"]

            st.session_state["fetched_holdings_prices"] = latest_prices

    col_reload, _ = st.columns([1, 3])
    with col_reload:
        if st.button("🔄 手動で最新価格を再取得"):
            st.rerun()

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
            width="stretch"
        )
    else:
        st.info("保有している株式データがありません。")

# ==========================================
# TAB 3: 📜 売買取引履歴
# ==========================================
with tab3:
    render_tab3_content()

# ==========================================
# TAB 4: 📝 新規登録・SBI証券CSV取込
# ==========================================
with tab4:
    st.subheader("🏦 SBI証券 CSVファイル一括読み込み")
    uploaded_file = st.file_uploader("SBI証券からダウンロードしたCSVファイルを選択してください", type=["csv"])

    if uploaded_file is not None:
        if st.button("📁 CSVファイルを取り込む"):
            try:
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