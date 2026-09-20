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
from google import genai  # 最新の google-genai SDK

# ==========================================
# ページ初期設定
# ==========================================
st.set_page_config(
    page_title="株式売買管理システム - Streamlit & Gemini AI",
    page_icon="📈",
    layout="wide"
)

DB_PATH = Path("portfolio_data.db")

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
        return [item.title.text for item in items if item.title]
    except Exception as e:
        st.error(f"ニュース取得エラー ({stock_code}): {e}")
        return []

def analyze_news_with_gemini(api_key, stock_name, news_headlines):
    """ google-genai SDKを利用したニュース感情分析 """
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
            "reason": "Gemini APIキーが設定されていないため分析をスキップしました。"
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
        )
        raw_text = response.text.strip()
        # JSONブロックの抽出処理（Markdown装飾の対策）
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(raw_text)
    except Exception as e:
        return {
            "sentiment_score": 0,
            "category": "解析エラー",
            "reason": f"Gemini APIエラー: {str(e)[:60]}"
        }

@st.cache_data(ttl=86400)
def get_tse_stock_list():
    """ JPX公式の最新上場銘柄一覧を取得 """
    base_page = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        ssl_context = ssl._create_unverified_context()
        page_req = urllib.request.Request(base_page, headers=headers)
        with urllib.request.urlopen(page_req, context=ssl_context) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        match = re.search(r'href="([^"]*data_j\.(?:xls|xlsx))"', html)
        excel_url = match.group(1) if match else "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        if not excel_url.startswith("http"):
            excel_url = "https://www.jpx.co.jp" + excel_url

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
        st.warning(f"JPX銘柄リスト取得エラー: {e}（基本データを使用します）")

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
    """ yfinanceから株価・各種テクニカル指標を取得 """
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
        rsi14 = 50.0 if (last_loss == 0 or pd.isna(last_loss)) else float((100 - (100 / (1 + (gain / loss)))).iloc[-1])

        info = ticker.info or {}
        stock_name = default_name or info.get('longName') or info.get('shortName') or f"銘柄 {clean_code}"

        raw_div = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0.0
        raw_val = float(raw_div)
        div_yield = raw_val / 100.0 if raw_val > 50.0 else (raw_val * 100.0 if 0.0 < raw_val < 0.2 else raw_val)

        return {
            "code": clean_code,
            "name": stock_name,
            "price": latest_price,
            "trading_value": trading_value,
            "sma25": sma25,
            "rsi14": rsi14,
            "deviation_rate": deviation_rate,
            "per": float(info.get('trailingPE') or info.get('forwardPE') or 0.0),
            "roe": float(info.get('returnOnEquity') or 0.0) * 100,
            "earnings_growth": float(info.get('earningsGrowth') or 0.0) * 100,
            "div_yield": round(div_yield, 2),
            "equity_ratio": (info.get('totalStockholderEquity', 0) / info.get('totalAssets', 1) * 100) if info.get('totalAssets') else 50.0,
            "operating_margin": float(info.get('operatingMargins') or 0.0) * 100,
            "payout_ratio": float(info.get('payoutRatio') or 0.0) * 100,
            "market_cap": float(info.get('marketCap') or 0.0) / 100_000_000
        }
    except Exception:
        return None

# ==========================================
# サイドバー & 基本設定
# ==========================================
st.sidebar.title("⚙️ 設定")
gemini_api_key = st.sidebar.text_input(
    "Gemini API キー", 
    value=os.getenv("GEMINI_API_KEY", ""), 
    type="password",
    help="ニュース感情分析機能を利用する際に使用します。"
)

st.title("📈 株式売買管理 ＆ 東証・Gemini AI リアルタイム診断")

tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 おすすめ購入株・買い時診断", 
    "📊 保有銘柄一覧・資産状況", 
    "📜 売買取引履歴", 
    "📝 新規登録・SBI証券CSV取込"
])

# ==========================================
# TAB 1: 🤖 おすすめ購入株・買い時診断
# ==========================================
with tab1:
    st.subheader("🎯 スクリーニング ＆ Gemini AI 診断")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        target_market = st.selectbox("対象市場", ["プライム", "スタンダード", "グロース"])
    with col2:
        st.write("") # 間隔調整
        btn_run_screening = st.button("🚀 最新銘柄をスクリーニング", use_container_width=True)

    if btn_run_screening:
        with st.spinner(f"【{target_market}】市場の銘柄データを取得・分析中..."):
            tse_stocks = get_tse_stock_list()
            filtered_stocks = {k: v for k, v in tse_stocks.items() if target_market in v.get("market", "")}
            
            candidates = []
            progress_bar = st.progress(0)
            items_list = list(filtered_stocks.items())[:20] # レスポンス確保のため20件制限
            
            for idx, (code, info) in enumerate(items_list):
                data = fetch_stock_full_data(code, default_name=info.get("name", ""))
                if data:
                    price = data["price"]
                    stop_loss = min(price * 0.95, data["sma25"] * 0.98)
                    signal = "◎ 上昇トレンド" if price > data["sma25"] else "△ 様子見"
                    candidates.append({
                        "銘柄コード": code,
                        "銘柄名": data["name"],
                        "推奨目的": f"スイング ({target_market})",
                        "現在株価": f"¥{price:,.1f}",
                        "予想配当利回り": f"{data['div_yield']:.2f}%" if data['div_yield'] > 0 else "---",
                        "買い時判定": signal,
                        "損切目安": f"¥{stop_loss:,.1f}",
                        "根拠/指標": f"RSI:{data['rsi14']:.1f}%, PER:{data['per']:.1f}倍"
                    })
                progress_bar.progress((idx + 1) / len(items_list))
            progress_bar.empty()
            
            if candidates:
                st.session_state["screening_results"] = pd.DataFrame(candidates)
                st.success(f"スクリーニング完了（{len(candidates)} 件抽出）")

    if "screening_results" in st.session_state:
        st.dataframe(st.session_state["screening_results"], use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Gemini AI リアルタイム統合診断")
    
    diag_col1, diag_col2 = st.columns([1, 2])
    with diag_col1:
        check_code = st.text_input("銘柄コード（半角数字）", placeholder="例: 7203")
        btn_diag = st.button("AIリアルタイム診断を実行", type="primary")

    if btn_diag and check_code:
        with st.spinner(f"銘柄 {check_code} の情報取得・AI分析を実行中..."):
            tse_stocks = get_tse_stock_list()
            info = tse_stocks.get(check_code, {})
            data = fetch_stock_full_data(check_code, default_name=info.get("name", ""))
            
            if not data:
                st.error(f"銘柄コード {check_code} のデータが見つかりませんでした。")
            else:
                news_headlines = fetch_latest_stock_news(check_code, data["name"])
                ai_res = analyze_news_with_gemini(gemini_api_key, data["name"], news_headlines)
                
                # スコア判定
                f_pass = (data["earnings_growth"] >= 10.0) and (data["roe"] >= 8.0)
                t_pass = (data["rsi14"] <= 40.0) or (data["deviation_rate"] <= -10.0)
                sentiment = ai_res.get("sentiment_score", 0)
                
                total_score = 50 + (20 if f_pass else 0) + (20 if t_pass else 0) + (sentiment * 0.3)
                
                if sentiment <= -50:
                    status_color, signal = "error", "⚠️ 危険・買わないこと (悪材料検知)"
                elif total_score >= 80:
                    status_color, signal = "success", "★ 絶好の買い場 (Sランク)"
                elif total_score >= 60:
                    status_color, signal = "info", "〇 買い検討可能 (Aランク)"
                else:
                    status_color, signal = "warning", "△ 様子見推奨 (B/Cランク)"

                st.subheader(f"判定結果: {signal}")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("総合スコア", f"{total_score:.0f} 点")
                col_b.metric("現在株価", f"¥{data['price']:,.1f}")
                col_c.metric("ニュース感情スコア", f"{sentiment:+d}")

                st.markdown(f"**🤖 Gemini AI 分析理由:** {ai_res.get('reason', 'なし')}")
                
                with st.expander("詳細な指標・データ"):
                    st.json(data)

# ==========================================
# TAB 2: 📊 保有銘柄一覧・資産状況
# ==========================================
with tab2:
    st.subheader("📊 保有資産状況")
    
    def calculate_portfolio():
        conn = sqlite3.connect(DB_PATH)
        df_trans = pd.read_sql_query("SELECT * FROM transactions ORDER BY trade_date ASC, id ASC", conn)
        conn.close()

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
            c_price = p["last_price"]
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

    df_holdings, total_val, unrealized_pl, realized_pl = calculate_portfolio()

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("総評価額", f"¥{total_val:,.0f}")
    col_m2.metric("評価損益", f"¥{unrealized_pl:+,.0f}")
    col_m3.metric("確定損益", f"¥{realized_pl:+,.0f}")

    if not df_holdings.empty:
        st.dataframe(df_holdings, use_container_width=True)
    else:
        st.info("現在、保有している銘柄はありません。")

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
        
        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            target_id = st.number_input("削除する履歴IDを指定", min_value=1, step=1)
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