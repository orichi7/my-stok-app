import os
import streamlit as st
import sys
import sqlite3
import datetime
import io
import urllib.request
import urllib.parse
import ssl
from pathlib import Path
import json
import re
import random
import threading
import sqlite3

import yfinance as yf
import pandas as pd
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import ttk, messagebox, filedialog
from bs4 import BeautifulSoup

# 新標準SDKを使用
import google.generativeai as genai

# ==========================================
# 設定: Gemini APIキーは環境変数から安全に取得
# (環境変数 GEMINI_API_KEY にキーを設定してください)
# ==========================================


def fetch_latest_stock_news(stock_code, stock_name):
    """ Google ニュースから該当銘柄の直近ニュースの見出しを取得する """
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
        print(f"ニュース取得エラー ({stock_code}): {e}")
        return []


def analyze_news_with_gemini(stock_name, news_headlines):
    """ 利用可能なGeminiモデルを自動検出してニュース分析を実施 """
    if not news_headlines:
        return {
            "sentiment_score": 0,
            "category": "ニュースなし",
            "reason": "直近のニュース見出しが取得できませんでした。"
        }

    if not GEMINI_API_KEY:
        return {
            "sentiment_score": 0,
            "category": "APIキー未設定",
            "reason": "環境変数 GEMINI_API_KEY が設定されていないため、ニュース感情分析をスキップしました。"
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

    genai.configure(api_key=GEMINI_API_KEY)

    try:
        # APIキーで利用可能なモデル一覧を取得し、generateContentが使えるモデルを自動抽出
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # 'models/gemini-xxx' の形式から 'gemini-xxx' を抽出
                name = m.name.replace("models/", "")
                available_models.append(name)
        
        # 'flash' がつく高速モデルを優先し、なければ一覧の先頭を使用
        flash_models = [m for m in available_models if "flash" in m]
        candidate_models = flash_models + [m for m in available_models if m not in flash_models]

    except Exception as e:
        print(f"モデル一覧取得エラー: {e}")
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
            print(f"モデル '{model_name}' の試行失敗: {e}")
            continue

    return {
        "sentiment_score": 0,
        "category": "解析エラー",
        "reason": f"通信エラー: {last_error[:60]}"
    }


def get_tse_stock_list():
    """ JPX公式の最新上場銘柄一覧（Excel）を安全に取得 """
    base_page = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
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
        print(f"JPX自動取得スキップ (フォールバックデータを使用): {e}")

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
    """ 株価データを安全に取得（404エラー・上場廃止対策済み） """
    try:
        clean_code = str(code).replace(".T", "").replace(".t", "").strip()
        symbol = f"{clean_code}.T" if clean_code.isdigit() else clean_code

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d")
        
        if hist.empty or len(hist) < 1:
            print(f"⚠️ 株価データが存在しないか取得できませんでした: {symbol}")
            return None

        latest_price = float(hist['Close'].iloc[-1])
        latest_volume = float(hist['Volume'].iloc[-1])
        trading_value = latest_price * latest_volume

        # テクニカル指標の計算（データ件数不足時の保護付き）
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
    except Exception as e:
        print(f"⚠️ データ取得スキップ ({code}): {e}")
        return None

# ==========================================
# 2. メイン画面の初期処理 ＆ サイドバー設定
# ==========================================
# ★★★【ここに組み込みます！】★★★
st.sidebar.title("⚙️ システム設定")
api_key_input = st.sidebar.text_input(
    "Gemini API キー",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password"
)

# 画面タイトルの表示
st.title("📈 株式売買管理 ＆ 東証・Gemini AI リアルタイム診断")

# タブの作成
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 おすすめ購入株・買い時診断",
    "📊 保有銘柄一覧・資産状況",
    "📜 売買取引履歴",
    "📝 新規登録・SBI証券CSV取込"
])

class StockManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("株式売買管理システム - JPX連携＆Gemini AI 統合診断版")
        self.root.geometry("1280x830")
        self.root.configure(bg="#f4f6f9")

        script_dir = Path(__file__).resolve().parent
        self.db_path = script_dir / "portfolio_data.db"
        self.current_selected_market = "プライム"
        self._is_refreshing_holdings = False  # 連打・重複取得防止フラグ

        self.init_database()

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", font=("Meiryo", 10))
        self.style.configure("Treeview.Heading", font=("Meiryo", 10, "bold"), background="#2c3e50", foreground="white")
        self.style.configure("Treeview", font=("Meiryo", 10), rowheight=28)

        header_frame = tk.Frame(self.root, bg="#1a252f", height=60)
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(header_frame, text="📈 株式売買管理 ＆ 東証・Gemini AI リアルタイム診断", font=("Meiryo", 15, "bold"), fg="white", bg="#1a252f")
        header_label.pack(side=tk.LEFT, padx=20, pady=15)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.tab_advisor = ttk.Frame(self.notebook)
        self.tab_holdings = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)
        self.tab_add_trade = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_advisor, text="  🤖 おすすめ購入株・買い時診断  ")
        self.notebook.add(self.tab_holdings, text="  保有銘柄一覧・資産状況  ")
        self.notebook.add(self.tab_history, text="  売買取引履歴  ")
        self.notebook.add(self.tab_add_trade, text="  新規登録・SBI証券CSV取込  ")

        # タブ移動イベントをバインド
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.setup_advisor_tab()
        self.setup_holdings_tab()
        self.setup_history_tab()
        self.setup_add_trade_tab()

        self.load_holdings_data()
        self.load_history_data()

    def init_database(self):
        db_uri = self.db_path.as_uri()
        self.conn = sqlite3.connect(db_uri, uri=True)
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def on_tab_changed(self, event):
        """ タブ移動時に「保有銘柄・資産状況」タブかつデータが存在する場合、自動で株価を最新化 """
        selected_tab = self.notebook.select()
        
        # 選択されたタブが「保有銘柄・資産状況」タブの場合
        if selected_tab == str(self.tab_holdings):
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM transactions")
            count = cursor.fetchone()[0]

            # 取引データが存在し、かつ現在更新処理中でなければ実行
            if count > 0 and not getattr(self, "_is_refreshing_holdings", False):
                self.refresh_holdings_current_prices()

    # --- プログレスバー・インジケーター制御 ---
    def show_loading(self, message="データ取得中..."):
        """ インジケーター表示＆全ボタン無効化 """
        self.lbl_status.config(text=message)
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.start(10)
        for btn in [self.btn_refresh, self.btn_prime, self.btn_standard, self.btn_growth, self.btn_diag]:
            try:
                btn.config(state=tk.DISABLED)
            except Exception:
                pass
        self.root.update_idletasks()

    def hide_loading(self):
        """ 完了時にプログレスバーを非表示にし、すべての操作ボタンを確実に復元 """
        self.progress_bar["value"] = 0
        self.progress_bar.pack_forget()
        self.lbl_status.config(text="🎯 各市場ボタンを押して最新おすすめ銘柄を取得できます")
        
        # 診断ボタンを含むすべてのボタンを有効化
        for btn in [self.btn_refresh, self.btn_prime, self.btn_standard, self.btn_growth, self.btn_diag]:
            try:
                btn.config(state=tk.NORMAL)
            except Exception:
                pass

    # 1. 「おすすめ銘柄を更新」ボタンの処理
    def update_existing_advisor_data(self):
        existing_items = self.tree_recommend.get_children()
        
        # ① データが表示されていない場合
        if not existing_items:
            messagebox.showinfo(
                "お知らせ", 
                "現在表示されているデータがありません。\n"
                "「プライム」「スタンダード」「グロース」などの市場ボタンを押下してデータを表示させてください。"
            )
            return

        # ② データが表示されている場合：現在表示中の銘柄の株価・指標を更新
        existing_codes_with_names = []
        for item_id in existing_items:
            vals = self.tree_recommend.item(item_id, "values")
            code = vals[0]
            name = vals[1]
            existing_codes_with_names.append((code, name))

        def task():
            updated_rows = []
            for code, name in existing_codes_with_names:
                data = fetch_stock_full_data(code, default_name=name)
                if not data:
                    continue

                stock_name = data["name"]
                price = data["price"]

                stop_loss_swing = min(price * 0.95, data["sma25"] * 0.98)
                c1_sma = (price > data["sma25"])

                signal = "◎ 上昇トレンド" if c1_sma else "△ 様子見"
                advice = f"目標:¥{price*1.04:,.0f} (RSI:{data['rsi14']:.1f}%, PER:{data['per']:.1f}倍)"
                strategy = f"①スイング({self.current_selected_market})"
                stop_loss_str = f"¥{stop_loss_swing:,.1f}"
                div_str = f"{data['div_yield']:.2f}%" if data["div_yield"] > 0 else "---"

                updated_rows.append((
                    code, stock_name, strategy, f"¥{price:,.1f}", div_str, signal, stop_loss_str, advice
                ))

            def update_ui():
                for item in self.tree_recommend.get_children():
                    self.tree_recommend.delete(item)
                for row in updated_rows:
                    self.tree_recommend.insert("", tk.END, values=row)
                self.hide_loading()
                messagebox.showinfo("更新完了", "表示中銘柄の最新価格データを更新しました！")

            self.root.after(0, update_ui)

        self.show_loading("表示中の株価最新データを取得中...")
        threading.Thread(target=task, daemon=True).start()

    def show_progress(self, current, total, message="データ取得中..."):
        """ 進捗率（%）付きでプログレスバーを更新 """
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate", maximum=total)
        self.progress_bar["value"] = current
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.lbl_status.config(text=f"⏳ {message} [{current}/{total} ({percent}%)]")
        self.root.update_idletasks()

    def hide_loading(self):
        """ 完了時にプログレスバーを非表示にし、全ボタンを復元 """
        self.progress_bar.stop()
        self.progress_bar["value"] = 0
        self.progress_bar.pack_forget()
        self.lbl_status.config(text="🎯 各市場ボタンを押して最新おすすめ銘柄を取得できます")
        
        # 診断ボタン(btn_diag)を含むすべてのボタンを確実に復元
        for btn in [self.btn_refresh, self.btn_prime, self.btn_standard, self.btn_growth, self.btn_diag]:
            try:
                btn.config(state=tk.NORMAL)
            except Exception:
                pass

    # 2. 各市場ボタン押下時のデータ取得処理 (インジケーター付き)
    def load_advisor_data(self, target_market=None, show_popup=False):
        """ 別スレッドでデータ取得・進行状況をプログレスバーに表示 """
        import threading  # スレッド用モジュール

        if target_market is not None:
            self.current_selected_market = target_market
        else:
            target_market = self.current_selected_market

        # ボタンを無効化
        for btn in [self.btn_refresh, self.btn_prime, self.btn_standard, self.btn_growth]:
            btn.config(state=tk.DISABLED)

        def task():
            tse_stocks = get_tse_stock_list()
            target_kw = target_market.strip()
            filtered_stocks = {}

            for code, info in tse_stocks.items():
                stock_name = info.get("name", "") if isinstance(info, dict) else str(info)
                market_name = info.get("market", "") if isinstance(info, dict) else ""

                if target_kw in market_name:
                    filtered_stocks[code] = stock_name
                    if len(filtered_stocks) >= 30:  # 取得件数制限
                        break

            if not filtered_stocks:
                def on_empty():
                    self.hide_loading()
                    if show_popup:
                        messagebox.showinfo("結果", f"【{target_market}市場】の対象銘柄が見つかりませんでした。")
                self.root.after(0, on_empty)
                return

            swing_candidates = []
            income_candidates = []
            total_count = len(filtered_stocks)

            # --- プログレスバーを進めながら順次データ取得 ---
            for idx, (code, name) in enumerate(filtered_stocks.items(), start=1):
                # UI側のプログレスバーを更新
                self.root.after(0, lambda i=idx, c=code: self.show_progress(i, total_count, f"【{target_market}】取得中: {c}"))

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
                swing_candidates.append((swing_score, code, data["name"], f"①スイング({target_market})", price, data["div_yield"], signal1, f"¥{stop_loss_swing:,.1f}", advice1))

                c2_yield = (2.5 <= data["div_yield"] <= 5.0) if target_market != "グロース" else (data["earnings_growth"] >= 15.0)
                c2_equity = (data["equity_ratio"] >= 40.0)
                c2_margin = (data["operating_margin"] >= 8.0)
                c2_payout = (20.0 <= data["payout_ratio"] <= 60.0)

                ai_macro_score_income = 2.0 if (c2_yield and c2_equity and c2_margin) else 0.0
                income_score = sum([c2_yield, c2_equity, c2_margin, c2_payout]) + ai_macro_score_income

                strat_label = "②高成長・長期" if target_market == "グロース" else "②配当・長期保有"
                signal2 = "★ 優良財務" if ai_macro_score_income > 0 else ("〇 安定保有" if c2_yield else "△ 保留")
                advice2 = f"利回り:{data['div_yield']:.2f}% (自己資本:{data['equity_ratio']:.0f}%, 営業益率:{data['operating_margin']:.1f}%)"
                income_candidates.append((income_score, code, data["name"], strat_label, price, data["div_yield"], signal2, f"¥{stop_loss_income:,.1f}", advice2))

            swing_candidates.sort(key=lambda x: x[0], reverse=True)
            income_candidates.sort(key=lambda x: x[0], reverse=True)

            selected_items = swing_candidates[:5] + income_candidates[:5]

            # メインスレッドでTreeview描画とUI復元
            def update_ui():
                for item in self.tree_recommend.get_children():
                    self.tree_recommend.delete(item)

                for item in selected_items:
                    _, code, s_name, strategy, price, div, signal, stop_loss_str, advice = item
                    div_str = f"{div:.2f}%" if div > 0 else "---"
                    self.tree_recommend.insert("", tk.END, values=(
                        code, s_name, strategy, f"¥{price:,.1f}", div_str, signal, stop_loss_str, advice
                    ))

                self.hide_loading()
                if show_popup:
                    count = len(self.tree_recommend.get_children())
                    messagebox.showinfo("スクリーニング完了", f"【{target_market}市場】の最新診断が完了しました！（抽出: {count}銘柄）")

            self.root.after(0, update_ui)

        threading.Thread(target=task, daemon=True).start()

    # --- AIリアルタイム診断処理 ---
    def diagnose_stock(self):
        code = self.ent_check_code.get().strip()
        if not code:
            messagebox.showwarning("入力エラー", "銘柄コードを入力してください。")
            return

        # 操作ボタンを一括で無効化
        for btn in [self.btn_refresh, self.btn_prime, self.btn_standard, self.btn_growth, self.btn_diag]:
            try:
                btn.config(state=tk.DISABLED)
            except Exception:
                pass

        def update_step(current, total, msg):
            """ スレッドからメインUIのプログレスバーを更新 """
            self.root.after(0, lambda: self.show_progress(current, total, f"【AI診断】銘柄 {code}: {msg}"))

        def task():
            total_steps = 4
            try:
                # 1. 株価・指標取得
                update_step(1, total_steps, "株価・財務指標を取得中...")
                tse_stocks = get_tse_stock_list()
                info = tse_stocks.get(code, {})
                default_name = info.get("name", "") if isinstance(info, dict) else str(info)

                data = fetch_stock_full_data(code, default_name=default_name)
                if not data:
                    def on_fail():
                        self.hide_loading()
                        messagebox.showerror("エラー", f"銘柄コード {code} のデータを取得できませんでした。")
                    self.root.after(0, on_fail)
                    return

                stock_name = data["name"]
                price = data["price"]

                # 2. 最新ニュース取得
                update_step(2, total_steps, f"「{stock_name}」の最新ニュースを取得中...")
                news_headlines = fetch_latest_stock_news(code, stock_name)

                # 3. Gemini AI による感情分析
                update_step(3, total_steps, "Gemini AI でニュース感情分析を実行中...")
                ai_res = analyze_news_with_gemini(stock_name, news_headlines)

                # 4. 総合スコア算出と判定
                update_step(4, total_steps, "総合判定レポートを作成中...")
                
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

                report = (
                    f"【Gemini AI 統合診断結果】 {stock_name} ({code})\n"
                    f"総合評価: 【 {signal} 】 (総合スコア: {total_score:.0f}点)\n"
                    f"{eval_msg}\n\n"
                    f"--- 🤖 Gemini AI ニュース感情分析 ---\n"
                    f"・ニュース分類: {ai_category} (感情スコア: {sentiment_score:+d})\n"
                    f"・AI分析理由: {ai_reason}\n\n"
                    f"--- 📊 ファンダメンタルズ判定 ({'合格' if fundamentals_pass else '要確認'}) ---\n"
                    f"・営業利益変化率: {data['earnings_growth']:+.1f}%\n"
                    f"・ROE: {data['roe']:.1f}% | PER: {data['per']:.1f}倍\n\n"
                    f"--- 📈 テクニカル / 株価目安 ---\n"
                    f"・現在株価: ¥{price:,.1f}\n"
                    f"・RSI (14日): {data['rsi14']:.1f}% | 25日乖離率: {data['deviation_rate']:+.1f}%\n"
                    f"・推奨損切ライン: ¥{suggested_stop_loss:,.1f} 付近\n"
                )

                def show_result():
                    # プログレスバー消去＆ボタン状態復元
                    self.hide_loading()
                    # 入力欄の初期化
                    self.ent_check_code.delete(0, tk.END)
                    # UI描画を確実に完了させてからポップアップを表示
                    self.root.update_idletasks()
                    messagebox.showinfo("Gemini AI 統合リアルタイム診断", report)

                self.root.after(0, show_result)

            except Exception as e:
                def on_error():
                    self.hide_loading()
                    messagebox.showerror("診断エラー", f"処理中にエラーが発生しました:\n{e}")
                self.root.after(0, on_error)

        # 0%から開始
        self.show_progress(0, 4, f"【AI診断】銘柄 {code}: 準備中...")
        threading.Thread(target=task, daemon=True).start()

    def setup_advisor_tab(self):
        info_frame = tk.Frame(self.tab_advisor, bg="#e8f8f5", bd=1, relief=tk.RIDGE)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        # ステータスラベル（self.lbl_statusに変更）
        self.lbl_status = tk.Label(
            info_frame,
            text="🎯 各市場ボタンを押して最新おすすめ銘柄を取得できます",
            font=("Meiryo", 10, "bold"),
            bg="#e8f8f5",
            fg="#16a085",
        )
        self.lbl_status.pack(padx=15, pady=10, side=tk.LEFT)

        # プログレスバーの追加 (determinateモード)
        self.progress_bar = ttk.Progressbar(info_frame, mode="determinate", length=180)

        btn_frame = tk.Frame(info_frame, bg="#e8f8f5")
        btn_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        # 1. 「おすすめ銘柄を更新」ボタン
        self.btn_refresh = tk.Button(btn_frame, text="🔄 おすすめ銘柄を更新", bg="#d35400", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, padx=10, pady=4, command=self.update_existing_advisor_data)
        self.btn_refresh.pack(side=tk.LEFT, padx=6)

        self.btn_prime = tk.Button(btn_frame, text="🏛️ プライム", bg="#2c3e50", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, padx=8, pady=4, command=lambda: self.load_advisor_data(target_market="プライム", show_popup=True))
        self.btn_prime.pack(side=tk.LEFT, padx=4)

        self.btn_standard = tk.Button(btn_frame, text="🏢 スタンダード", bg="#2980b9", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, padx=8, pady=4, command=lambda: self.load_advisor_data(target_market="スタンダード", show_popup=True))
        self.btn_standard.pack(side=tk.LEFT, padx=4)

        self.btn_growth = tk.Button(btn_frame, text="🚀 グロース", bg="#27ae60", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, padx=8, pady=4, command=lambda: self.load_advisor_data(target_market="グロース", show_popup=True))
        self.btn_growth.pack(side=tk.LEFT, padx=4)

        table_frame = tk.Frame(self.tab_advisor)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("code", "name", "strategy", "price", "div_yield", "signal", "stop_loss", "advice")
        self.tree_recommend = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

        headers = {
            "code": "銘柄コード", "name": "銘柄名 (日本語)", "strategy": "推奨目的",
            "price": "現在株価", "div_yield": "予想配当利回り", "signal": "買い時判定",
            "stop_loss": "損切ライン(目安)", "advice": "判定理由 / スクリーニング根拠"
        }

        for col, heading in headers.items():
            self.tree_recommend.heading(col, text=heading)
            width = 230 if col == "advice" else (160 if col == "name" else (110 if col == "stop_loss" else 100))
            self.tree_recommend.column(col, anchor=tk.CENTER, width=width)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree_recommend.yview)
        self.tree_recommend.configure(yscrollcommand=scrollbar.set)
        self.tree_recommend.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        diag_frame = tk.LabelFrame(self.tab_advisor, text=" 🤖 Gemini AI ニュース統合リアルタイム診断 ", font=("Meiryo", 10, "bold"), bg="white", padx=15, pady=10)
        diag_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(diag_frame, text="銘柄コード:", bg="white").pack(side=tk.LEFT, padx=5)
        self.ent_check_code = tk.Entry(diag_frame, width=12)
        self.ent_check_code.pack(side=tk.LEFT, padx=5)

        self.btn_diag = tk.Button(diag_frame, text="Gemini AI でリアルタイム診断を実行", bg="#8e44ad", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, command=self.diagnose_stock)
        self.btn_diag.pack(side=tk.LEFT, padx=15)

    def setup_holdings_tab(self):
        summary_frame = tk.Frame(self.tab_holdings, bg="#ecf0f1", bd=1, relief=tk.RIDGE)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_total_val = tk.Label(summary_frame, text="総評価額: ¥0", font=("Meiryo", 12, "bold"), bg="#ecf0f1", fg="#2c3e50")
        self.lbl_total_val.pack(side=tk.LEFT, padx=15, pady=12)

        self.lbl_unrealized_pl = tk.Label(summary_frame, text="評価損益: ¥0 (0.00%)", font=("Meiryo", 12, "bold"), bg="#ecf0f1", fg="#27ae60")
        self.lbl_unrealized_pl.pack(side=tk.LEFT, padx=15, pady=12)

        self.lbl_realized_pl = tk.Label(summary_frame, text="確定損益: ¥0", font=("Meiryo", 12, "bold"), bg="#ecf0f1", fg="#2980b9")
        self.lbl_realized_pl.pack(side=tk.LEFT, padx=15, pady=12)

        # 「保有データを更新」ボタン
        self.btn_refresh_h = tk.Button(summary_frame, text="🔄 保有データを更新", bg="#2980b9", fg="white", font=("Meiryo", 9, "bold"), relief=tk.FLAT, command=self.refresh_holdings_current_prices)
        self.btn_refresh_h.pack(side=tk.RIGHT, padx=15, pady=12)

        # 保有タブ用のプログレスバー＆進捗ステータスラベル
        self.progress_bar_h = ttk.Progressbar(summary_frame, mode="determinate", length=160)
        self.lbl_status_h = tk.Label(summary_frame, text="", font=("Meiryo", 9, "bold"), bg="#ecf0f1", fg="#2980b9")
        self.lbl_status_h.pack(side=tk.RIGHT, padx=5)

        table_frame = tk.Frame(self.tab_holdings)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("code", "name", "shares", "avg_price", "current_price", "market_val", "profit_loss", "return_rate")
        self.tree_holdings = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

        self.tree_holdings.tag_configure("loss_row", background="#fadbd8", foreground="#78281f")

        headers = {
            "code": "銘柄コード", "name": "銘柄名", "shares": "保有株数",
            "avg_price": "平均取得単価", "current_price": "現在値",
            "market_val": "評価額", "profit_loss": "評価損益", "return_rate": "損益率"
        }
        for col, heading in headers.items():
            self.tree_holdings.heading(col, text=heading)
            self.tree_holdings.column(col, anchor=tk.CENTER, width=120)

        scrollbar_h = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree_holdings.yview)
        self.tree_holdings.configure(yscrollcommand=scrollbar_h.set)
        self.tree_holdings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_h.pack(side=tk.RIGHT, fill=tk.Y)

    # 保有タブ専用の進捗インジケーター表示・非表示制御
    def show_holdings_progress(self, current, total, message="取得中..."):
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar_h.config(mode="determinate", maximum=total)
        self.progress_bar_h["value"] = current
        self.progress_bar_h.pack(side=tk.RIGHT, padx=10)
        self.lbl_status_h.config(text=f"⏳ {message} [{current}/{total} 銘柄 ({percent}%)]")
        self.btn_refresh_h.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def hide_holdings_loading(self):
        self.progress_bar_h["value"] = 0
        self.progress_bar_h.pack_forget()
        self.lbl_status_h.config(text="")
        self.btn_refresh_h.config(state=tk.NORMAL)

    def refresh_holdings_current_prices(self):
        """ 保有銘柄の最新株価を進捗表示付きで取得・マップして更新 """
        if getattr(self, "_is_refreshing_holdings", False):
            return
        self._is_refreshing_holdings = True

        def task():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT code, name FROM transactions")
                    codes = cursor.fetchall()

                if not codes:
                    def on_empty():
                        self.hide_holdings_loading()
                        self._is_refreshing_holdings = False
                    self.root.after(0, on_empty)
                    return

                updated_prices = {}
                total_count = len(codes)

                for idx, (raw_code, name) in enumerate(codes, start=1):
                    clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
                    
                    # スレッド内からUI側のプログレスバーを更新
                    self.root.after(0, lambda i=idx, c=clean_code: self.show_holdings_progress(i, total_count, f"保有株価取得中: {c}"))

                    data = fetch_stock_full_data(clean_code, default_name=name)
                    if data and "price" in data:
                        price = data["price"]
                        updated_prices[str(raw_code).strip()] = price
                        updated_prices[clean_code] = price
                        updated_prices[f"{clean_code}.T"] = price

                def update_ui():
                    self.hide_holdings_loading()
                    self._is_refreshing_holdings = False
                    self.load_holdings_data(fetched_prices=updated_prices)

                self.root.after(0, update_ui)

            except Exception as e:
                def on_error():
                    self.hide_holdings_loading()
                    self._is_refreshing_holdings = False
                    print(f"保有株価取得エラー: {e}")
                self.root.after(0, on_error)

        threading.Thread(target=task, daemon=True).start()

    def load_holdings_data(self, fetched_prices=None):
        """ 保有銘柄一覧・資産状況の正確な再計算処理 """
        for item in self.tree_holdings.get_children():
            self.tree_holdings.delete(item)

        cursor = self.conn.cursor()
        cursor.execute("SELECT code, name, trade_type, shares, price, fee FROM transactions ORDER BY trade_date ASC, id ASC")
        rows = cursor.fetchall()

        portfolio = {}
        realized_pl = 0.0

        for raw_code, name, t_type, shares, price, fee in rows:
            # 1. 銘柄コードの表記揺れを統一（2695.T -> 2695）
            clean_code = str(raw_code).replace(".T", "").replace(".t", "").strip()
            
            if clean_code not in portfolio:
                portfolio[clean_code] = {
                    "raw_code": raw_code,
                    "name": name,
                    "shares": 0.0,
                    "total_cost": 0.0,  # 総取得コスト
                    "last_price": price
                }

            portfolio[clean_code]["last_price"] = price
            t_str = str(t_type).strip()

            # 2. 取引区分の判定を強化（"買" を含む判定）
            if any(k in t_str for k in ["買", "現買", "買付"]):
                current_shares = portfolio[clean_code]["shares"]
                buy_cost = (shares * price) + fee
                
                portfolio[clean_code]["shares"] += shares
                portfolio[clean_code]["total_cost"] += buy_cost

            elif any(k in t_str for k in ["売", "現売", "売却"]):
                current_shares = portfolio[clean_code]["shares"]
                
                if current_shares > 0:
                    # 売却時点の平均取得単価
                    avg_p = portfolio[clean_code]["total_cost"] / current_shares
                    
                    sell_proceeds = (shares * price) - fee
                    cost_basis = avg_p * shares
                    realized_pl += (sell_proceeds - cost_basis)

                    # 保有株数および取得原価を比例減算
                    portfolio[clean_code]["shares"] -= shares
                    portfolio[clean_code]["total_cost"] -= cost_basis
                    
                    if portfolio[clean_code]["shares"] <= 0:
                        portfolio[clean_code]["shares"] = 0.0
                        portfolio[clean_code]["total_cost"] = 0.0

        total_market_val = 0.0
        total_cost_val = 0.0

        for clean_code, data in portfolio.items():
            shares = data["shares"]
            if shares <= 0:
                continue

            # 3. 加重平均による平均取得単価の正確な算出
            avg_price = data["total_cost"] / shares if shares > 0 else 0.0
            raw_code = data["raw_code"]

            # 現在値の検索
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

            row_tags = ("loss_row",) if pl < 0 else ()

            self.tree_holdings.insert("", tk.END, values=(
                clean_code,
                data["name"],
                f"{int(shares):,}株" if shares.is_integer() else f"{shares:,.2f}株",
                f"¥{avg_price:,.1f}",
                f"¥{c_price:,.1f}",
                f"¥{market_val:,.0f}",
                f"¥{pl:+,.0f}",
                f"{ret_rate:+.2f}%"
            ), tags=row_tags)

        total_unrealized_pl = total_market_val - total_cost_val
        total_unrealized_rate = (total_unrealized_pl / total_cost_val * 100) if total_cost_val > 0 else 0.0

        self.lbl_total_val.config(text=f"総評価額: ¥{total_market_val:,.0f}")
        self.lbl_unrealized_pl.config(text=f"評価損益: ¥{total_unrealized_pl:+,.0f} ({total_unrealized_rate:+.2f}%)")
        self.lbl_realized_pl.config(text=f"確定損益: ¥{realized_pl:+,.0f}")

    def delete_all_transactions(self):
        """ 売買取引履歴を全て削除する """
        # 誤操作防止の確認ダイアログ
        if not messagebox.askyesno("全削除の確認", "本当に全ての売買取引履歴を削除しますか？\n※この操作は元に戻せません。"):
            return

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM transactions")
        self.conn.commit()

        messagebox.showinfo("完了", "全ての売買取引履歴を削除しました。")
        
        # 画面表示の更新（履歴および保有株の再計算）
        self.load_history_data()
        self.load_holdings_data()

    def setup_history_tab(self):
        """ 売買取引履歴タブのUI構築 """
        # --- 上部：操作ボタンエリア ---
        btn_frame = tk.Frame(self.tab_history, bg="white")
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        # 【左側】選択行に対する個別操作ボタン
        btn_edit = tk.Button(
            btn_frame, 
            text="✏️ 選択した履歴を編集", 
            font=("Meiryo", 9), 
            bg="#3498db", 
            fg="white", 
            relief=tk.FLAT, 
            padx=10, 
            pady=4, 
            command=self.edit_selected_transaction
        )
        btn_edit.pack(side=tk.LEFT, padx=(0, 5))

        btn_delete_selected = tk.Button(
            btn_frame, 
            text="🗑️ 選択した履歴を削除", 
            font=("Meiryo", 9), 
            bg="#e67e22", 
            fg="white", 
            relief=tk.FLAT, 
            padx=10, 
            pady=4, 
            command=self.delete_selected_transaction
        )
        btn_delete_selected.pack(side=tk.LEFT)

        # 【右側】全削除ボタン（「再読み込み」ボタンは削除）
        btn_delete_all = tk.Button(
            btn_frame, 
            text="⚠️ 取引履歴を全削除", 
            font=("Meiryo", 9, "bold"), 
            bg="#e74c3c", 
            fg="white", 
            relief=tk.FLAT, 
            padx=12, 
            pady=4, 
            command=self.delete_all_transactions
        )
        btn_delete_all.pack(side=tk.RIGHT)

        # --- 中央：取引履歴テーブルエリア ---
        table_frame = tk.Frame(self.tab_history, bg="white")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # スクロールバーの設定
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        # 取引履歴表示用 Treeview
        columns = ("id", "trade_date", "code", "name", "trade_type", "shares", "price", "fee", "total_amount")
        self.tree_history = ttk.Treeview(
            table_frame, 
            columns=columns, 
            show="headings", 
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )

        scrollbar_y.config(command=self.tree_history.yview)
        scrollbar_x.config(command=self.tree_history.xview)

        # 各列のヘッダー名・幅の設定
        headers = {
            "id": ("ID", 50, "center"),
            "trade_date": ("取引日", 100, "center"),
            "code": ("銘柄コード", 90, "center"),
            "name": ("銘柄名", 180, "w"),
            "trade_type": ("取引区分", 80, "center"),
            "shares": ("株数", 90, "e"),
            "price": ("取引単価", 100, "e"),
            "fee": ("手数料", 80, "e"),
            "total_amount": ("受渡金額", 110, "e")
        }

        for col, (text, width, anchor) in headers.items():
            self.tree_history.heading(col, text=text)
            self.tree_history.column(col, width=width, anchor=anchor)

        self.tree_history.pack(fill=tk.BOTH, expand=True)

        # 初期データの読み込み
        self.load_history_data()
        
    def edit_selected_transaction(self):
        """ 選択された取引履歴を編集するダイアログを表示 """
        selected = self.tree_history.selection()
        if not selected:
            messagebox.showwarning("選択エラー", "編集する取引履歴を選択してください。")
            return

        item = self.tree_history.item(selected[0])
        values = item["values"]
        t_id = values[0]

        # データベースから最新の生データを取得
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, trade_date, code, name, trade_type, shares, price, fee FROM transactions WHERE id = ?", (t_id,))
        row = cursor.fetchone()
        if not row:
            return

        _, tdate, code, name, ttype, shares, price, fee = row

        # 編集モーダルウィンドウの作成
        dlg = tk.Toplevel(self.root)
        dlg.title(f"取引履歴の編集 (ID: {t_id})")
        dlg.geometry("380x360")
        dlg.grab_set()  # モーダル化

        fields = [
            ("取引日 (YYYY-MM-DD):", tdate),
            ("銘柄コード:", code),
            ("銘柄名:", name),
            ("取引区分 (現買/現売等):", ttype),
            ("株数:", str(shares)),
            ("取引単価 (円):", str(price)),
            ("手数料 (円):", str(fee))
        ]

        entries = {}
        for idx, (label_text, default_val) in enumerate(fields):
            tk.Label(dlg, text=label_text, font=("Helvetica", 9, "bold")).grid(row=idx, column=0, sticky="e", padx=10, pady=5)
            ent = tk.Entry(dlg, width=22)
            ent.insert(0, str(default_val))
            ent.grid(row=idx, column=1, padx=10, pady=5)
            entries[label_text] = ent

        def save_changes():
            try:
                new_date = entries["取引日 (YYYY-MM-DD):"].get().strip()
                new_code = entries["銘柄コード:"].get().strip().replace(".T", "").replace(".t", "")
                new_name = entries["銘柄名:"].get().strip()
                new_type = entries["取引区分 (現買/現売等):"].get().strip()
                new_shares = float(entries["株数:"].get().replace(",", ""))
                new_price = float(entries["取引単価 (円):"].get().replace(",", ""))
                new_fee = float(entries["手数料 (円):"].get().replace(",", ""))

                if not new_code or new_shares <= 0 or new_price < 0:
                    raise ValueError("入力値が正しくありません。")

                cursor.execute("""
                    UPDATE transactions 
                    SET trade_date = ?, code = ?, name = ?, trade_type = ?, shares = ?, price = ?, fee = ?
                    WHERE id = ?
                """, (new_date, new_code, new_name, new_type, new_shares, new_price, new_fee, t_id))
                self.conn.commit()

                dlg.destroy()
                messagebox.showinfo("更新完了", "取引履歴を更新しました！")
                
                # 表示の再ロード
                self.load_history_data()
                self.load_holdings_data()

            except Exception as e:
                messagebox.showerror("入力エラー", f"保存に失敗しました。入力内容を確認してください。\n{e}")

        tk.Button(dlg, text="💾 保存", font=("Helvetica", 10, "bold"), bg="#e1f5fe", command=save_changes, width=12).grid(row=len(fields), column=0, columnspan=2, pady=15)

    def delete_selected_transaction(self):
        """ 選択された取引履歴を削除 """
        selected = self.tree_history.selection()
        if not selected:
            messagebox.showwarning("選択エラー", "削除する取引履歴を選択してください。")
            return

        item = self.tree_history.item(selected[0])
        t_id = item["values"][0]

        if messagebox.askyesno("削除確認", f"ID: {t_id} の取引履歴を削除してもよろしいですか？\n※この操作は取り消せません。"):
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = ?", (t_id,))
            self.conn.commit()

            messagebox.showinfo("削除完了", "取引履歴を削除しました。")
            self.load_history_data()
            self.load_holdings_data()

    def load_history_data(self):
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, trade_date, code, name, trade_type, shares, price, fee FROM transactions ORDER BY trade_date DESC, id DESC")
        rows = cursor.fetchall()

        for tid, tdate, code, name, ttype, shares, price, fee in rows:
            t_str = str(ttype).strip()
            if "買" in t_str:
                total = (shares * price) + fee
            else:
                total = (shares * price) - fee

            self.tree_history.insert("", tk.END, values=(
                tid, tdate, code, name, ttype,
                f"{shares:,}株", f"¥{price:,.1f}", f"¥{fee:,.0f}", f"¥{total:,.0f}"
            ))

    def setup_add_trade_tab(self):
        """ 新規登録・SBI証券CSV取込タブのUI構築 """
        # --- SBI証券 CSVファイル一括読み込みエリア ---
        sbi_frame = tk.LabelFrame(
            self.tab_add_trade, text=" 🏦 SBI証券 CSVファイル一括読み込み ", 
            font=("Meiryo", 11, "bold"), padx=20, pady=15, bg="#ebf5fb"
        )
        sbi_frame.pack(padx=15, pady=15, fill=tk.X)

        btn_csv_import = tk.Button(
            sbi_frame, text="📁 SBI証券のCSVファイルを選択して取り込む", 
            font=("Meiryo", 10, "bold"), bg="#2980b9", fg="white", relief=tk.FLAT, padx=15, pady=6, 
            command=self.import_sbi_csv
        )
        btn_csv_import.pack(anchor="w", pady=5)

        # --- 手動登録エリア ---
        form_frame = tk.LabelFrame(
            self.tab_add_trade, text=" 📝 個別取引の手動登録 ", 
            font=("Meiryo", 10, "bold"), bg="white", padx=20, pady=15
        )
        form_frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(form_frame, text="取引日 (YYYY-MM-DD):", bg="white").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ent_date = tk.Entry(form_frame, width=15)
        self.ent_date.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        self.ent_date.grid(row=0, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="銘柄コード:", bg="white").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ent_code = tk.Entry(form_frame, width=15)
        self.ent_code.grid(row=1, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="銘柄名:", bg="white").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.ent_name = tk.Entry(form_frame, width=25)
        self.ent_name.grid(row=2, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="取引区分:", bg="white").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.cmb_type = ttk.Combobox(form_frame, values=["買付", "売却"], state="readonly", width=12)
        self.cmb_type.current(0)
        self.cmb_type.grid(row=3, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="株数:", bg="white").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.ent_shares = tk.Entry(form_frame, width=15)
        self.ent_shares.grid(row=4, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="取引単価 (円):", bg="white").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.ent_price = tk.Entry(form_frame, width=15)
        self.ent_price.grid(row=5, column=1, sticky=tk.W, pady=5)

        tk.Label(form_frame, text="手数料 (円):", bg="white").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.ent_fee = tk.Entry(form_frame, width=15)
        self.ent_fee.insert(0, "0")
        self.ent_fee.grid(row=6, column=1, sticky=tk.W, pady=5)

        btn_add = tk.Button(
            form_frame, text="取引を登録する", bg="#27ae60", fg="white", 
            font=("Meiryo", 10, "bold"), relief=tk.FLAT, padx=15, pady=5, 
            command=self.add_transaction
        )
        btn_add.grid(row=7, column=0, columnspan=2, pady=15)

    def import_sbi_csv(self):
        """ SBI証券のCSVファイルを読み込み、データベースへ一括登録する """
        file_path = filedialog.askopenfilename(
            title="SBI証券 CSVファイルの選択", 
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        lines = []
        # 文字コードを順に試してファイルを読み込み
        for enc in ["cp932", "utf-8", "shift_jis"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                if lines:
                    break
            except Exception:
                continue

        if not lines:
            messagebox.showerror("エラー", "CSVファイルを読み込めませんでした。文字コードをご確認ください。")
            return

        imported_count = 0
        cursor = self.conn.cursor()
        reader = csv.reader(lines)

        for row in reader:
            # SBI証券の取引明細に必要な最小列数がない行はスキップ
            if not row or len(row) < 10:
                continue

            trade_date_raw = row[0].strip()
            stock_name = row[1].strip()
            stock_code = row[2].strip()
            trade_type_raw = row[4].strip()

            # 取引区分の判定（買付 / 売却）
            if "買" in trade_type_raw:
                trade_type = "買付"
            elif "売" in trade_type_raw:
                trade_type = "売却"
            else:
                continue

            try:
                # 数値データのクレンジング（カンマ除去など）
                shares = int(float(row[8].replace(",", "").strip()))
                price = float(row[9].replace(",", "").strip())
                
                fee_str = row[10].replace(",", "").strip()
                fee = float(fee_str) if fee_str and fee_str != "--" else 0.0
                
                # 日付フォーマットの調整 (YYYY/MM/DD -> YYYY-MM-DD)
                trade_date = trade_date_raw.replace("/", "-")

                # 取引履歴テーブルへの挿入
                cursor.execute("""
                    INSERT INTO transactions (code, name, trade_type, shares, price, fee, trade_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (stock_code, stock_name, trade_type, shares, price, fee, trade_date))

                # 最新価格（現在値）の更新
                cursor.execute("""
                    INSERT INTO current_prices (code, current_price) VALUES (?, ?)
                    ON CONFLICT(code) DO UPDATE SET current_price = excluded.current_price
                """, (stock_code, price))

                imported_count += 1
            except ValueError:
                continue

        # データベースへの反映
        self.conn.commit()

        if imported_count > 0:
            messagebox.showinfo("成功", f"SBI証券の取引明細から {imported_count} 件を正しく取り込みました！")
            
            # 【エラー修正箇所】安全に画面表示を更新
            if hasattr(self, "load_holdings_data"):
                self.load_holdings_data()
            if hasattr(self, "load_history_data"):
                self.load_history_data()
        else:
            messagebox.showwarning("警告", "取引データを正常に抽出できませんでした。ファイルのフォーマットをご確認ください。")

    def add_transaction(self):
        tdate = self.ent_date.get().strip()
        code = self.ent_code.get().strip()
        name = self.ent_name.get().strip()
        ttype = self.cmb_type.get()
        shares_str = self.ent_shares.get().strip()
        price_str = self.ent_price.get().strip()
        fee_str = self.ent_fee.get().strip()

        if not (tdate and code and name and shares_str and price_str):
            messagebox.showwarning("入力エラー", "必須項目をすべて入力してください。")
            return

        try:
            shares = int(shares_str)
            price = float(price_str)
            fee = float(fee_str) if fee_str else 0.0
        except ValueError:
            messagebox.showerror("入力エラー", "株数・単価・手数料は半角数値で入力してください。")
            return

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO transactions (code, name, trade_type, shares, price, fee, trade_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (code, name, ttype, shares, price, fee, tdate))
        self.conn.commit()

        messagebox.showinfo("登録完了", f"{name} ({code}) の取引データを追加しました。")

        self.load_holdings_data()
        self.load_history_data()

        # 3-② 個別取引登録後に各入力フォームを初期化
        self.ent_code.delete(0, tk.END)
        self.ent_name.delete(0, tk.END)
        self.ent_shares.delete(0, tk.END)
        self.ent_price.delete(0, tk.END)
        self.ent_fee.delete(0, tk.END)
        self.ent_fee.insert(0, "0")


if __name__ == "__main__":
    root = tk.Tk()
    app = StockManagerApp(root)
    root.mainloop()