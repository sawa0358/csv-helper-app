import os
import io
import pandas as pd
import google.generativeai as genai
import json
# import boto3 # S3機能は一時的に無効化
from flask import Flask, request, jsonify, render_template_string, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback

# .envファイルから環境変数を読み込む
load_dotenv()

# Flaskアプリケーションの初期化
app = Flask(__name__)

# --- 環境変数と定数の設定 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Geminiモデルが正常に初期化されました。")
    else:
        print("警告: GEMINI_API_KEYが設定されていません。AI機能は無効になります。")
except Exception as e:
    print(f"エラー: Geminiモデルの初期化に失敗しました。 {e}")

# =================================================================
# ヘルパー関数
# =================================================================

def read_csv_from_stream(file_stream):
    """ファイルストリームからCSVをPandas DataFrameとして読み込む"""
    encodings_to_try = ['utf-8-sig', 'utf-8', 'shift-jis', 'cp932']
    for encoding in encodings_to_try:
        try:
            file_stream.seek(0)
            # すべての列を文字列として読み込み、型の不一致によるマージエラーを防ぐ
            return pd.read_csv(file_stream, encoding=encoding, dtype=str)
        except Exception:
            continue
    raise ValueError("ファイルの文字コードを認識できませんでした。UTF-8またはShift-JISで保存してください。")

# =================================================================
# Flask ルート
# =================================================================

@app.route('/')
def index():
    """メインのHTMLページをレンダリングする"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "index.htmlが見つかりません。先にファイルを作成してください。", 404

@app.route('/api/process', methods=['POST'])
def process_csv():
    """CSVを処理するメインAPI"""
    try:
        # === ステップ1: ファイル読み込み ===
        latest_file = request.files.get('latest_file')
        if not latest_file:
            return jsonify({'error': '「最新の案件ファイル」がアップロードされていません。'}), 400
        
        df_latest = read_csv_from_stream(latest_file.stream)
        original_row_count = len(df_latest)
        processing_log = [f"最新ファイル「{secure_filename(latest_file.filename)}」を読み込みました。({original_row_count}行)"]

        # === ステップ2: 【最優先】差分抽出 ===
        previous_file = request.files.get('previous_file')
        if previous_file:
            try:
                rows_before_diff = len(df_latest)
                df_previous = read_csv_from_stream(previous_file.stream)
                
                common_columns = list(set(df_latest.columns) & set(df_previous.columns))
                if not common_columns:
                    raise ValueError("差分比較のため、2つのファイル間で共通の列が1つも見つかりませんでした。")

                # 高速なマージ（結合）を使って差分を検出
                merged_df = pd.merge(df_latest, df_previous, on=common_columns, how='left', indicator=True)
                
                # 最新のファイルにしか存在しない行（_merge == 'left_only'）だけを抽出
                diff_df = merged_df[merged_df['_merge'] == 'left_only']
                
                # 不要になった_merge列を削除して、元のdf_latestを置き換える
                df_latest = diff_df.drop(columns=['_merge'])

                processing_log.append(f"差分抽出: 新規案件に絞り込みました。({rows_before_diff}行 -> {len(df_latest)}行)")
            
            except Exception as e:
                # 差分抽出に失敗した場合は、ここで処理を中断し、以降の処理を行わない
                error_message = f"差分抽出中にエラーが発生したため、以降の処理を中断しました。エラー: {e}"
                processing_log.append(f"【重要】{error_message}")
                return jsonify({'message': '処理が中断されました。','log': processing_log,'rowCount': 0,'csvData': ''})

        # 以降の処理は、データが存在する場合（df_latestが空でない場合）のみ実行
        if not df_latest.empty:
            # === ステップ3: 下準備フィルター ===
            filter_date_column = request.form.get('filter_date_column')
            filter_date_value = request.form.get('filter_date_value')
            if filter_date_column and filter_date_value:
                if filter_date_column in df_latest.columns:
                    rows_before = len(df_latest)
                    df_latest[filter_date_column] = pd.to_datetime(df_latest[filter_date_column], errors='coerce')
                    df_latest.dropna(subset=[filter_date_column], inplace=True)
                    df_latest = df_latest[df_latest[filter_date_column] >= pd.to_datetime(filter_date_value)]
                    processing_log.append(f"日付フィルタ: {rows_before}行 -> {len(df_latest)}行")

            keyword_column = request.form.get('keyword_column')
            keywords_str = request.form.get('keywords')
            search_type = request.form.get('search_type')
            if keyword_column and keywords_str:
                if keyword_column in df_latest.columns:
                    keywords = [kw.strip() for kw in keywords_str.splitlines() if kw.strip()]
                    if keywords:
                        rows_before = len(df_latest)
                        if search_type == 'OR':
                            condition = df_latest[keyword_column].str.contains('|'.join(keywords), na=False)
                        else: # AND
                            condition = pd.concat([df_latest[keyword_column].str.contains(kw, na=False) for kw in keywords], axis=1).all(axis=1)
                        df_latest = df_latest[condition]
                        processing_log.append(f"キーワード検索: {rows_before}行 -> {len(df_latest)}行")

            # === ステップ4: AIによる日付自動整形 ===
            ai_date_format_enabled = request.form.get('ai_date_format_enabled') == 'on'
            ai_date_format_column = request.form.get('ai_date_format_column')
            if ai_date_format_enabled and ai_date_format_column and model:
                if ai_date_format_column in df_latest.columns:
                    processing_log.append(f"AIによる日付自動整形を開始 (対象列: {ai_date_format_column})")
                    original_date_series = df_latest[ai_date_format_column].fillna('').astype(str)
                    all_formatted_dates = []
                    batch_size = 100
                    try:
                        for i in range(0, len(original_date_series), batch_size):
                            batch_data_list = original_date_series.iloc[i:i+batch_size].to_list()
                            processing_log.append(f"  - 日付整形バッチ処理中... ({i+1}～{min(i+batch_size, len(original_date_series))}行目)")
                            date_formatting_prompt = f"""
# あなたのタスク
あなたは、日本の様々な日付表現を、厳格なルールに従って「YYYY-MM-DD」形式の文字列に変換する、超高性能な日付整形専門AIです。
これからJSON形式の文字列配列を受け取ります。各文字列をルールに従って解析し、変換結果をJSON配列で返してください。

# 厳格なルール
- **出力形式:** 必ず入力と同じ数の要素を持つJSON配列の文字列（例: `["2025-07-01", "", "2025-08-15"]`）として回答してください。会話や説明、マークダウン(` ```json ... ```)は一切含めないでください。
- **基本変換:** 和暦(令和,平成,昭和)、西暦、区切り文字（「.」「・」「/」など）を解釈し、「YYYY-MM-DD」に変換してください。（例: 「R7.7.1」→「2025-07-01」）
- **曜日と不要な文字の削除:** 曜日（(月),(火)など）や前後の不要な文字列はすべて削除してください。
- **文字列のみの場合:** 日付と解釈できない文字列のみの場合は、空文字列（""）にしてください。
- **範囲表現:** 「A〜B」のような範囲を示す場合は、必ず「未来の方の日付」だけを残してください。AとBのどちらが未来かは、日付を比較して判断してください。（例: 「2025-07-01〜令和7年6月30日」→「2025-07-01」）
- **複数日付:** 複数の日付が並んでいる場合も、「未来の方の日付」だけを残してください。
- **文字と日付の混在:** 「A:文字〜B:日付」のように、日付と解釈できないものが含まれる場合は、日付と解釈できる方だけを変換対象にしてください。

# 変換対象のJSON配列
{json.dumps(batch_data_list)}
"""
                            response = model.generate_content(date_formatting_prompt)
                            cleaned_response_text = response.text.strip().replace("`", "").replace("json", "")
                            batch_formatted_dates = json.loads(cleaned_response_text)
                            if isinstance(batch_formatted_dates, list) and len(batch_formatted_dates) == len(batch_data_list):
                                all_formatted_dates.extend(batch_formatted_dates)
                            else:
                                all_formatted_dates.extend(batch_data_list)
                        if len(all_formatted_dates) == len(df_latest):
                            df_latest[ai_date_format_column] = all_formatted_dates
                            processing_log.append("AIによる日付自動整形が完了しました。")
                    except Exception as e:
                        processing_log.append(f"警告: AI日付整形中にエラー: {e}")

        # === ステップ5: ユーザー指示のAIプロンプト処理 ===
        ai_processing_prompt = request.form.get('ai_prompt')
        final_df = df_latest.copy()
        if ai_processing_prompt and model and not final_df.empty:
            processing_log.append("ユーザー指示のAIプロンプト処理を開始します...")
            filtered_csv_data = final_df.to_csv(index=False)
            final_prompt = f"""
# あなたのタスク
あなたは、CSVデータを自在に加工・整形する、優秀なデータ処理専門家です。
これから渡される「処理前のCSVデータ」を、ユーザーからの「処理内容の指示」に厳密に従って処理し、その結果だけを、新しいCSVデータとして出力してください。

# ルール
- 出力は、ヘッダー行を含む、完全なCSV形式の文字列だけにしてください。
- 説明や前置き、`「はい、処理しました」`のような会話、```csv ... ``` といったマークダウンは一切含めないでください。
- 指示にない列を勝手に削除したり、列の順序を変えたりしないでください。
- 指示が曖昧な場合は、最も一般的で妥当だと思われる解釈で処理を実行してください。

# 処理前のCSVデータ
```csv
{filtered_csv_data}
```

# 処理内容の指示
{ai_processing_prompt}
"""
            try:
                response = model.generate_content(final_prompt)
                processed_csv_text = response.text.strip().replace("`", "")
                if processed_csv_text and len(processed_csv_text.splitlines()) > 1:
                    final_df = pd.read_csv(io.StringIO(processed_csv_text))
                    processing_log.append("ユーザー指示のAIプロンプト処理が完了しました。")
                else:
                    processing_log.append("警告: AIがプロンプト処理で有効なCSVを返しませんでした。")
            except Exception as e:
                processing_log.append(f"警告: AIプロンプト処理中にエラー: {e}")
        
        # === ステップ6: 最終結果を返却 ===
        return jsonify({
            'message': '処理が正常に完了しました。',
            'log': processing_log,
            'rowCount': len(final_df),
            'csvData': final_df.to_csv(index=False, encoding='utf-8-sig')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'サーバーで予期せぬエラーが発生しました: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """アップロードされたCSVの内容についてAIと対話する"""
    if not model:
        return jsonify({'error': 'AI機能が設定されていないため、チャットは実行できません。'}), 503
    data = request.get_json()
    if not data: return jsonify({'error': 'リクエストデータが不正です。'}), 400
    user_question = data.get('question')
    csv_content_string = data.get('csv_content')
    if not user_question: return jsonify({'error': '質問が入力されていません。'}), 400
    if not csv_content_string: return jsonify({'error': '分析対象のCSVデータが見つかりません。'}), 400
    try:
        df = pd.read_csv(io.StringIO(csv_content_string))
        csv_for_prompt = df.to_string(index=False)
        prompt = f"""あなたは優秀なデータアナリストです。以下のCSVデータの内容を分析し、ユーザーからの質問に簡潔かつ的確に答えてください。表形式での回答が適切と判断した場合は、マークダウン形式のテーブルを使用して回答してください。

# CSVデータ:
```text
{csv_for_prompt}
```

# ユーザーからの質問:
{user_question}

# 回答:
"""
        response = model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'AIとの対話中にエラーが発生しました: {str(e)}'}), 500

# =================================================================
# アプリケーション実行
# =================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
