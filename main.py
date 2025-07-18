import os
import io
import pandas as pd
import google.generativeai as genai
import json
import boto3
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify, render_template_string, Response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback

# .envファイルから環境変数を読み込む
load_dotenv()

# Flaskアプリケーションの初期化
app = Flask(__name__)

# --- 環境変数と定数の設定 ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_POINTER_FILE_KEY = "__latest_filename_pointer.txt" 
S3_TEMPLATES_KEY = "prompt_templates.json" # ★★★ テンプレート保存用のファイル名 ★★★

# --- AIモデルとS3クライアントの初期化 ---
model = None
try:
    if GEMINI_API_KEY:
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("Geminiモデルが正常に初期化されました。")
    else:
        print("警告: GEMINI_API_KEYが設定されていません。AI機能は無効になります。")
except Exception as e:
    print(f"エラー: Geminiモデルの初期化に失敗しました。 {e}")

s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("S3クライアントが正常に初期化されました。")
    except Exception as e:
        print(f"エラー: S3クライアントの初期化に失敗しました。 {e}")
else:
    print("警告: S3接続情報が不足しているため、S3連携機能は無効になります。")

def read_csv_from_stream(file_stream):
    encodings_to_try = ['utf-8-sig', 'utf-8', 'shift-jis', 'cp932']
    for encoding in encodings_to_try:
        try:
            file_stream.seek(0)
            return pd.read_csv(file_stream, encoding=encoding, dtype=str)
        except Exception:
            continue
    raise ValueError("ファイルの文字コードを認識できませんでした。UTF-8またはShift-JISで保存してください。")

@app.route('/')
def index():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "index.htmlが見つかりません。先にファイルを作成してください。", 404

@app.route('/api/process', methods=['POST'])
def process_csv():
    # (この関数の中身は変更ありません)
    try:
        latest_file = request.files.get('latest_file')
        if not latest_file:
            return jsonify({'error': '「最新の案件ファイル」がアップロードされていません。'}), 400
        
        df_latest = read_csv_from_stream(latest_file.stream)
        original_row_count = len(df_latest)
        processing_log = [f"最新ファイル「{secure_filename(latest_file.filename)}」を読み込みました。({original_row_count}行)"]

        previous_file = request.files.get('previous_file')
        if previous_file:
            try:
                rows_before_diff = len(df_latest)
                df_previous = read_csv_from_stream(previous_file.stream)
                common_columns = list(set(df_latest.columns) & set(df_previous.columns))
                if not common_columns:
                    raise ValueError("差分比較のため、2つのファイル間で共通の列が1つも見つかりませんでした。")
                merged_df = pd.merge(df_latest, df_previous, on=common_columns, how='left', indicator=True)
                diff_df = merged_df[merged_df['_merge'] == 'left_only']
                df_latest = diff_df.drop(columns=['_merge'])
                processing_log.append(f"差分抽出: 新規案件に絞り込みました。({rows_before_diff}行 -> {len(df_latest)}行)")
            except Exception as e:
                error_message = f"差分抽出中にエラーが発生したため、以降の処理を中断しました。エラー: {e}"
                processing_log.append(f"【重要】{error_message}")
                return jsonify({'message': '処理が中断されました。','log': processing_log,'rowCount': 0,'csvData': ''})

        if not df_latest.empty:
            def apply_date_filter(df, col_name, date_val, log_list, filter_num):
                if col_name and date_val:
                    if col_name in df.columns:
                        rows_before = len(df)
                        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                        df.dropna(subset=[col_name], inplace=True)
                        df = df[df[col_name] >= pd.to_datetime(date_val)]
                        log_list.append(f"日付フィルタ{filter_num}: 「{col_name}」で {rows_before}行 -> {len(df)}行")
                return df

            df_latest = apply_date_filter(df_latest, request.form.get('filter_date_column_1'), request.form.get('filter_date_value_1'), processing_log, "①")
            df_latest = apply_date_filter(df_latest, request.form.get('filter_date_column_2'), request.form.get('filter_date_value_2'), processing_log, "②")

            keyword_column = request.form.get('keyword_column')
            keywords_str = request.form.get('keywords')
            search_type = request.form.get('search_type')
            if keyword_column and keywords_str and keyword_column in df_latest.columns:
                keywords = [kw.strip() for kw in keywords_str.splitlines() if kw.strip()]
                if keywords:
                    rows_before = len(df_latest)
                    condition = df_latest[keyword_column].str.contains('|'.join(keywords), na=False) if search_type == 'OR' else pd.concat([df_latest[keyword_column].str.contains(kw, na=False) for kw in keywords], axis=1).all(axis=1)
                    df_latest = df_latest[condition]
                    processing_log.append(f"キーワード検索: {rows_before}行 -> {len(df_latest)}行")

            ai_date_format_enabled = request.form.get('ai_date_format_enabled') == 'on'
            ai_date_format_column = request.form.get('ai_date_format_column')
            if ai_date_format_enabled and ai_date_format_column and model and ai_date_format_column in df_latest.columns:
                processing_log.append(f"AIによる日付自動整形を開始 (対象列: {ai_date_format_column})")
                original_date_series = df_latest[ai_date_format_column].fillna('').astype(str)
                all_formatted_dates = []
                batch_size = 100
                try:
                    for i in range(0, len(original_date_series), batch_size):
                        batch_data_list = original_date_series.iloc[i:i+batch_size].to_list()
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
                        all_formatted_dates.extend(batch_formatted_dates if isinstance(batch_formatted_dates, list) and len(batch_formatted_dates) == len(batch_data_list) else batch_data_list)
                    if len(all_formatted_dates) == len(df_latest):
                        df_latest[ai_date_format_column] = all_formatted_dates
                        processing_log.append("AIによる日付自動整形が完了しました。")
                except Exception as e:
                    processing_log.append(f"警告: AI日付整形中にエラーが発生したため、スキップしました。エラー: {e}")

        ai_processing_prompt = request.form.get('ai_prompt')
        final_df = df_latest.copy()
        if ai_processing_prompt and model and not final_df.empty:
            processing_log.append("ユーザー指示のAIプロンプト処理を開始します...")
            csv_for_prompt = final_df.to_csv(index=True)
            final_prompt = f"""
# あなたのタスク
あなたは、CSVデータの中から、ユーザーが定義したルールに合致する行を見つけ出し、その**行番号（インデックス）**だけを返す、超高性能なデータフィルタリング専門AIです。

# ユーザーの指示
{ai_processing_prompt}

# 処理前のCSVデータ（先頭に行番号が付いています）
```csv
{csv_for_prompt}
```

# 実行手順
1. まず、「ユーザーの指示」を注意深く読み、抽出条件と除外条件を正確に理解してください。
2. 指示の中に「〇〇列を調べて」のように特定の列名が指定されている場合、その列の値を最優先で評価してください。なければ、すべての列を総合的に判断してください。
3. 次に、「処理前のCSVデータ」を一行ずつ確認します。
4. 各行が「ユーザーの指示」に合致するかを判断します。
5. 条件に合致した行の**行番号（インデックス）**だけを、結果として蓄積します。

# 絶対的なルール
- **出力は、条件に合致した行の行番号（インデックス）を、JSON形式の数値配列として、ただ一つだけ返してください。**
- **例:** `[0, 5, 12, 23]`
- 会話や説明、マークダウン(` ```json ... ```)など、余計な情報は一切含めないでください。
- もし、どの行も条件に合致しなかった場合は、空の配列 `[]` を返してください。
"""
            try:
                generation_config = genai.types.GenerationConfig(temperature=0)
                safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE','HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE','HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE','HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',}
                response = model.generate_content(final_prompt, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 180})
                cleaned_response_text = response.text.strip().replace("`", "").replace("json", "")
                matched_indices = json.loads(cleaned_response_text)
                if isinstance(matched_indices, list):
                    valid_indices = [idx for idx in matched_indices if idx in final_df.index]
                    final_df = final_df.loc[valid_indices]
                    processing_log.append(f"ユーザー指示のAIプロンプト処理が完了しました。{len(valid_indices)}件の行が合致しました。")
            except Exception as e:
                processing_log.append(f"警告: AIプロンプト処理中にエラーが発生しました。AI処理前のデータを結果とします。エラー: {e}")
        
        return jsonify({
            'message': '処理が正常に完了しました。',
            'log': processing_log,
            'rowCount': len(final_df),
            'csvData': final_df.to_csv(index=False, encoding='utf-8-sig')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'サーバーで予期せぬエラーが発生しました: {str(e)}'}), 500

@app.route('/api/save_latest_file', methods=['POST'])
def save_latest_file_to_s3():
    if not s3_client: return jsonify({'error': 'S3が設定されていません。'}), 503
    file_to_save = request.files.get('file_to_save')
    if not file_to_save: return jsonify({'error': '保存するファイルが見つかりません。'}), 400
    try:
        original_filename = secure_filename(file_to_save.filename)
        file_to_save.seek(0)
        s3_client.upload_fileobj(file_to_save, S3_BUCKET_NAME, original_filename)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=S3_POINTER_FILE_KEY, Body=original_filename.encode('utf-8'))
        return jsonify({'message': f'ファイル「{original_filename}」をS3に保存しました。'})
    except ClientError as e:
        return jsonify({'error': f'S3へのファイル保存に失敗しました: {e}'}), 500

@app.route('/api/load_previous_file', methods=['GET'])
def load_previous_file_from_s3():
    if not s3_client: return jsonify({'error': 'S3が設定されていません。'}), 503
    try:
        pointer_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_POINTER_FILE_KEY)
        previous_filename = pointer_object['Body'].read().decode('utf-8')
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=previous_filename)
        file_content = s3_object['Body'].read()
        return Response(file_content, mimetype='text/csv', headers={'Content-Disposition': f'attachment;filename={previous_filename}'})
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return jsonify({'error': 'S3に前回ファイルが見つかりませんでした。'}), 404
        return jsonify({'error': f'S3からのファイル取得に失敗しました: {e}'}), 500

# ★★★ ここからテンプレート用の新しい機能 ★★★

@app.route('/api/templates', methods=['GET'])
def get_templates_from_s3():
    """S3からテンプレートファイル(JSON)を取得する"""
    if not s3_client: return jsonify({'error': 'S3が設定されていません。'}), 503
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_TEMPLATES_KEY)
        templates_content = s3_object['Body'].read().decode('utf-8')
        return jsonify(json.loads(templates_content))
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return jsonify([]) # ファイルがなければ空のリストを返す
        return jsonify({'error': f'S3からのテンプレート取得に失敗: {e}'}), 500

@app.route('/api/templates', methods=['POST'])
def save_templates_to_s3():
    """テンプレートファイル(JSON)をS3に保存する"""
    if not s3_client: return jsonify({'error': 'S3が設定されていません。'}), 503
    
    templates_data = request.get_json()
    if templates_data is None:
        return jsonify({'error': '保存するテンプレートデータがありません。'}), 400
        
    try:
        # Pythonのリスト/辞書をJSON形式の文字列に変換し、S3にアップロード
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=S3_TEMPLATES_KEY,
            Body=json.dumps(templates_data, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        return jsonify({'message': 'テンプレートをS3に保存しました。'})
    except ClientError as e:
        return jsonify({'error': f'S3へのテンプレート保存に失敗: {e}'}), 500

# ★★★ ここまでテンプレート用の新しい機能 ★★★

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    # (この部分は変更なし)
    if not model: return jsonify({'error': 'AI機能が設定されていないため、チャットは実行できません。'}), 503
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
