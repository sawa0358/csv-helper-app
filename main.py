import os
import io
import pandas as pd
# import google.generativeai as genai # AI機能は後で追加します
import boto3 # S3機能は後で追加します
from flask import Flask, request, jsonify, render_template_string, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback

# .envファイルから環境変数を読み込む
load_dotenv()

# Flaskアプリケーションの初期化
app = Flask(__name__)

# --- 環境変数と定数の設定 ---
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_PREVIOUS_FILE_KEY = "previous_file.csv" # S3に保存する際ファイル名

# --- AIモデルとS3クライアントの初期化 ---
# AIモデルは後で設定します
s3_client = None
if all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
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
    print("警告: S3関連の環境変数が不足しています。S3連携機能は無効になります。")


# =================================================================
# ヘルパー関数 (Helper Functions)
# =================================================================

def read_csv_from_stream(file_stream):
    """ファイルストリームからCSVをPandas DataFrameとして読み込む"""
    encodings_to_try = ['utf-8-sig', 'utf-8', 'shift-jis', 'cp932']
    for encoding in encodings_to_try:
        try:
            file_stream.seek(0) # 試すたびにファイルの先頭に戻る
            return pd.read_csv(file_stream, encoding=encoding)
        except Exception:
            continue # 失敗したら次のエンコーディングを試す
    
    # すべてのエンコーディングで失敗した場合
    raise ValueError("ファイルの文字コードを認識できませんでした。UTF-8またはShift-JISで保存してください。")

# =================================================================
# Flask ルート (APIエンドポイント)
# =================================================================

@app.route('/')
def index():
    """メインのHTMLページをレンダリングする"""
    # index.htmlを直接読み込んでテンプレートとして使用
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return "index.htmlが見つかりません。先にファイルを作成してください。", 404
@app.route('/api/process', methods=['POST'])
def process_csv():
    """CSVの整形・分析処理を行うメインAPI"""
    try:
        # --- ファイルの読み込み ---
        latest_file = request.files.get('latest_file')
        previous_file = request.files.get('previous_file')
        
        if not latest_file:
            return jsonify({'error': '「最新の案件ファイル」がアップロードされていません。'}), 400

        df_latest = read_csv_from_stream(latest_file.stream)
        
        original_row_count = len(df_latest)
        processing_log = [f"最新ファイル「{secure_filename(latest_file.filename)}」を読み込みました。({original_row_count}行)"]

        # --- ファイルの比較・差分抽出 ---
        if previous_file:
            df_previous = read_csv_from_stream(previous_file.stream)
            processing_log.append(f"前回ファイル「{secure_filename(previous_file.filename)}」を読み込みました。({len(df_previous)}行)")
            
            try:
                df_concat = pd.concat([df_latest, df_previous])
                df_unique = df_concat.drop_duplicates(keep=False)
                df_latest = df_latest.loc[df_unique.index].dropna(how='all')
                new_rows = len(df_latest)
                processing_log.append(f"差分抽出の結果、{new_rows}件の新規案件が見つかりました。")
            except Exception as e:
                 processing_log.append(f"警告: ファイル比較中にエラーが発生しました。差分抽出はスキップされます。詳細: {e}")

        # --- 日付によるフィルタリング ---
        filter_date_column = request.form.get('filter_date_column')
        filter_date_value = request.form.get('filter_date_value')
        if filter_date_column and filter_date_value:
            if filter_date_column not in df_latest.columns:
                 return jsonify({'error': f'フィルタリング対象の列「{filter_date_column}」がファイルに存在しません。'}), 400
            
            rows_before_filter = len(df_latest)
            df_latest[filter_date_column] = pd.to_datetime(df_latest[filter_date_column], errors='coerce')
            df_latest = df_latest[df_latest[filter_date_column] >= pd.to_datetime(filter_date_value)]
            rows_after_filter = len(df_latest)
            processing_log.append(f"日付フィルタリング: 「{filter_date_column}」が {filter_date_value} 以降のデータに絞り込みました。({rows_before_filter}行 -> {rows_after_filter}行)")

        # --- キーワードによる絞り込み ---
        keyword_column = request.form.get('keyword_column')
        keywords_str = request.form.get('keywords')
        search_type = request.form.get('search_type') # 'AND' or 'OR'
        if keyword_column and keywords_str:
            if keyword_column not in df_latest.columns:
                return jsonify({'error': f'キーワード検索対象の列「{keyword_column}」がファイルに存在しません。'}), 400
            
            # キーワードを改行で分割し、空の行は無視する
            keywords = [kw.strip() for kw in keywords_str.splitlines() if kw.strip()]
            if keywords:
                rows_before_filter = len(df_latest)
                # 検索対象の列を文字列に変換しておく（数値などが含まれていても安全に検索できるように）
                search_series = df_latest[keyword_column].astype(str)

                if search_type == 'AND':
                    # 全てのキーワードを含む行を検索
                    condition = pd.Series(True, index=df_latest.index)
                    for kw in keywords:
                        condition &= search_series.str.contains(kw, na=False)
                else: # OR検索
                    # いずれかのキーワードを含む行を検索
                    condition = pd.Series(False, index=df_latest.index)
                    for kw in keywords:
                        condition |= search_series.str.contains(kw, na=False)
                
                df_latest = df_latest[condition]
                rows_after_filter = len(df_latest)
                processing_log.append(f"キーワード検索: 「{keyword_column}」で {search_type} 検索を実行しました。({rows_before_filter}行 -> {rows_after_filter}行)")

        # --- S3へのファイル保存 ---
        if s3_client:
            try:
                csv_buffer = io.StringIO()
                df_latest.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=S3_PREVIOUS_FILE_KEY,
                    Body=csv_buffer.getvalue()
                )
                processing_log.append(f"処理結果を次回使用のためS3バケットに「{S3_PREVIOUS_FILE_KEY}」として保存しました。")
            except Exception as e:
                processing_log.append(f"警告: S3へのファイル保存に失敗しました。詳細: {e}")

        # --- 処理結果を画面に返す ---
        return jsonify({
            'message': '処理が正常に完了しました。',
            'log': processing_log,
            'rowCount': len(df_latest),
            'csvData': df_latest.to_csv(index=False, encoding='utf-8-sig')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'サーバーで予期せぬエラーが発生しました: {str(e)}'}), 500
    
    @app.route('/api/load_previous', methods=['GET'])
    def load_previous_file_from_s3():
        """S3から前回保存したファイルを取得する"""
        if not s3_client:
            return jsonify({'error': 'S3が設定されていないため、前回ファイルを取得できません。'}), 503
        
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_PREVIOUS_FILE_KEY)
            file_content = response['Body'].read()
            
            # ファイルをクライアントに送信
            return send_file(
                io.BytesIO(file_content),
                mimetype='text/csv',
                as_attachment=True,
                download_name=S3_PREVIOUS_FILE_KEY
            )
        except s3_client.exceptions.NoSuchKey:
            return jsonify({'error': f'S3バケットに「{S3_PREVIOUS_FILE_KEY}」が見つかりません。'}), 404
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'S3からのファイル取得中にエラーが発生しました: {str(e)}'}), 500


# =================================================================
# アプリケーション実行
# =================================================================
if __name__ == '__main__':
    # Procfileでgunicornを使うため、デバッグモードは開発時のみ有効
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))