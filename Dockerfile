# ベースとなるPythonの環境を指定
FROM python:3.11-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 必要なライブラリをインストールするため、まず要件定義ファイルをコピー
COPY requirements.txt .

# gunicorn（本番用サーバー）を含めてライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# プロジェクトの全てのファイルをコンテナにコピー
COPY . .

# コンテナがリクエストを待ち受けるポート番号を指定
EXPOSE 8080

# タイムアウト時間を180秒に延長（Herokuでは環境変数PORTを使用）
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 180 main:app"]
