import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# .envファイルから設定を読み込む
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = os.environ.get("S3_BUCKET_NAME")

print("--- S3接続診断を開始します ---")
print(f"バケット名: {s3_bucket_name}")

# 1. 設定が.envファイルに存在するかチェック
if not all([aws_access_key_id, aws_secret_access_key, s3_bucket_name]):
    print("\n[診断結果: 失敗 ❌]")
    print("エラー: .envファイルに必要な情報（AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME）が設定されていません。")
    exit()
else:
    print("✅ .envファイルの設定は読み込めました。")

# 2. S3クライアントの初期化と認証情報の検証
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    print("✅ S3クライアントの初期化に成功しました。")
except Exception as e:
    print(f"\n[診断結果: 失敗 ❌]")
    print(f"エラー: S3クライアントの初期化中に予期せぬエラーが発生しました: {e}")
    exit()

# 3. バケットの存在とアクセス権のチェック
try:
    s3_client.head_bucket(Bucket=s3_bucket_name)
    print(f"✅ バケット「{s3_bucket_name}」へのアクセスに成功しました。")
except NoCredentialsError:
    print("\n[診断結果: 失敗 ❌]")
    print("エラー: AWSの認証情報（アクセスキー）が見つからないか、正しくありません。")
    exit()
except ClientError as e:
    error_code = e.response.get("Error", {}).get("Code")
    if error_code == '404':
        print(f"\n[診断結果: 失敗 ❌]")
        print(f"エラー: 指定されたバケット「{s3_bucket_name}」が見つかりません。バケット名が正しいか確認してください。")
    elif error_code == '403':
        print(f"\n[診断結果: 失敗 ❌]")
        print(f"エラー: バケット「{s3_bucket_name}」へのアクセスが拒否されました。IAMの権限設定を確認してください。")
    else:
        print(f"\n[診断結果: 失敗 ❌]")
        print(f"エラー: バケットへのアクセス中にエラーが発生しました: {e}")
    exit()
except Exception as e:
    print(f"\n[診断結果: 失敗 ❌]")
    print(f"エラー: 予期せぬエラーが発生しました: {e}")
    exit()

# 4. テストファイルのアップロードと削除
try:
    test_file_key = "s3_connection_test.txt"
    test_file_content = "S3 connection test successful."
    
    # アップロード
    s3_client.put_object(Bucket=s3_bucket_name, Key=test_file_key, Body=test_file_content)
    print(f"✅ テストファイルのアップロードに成功しました。")

    # 削除
    s3_client.delete_object(Bucket=s3_bucket_name, Key=test_file_key)
    print(f"✅ テストファイルの削除に成功しました。")

except Exception as e:
    print(f"\n[診断結果: 失敗 ❌]")
    print(f"エラー: ファイルのアップロード/削除テスト中にエラーが発生しました。IAMの権限（PutObject, DeleteObject）を確認してください。エラー: {e}")
    exit()


print("\n[最終診断結果: 成功 ✅]")
print("おめでとうございます！S3への接続設定はすべて正常です。")
print("--------------------------")
