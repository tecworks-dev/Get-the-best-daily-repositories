<section align="center">
  <a href="https://github.com/leochen-g/dify-schedule" target="_blank">
    <img src="./static/logo.png" alt="Dify" width="260" />
  </a>
</section>

<h1 align="center">Difyワークフロースケジューラー</h1>

<p align="center">Dify公式にスケジュール機能が実装されていないため、定期的なワークフロー実行が困難でした。そこで、GitHub Actionsを活用して毎日定期的にワークフローを実行し、通知を送信する本プロジェクトが誕生しました。</p>

## 使用方法
自動化ワークフローには、オンラインでの即時利用とローカルでの自己ホスティングの2つの方法があります：

- クイックスタート（オンライン）：[使い方を見る](#使い方)
- Qinglongパネルでのタスク追加：[Qinglong使用方法を見る](#Qinglong使用方法)

## 使い方

自動実行タスク：複数のDifyワークフローに対応\
自動実行時間：北京時間 午前06:30

1. [リポジトリをフォーク](https://github.com/leochen-g/dify-schedule)

2. リポジトリ -> Settings -> Secrets -> New repository secretで、以下の環境変数を追加：

   | 名前 | 値 | 必須 |
   | --- | --- | --- |
   | DIFY_BASE_URL | DifyのベースURL。自己ホスティングの場合は、パブリックアクセス可能なURLを指定。未設定の場合はhttps://api.dify.ai/v1がデフォルト | 任意 |
   | DIFY_TOKENS | DifyワークフローのAPIキー。必須項目。複数指定する場合は`;`で区切る | 必須 |
   | DIFY_INPUTS | Difyワークフローで必要な変数。必須項目がある場合は必ず設定してください。JSON形式である必要があります | 任意 |
   | EMAIL_USER | 送信元メールアドレス（SMTP有効化が必要） | 任意 |
   | EMAIL_PASS | 送信元メールパスワード（SMTPパスワード） | 任意 |
   | EMAIL_TO | 受信者メールアドレス。複数指定する場合は`, `で区切る（例：`a@163.com, b@qq.com`） | 任意 |
   | DINGDING_WEBHOOK | DingDingボットのWEBHOOK URL | 任意 |
   | PUSHPLUS_TOKEN | [Pushplus](http://www.pushplus.plus/)で取得したトークン。WeChatプッシュ通知に対応 | 任意 |
   | SERVERPUSHKEY | [Server酱](https://sct.ftqq.com//)で取得したキー。WeChatプッシュ通知に対応 | 任意 |
   | WEIXIN_WEBHOOK | 企業WeChatボットのWEBHOOK URL | 任意 |
   | FEISHU_WEBHOOK | Feishuボットのウェブフック URL | 任意 |
   | AIBOTK_KEY | [AI秘書](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)のAPIキー | 任意 |
   | AIBOTK_ROOM_RECIVER | [AI秘書](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)で送信するグループ名 | 任意 |
   | AIBOTK_CONTACT_RECIVER | [AI秘書](https://wechat.aibotk.com?r=dBL0Bn&f=difySchedule)で送信する友達のニックネーム | 任意 |

4. リポジトリ -> Actionsで、Workflowsを確認して有効化。

## プレビュー

| Difyワークフロー実行 - WeChat | Difyワークフロー実行 - メール |
|:-----------:| :-----------:|
| ![Difyワークフロー実行](./static/chat.png) |![Difyワークフロー実行](./static/chat2.png) |

## Qinglong使用方法

1. Qinglongパネルでサブスクリプションを追加

```shell
ql repo https://github.com/leochen-g/ql-juejinhelper.git "ql_" "utils" "sdk"
```

2. パネルメニューの依存関係管理からNodeJsの依存関係として`axios`を追加

3. 環境変数に`DIFY_TOKENS`と`DIFY_BASE_URL`を追加

Qinglongパネルで環境変数`DIFY_TOKENS`と`DIFY_BASE_URL`を追加。複数のワークフロートークンは`;`で区切って指定可能

4. デフォルトでQinglong組み込みの通知を使用。適宜設定してください

## よくある質問

### Difyワークフロートークンの取得方法

Difyウェブサイトにアクセス

> ワークフローアプリケーションのみ対応。他のアプリケーションタイプは現在サポートされていません：

<img width="1156" alt="gettoken" src="./static/dify1.png">
<img width="1156" alt="gettoken" src="./static/dify2.png">

### アクセスできない場合

自己ホスティングのDifyを使用している場合は、パブリックアクセスが可能であることを確認してください。ローカルデプロイされたサービスにGitHub Workflowsからアクセスすることはできません。

### 実行エラーについて

1. まず、アプリケーションがワークフローアプリケーションであることを確認してください。現在はワークフローアプリケーションのみサポートしています

2. 必須入力変数がある場合は、環境変数`DIFY_INPUTS`を設定してください。JSON形式である必要があります。オンラインJSONバリデーターで確認してから入力することをお勧めします

3. エラーメッセージを注意深く確認し、指示に従って問題を解決するか、ログを添付してissuesを作成してください。その際は機密情報を必ず削除してください

## 貢献

アイデアは[プルリクエスト](https://github.com/leochen-g/dify-schedule/pulls)または[GitHubイシュー](https://github.com/leochen-g/dify-schedule/issues)として提出できます。

## サポート
### ☕️WeChatでの投げ銭
<img src="./static/zan.jpg" alt="WeChatでの投げ銭" width="300" />

## ライセンス

[MIT](./LICENSE)
