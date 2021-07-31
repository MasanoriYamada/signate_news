# 日本取引所グループ ニュース分析チャレンジ



## 環境setup

```sh
docker-compose up -d  # コンテナ起動
docker exec -it signate_news /bin/bash  # コンテナの中に入る
# pythonの実行
python xxx.py
# jupyter notebookの起動
jupyter notebook --port 7777 --ip=0.0.0.0 --allow-root 

exit # コンテナを出る
docker-compose down  # コンテナの削除
```

今回の計算ではgpu環境は不要です.

dockerとdocker-composeがあるcpu環境で動作します.



# 学習と推論手順

1. 学習: train.ipynb => checkpoints.pickleを生成 (実際に提出したモデルはsubmit/src/model/checkpoints.pickleにすでにおいてあります.)

2. checkpoints.pickleをsubmit/src/model/checkpoints.pickleにコピー

3. 予測: submit/src/predictor.py (sbumit以下のディレクトリがsignateに提出したものになります)



# 戦略

基本的な戦略は以下の記事の通り

zenn: https://zenn.dev/ymd/articles/a4b065d9b124e7

上の記事に加え、下降トレンドでチェックして(評価期間がSell in Mayなので)、シャープレシオを見ながらトップkのkの値を調整して最終的なモデルとしました。





