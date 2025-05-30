# Enhancing NER by Harnessing Multiple Datasets with Conditional Variational Autoencoders

基本的なNERのコードは[PURE](https://github.com/princeton-nlp/PURE/tree/main)を基に作成

## Docker
docker/で以下のコマンドでイメージ作成
```
docker build -t image_name .
```
コンテナ起動
```
docker run --gpus all --rm -it image_name /bin/bash
```
実験実行
```
accelerate launch src/run_entity.py \
    --do_train \
    --do_eval \
    --num_epoch 100 \
    --max_span_length 10 \
    --learning_rate=7e-4 \
    --task_learning_rate=7e-4 \
    --train_batch_size=16 \
    --eval_batch_size=64 \
    --context_window 150 \
    --task_list bc8 aioner_original\
    --data_dir {data_dir}\
    --model google-t5/t5-3b \
    --output_dir {output_dir} \
    --early_stopping 5\
    --train_shuffle \
    --target_label \
    --gradient_accumulation_steps 4 \
    --enc_hidden_dim 768 \
    --hidden_exp_cvae \
    --lookup_label_token_onehot "low" \
    --corpus_token_pos end \
    --use_t5 \
    --use_lora \
    --bool_quantize
```
