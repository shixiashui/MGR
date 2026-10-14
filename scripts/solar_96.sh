export CUDA_VISIBLE_DEVICES=2

model_name=SARAF

for seed in 42
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/solar/ \
      --data_path solar_AL.txt \
      --model_id solar_96_96 \
      --model $model_name \
      --data solar \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.01 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/solar/ \
      --data_path solar_AL.txt \
      --model_id solar_96_192 \
      --model $model_name \
      --data solar \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.001 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 5

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/solar/ \
      --data_path solar_AL.txt \
      --model_id solar_96_336 \
      --model $model_name \
      --data solar \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.001 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 5

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/solar/ \
      --data_path solar_AL.txt\
      --model_id solar_96_720 \
      --model $model_name \
      --data solar \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.001 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --topm 5
done