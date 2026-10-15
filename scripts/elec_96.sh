export CUDA_VISIBLE_DEVICES=2

model_name=SARAF

for seed in 42
do  
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ \
      --data_path electricity.csv \
      --model_id electricity_96_96 \
      --model $model_name \
      --data electricity \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --learning_rate 0.01 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.7 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ \
      --data_path electricity.csv \
      --model_id electricity_96_192 \
      --model $model_name \
      --data electricity \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 192 \
      --learning_rate 0.01 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.9 \
      --topm 20

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ \
      --data_path electricity.csv \
      --model_id electricity_96_336 \
      --model $model_name \
      --data electricity \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 336 \
      --learning_rate 0.01 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.5 \
      --topm 2

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ \
      --data_path electricity.csv \
      --model_id electricity_96_720 \
      --model $model_name \
      --data electricity \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --learning_rate 0.01 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --seed $seed \
      --itr 1 \
      --time_aware_weight 0.9 \
      --topm 20
done