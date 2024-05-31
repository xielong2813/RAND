
mkdir -p logs/LongForecasting

gpu=0
station_type=adaptive
features=M
model_name=RAND
pred_len=96

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/exchange_rate \
    --data_path exchange_rate.csv \
    --model_id exchange_336_$pred_len \
    --model RAND \
    --data custom \
    --features $features \
    --seq_len 336 \
    --L_slice 48 \
    --label_len 168 \
    --batch_size 500 \
    --pred_len $pred_len \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --gpu $gpu \
    --station_type $station_type \
    --period_len 6 \
    --learning_rate 0.001 \
    --station_lr 0.001 \
    --itr 3
