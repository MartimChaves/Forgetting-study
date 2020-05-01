python3 parallel.py --M 40 --M 80 --epoch_1st 151

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --experiment_name ssl_soloParallel x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --experiment_name ssl_soloForget x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --relabel --experiment_name ssl_soloRelabel x

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --balanced_set --experiment_name ssl_soloP_balanced x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --balanced_set --experiment_name ssl_soloF_balanced x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --relabel --balanced_set --experiment_name ssl_soloR_balanced x

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --balanced_set --experiment_name ssl_FP_balanced x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --balanced_set --experiment_name ssl_FR_balanced x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --balanced_set --experiment_name ssl_PR_balanced x

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --balanced_set --agree_on_clean --experiment_name ssl_FP_balanced_agree x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --balanced_set --agree_on_clean --experiment_name ssl_FR_balanced_agree x
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --balanced_set --agree_on_clean --experiment_name ssl_PR_balanced_agree x

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --agree_on_clean --experiment_name ssl_FP_agree
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --agree_on_clean --experiment_name ssl_FR_agree
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --agree_on_clean --experiment_name ssl_PR_agree

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --forget --agree_on_clean --experiment_name ssl_PRF_agree
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --forget --balanced_set --experiment_name ssl_PRF_balanced

-----

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --experiment_name ssl_soloParallel_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --experiment_name ssl_soloForget_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --relabel --experiment_name ssl_soloRelabel_th0.1 --threshold 0.10

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --balanced_set --experiment_name ssl_soloP_balanced_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --balanced_set --experiment_name ssl_soloF_balanced_th0.1 --threshold 0.10 x +th0.4
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --relabel --balanced_set --experiment_name ssl_soloR_balanced_th0.1 --threshold 0.10

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --balanced_set --experiment_name ssl_FP_balanced_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --balanced_set --experiment_name ssl_FR_balanced_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --balanced_set --experiment_name ssl_PR_balanced_th0.1 --threshold 0.10

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --balanced_set --agree_on_clean --experiment_name ssl_FP_balanced_agree_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --balanced_set --agree_on_clean --experiment_name ssl_FR_balanced_agree_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --balanced_set --agree_on_clean --experiment_name ssl_PR_balanced_agree_th0.1 --threshold 0.10

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --parallel --agree_on_clean --experiment_name ssl_FP_agree_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --relabel --agree_on_clean --experiment_name ssl_FR_agree_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --agree_on_clean --experiment_name ssl_PR_agree_th0.1 --threshold 0.10

python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --forget --agree_on_clean --experiment_name ssl_PRF_agree_th0.1 --threshold 0.10
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --relabel --forget --balanced_set --experiment_name ssl_PRF_balanced_th0.1 --threshold 0.10

