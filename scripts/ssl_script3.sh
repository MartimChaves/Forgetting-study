python3 relabel.py --lr_2nd 0.1  --M 75 --M 110 --epoch_1st 130 --warmup_e 70 --experiment_name relabel_4_ssl
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --relabel --experiment_name ssl_soloRelabel
