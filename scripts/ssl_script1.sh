python3 main.py --lr_2nd 0.001 --epoch_1st 151 --epoch_2nd 10 --M 40 --M 80 --save_best_AUC_model --track_CE --step_number 1500 --experiment_name forget_4_ssl
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --experiment_name ssl_soloForget
