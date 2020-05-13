python3 main.py --lr_2nd 0.001 --epoch_1st 152 --epoch_2nd 10 --M 40 --M 80 --save_best_AUC_model --track_CE --step_number 1500 --experiment_name cifar10_random_forget_4_ssl --first_stage_num_classes 10 --first_stage_noise_type "random_in_noise" --second_stage_num_classes 10 --second_stage_data_name "cifar100" --second_stage_subset 2 14 23 35 48 51 69 74 87 90 --first_stage_data_name "cifar10"
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --experiment_name ssl_cifar10ranForget --train_root "./datasets/cifar10/data" --num_classes 10 --dataset "cifar10" --use_bmm --double_run --noise_type "random_in_noise"