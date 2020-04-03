python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifarReal1 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifarReal2 --step_number 10000 --lr_2nd 0.01 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifarReal3 --step_number 10000 --lr_2nd 0.001 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifarReal4 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --epoch_2nd 6 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --unfreeze_secondStage 6 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifarReal5 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --epoch_2nd 11 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 10 --unfreeze_secondStage 11 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500

