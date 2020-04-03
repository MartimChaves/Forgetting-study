python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name cifarRan1 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name cifarRan2 --step_number 10000 --lr_2nd 0.01 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name cifarRan3 --step_number 10000 --lr_2nd 0.001 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name cifarRan4 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --epoch_2nd 6 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --unfreeze_secondStage 6 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name cifarRan5 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --epoch_2nd 11 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 10 --unfreeze_secondStage 11 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500
