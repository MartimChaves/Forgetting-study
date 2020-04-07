python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name cifar10_real_test --step_number 10000 --lr_2nd 0.001 --epoch_1st 201 --second_stage_data_name "cifar10" --second_stg_max_median_loss 1500 --first_stage_data_name "cifar10" --save_best_AUC_model --track_CE

python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --second_stage_num_classes 10 --experiment_name cifar10_complete_test --step_number 10000 --lr_2nd 0.001 --epoch_1st 200 --second_stage_data_name "svhn" --second_stg_max_median_loss 1500 --first_stage_data_name "cifar10" --freeze_earlySecondStage --freeze_epochWise --unfreeze_secondStage 6 --epoch_2nd 6

python3 parallel.py --M 40 --M 80 --epoch_1st 150
