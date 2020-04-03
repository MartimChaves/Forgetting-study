python3 main.py --experiment_name cifar10_complete1 --step_number 10000 --lr_2nd 0.1 --epoch_2nd 1 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete2 --step_number 10000 --lr_2nd 0.01 --epoch_2nd 1 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete3 --step_number 10000 --lr_2nd 0.001 --epoch_2nd 1 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete4 --step_number 10000 --lr_2nd 0.1 --epoch_2nd 6 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --unfreeze_secondStage 6 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete5 --step_number 10000 --lr_2nd 0.1 --epoch_2nd 6 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 5 --unfreeze_secondStage 6 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete6 --step_number 10000 --lr_2nd 0.1 --epoch_2nd 11 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 10 --unfreeze_secondStage 11 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500
python3 main.py --experiment_name cifar10_complete7 --step_number 10000 --lr_2nd 0.1 --epoch_2nd 17 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 10 --M_2nd 15 --unfreeze_secondStage 16 --first_stage_noise_type "real_in_noise" --second_stg_max_median_loss 1500


