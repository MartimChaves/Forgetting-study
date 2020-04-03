python3 main.py --experiment_name test_2 --epoch_1st 200 --first_stage_subset 3 5 4 7 --second_stage_subset 0 1 2 9 --step_number 15
python3 main.py --experiment_name subset1_3_5 --first_stage_subset [3,5] --epoch_1st 200
python3 main.py --experiment_name subset1_3_5_subset2_1_3_5_9 --first_stage_subset 0 2 --second_stage_subset 1 3 5 9 --epoch_1st 200

python3 main.py --experiment_name subset1_1_9_3_5_4_7subset2_0_2_6_8 --first_stage_subset 1 9 3 5 4 7 --second_stage_subset 0 2 6 8 --epoch_1st 200 --step_number 10

python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 1 2 3 4 5 6 7 8 9 --second_stage_num_classes 10 --second_stage_data_name "svhn" --second_stage_subset 0 1 2 3 4 5 6 7 8 9 --experiment_name 1_cifar10Complete_2_svhnComplete_lr_0.01 --step_number 8 --lr_2nd 0.01 --epoch_1st 150
python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 1 2 3 4 5 6 7 8 9 --second_stage_num_classes 10 --second_stage_data_name "svhn" --second_stage_subset 0 1 2 3 4 5 6 7 8 9 --experiment_name 1_cifar10Complete_2_svhnComplete_lr_0.001 --step_number 20 --lr_2nd 0.001 --epoch_1st 100
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name 1_cifar10sub_2_cifar10sub_random_lr_0.01 --step_number 10 --lr_2nd 0.01
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name 1_cifar10sub_2_cifar10sub_random_lr_0.001 --step_number 20 --lr_2nd 0.001
python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 10 --second_stage_subset 1 4 7 9 --experiment_name 1_cifar10sub_2_cifar10sub_real_lr_0.01 --step_number 10 --lr_2nd 0.01
python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 10 --second_stage_subset 1 4 7 9 --experiment_name 1_cifar10sub_2_cifar10sub_real_lr_0.001 --step_number 20 --lr_2nd 0.001

python3 main.py --experiment_name 3_cifar10Complete_2_svhnComplete_lr_0.1_v2 --step_number 50 --lr_2nd 0.1 --unfreeze_secondStage 10
python3 main.py --experiment_name 3_cifar10Complete_2_svhnComplete_lr_0.01_v2 --step_number 500 --lr_2nd 0.01 --unfreeze_secondStage 20
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name 3_cifar10sub_2_cifar10sub_random_lr_0.1 --step_number 50 --lr_2nd 0.1
python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name 3_cifar10sub_2_cifar10sub_random_lr_0.01 --step_number 100 --lr_2nd 0.01 --unfreeze_secondStage 20
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name 3_cifar10sub_2_cifar10sub_real_lr_0.1 --step_number 50 --lr_2nd 0.1
python3 main.py --first_stage_num_classes 6 --first_stage_noise_ration 0.4 --first_stage_noise_type "real_in_noise" --first_stage_subset 0 2 3 5 6 8 --second_stage_num_classes 4 --second_stage_subset 1 4 7 9 --experiment_name 3_cifar10sub_2_cifar10sub_real_lr_0.01 --step_number 100 --lr_2nd 0.01 --unfreeze_secondStage 20 

Traceback (most recent call last):
  File "main.py", line 302, in <module>
    title = measure_info[measure]['title']
  File "main.py", line 263, in main

  File "/home/martim/Documents/work_insight/study_forgetting_v2/utils/utils.py", line 167, in train_CrossEntropy
    loss.backward()
  File "/home/martim/Documents/work_insight/study_forgetting_v2/utils/utils.py", line 219, in accuracy_v2
    break
RuntimeError: invalid argument 5: k not in range for dimension at /pytorch/aten/src/THC/generic/THCTensorTopK.cu:21



python3 main.py --experiment_name 4_cifar10Complete_2_svhnComplete_lr_0.01_v2_noFreeze --step_number 10000 --lr_2nd 0.01 --epoch_2nd 1 --freeze_earlySecondStage False -CHECK

python3 main.py --experiment_name 4_cifar10Complete_2_svhnComplete_lr_0.01_v2_bigFreeze --step_number 10000 --lr_2nd 0.1 --epoch_2nd 6 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 10 --unfreeze_secondStage 6 

python3 main.py --experiment_name 4_cifar10Complete_2_svhnComplete_lr_0.01_v2_bigEpoch --step_number 10000 --lr_2nd 0.1 --epoch_2nd 9 --freeze_epochWise --freeze_earlySecondStage --M_2nd 5 --M_2nd 15 --unfreeze_secondStage 9 

python3 main.py --first_stage_num_classes 5 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 0 1 2 3 4 --second_stage_num_classes 5 --second_stage_subset 5 6 7 8 9 --experiment_name 4_cifar10sub_2_cifar10sub_random_lr_0.01 --step_number 200 --lr_2nd 0.01 --M 100 --M 150 --epoch_1st 201 --freeze_epochWise False --unfreeze_secondStage 25 


python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 2 14 23 35 48 51 69 74 87 90 --second_stage_num_classes 10 --second_stage_subset 4 19 26 31 44 58 63 72 85 97 --experiment_name cifar100Ran1 --step_number 10000 --lr_2nd 0.1 --epoch_1st 201 --M 100 --M 150 --second_stage_data_name "cifar100" --second_stg_max_median_loss 1500 --first_stage_data_name "cifar100"
python3 main.py --first_stage_num_classes 10 --first_stage_noise_ration 0.4 --first_stage_noise_type "random_in_noise" --first_stage_subset 2 14 23 35 48 51 69 74 87 90 --second_stage_num_classes 10 --second_stage_subset 4 19 26 31 44 58 63 72 85 97 --experiment_name cifar100Ran2 --step_number 10000 --lr_2nd 0.01 --epoch_1st 201 --second_stage_data_name "cifar100" --second_stg_max_median_loss 1500 --first_stage_data_name "cifar100"



