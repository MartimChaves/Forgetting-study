python3 parallel.py --M 40 --M 80 --epoch_1st 152 --first_stage_data_name "cifar10" --first_stage_noise_type "real_in_noise" --first_stage_num_classes 10 --experiment_name cifar10real_parallel_4_ssl
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --parallel --experiment_name ssl_cifar10realParallel --train_root "./datasets/cifar10/data" --num_classes 10 --dataset "cifar10" --use_bmm --double_run --noise_type "real_in_noise"