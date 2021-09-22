# INSIGHT work - Studying how a network forgets 
Code developed at INSIGHT with the aim of further understanding how a neural network forgets, and using that in the context of noisy label learning.

## Requirements
```
pip install -r requirements.txt
```

## Usage
## Data separation methods
### Forgetting method
```
python3 main.py --lr_2nd 0.001 --epoch_1st 150 --epoch_2nd 10 --M 40 --M 80 --save_best_AUC_model --track_CE --step_number 1500 --experiment_name cifar10_random_forget_4_ssl --first_stage_num_classes 10 --first_stage_noise_type "random_in_noise" --second_stage_num_classes 10 --second_stage_data_name "cifar100" --second_stage_subset 2 14 23 35 48 51 69 74 87 90 --first_stage_data_name "cifar10"
```

Note that if you use a subset of cifar100 please include which classes are being used and the number of classes.

### Parallel Cross Tracking method
```
python3 parallel.py --M 40 --M 80 --epoch_1st 151 --first_stage_data_name "cifar10" --first_stage_noise_type "real_in_noise" --first_stage_num_classes 10 --experiment_name cifar10real_parallel_4_ssl
```

### Relabel method

Code taken from [here](https://github.com/EricArazo/PseudoLabeling) (thank you!).

```
python3 relabel.py --lr_2nd 0.1  --M 75 --M 110 --epoch_1st 130 --warmup_e 70 --experiment_name cifar100_relabel_4_ssl --first_stage_data_name "cifar100" --first_stage_noise_type "random_in_noise" --first_stage_num_classes 100
```

## Noisy label learning
### Semi supervised learning (SSL)
Code taken from [here](https://github.com/EricArazo/PseudoLabeling) (thank you!).
```
python3 ssl.py --epoch 400 --epoch_begin 10 --M 250 --M 350 --forget --experiment_name ssl_cifar10ranForget --train_root "./datasets/cifar10/data" --num_classes 10 --dataset "cifar10" --use_bmm --double_run --noise_type "random_in_noise"
```

Note that SSL requires having previously run one of the data separation methods.
You can choose the gpu that will be used with the argument --cuda_dev.

Other .py files don't have any parseable arguments.

#### Further Reading
More information can be found [here](https://drive.google.com/file/d/12Ru6YAR-RDHGfxbxKNAwFFu6_ncTiGBX/view?fbclid=IwAR0cMKias6FqlC8EdK3jCrcNaYPRyJqKSO3u1Edb38lfMd4PXm4b6xiyh6Q).

#### Contact
Any question or doubt please email at: mgrc99@gmail.com
