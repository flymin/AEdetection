#!/bin/bash
python test_magnet.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --load_dir results/MagNet-cifar10-2021-05-26-19-45-41 --ae_path "results/whitebox/PGD/cifar10_PGDinf_2048.pt;results/whitebox/PGDL2/cifar10_PGDL2_2416.pt;results/whitebox/BIML2/cifar10_BIML2_3296.pt;results/whitebox/BIM/cifar10_BIMinf_2352.pt;results/whitebox/DF/cifar10_DFinf_2408.pt;results/whitebox/DFL2/cifar10_DFL2_2408.pt;results/whitebox/CW/cifar10_CW_2416.pt;results/whitebox/CWinf/cifar10_CWinf_2416.pt;results/whitebox/EADA/cifar10_EADA_2416.pt;results/whitebox/EADAL1/cifar10_EADAL1_2416.pt" --batch_size 64 \
&& \
python test_magnet.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset MNIST --load_dir results/MagNet-MNIST-2021-05-25-20-17-12 --ae_path "results/whitebox/PGD/MNIST_PGDinf_2416.pt;results/whitebox/PGDL2/MNIST_PGDL2_2416.pt;results/whitebox/BIM/MNIST_BIMinf_2416.pt;results/whitebox/BIML2/MNIST_BIML2_2416.pt;results/whitebox/DF/MNIST_DFinf_2408.pt;results/whitebox/DFL2/MNIST_DFL2_2408.pt;results/whitebox/CW/MNIST_CW_2416.pt;results/whitebox/CWinf/MNIST_CWinf_2416.pt;results/whitebox/EADA/MNIST_EADA_2416.pt;results/whitebox/EADAL1/MNIST_EADAL1_2416.pt" \
&& \
python test_magnet.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset gtsrb --load_dir results/MagNet-gtsrb-2021-05-26-20-58-17 --ae_path "results/whitebox/PGD/gtsrb_PGDinf_2404.pt;results/whitebox/PGDL2/gtsrb_PGDL2_2404.pt;results/whitebox/BIM/gtsrb_BIMinf_2404.pt;results/whitebox/BIML2/gtsrb_BIML2_2404.pt;results/whitebox/DF/gtsrb_DFinf_2402.pt;results/whitebox/DFL2/gtsrb_DFL2_2402.pt;results/whitebox/CW/gtsrb_CW_2404.pt;results/whitebox/CWinf/gtsrb_CWinf_2408.pt;results/whitebox/EADA/gtsrb_EADA_2416.pt;results/whitebox/EADAL1/gtsrb_EADAL1_2416.pt" 

python test_fs.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --detection "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1" --ae_path "results/whitebox/PGD/cifar10_PGDinf_2048.pt;results/whitebox/PGDL2/cifar10_PGDL2_2416.pt;results/whitebox/BIM/cifar10_BIMinf_2352.pt;results/whitebox/BIML2/cifar10_BIML2_3296.pt;results/whitebox/DF/cifar10_DFinf_2408.pt;results/whitebox/DFL2/cifar10_DFL2_2408.pt;results/whitebox/CW/cifar10_CW_2416.pt;results/whitebox/CWinf/cifar10_CWinf_2416.pt;results/whitebox/EADA/cifar10_EADA_2416.pt;results/whitebox/EADAL1/cifar10_EADAL1_2416.pt" --batch_size 32 \
&& \
python test_fs.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset MNIST --detection "FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1" --ae_path "results/whitebox/PGD/MNIST_PGDinf_2416.pt;results/whitebox/PGDL2/MNIST_PGDL2_2416.pt;results/whitebox/BIM/MNIST_BIMinf_2416.pt;results/whitebox/BIML2/MNIST_BIML2_2416.pt;results/whitebox/DF/MNIST_DFinf_2408.pt;results/whitebox/DFL2/MNIST_DFL2_2408.pt;results/whitebox/CW/MNIST_CW_2416.pt;results/whitebox/CWinf/MNIST_CWinf_2416.pt;results/whitebox/EADA/MNIST_EADA_2416.pt;results/whitebox/EADAL1/MNIST_EADAL1_2416.pt" \
&& \
python test_fs.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset gtsrb --detection "FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1" --ae_path "results/whitebox/PGD/gtsrb_PGDinf_2404.pt;results/whitebox/PGDL2/gtsrb_PGDL2_2404.pt;results/whitebox/BIM/gtsrb_BIMinf_2404.pt;results/whitebox/BIML2/gtsrb_BIML2_2404.pt;results/whitebox/DF/gtsrb_DFinf_2402.pt;results/whitebox/DFL2/gtsrb_DFL2_2402.pt;results/whitebox/CW/gtsrb_CW_2404.pt;results/whitebox/CWinf/gtsrb_CWinf_2408.pt;results/whitebox/EADA/gtsrb_EADA_2416.pt;results/whitebox/EADAL1/gtsrb_EADAL1_2416.pt"


# Trapdoor
python test_trapdoor.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset MNIST --ae_path "results/TrapdoorAE/PGD/MNIST_PGDinf_28_2432.pt;results/TrapdoorAE/PGDL2/MNIST_PGDL2_28_2432.pt;results/TrapdoorAE/BIM/MNIST_BIMinf_28_2432.pt;results/TrapdoorAE/BIML2/MNIST_BIML2_28_2432.pt;results/TrapdoorAE/DF/MNIST_DFinf_28_10000.pt;results/TrapdoorAE/DFL2/MNIST_DFL2_28_10000.pt;results/TrapdoorAE/CW/MNIST_CW_28_10000.pt;results/TrapdoorAE/CWinf/MNIST_CWinf_28_10000.pt;results/TrapdoorAE/EADA/MNIST_EADA_28_10000.pt;results/TrapdoorAE/EADAL1/MNIST_EADAL1_28_10000.pt"

# python test_trapdoor.py --dataset MNIST --ae_path "results/TrapdoorAE/PGD/MNIST_PGDinf_2432.pt;results/TrapdoorAE/PGDL2/MNIST_PGDL2_2432.pt;results/TrapdoorAE/BIM/MNIST_BIMinf_2432.pt;results/TrapdoorAE/BIML2/MNIST_BIML2_2432.pt;results/TrapdoorAE/DF/MNIST_DFinf_10000.pt;results/TrapdoorAE/DFL2/MNIST_DFL2_10000.pt;results/TrapdoorAE/CW/MNIST_CW_10000.pt"

python test_trapdoor.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset gtsrb --ae_path "results/TrapdoorAE/PGD/gtsrb_PGDinf_2496.pt;results/TrapdoorAE/PGDL2/gtsrb_PGDL2_2432.pt;results/TrapdoorAE/BIM/gtsrb_BIMinf_2432.pt;results/TrapdoorAE/BIML2/gtsrb_BIML2_2432.pt;results/TrapdoorAE/DF/gtsrb_DFinf_12630.pt;results/TrapdoorAE/DFL2/gtsrb_DFL2_12630.pt;results/TrapdoorAE/CW/gtsrb_CW_12630.pt;results/TrapdoorAE/CWinf/gtsrb_CWinf_32_12630.pt;results/TrapdoorAE/EADA/gtsrb_EADA_32_12630.pt;results/TrapdoorAE/EADAL1/gtsrb_EADAL1_32_12630.pt"

python test_trapdoor.py --drop_rate 0.01 --results_dir ./results/ae_test_0.01 --dataset cifar10 --ae_path "results/TrapdoorAE/PGD/cifar10_PGDinf_2432.pt;results/TrapdoorAE/PGDL2/cifar10_PGDL2_2432.pt;results/TrapdoorAE/BIM/cifar10_BIMinf_2432.pt;results/TrapdoorAE/BIML2/cifar10_BIML2_2432.pt;results/TrapdoorAE/DF/cifar10_DFinf_10000.pt;results/TrapdoorAE/DFL2/cifar10_DFL2_10000.pt;results/TrapdoorAE/CW/cifar10_CW_10000.pt;results/TrapdoorAE/CWinf/cifar10_CWinf_32_10000.pt;results/TrapdoorAE/EADA/cifar10_EADA_32_10000.pt;results/TrapdoorAE/EADAL1/cifar10_EADAL1_32_10000.pt"
