gpu='0'

# load_path='../results/LA/checkpoints'
# python test.py --gpu $gpu \
#                      --data_dir '../../../Datasets/LA_dataset' \
#                      --list_dir '../datalist/LA' \
#                      --load_path $load_path

# load_path='../results/AD_0/checkpoints'
# python test.py --gpu $gpu \
#                     --data_dir '../../../Datasets/TBAD128'  \
#                     --list_dir '../datalist/AD/AD_0' \
#                     --num_classes 3 \
#                     --load_path $load_path

load_path='../results/Pancreas/checkpoints'
python test.py --gpu $gpu \
                     --data_dir '../../../Datasets/Pancreas-processed' \
                     --list_dir '../datalist/Pancreas' \
                     --load_path $load_path