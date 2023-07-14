gpu='0'

# exp='LA'
# data_dir='../../../Datasets/LA_dataset'
# list_dir='../datalist/LA'

# python train.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --exp $exp

# exp='Pancreas'
# data_dir='../../../Datasets/Pancreas-processed'
# list_dir='../datalist/Pancreas'

# python train.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --exp $exp 

exp='AD_0'
data_dir='../../../Datasets/TBAD128'
list_dir='../datalist/AD/AD_0'

python train.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --num_classes 3 --exp $exp