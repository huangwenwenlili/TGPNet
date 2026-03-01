#CUDA_VISIBLE_DEVICES=0 python ./basicsr/train.py -opt option/TGPNet_train_1080.yml

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 ./basicsr/train.py -opt option/TGPNet_train.yml --launcher pytorch

