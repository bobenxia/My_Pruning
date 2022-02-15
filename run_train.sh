# python path
FILE_PATH="$( cd "$( dirname $0 )" && pwd )"
cd $FILE_PATH
echo $FILE_PATH

# # check pytorch
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.distributed.is_available())"

# install requirements
pip3 install scikit-learn
pip3 install ptflops
pip3 install pynvml
pip3 install pandas
pip3 install coloredlogs
pip3 install h5py
# pip3 install loguru==0.5.2
# pip3 install thop
# pip3 install tabulate

# # ln dataset from TOS to code
# ln -s /Tos ./datasets/COCO

# debug
# export NCCL_DEBUG=INFO

# wandb usage
# WANDB_API_KEY = ""

# launch training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 prune_resnet50_cifar10_CSGD.py --mode train_with_csgd