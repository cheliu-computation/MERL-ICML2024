task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/home/cl522/github_repo/ETP/finetune/ckpt/chapman/$task_name"

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset chapman \
    --pretrain_path $pretrain_path \
    --ratio 1 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 100 \
    --name $task_name

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset chapman \
    --pretrain_path $pretrain_path \
    --ratio 10 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 100 \
    --name $task_name

python main_single.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset chapman \
    --pretrain_path $pretrain_path \
    --ratio 100 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 100 \
    --name $task_name