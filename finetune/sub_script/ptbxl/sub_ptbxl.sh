task_name=$1
backbone=$2
pretrain_path=$3

bash sub_ptbxl_form.sh $task_name $backbone $pretrain_path
bash sub_ptbxl_rhythm.sh $task_name $backbone $pretrain_path
bash sub_ptbxl_super_class.sh $task_name $backbone $pretrain_path
bash sub_ptbxl_sub_class.sh $task_name $backbone $pretrain_path