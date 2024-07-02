taskname='your_taskname'
backbone='resnet18'
pretrain_path='your_pretrained_encoder.pth'

cd icbeb
bash sub_icbeb.sh $taskname $backbone $pretrain_path

cd ..
cd chapman
bash sub_chapman.sh $taskname $backbone $pretrain_path

cd ..
cd ptbxl
bash sub_ptbxl.sh $taskname $backbone $pretrain_path
