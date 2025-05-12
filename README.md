# pipeline_model_update
# Create output_folder
# Run Script:
## !python train.py --mode evaluate --data "" --checkpoint "" --saveas "" 
# Training:
## !python train.py --mode train --epoch 3 --data /kaggle/input/isic2018/ISIC2018 --lr0 0.1 --saveas "/kaggle/working/outputSwinUNet1" --batchsize 16
# Pretrain:
## !python train.py --mode pretrain --epoch 50 --lr0 0.1 --batchsize 16 --data /kaggle/input/isic2018/ISIC2018 --checkpoint /kaggle/working/last_model_down.pth --saveas /kaggle/working/outputSwinUNet1
# Evaluate:
!python train.py --mode evaluate --data /kaggle/input/isic2018/ISIC2018 --checkpoint /kaggle/working/last_model_down.pth --saveas /kaggle/working/outputSwinUNet1
