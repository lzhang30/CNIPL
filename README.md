# CMILP
A description projection for our Cross-Neighbor-Information based Pseudo-Label for Better Semi-Scribble-Supervised Medical Image Segmentation

# Data Preprocess

To prepare the dataset, you can follow the work of [WLSMIS](https://github.com/HiLab-git/WSL4MIS)

You can also run ```python code/scribbles_generator.py``` for scribble label generating and

```python code/dataloaders/word_data_processing.py``` for data preprocessing.

Our work is based on both the 2D medical volumes.


The scribble annotations we used for WORD dataset are available at `./WORD_scribble_labels`



# Model Training
Run
```
cd code
python train_CNIPL.py --gpu 0  --check 500 --labeled_ratio 8 --early_stop 20000 --fold fold1 --num_classes 4  --root_path ../data/ACDC --exp ACDC_CNIPL  --max_iterations 60000 --batch_size 16 
python train_CNIPL.py --gpu 0  --check 500 --labeled_ratio 8 --early_stop 20000 --fold fold2 --num_classes 4  --root_path ../data/ACDC --exp ACDC_CNIPL  --max_iterations 60000 --batch_size 16 
python train_CNIPL.py --gpu 0  --check 500 --labeled_ratio 8 --early_stop 20000 --fold fold3 --num_classes 4  --root_path ../data/ACDC --exp ACDC_CNIPL  --max_iterations 60000 --batch_size 16 
python train_CNIPL.py --gpu 0  --check 500 --labeled_ratio 8 --early_stop 20000 --fold fold4 --num_classes 4  --root_path ../data/ACDC --exp ACDC_CNIPL  --max_iterations 60000 --batch_size 16 
python train_CNIPL.py --gpu 0  --check 500 --labeled_ratio 8 --early_stop 20000 --fold fold5 --num_classes 4  --root_path ../data/ACDC --exp ACDC_CNIPL  --max_iterations 60000 --batch_size 16
python val_ours.py

```
for model training and evaluating.
Have fun.
