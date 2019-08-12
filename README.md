# FUNIT-Tensorflow
## : Few-Shot Unsupervised Image-to-Image Translation (ICCV 2019)

### [Paper](https://arxiv.org/abs/1905.01723) | [Official Pytorch code](https://github.com/NVlabs/FUNIT)

## Pytorch Implementation
Will be soon

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── train
           ├── class1 (class folder)
               ├── xxx.jpg (class1 image)
               ├── yyy.png
               ├── ...
           ├── class2
               ├── aaa.jpg (class2 image)
               ├── bbb.png
               ├── ...
           ├── class3
           ├── ...
       ├── test
           ├── content (content folder)
               ├── zzz.jpg (any content image)
               ├── www.png
               ├── ...
           ├── class (class folder)
               ├── ccc.jpg (unseen target class image)
               ├── ddd.jpg
               ├── ...
```

### Train
```
> python main.py --dataset flower
```

### Test
```
> python main.py --dataset flower --phase test
```

## Architecture

## Author
[Junho Kim](http://bit.ly/jhkim_ai)
