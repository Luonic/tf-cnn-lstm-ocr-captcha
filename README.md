# tf-cnn-lstm-ocr-captcha
Code for training LSTM neural network on top of convolutional features for captcha recognition in Moscow subway

All code tested with python 2.7, TF 1.0

1) To download sythetically generated captchas run ```python downloader.py``` Images will be stored in "data/train"
2) Run to_tfrecords.py to create *.tfrecords files from images
3) Run ocr_train.py to train network. In different terminal run
```
export CUDA_VISIBLE_DEVICES=""
python ocr_eval.py
```
4) After training edit chekpoint path in ```freeze.sh``` with your best checkpoint path and then run it
5) ???
6) PROFIT!

Sample of captcha image:

![sample](/data/test/2a48-sample.png)
