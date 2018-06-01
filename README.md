# Adaptive Scaling for Sparse Detection in Information Extraction

This is the source code for paper "Adaptive Scaling for Sparse Detection in Information Extraction" in ACL2018.

## Requirements

* Tensorflow >= 1.2.0

## Usage
First, please unzip the word2vec embeddings in "CNN/"

* gzip -d CNN/en_word2vec.dat.gz
* gzip -d CNN/zh_word2vec.dat.gz

Then enter CNN dir, run the program like

* python CNN.py config_f1_zh.cfg

Hyperparameters in our paper are saved in configure file "config_f1_en.cfg" or "config_f1_zh.cfg".

## Citation
Please cite:
* Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun. *Adaptive Scaling for Sparse Detection in Information Extraction*. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.

```
@InProceedings{lin-Etal:2018:ACL2018adaptive,
  author    = {Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le},
  title     = {Adaptive Scaling for Sparse Detection in Information Extraction},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2018},
  address   = {Melbourne, Australia},
  location = {Association for Computational Linguistics},
}
```

## Contact
If you have any question or want to request for the data(only if you have the license from LDC), please contact me by
* hongyu2016@iscas.ac.cn
