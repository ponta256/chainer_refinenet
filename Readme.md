python train.py -g 1 -tt  data/feel.txt -tr ./data/feel_images/ -ta ./data/feel_classes/ -e 500 -b 16 --lr=0.0001
python predict.py -g 1 -i ./data/voc2012_images/2011_003256.jpg -w weight/chainer_refinenet_140.weight
