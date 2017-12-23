import chainer
from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2

from refinenet import RefineResNet
from color_map import make_color_map

parser = argparse.ArgumentParser(description='RefineNet on Chainer (predict)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
          help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_path', '-i', default=None, type=str)
parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
args = parser.parse_args()

img_name = args.image_path.split("/")[-1].split(".")[0]

color_map = make_color_map()
model = RefineResNet()
serializers.load_npz(args.weight, model)

img = Image.open(args.image_path)
img = img.resize((224,224))
mean = np.array([103.939, 116.779, 123.68])
img -= mean
x = np.asarray(img, dtype=np.float32)
x = x.transpose(2, 0, 1)
x = np.expand_dims(x, axis=0)

with chainer.using_config('train', False):
  pred = model(x).data

print(pred[0].shape)
x = pred[0].copy()

pred = pred[0].argmax(axis=0)

row, col = pred.shape
dst = np.ones((row, col, 3))
for i in range(21):
  dst[pred == i] = color_map[i]
img = Image.fromarray(np.uint8(dst))

b,g,r = img.split()
img = Image.merge("RGB", (r, g, b))

trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
w, h = img.size
for x in range(w):
  for y in range(h):
    pixel = img.getpixel((x, y))
    if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0) or \
       (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
      continue
    trans.putpixel((x, y), pixel)

if not os.path.exists("out"):
  os.mkdir("out")

o = Image.open(args.image_path)
ow, oh = o.size
o.save("out/original.jpg")

trans.save("out/pred.png")
o = cv2.imread("out/original.jpg", 1)
p = cv2.imread("out/pred.png", 1)

cv2.imshow("image", p) 
while cv2.waitKey(33) != 27:
  pass
    
p = cv2.resize(p, (ow, oh))
pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

# cv2.imwrite("out/pred_{}.png".format(img_name), pred)
os.remove("out/original.jpg")
os.remove("out/pred.png")

cv2.imshow("image", pred) 
while cv2.waitKey(33) != 27:
  pass
