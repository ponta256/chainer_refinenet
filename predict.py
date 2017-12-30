import chainer
from chainer import cuda, optimizers, serializers, Variable
import numpy as np
from PIL import Image
import os
import argparse
import cv2

from refinenet import RefineResNet
from color_map import make_color_map

def predict(image, weight, class_num, gpu=-1):
  model = RefineResNet(class_num)
  serializers.load_npz(weight, model)

  if gpu >= 0:
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()
  xp = np if args.gpu < 0 else cuda.cupy
  
  img = image.resize((224,224))
  mean = np.array([103.939, 116.779, 123.68])
  img -= mean
  x = xp.asarray(img, dtype=xp.float32)
  x = x.transpose(2, 0, 1)
  x = xp.expand_dims(x, axis=0)

  with chainer.using_config('train', False):
    pred = model(x).data

    return pred

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='RefineNet on Chainer (predict)')
  parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--image_path', '-i', default=None, type=str)
  parser.add_argument('--class_num', '-n', default=21, type=int)
  parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
  args = parser.parse_args()

  img = Image.open(args.image_path)
  pred = predict(img, args.weight, args.class_num, args.gpu)

  x = pred[0].copy()
  pred = pred[0].argmax(axis=0)

  row, col = pred.shape

  xp = np if args.gpu < 0 else cuda.cupy
  dst = xp.ones((row, col, 3))

  color_map = make_color_map()
  for i in range(args.class_num):
    dst[pred == i] = color_map[i]

  if args.gpu >= 0:
    dst = cuda.to_cpu(dst)
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

  p = cv2.resize(p, (ow, oh))
  pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

  # cv2.imwrite("out/pred_{}.png".format(img_name), pred)
  os.remove("out/original.jpg")
  os.remove("out/pred.png")

  cv2.imshow("image", pred) 
  while cv2.waitKey(33) != 27:
    pass

  cv2.imwrite("out.jpg", pred)


