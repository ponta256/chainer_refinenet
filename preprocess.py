
import numpy as np
from PIL import Image
import random

def load_data(path, crop=False, mode="label", xp=np, hflip=False, rscale=False, rcrop=False, xs=0, ys=0, rs=256):

  '''
  1. Pickrandom L in range[256,480]
  2. Resize training image, shortside=L
  3. Sample random 224x224 patch
  '''

  img = Image.open(path)
  if crop:
    w, h = img.size    
    if rcrop:
      size = rs
    else:
      size = 224

    if w < h:
      img = img.resize((size, size*h//w))
      w, h = img.size                  
      if not rcrop:
        xs = 0
        ys = (h-224)//2

    else:
      img = img.resize((size*w//h, size))
      w, h = img.size                        
      if not rcrop:
        xs = (w-224)//2
        ys = 0
      
  if mode=="label":
    y = xp.asarray(img, dtype=xp.int32)
    y = y[ys:ys+224,xs:xs+224]
      
    if hflip:
      y = y[:,::-1]

    mask = y == 255
    y[mask] = -1

    # print(y)
    return y

  elif mode=="data":
    mean = np.array([103.939, 116.779, 123.68])
    # print(img.size, img.mode)
    if img.mode == 'L':
      rgbimg = Image.new("RGB", img.size)
      rgbimg.paste(img)
      img = rgbimg
      
    img -= mean
    x = xp.asarray(img, dtype=xp.float32)
    x = x.transpose(2, 0, 1)
    x = x[:,ys:ys+224,xs:xs+224]

    if hflip:
      x = x[:,:,::-1]

    return x

