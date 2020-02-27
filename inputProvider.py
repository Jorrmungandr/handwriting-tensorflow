from PIL import Image
import PIL.ImageOps
import os

size = 28, 28

inputs = []

for r, d, f in os.walk(os.path.dirname('inputs/')):
  for file in f:
    img = Image.open('inputs/{}'.format(file)).resize(size, Image.ANTIALIAS).convert('L')
    img = PIL.ImageOps.invert(img)
    WIDTH, HEIGHT = img.size
    data = list(img.getdata())
    resultData = []
    for num in data:
      if (num > 110):
        resultData.append(num)
      else:
        resultData.append(0)
    twodData = [resultData[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
    inputs.append(list(twodData))
    # for row in twodData:
    #   print(' '.join('{:3}'.format(value) for value in row))