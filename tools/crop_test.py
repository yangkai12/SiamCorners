from PIL import Image
import matplotlib.pyplot as plt
img = Image.open("/media/yangkai/data1/home/yangkai/study/code/pysot-master12/tools/127_bbox.jpg")
plt.imshow(img)
plt.show()
img_c = img.crop([img.size[0]/4,img.size[1]/4,img.size[0]*3/4,img.size[1]*3/4])
plt.imshow(img_c)
plt.show()
