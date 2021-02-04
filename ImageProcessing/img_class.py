from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('images/chestxray.jpg')
# img.show()


print()
new_img = img.resize((200,190))
# new_img.show()
pix_val = list(new_img.getdata())
# print(pix_val)
# tmp_img = new_img.convert('RGB')
#
# r = new_img.split()
#
# r = r.point(lambda i: i*0.625)

for i in range(len(pix_val)):
    pix_val[i] = pix_val[i]*1

output = Image.fromarray((np.array(pix_val)).reshape(200,190), 'L')
output.show()