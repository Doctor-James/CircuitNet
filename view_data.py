import numpy as np
import matplotlib.pyplot as plt

feature = np.load('/home/users/jialv.zou/datasets/CircuitNet/train/congestion/feature/1-RISCY-a-1-c2-u0.7-m1-p1-f0.npy')
label = np.load('/home/users/jialv.zou/datasets/CircuitNet/train/congestion/label/1-RISCY-a-1-c2-u0.7-m1-p1-f0.npy')
Macro_region = feature[:,:,:1]
RUDY = feature[:,:,1:2]
Pin_RUDY = feature[:,:,2:]
plt.imshow(feature)
plt.savefig('./images/feature.jpg')
# plt.imshow(Macro_region)
# plt.show()