import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## view features
    feature = 255* np.load('/xxx/CircuitNet/train_congesion/congestion/feature/1-RISCY-a-1-c2-u0.7-m1-p1-f0.npy')
    label = 255* np.load('/xxx/CircuitNet/train_congesion/congestion/label/1-RISCY-a-1-c2-u0.7-m1-p1-f0.npy')
    Macro_region = feature[:,:,:1]
    RUDY = feature[:,:,1:2]
    Pin_RUDY = feature[:,:,2:]
    cv2.imwrite('./images/feature.jpg', feature)
    cv2.imwrite('./images/label.jpg', label)
    cv2.imwrite('./images/Macro_region.jpg', Macro_region)
    cv2.imwrite('./images/RUDY.jpg', RUDY)
    cv2.imwrite('./images/Pin_RUDY.jpg', Pin_RUDY)

    ## instance_placement
    path = '/xxx/CircuitNet/graph_features/instance_placement/1-RISCY-a-1-c2-u0.7-m1-p1-f0'
    instance_placement = np.load(path, allow_pickle=True).item()
    img = np.zeros([400, 400, 1], np.uint8)
    for key, value in instance_placement.items():
        xl = int(value[0])
        yt = int(value[1])
        xr = int(value[2])
        yb = int(value[3])
        img = cv2.rectangle(img, (xl, yt), (xr, yb), 255, -1)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite('./images/instance_placement.jpg', img)
