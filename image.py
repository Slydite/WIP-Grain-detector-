    
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io          # Only needed for web grabbing images

# Load some image with circles from web
image = io.imread('https://i.imgur.com/k0k5C3l.jpeg')
plt.figure(1), plt.imshow(image), plt.title('original image'), plt.tight_layout()

# Mimic watershed result using findContours and drawContours
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
markers = np.zeros_like(gray).astype(np.int32)
for i, cnt in enumerate(cnts):
    markers = cv2.drawContours(markers, [cnt], -1, i+1, cv2.FILLED)
plt.figure(2), plt.imshow(markers), plt.title('markers'), plt.colorbar(), plt.tight_layout()
plt.show()

# Assuming we only have markers now; iterate all values and crop image part
for i in np.arange(1, np.max(markers[:, :])+1):
    pixels = np.array(np.where(markers == i)).astype(np.int32)
    x1 = np.min(pixels[1, :])
    x2 = np.max(pixels[1, :])
    y1 = np.min(pixels[0, :])
    y2 = np.max(pixels[0, :])
    cv2.imwrite(str(i) + '.png', image[y1:y2, x1:x2, :])