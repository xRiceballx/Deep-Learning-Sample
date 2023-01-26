import cv2
from matplotlib import pyplot as plt

"""
#OpenCVによる画像読み込みと表示、出力
img = cv2.imread("sampleimage/histgram38.png")

cv2.imshow("histgram38", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("sampleimage/histgram38-alt.png", img)

"""

#読み込んだ画像のRGB度合いをヒストグラムで表示
img = cv2.imread("sampleimage/cat.jpg")
color = ("b","g","r")

for i, col in enumerate(color):
    hist = cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0,256])

    plt.plot(hist,color=col)
    plt.xlim([0,256])
    
plt.show()
