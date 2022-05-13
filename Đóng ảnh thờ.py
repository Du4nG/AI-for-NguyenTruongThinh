import cv2
vid = cv2.VideoCapture('tự_quay_video_vài_giây_rồi_bỏ_vào.mp4')
success,image = vid.read()
count = 0
while success:
    img = image[200:500,500:800] # canh mặt ở đâu mà đóng khung vào nhé
    img = cv2.resize(img,(150,150),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("frame%d.jpg" % count, img)      
    success, img = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
