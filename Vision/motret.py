import cv2
import time

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

flag = 0
flag2 = 0
count = 0
cek = 0

start = 10
tiap = 5
batas = 10

while True :
    status, img1 = cap1.read()
    status, img2 = cap2.read()

    if flag == 0 :
        awal = time.time()
        flag = 1
    
    akhir = time.time()
    waktu = akhir - awal
    print(waktu)
    
    if waktu >= start :
        if flag2 == 0 :
            awal2 = time.time()
            flag2 = 1

        akhir2 = time.time()
        delay = int(akhir2 - awal2)
     
        
        if delay % tiap == 0 and cek == 0:
            print(delay)
            img_name1 ="mangrove{}.png".format(count)
            img_name2 ="ikan{}.png".format(count)
            cv2.imwrite(img_name1,img1)
            cv2.imwrite(img_name2,img2)
            count = count + 1
            cek = 1
            
        elif delay % tiap != 0 :
            cek = 0
        
        if count > batas :
            count = 0
     

    cv2.imshow('raw image1', img1)
    cv2.imshow('raw image2', img2)

    if cv2.waitKey(1) == ord('q'):
        break      

cv2.destroyAllWindows()
