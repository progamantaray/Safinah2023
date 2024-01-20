import cv2
import numpy as np
import serial
import time

INVERS = 'n' #ganti y jika bola merah di kanan

#List masking merah
listLowRed = [np.array([0,26,53]),np.array([163,43,142]),np.array([0,81,127])] # [np.array([0, 111, 109])]
listUpRed = [np.array([7,255,168]),np.array([180,255,255]),np.array([4,244,249])] #[np.array([3, 190, 178])] 

#List masking hijau
listLowGreen = [np.array([70,145,24]),np.array([76, 121, 17 ]), np.array([62,147,0])] #[np.array([83, 28, 101])] 
listUpGreen = [np.array([83,255,89]),np.array([97, 255, 255]),np.array([179, 255, 255])] #[np.array([94, 116, 158])] 

#Batas Luas yang bisa didetect
batasLuasAtasMerah = 10000  
batasLuasBawahMerah = 300  

batasLuasAtasHijau = 8000
batasLuasBawahHijau = 300

batasLuasAtasKuning = 10000
batasLuasBawahKuning = 200

#setpoint stopping
LuasDermaga = 9000

#inisiasi arduino
port = '/dev/ttyUSB0'  
bautRate = 9600

#inisiasi video
path = 'D:\Resources/raw.mp4'
scalling = 1

#Atur jarak manuver
areaLurus = 20
areaBelokTipis = 40
areaBelokBesar = 40

#Atur ROI
blockAtas = 130
blockTengah = 90
blockKanan = 30
blockKiri = 30
blockBawah = 180
segitigaVertikal = 180
segitigaHorizontal = 160

##Balik bola kanan sama kiri
if INVERS == 'y':

    listLowRed2 = listLowRed
    listUpRed2 = listUpRed

    listLowRed = listLowGreen
    listUpRed = listUpGreen
    listLowGreen = listLowRed2
    listUpGreen = listUpRed2


    batasLuasAtasMerah2 = batasLuasAtasMerah
    batasLuasBawahMerah2 = batasLuasBawahMerah

    batasLuasAtasMerah = batasLuasAtasHijau
    batasLuasBawahMerah = batasLuasBawahHijau
    batasLuasAtasHijau = batasLuasAtasMerah2
    batasLuasBawahHijau = batasLuasBawahMerah2

    ################################

#Rescale video
def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

#Processing bola merah
def detect_red(image, lower_red, upper_red):
    frame =  cv2.GaussianBlur(image, (31,31), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    global mask_red
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    contours,hierarchy = cv2.findContours(mask_red, 1, cv2.CHAIN_APPROX_NONE)
    
    red_detected = False
  
    contoursList = []
    for i in contours:
        area1 = cv2.contourArea(i)

        if area1 < batasLuasAtasMerah and area1 > batasLuasBawahMerah:
            contoursList.append(i)
            
    contours = tuple(contoursList)

    global cyM

    ##Image segmenting MERAH (KIRI)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        area1 = cv2.contourArea(c)
        
        M = cv2.moments(c)
        if M['m00']!=0:
            cxM = int(M['m10']/M['m00'])
            cyM = int(M['m01']/M['m00'])
            cv2.rectangle(image, (cxM-20,cyM-20), (cxM+20,cyM+20), (255,255,255), 1)
        red_detected = True

    else:
            if cxH > frametengah :
                cxM = -5
                cyM = image.shape[0]//2+10
            else :
                cxM = image.shape[1]+5
                cyM = image.shape[0]//2+10
    cv2.drawContours(image, contours, -1, (255,255,255),1)
    
    return red_detected, cxM

#Processing bola hijau
def detect_green(image, lower_green, upper_green):
    frame =  cv2.GaussianBlur(image, (31,31), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    global mask_green
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    contours2,hierarchy = cv2.findContours(mask_green, 1, cv2.CHAIN_APPROX_NONE)
    
    green_detected = False
  
    
    global cyH
    contoursList2 = []
    for i in contours2:
        area2 = cv2.contourArea(i)

        if area2 < batasLuasAtasHijau and area2 > batasLuasBawahHijau:
            contoursList2.append(i)
            
    contours2 = tuple(contoursList2)


    ##Image segmenting HIJAU (KANAN)
    if len(contours2) > 0:
        c = max(contours2, key=cv2.contourArea)
        
        area2 = cv2.contourArea(c)
        
        M = cv2.moments(c)
        if M['m00']!=0:
            cxH = int(M['m10']/M['m00'])
            cyH = int(M['m01']/M['m00'])
            cv2.rectangle(image, (cxH-20,cyH-20), (cxH+20,cyH+20), (255,255,255), 1)
        green_detected = True

    else:
            if cxM < frametengah :
                cxH = image.shape[1]+5
                cyH = image.shape[0]//2+10
            else :
                cxH = -5
                cyH = image.shape[0]//2+10
    
    cv2.drawContours(image, contours2, -1, (255,255,255),1)

    return green_detected, cxH

#Fungsi Utama
def main():
    cap = cv2.VideoCapture(path)

    #Masking kuning
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    #inisiasi index array
    nRed = 0
    nGreen = 0

    global cxH
    global cxM
    cxH = 0
    cxM = 3000
    
    ##Koordinat area manuver
    batasLurus = {
        "+"     : areaLurus,
        "-"     : -areaLurus,
    }

    batasBelokTipis = {
        "+kanan"    : (areaLurus + areaBelokTipis),
        "+kiri"     : (areaLurus) -1,
        
        "-kanan"    : -(areaLurus) +1,
        "-kiri"     : -(areaLurus + areaBelokTipis),
    }

    batasBelokBesar = {
        "+kanan"    : (areaLurus + areaBelokTipis + areaBelokBesar),
        "+kiri"     : (areaLurus + areaBelokTipis) -1,

        "-kanan"    : -(areaLurus + areaBelokTipis) +1,
        "-kiri"     : -(areaLurus + areaBelokTipis + areaBelokBesar),
    }

    #Nyambungin ke arduino
    try:
        ser = serial.Serial(port, bautRate, timeout=None)
        arduino = 'ada'
        #startNyelem = 1000
    except:
        arduino = 'gada'
        print('------------gada arduino------------')

    while True:
        status, frame = cap.read()
        # frame = cv2.flip(frame,1)
       
        frame = rescale(frame,scalling)
        global frametengah
        frametengah = frame.shape[1]//2
        # print("nRed =",nRed)
        # print("nGreen =",nGreen)
        # print("cxM",cxM)
        # print("cxH",cxH)

        #Area of interest
        cv2.rectangle(frame, (0,0), (frame.shape[1],blockAtas), (0,0,0), thickness=-1)  #y=90  #y=110   ###
        cv2.rectangle(frame, ((frame.shape[1]//2)-blockTengah,0), ((frame.shape[1]//2)+blockTengah,frame.shape[0]), (0,0,0), thickness=-1) ###
        cv2.line(frame, (frame.shape[1]//5,blockAtas-50), (-5,blockAtas+10), (0,0,0), thickness=blockKiri) ###
        cv2.line(frame, (frame.shape[1]-frame.shape[1]//5,blockAtas-50), (frame.shape[1]+5,blockAtas+10), (0,0,0), thickness=blockKanan) ###
        cv2.rectangle(frame, (0,(frame.shape[0]//2)+blockBawah), (frame.shape[1],frame.shape[0]), (0,0,0), thickness=-1)
        # points = np.array([[10+segitigaHorizontal, 180+segitigaVertikal], [200+segitigaHorizontal, 110+segitigaVertikal], [200+segitigaHorizontal, 180+segitigaVertikal]]) 
        # cv2.fillPoly(frame, pts=[points], color=(0, 0, 0))#merah
        # points2 = np.array([[650+segitigaHorizontal, 180+segitigaVertikal], [440+segitigaHorizontal, 110+segitigaVertikal], [440+segitigaHorizontal, 180+segitigaVertikal]])
        # cv2.fillPoly(frame, pts=[points2], color=(0, 0, 0))#hijau

        # Detect bola merah dan hijau
        red_detected,cxM = detect_red(frame, listLowRed[nRed], listUpRed[nRed])
        green_detected,cxH = detect_green(frame, listLowGreen[nGreen], listUpGreen[nGreen])
        
        #pindah index masking jika salah satu bola tak terdeteksi
        if not red_detected and green_detected:
            nRed = nRed + 1
            if nRed == len(listLowRed) :
                nRed = 0
        if not green_detected and red_detected:
            nGreen = nGreen + 1
            if nGreen == len(listLowGreen) :
                nGreen= 0

        # jika kedua bola tak terdeteksi pindah mode dermaga
        if not red_detected and not green_detected:
            status, frame = cap.read()
            frame = rescale(frame,scalling)
            cv2.putText(frame, "mode dermaga", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            contours3, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            global area3
            area3 = 0
            contoursList3 = []
            for i in contours3:
                area3 = cv2.contourArea(i)

                if area3 < batasLuasAtasKuning and area3 > batasLuasBawahKuning:
                    contoursList3.append(i)
                

            contours3 = tuple(contoursList3)

            global cxK
            global cyK

            ##Image segmenting DERMAGA
            if len(contours3) > 0:
                c = max(contours3, key=cv2.contourArea)
                
                area3 = cv2.contourArea(c)
                
                M = cv2.moments(c)
                if M['m00']!=0:
                    cxK = int(M['m10']/M['m00'])
                    cyK = int(M['m01']/M['m00'])
                    cv2.rectangle(frame, (cxK-50,cyK-50), (cxK+50,cyK+50), (255,255,255), 1)
                cv2.putText(frame, "Luas = "+str(area3), (cxK-53,cyK-53), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else :
                cxK = frame.shape[1]//2
                cyK = frame.shape[0]//2

            #manuver dermaga
            #-------------START----------#
            aKanan = cxK+batasLurus['+']
            aKiri = cxK+batasLurus['-']
            bKanan = cxK+batasBelokTipis['+kanan']
            bKiri = cxK+batasBelokTipis['-kiri']
            cKanan = cxK+batasBelokBesar['+kanan']
            cKiri = cxK+batasBelokBesar['-kiri']

        
            cv2.line(frame, (0,frame.shape[0]//2), (frame.shape[1],frame.shape[0]//2), (0,0,0), thickness=2)
            cv2.circle(frame, (frametengah,frame.shape[0]//2), 10, (0,0,255), thickness=2)
            
            cv2.line(frame, (aKanan,0), (aKanan,frame.shape[0]), (255,255,255), thickness=2)
            cv2.line(frame, (aKiri,0), (aKiri,frame.shape[0]), (255,255,255), thickness=2)
            cv2.line(frame, (bKanan,0), (bKanan,frame.shape[0]), (255,255,255), thickness=2)
            cv2.line(frame, (bKiri,0), (bKiri,frame.shape[0]), (255,255,255), thickness=2)
            cv2.line(frame, (cKanan,0), (cKanan,frame.shape[0]), (255,255,255), thickness=2)
            cv2.line(frame, (cKiri,0), (cKiri,frame.shape[0]), (255,255,255), thickness=2)

            if arduino == 'ada' :
                ##Lurus
                if frametengah > aKiri and frametengah < aKanan :
                    ser.write('a'.encode())
                    print('a')

                ##Belok kanan tipis
                elif frametengah > aKanan and frametengah < bKanan:
                    ser.write('b'.encode())
                    print('b')

                ##Belok kiri tipis
                elif frametengah > bKiri and frametengah < aKiri:
                    ser.write('c'.encode())
                    print('c')

                ##Belok kanan besar
                elif frametengah > bKanan and frametengah < cKanan:
                    ser.write('d'.encode())
                    print('d')

                ##Belok kiri besar
                elif frametengah > cKiri and frametengah < bKiri:
                    ser.write('e'.encode())
                    print('e')
  
                ##Belok kanan tajam
                elif frametengah > cKanan:
                    ser.write('f'.encode())
                    print('f')

                ##Belok kiri tajam
                elif frametengah < cKiri :
                    ser.write('g'.encode())
                    print('g')
                
                if area3 > LuasDermaga :
                    ser.write('h'.encode())
                    print('h')
            #-------------END------------#


        if red_detected or green_detected :
            cv2.putText(frame, "mode bola", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame, (frame.shape[1]//2,0), (frame.shape[1]//2,frame.shape[0]), (0,0,255), thickness=2) ###
            cv2.line(frame, (cxM,cyM), (cxH,cyH), (255,255,255), thickness=2)
            cxDOT = (cxH + cxM) // 2
            cyDOT = (cyH + cyM) // 2
            cv2.circle(frame, (cxDOT,cyDOT), 10, (0,0,255), thickness=2)

            #Area - area manuver kapal
            cv2.line(frame, (frame.shape[1]//2 + batasLurus['-'], 0), (frame.shape[1]//2 + batasLurus['-'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.line(frame, (frame.shape[1]//2 + batasLurus['+'],0), (frame.shape[1]//2 + batasLurus['+'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.putText(frame, 'a', (frame.shape[1]//2,frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), thickness=1) ###

            cv2.line(frame, (frame.shape[1]//2 + batasBelokTipis['-kiri'], 0), (frame.shape[1]//2 + batasBelokTipis['-kiri'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.line(frame, (frame.shape[1]//2 + batasBelokTipis['+kanan'],0), (frame.shape[1]//2 + batasBelokTipis['+kanan'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.putText(frame, 'b', (frame.shape[1]//2 + ((batasBelokTipis['-kiri']+batasLurus['-'])//2) , frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), thickness=1) ###
            cv2.putText(frame, 'c', (frame.shape[1]//2 + ((batasBelokTipis['+kanan']+batasLurus['+'])//2) , frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), thickness=1) ###

            cv2.line(frame, (frame.shape[1]//2 + batasBelokBesar['-kiri'], 0), (frame.shape[1]//2 + batasBelokBesar['-kiri'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.line(frame, (frame.shape[1]//2 + batasBelokBesar['+kanan'],0), (frame.shape[1]//2 + batasBelokBesar['+kanan'],frame.shape[0]), (255,255,255), thickness=1) ###
            cv2.putText(frame, 'd', (frame.shape[1]//2 + ((batasBelokBesar['-kiri']+batasBelokTipis['-kiri'])//2) , frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), thickness=1) ###
            cv2.putText(frame, 'e', (frame.shape[1]//2 + ((batasBelokBesar['+kanan']+batasBelokTipis['+kanan'])//2) , frame.shape[0]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255,255,255), thickness=1) ###

            ##jarak sumbu x lingkaran, ke tengah
            cx = cxDOT - frame.shape[1]/2
            print(cx)
            ##Ngirim data ke arduino
            
            if arduino == 'ada' :
                ##Lurus
                if cx > batasLurus['-'] and cx < batasLurus['+']:
                    ser.write('a'.encode())
                    print('a')

                ##Belok kiri tipis
                elif cx > batasBelokTipis["-kiri"] and cx < batasBelokTipis["-kanan"]:
                    ser.write('b'.encode())
                    print('b')

                ##Belok kanan tipis
                elif cx > batasBelokTipis["+kiri"] and cx < batasBelokTipis["+kanan"]:
                    ser.write('c'.encode())
                    print('c')

                ##Belok kiri besar
                elif cx > batasBelokBesar['-kiri'] and cx < batasBelokBesar['-kanan']:
                    ser.write('d'.encode())
                    print('d')

                 ##Belok kanan besar
                elif cx > batasBelokBesar['+kiri'] and cx < batasBelokBesar['+kanan']:
                    ser.write('e'.encode())
                    print('e')

                ##Belok kiri tajam
                elif cx < batasBelokBesar['-kiri'] :
                    ser.write('f'.encode())
                    print('f')
  
                ##Belok kanan tajam
                elif cx > batasBelokBesar['+kanan']:
                    ser.write('g'.encode())
                    print('g')


        cv2.imshow("Frame", frame)
        # cv2.imshow("merah", mask_red )
        # cv2.imshow("hijau",mask_green)
        # cv2.imshow("final", result)
        # time.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
