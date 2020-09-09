import cv2

face_cascade = cv2.CascadeClassifier('face.xml')
smile_cascade=cv2.CascadeClassifier('smile.xml')
righteye=cv2.CascadeClassifier('righteye.xml')
leftteye=cv2.CascadeClassifier('lefteye.xml')
img =cv2.imread("akp.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#complete face
for (x, y , w ,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
    cv2.putText(img,"Trideep",(x,y),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),2)
    #smile detection
    gray_smile=gray[y:y+h,x:x+w]
    color_smile=img[y:y+h,x:x+w]
    smile=smile_cascade.detectMultiScale(gray_smile,1.1,100)
    for (sx,sy,sw,sh) in smile:
        cv2.rectangle(color_smile,(sx,sy),(sx+sw,sy+sh),(0,0,255),5)
        cv2.putText(color_smile, "smile", (sx,sy), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0,0), 2)
    #right eye detection
    gray_reye=gray[y:y+h,x:x+h]
    col_reye =img[y:y + h, x:x + h]
    reye=righteye.detectMultiScale(gray_reye,1.1,15)
    for (rx,ry,rw,rh) in reye:
        cv2.rectangle(col_reye,(rx,ry),(rx+rw,ry+rh),(0,255,255),2)
        cv2.putText(col_reye,"R-eye",(rx,ry),cv2.FONT_HERSHEY_COMPLEX,.7,(0,0,0),2)
    #left eye detection
    gray_leye=gray[y:y+h,x:x+h]
    colo_leye=img[y:y+h,x:x+h]
    leye=leftteye.detectMultiScale(gray_leye,1.4,10)
    for (lx,ly,lw,lh) in leye:
        cv2.rectangle(colo_leye,(lx,ly),(lx+lw,lx+lh),(255,0,255),2)
        cv2.putText(colo_leye, "L-eye", (lx, ly), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 0), 2)

    cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()