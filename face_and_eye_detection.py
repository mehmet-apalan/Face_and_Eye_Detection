face_stages = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_stages  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
capture_video = cv2.VideoCapture(0)
while True:
    ret,PictureFromVideo = capture_video.read()
    gray = cv2.cvtColor(PictureFromVideo, cv2.COLOR_BGR2GRAY)
    face_pics = face_stages.detectMultiScale(gray, 1.1, 6)
    for x, y, w, h in face_pics:
        cv2.rectangle(PictureFromVideo, (x,y), (x+w, y+h), (255,0,0), 2) #lower left and upper right coordinates
        roi_gray = gray[y : y + h, x : x + w]
        roi_colored = PictureFromVideo[y : y + h, x : x + w]
        eyes = eye_stages.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colored, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow("img", PictureFromVideo)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
capture_video.release()
cv2.destroyAllWindows()
