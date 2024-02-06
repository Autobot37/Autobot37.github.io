import cv2
from deepface import DeepFace

known_face = cv2.imread('known_facee.jpg')
names = ["DLC 0"]
cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()

    faces = DeepFace.extract_faces(frame, enforce_detection=False)
    
    for face_dict in faces:
        face = face_dict['face']
        region = face_dict['facial_area']
        
        verified = DeepFace.verify(known_face, face, enforce_detection=False)
        
        if verified['verified']:
            print("verified")
            x, y, w, h = region['x'],region['y'],region['w'],region['h']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 20)
            cv2.putText(frame, names[0], (x, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            break

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    i+=1
    if i>10:
        break
cap.release() 
cv2.destroyAllWindows()