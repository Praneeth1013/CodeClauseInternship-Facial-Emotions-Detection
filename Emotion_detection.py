import cv2
from fer import FER

# Loading haarcascade model and FER model
emotion_model = FER()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

path = 'images/emotion.jpg' # Image path

image = cv2.imread(path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecting faces
faces = face_detector.detectMultiScale(gray_image,scaleFactor=1.3,minNeighbors=9,minSize=(30,30))

for (x, y, w, h) in faces:
    # convert the image back to BGR
    face_image = cv2.cvtColor(gray_image[y+5:y + h+5, x+5:x + w+5], cv2.COLOR_GRAY2BGR)

    # Detect the emotion
    emotion,value  = emotion_model.top_emotion(face_image)

    # Print the emotion and the accuracy
    print(f'{emotion} - {value}')

    # Draw the emotion on the face
    cv2.rectangle(image,(x+5,y+5),(x+w+5,y+h+5),(0,255,0),2)
    cv2.putText(image, emotion, (x, y+w+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),thickness = 2)


# Show the image
cv2.imshow(path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()