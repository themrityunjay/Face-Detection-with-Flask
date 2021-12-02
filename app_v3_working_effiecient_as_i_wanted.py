import cv2
import numpy as np
import os
from face_detection import *


from flask import Flask,request,Response,jsonify,send_from_directory

from face_detection import image_processing

# output_path="./detections/"

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app=Flask(__name__)

@app.route('/',methods=['POST'])
def get_detections():
### Image Read............................

    file_img = request.files["image"].read() ### byte file
    # npimg = np.fromstring(file_img,np.uint8)
    npimg = np.frombuffer(file_img,np.uint8)

    img=cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    
    img=image_processing(img)

######################################################################
### Image Operations for response ..................................

    # faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     # roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]

    cv2.imwrite("detections/detection.jpg",img)

#######################################################################

### Prepare image for response
    _, img_encoded = cv2.imencode('.jpg', img)

    response = img_encoded.tobytes()

### Response to browser
    return Response(response=response, status=200,  mimetype='image/png')


if __name__=='__main__':
    app.run(debug=True)