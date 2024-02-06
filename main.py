from flask import Flask, render_template, request, jsonify
import numpy as np
from deepface import DeepFace
import cv2
import os

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
    
@app.route('/') 
def register():
    return render_template("register.html")

@app.route('/save', methods=['POST']) 
def save():
    try:
        image = request.files['image']
        name = request.form['name'] + '.png' 
        image.save(os.path.join('images', name))  
        print(f"saved:{name} img")
    except Exception as e:
        print(e)
        return "Error saving image"
        
    return "Image saved"

@app.route("/home")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    frame = request.files["frame"]

    frame_cv2 = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), cv2.IMREAD_COLOR)
    frame_np = np.array(frame_cv2)

    json_response = {"verified": False, "region": None,"Name":None}

    for imgpath in os.listdir("images"):
        target_image_path =  os.path.join("images",imgpath)
        target_name = f"{imgpath[:-4]}"
        target_np = np.array(cv2.imread(target_image_path))
        print(target_np.shape,frame_np.shape)
        faces = DeepFace.extract_faces(frame_np, enforce_detection=False)

        for face_dict in faces:
            face = face_dict['face']
            region = face_dict['facial_area']
            
            verified = DeepFace.verify(target_np, face, enforce_detection=False)

            if verified['verified']:
                json_response["verified"] = True
                json_response["region"] = region
                json_response["Name"] = target_name
                print(target_name)
                return jsonify(json_response)

    return jsonify(json_response)

if __name__ == "__main__":
    app.run(debug=True,port=8090)
