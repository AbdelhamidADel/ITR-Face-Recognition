import os
import numpy as np
from flask import Flask, request, render_template
import pickle
import cv2
from model import w2d,get_cropped_image_if_2_eyes


#place of save images
UPLOAD_FOLDER = './static/images/'

app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/Home')
def home():
    return render_template('Home.html')


@app.route('/Recognition', methods=['GET', 'POST'])
def Recognition():
    if request.method == 'POST':
        # ----------------------------------------choose file------------------------------------------------
        file1 = request.files['file1']
        if 'file1' not in request.files:
            return render_template('Recognition.html',output='there is no file uploaded') 

        elif file1.filename.endswith("png") or file1.filename.endswith("jpg") or file1.filename.endswith("jpeg"):
            path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(path)
            # model process

            #pickle model
            with open('saved_model.pkl','rb') as file:
                model = pickle.load(file)
            try:
            #2-crop face
                cropped_image=get_cropped_image_if_2_eyes(path)

            
                #3-scalling
                scalled_raw_img = cv2.resize(cropped_image, dsize=(32, 32))
                img_har = w2d(cropped_image,'db1',5)
                scalled_img_har = cv2.resize(img_har, dsize=(32, 32))
                combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
                test=np.array(combined_img).reshape(1, 4096)
                #4-result
                prediction=model.predict(test)
                if prediction == 0:
                    name="She is Alexia Putellas"
                if prediction == 1:
                    name="He is Gianluigi Donnarumma"
                if prediction == 2:
                    name="He is Lionel Messi"
                if prediction == 3:
                    name="He is Pedri"
                if prediction == 4:
                    name="He is Robert Lewandowski"

            
                return render_template('Recognition.html',output=name) 
            except:
                return render_template('Recognition.html',output="There is a problem, try another picture")
          
        else :
            return render_template('Recognition.html',output="It wasn't a picture, next time please make sure you upload a picture!")                  
    return render_template('Recognition.html')
    
@app.route('/About')
def About():
    return render_template('About.html')
    


if __name__ == '__main__':
    app.run(debug=True)