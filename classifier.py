from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template
import os
import cv2
import numpy as np
from model import FacialExpressionModel

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
bootstrap = Bootstrap(app)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

modelFER = FacialExpressionModel("models/new_fer_best_model.h5")

class UploadForm(FlaskForm):
    upload = FileField('Seleccionar una imagen:', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg', 'JPEG', 'PNG', 'JPG'], 'Images only!')
    ])
    submit = SubmitField('Clasificar')


def get_prediction(img_path):
    img_FER = cv2.imread(img_path)
    img_gray_FER = cv2.cvtColor(img_FER, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(img_gray_FER, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = img_gray_FER[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = modelFER.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(img_FER, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(img_FER,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imwrite('static/result.jpg', img_FER)
    return pred

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.upload.data
        filename = secure_filename(f.filename)
        file_url = os.path.join('static', filename)
        f.save(file_url)
        form = None
        prediction = get_prediction(file_url)
        file_url = 'static/result.jpg'
    else:
        file_url = None
        prediction = None
    return render_template("index.html", form=form, file_url=file_url, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)