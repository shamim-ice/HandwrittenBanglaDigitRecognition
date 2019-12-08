from flask import Flask, request, render_template
import yaml
import cv2, numpy as np
from keras.models import load_model
from flask_mysqldb import MySQL
import re
import base64
import bangla
i=0
app = Flask(__name__)

yaml.warnings({'YAMLLoadWarning': False})
db=yaml.load(open('db.yaml'))
app.config['MYSQL_HOST']=db['mysql_host']
app.config['MYSQL_USER']=db['mysql_user']
app.config['MYSQL_PASSWORD']=db['mysql_password']
app.config['MYSQL_DB']=db['mysql_db']

mysql=MySQL(app)

global model


model = load_model('val_loss32.hdf5')


def ConvertImage(img):
    imgstr = re.search(b'base64,(.*)', img).group(1)
    with open('image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    global i
    img = request.get_data()
    imgstr = re.search(b'base64,(.*)', img).group(1)
    with open('E:/Code/htmljscss/imageData/img_{}.png'.format(i), 'wb') as output:
        output.write(base64.b64decode(imgstr))
    #ConvertImage(img)
    x = cv2.imread('E:/Code/htmljscss/imageData/img_{}.png'.format(i), 0)
    x = np.invert(x)
    #cv2.imshow('Image',x)
    #cv2.waitKey(0)
    x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA)
    x = x.reshape(1, 32, 32, 1)
    out = model.predict(x)
    response = out.argmax(axis=1)
    ret = bangla.convert_english_digit_to_bangla_digit(response)
    img_id = ('img_{}.png'.format(i))
    label = response[0]
    cur = mysql.connection.cursor()
    if i==0:
        cur.execute('delete from image_info')

    cur.execute('insert into image_info(img_id,label) values(%s,%s)', (img_id, label))
    mysql.connection.commit()
    cur.close()

    i+=1
    return ret


if __name__ == '__main__':
    app.run(debug=True)
