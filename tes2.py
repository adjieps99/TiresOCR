import cv2
import time
import pytz
import os
import datetime
import requests
import numpy as np
# from trainer import main as train
from mss import mss
from PIL import Image
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib
import urllib.request
import urllib.parse
import os
import tensorflow as tf
import shutil

ip_server_web = "34.101.182.235"
ip_server_ai = "34.101.120.217"

def get_status_process_metode():
#   """get status proses prediksi sedang berjalan atau tidak"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        status_start = r['start']
        status_metode = r['metode']
        return status_start,status_metode
    except requests.exceptions.ReadTimeout:
        print("timeout occured")
        return None,None
        
def get_status_run():
#   """get status motor di mikroskop apakah sedang berjalan atau tidak"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        status_run = r['run']
        return status_run
    except requests.exceptions.ReadTimeout:
        print("timeout occured")
        return None,None

def change_run_status():
    """mengubah status running dari 0 menjadi 1"""
    url= "http://cpepolio.biofarma.co.id/biofarma/pages/update_run/1"
    r = requests.put(url, data={"run": 1})

def get_delay():
    """mendapatkan delay untuk setiap class dari database"""
    dict_time = dict()
    r_time = requests.get(f"http://{ip_server_web}/biofarma/api/info/setting")
    r_time = r_time.json()
    for i in range(len(r_time)):
      dict_time[r_time[i]["kategori"]] = r_time[i]["detik"]
    return dict_time

# def get_ss(sct):
#     """screenshot gambar dari tampilan layar laptop untuk disimpan lalu di upload ke server"""
#     w, h = 1280, 960
#     xmin,ymin,xmax,ymax = 160,160,320,320
#     monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
#     img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
#     img = np.array(img)
#     img = cv2.rectangle(img, (xmin,ymin) , (xmax,ymax), (0,0,255),2)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img_ss = img[xmin:xmax, ymin:ymax] #xmin:xmax, ymin:ymax
#     return img_ss 

def get_ss(sct):
    """screenshot gambar dari tampilan layar laptop untuk disimpan lalu di upload ke server"""
    # w, h = 980, 768
    xmin,ymin,xmax,ymax = 283,0,1083,768
    # monitor = {'top': 0, 'left': 192, 'width': w, 'height': h}
    w, h = 1010, 812
    monitor = {'top': 112, 'left': 140, 'width': w, 'height': h}
    img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
    img = np.array(img)
    # img = cv2.rectangle(img, (xmin,ymin) , (xmax,ymax), (0,0,255),2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_ss = img[xmin:xmax, ymin:ymax] #xmin:xmax, ymin:ymax
    return img

def upload_predict(img_name):
    """upload gambar ke server untuk di prediksi"""
    url = f"http://34.101.120.217:8000/predict"
    files = {'file': open(img_name, 'rb')}
    r = requests.post(url, files=files) 
    r = r.json()
    r = int(r['label'])
    r -= 1
    return r

def start_time():
    """ambil waktu start dari server"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        time_start = r['timer_start']
        start_time_temp = time_start.replace("-","/")
        ribu  = int(start_time_temp.split('/', 1)[0])
        puluh = str(ribu - 2000)
        start_time_temp = start_time_temp.replace("2022", puluh)
        start_time = datetime.datetime.strptime(start_time_temp, '%y/%m/%d %H:%M:%S')
        return start_time
    except requests.exceptions.ReadTimeout:
        print("run timeout occured")
        return None,None

def get_previous_class():
    r = requests.get(f"http://{ip_server_web}/biofarma/api/info/cpe_kategori")
    r = r.json()
    category = list(r['data'][0].values())[1]
    category-=1
    return category

def upload_timecpe(previous_class,last_class,curve_time):
    """upload time cpe ke server untuk curve time graph"""
    url1 = f"http://{ip_server_web}/biofarma/api/info/curve1"
    url2 = f"http://{ip_server_web}/biofarma/api/info/curve2"
    url3 = f"http://{ip_server_web}/biofarma/api/info/curve3"
    if previous_class == 0 and last_class == 1:
        payload=f'cpe1={curve_time}'
        headers = {}
        response = requests.request("PUT", url1, headers=headers, data=payload)
    elif previous_class == 1 and last_class == 2:
        payload=f'cpe2={curve_time}'
        headers = {}
        response = requests.request("PUT", url2, headers=headers, data=payload)
    elif previous_class == 2 and last_class == 3:
        payload=f'cpe3={curve_time}'
        headers = {}
        response = requests.request("PUT", url3, headers=headers, data=payload)

def get_status_training():
#   """get status training apakah sedang berjalan atau tidak"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        status_training = r['training']
        return status_training
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def training():
#   """ start training """
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/updateTraining")
        r = r.json()
        return r
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def reset_training():
#   """ reset training """
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/reset-training")
        r = r.json()
        return r
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def get_file():
#   """ get training """
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/updateTraining")
        r = r.json()
        return r
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None


def make_soup(url):
    html = urllib.request.urlopen(url)
    return BeautifulSoup(html, features="html.parser")
    

def get_allImage(url, cpe):
    soup = make_soup(url)
    images  = [link["href"] for link in soup.find_all('a', href=True)]
    for image in images:
        filename = urllib.parse.unquote_plus(image)
        if image != '../':
            if cpe == 1:
                urllib.request.urlretrieve(url+image, os.path.join(os.path.dirname(__file__),r'C:\Users\biofa\OneDrive\Desktop\minipc-biofarma\images\schedule_training\CPE 1/{filename}'))   
            elif cpe == 2:
                urllib.request.urlretrieve(url+image, os.path.join(os.path.dirname(__file__),r'C:\Users\biofa\OneDrive\Desktop\minipc-biofarma\images\schedule_training\CPE 2/{filename}'))   
            elif cpe == 3:
                urllib.request.urlretrieve(url+image, os.path.join(os.path.dirname(__file__),r'C:\Users\biofa\OneDrive\Desktop\minipc-biofarma\images\schedule_training\CPE 3/{filename}'))   
            elif cpe == 4:
                urllib.request.urlretrieve(url+image, os.path.join(os.path.dirname(__file__),r'C:\Users\biofa\OneDrive\Desktop\minipc-biofarma\images\schedule_training\CPE 4/{filename}'))        
    
    return print('success')

def get_images(cpe):
    url1 = "http://34.101.182.235/biofarma/uploads/schedule_training/CPE%201/"
    url2 = "http://34.101.182.235/biofarma/uploads/schedule_training/CPE%202/"
    url3 = "http://34.101.182.235/biofarma/uploads/schedule_training/CPE%203/"
    url4 = "http://34.101.182.235/biofarma/uploads/schedule_training/CPE%204/"
    

    if cpe==1:
        get_allImage(url1, 1)
    elif cpe==2:
        get_allImage(url2, 2)
    elif cpe==3:
        get_allImage(url3, 3)
    elif cpe==4:
        get_allImage(url4, 4)

def get_status_predict_manual():
#   """get status predict"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        status_predict = r['predict']
        return status_predict
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def reset_predict():
#   """ reset predict """
    try:
        r = requests.get(f"http://cpepolio.biofarma.co.id/biofarma/pages/reset_predict")
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None
        
def send_notification(step):
    url = "http://cpepolio.biofarma.co.id/biofarma/api/notif"
    if step == 1:
        dict_notif = {"old_cpe": "CPE 1","new_cpe": "CPE 2"}
        requests.post(url,json=dict_notif)
    if step == 2:
        dict_notif = {"old_cpe": "CPE 2","new_cpe": "CPE 3"}
        requests.post(url,json=dict_notif)
    if step == 3:
        dict_notif = {"old_cpe": "CPE 3","new_cpe": "CPE 4"}
        requests.post(url,json=dict_notif)

def inference(model, image_path):
  #-----preprocessing
  image_ori = cv2.imread(image_path)
  image = cv2.resize(image_ori, (512, 512))
  image = image.astype(float)/255.0
  x = np.expand_dims(image, axis=0)

  #-----inference
  p = model.predict(x)
  p_class = np.argmax(p)
  return p, p_class,image_ori


def post_processing(p, p_c,image_ori,class_dict):
    label = class_dict[p_c]
    class_score = [] #class score untuk melihat kemiripan
        
    for i in range(p.shape[1]):
        score = p[:,i].item()*100
        class_score.append(float(score))

    class_score_list = class_score
    class_score = np.array(class_score)
    score_classification = class_score.max()
    
    # if score_classification > 50: #jika score lebih dari 50 maka akan menghasilkan output kelas yang benar
    #     label = label
    #     path_file_image = "images/schedule_training/{}".format(label)
        
    #     if os.path.exists(path_file_image):
    #         pass
    #     else:
    #         os.mkdir(path_file_image)
        
    #     path_after_prediction = path_file_image + "/{}".format(image_path)
    #     cv2.imwrite("{}".format(path_after_prediction),image_ori)  #save image for future training
    #     print(path_after_prediction)      
    # else:
    #     label = "Unknown"
    return label,score_classification,class_score_list


def send_data_pred(category,score_classification,img,class_score_list):
  url = "https://cpepolio.biofarma.co.id/biofarma/api/predict"
#   url = "http://34.101.182.235/biofarma/api/gambar?"
  img_filesize = os.stat(img).st_size
  payload={'kategori': category,'persentase': score_classification,"size":img_filesize,"cpe1":class_score_list[0],"cpe2":class_score_list[1],"cpe3":class_score_list[2],"cpe4":class_score_list[3]}
  files = {'file': open(img, 'rb')}
  headers = {}
  response = requests.request("POST", url, headers=headers, data=payload, files=files)

model = tf.keras.models.load_model('models/model_best.h5')
class_dict = {0:'1', 
              1:'2', 
              2:'3',
              3:'4'}
latest_cpe = ""
cpe1_time = ""
cpe2_time = ""
cpe3_time = ""
cpe4_time = ""
durasi_cpe12,durasi_cpe23,durasi_cpe34 = 0,0,0
key_percobaan = ""

def predict_manual(image):
    p, p_c,image_ori = inference(model, image)
    label,score_classification,class_score_list = post_processing(p, p_c,image_ori,class_dict)
    send_data_pred(label,score_classification,image,class_score_list)
    # delete_uploaded_images(upload_dir)
    return {"label":label,"s":str(score_classification)}

# # new to do list: buat predik manual di lokal
# def upload_predict_manual(img_name):
#     """upload gambar ke server untuk di prediksi"""
#     url = f"http://34.101.120.217:8000/predict_manual"
#     files = {'file': open(img_name, 'rb')}
#     r = requests.post(url, files=files) 
#     r = r.json()
#     r = int(r['label'])
#     r -= 1
#     return r

def get_time_predict():
#   """get status motor di mikroskop apakah sedang berjalan atau tidak"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/pages/live_timer")
        r = r.json()
        status_run = r['detik']
        return status_run
    except requests.exceptions.ReadTimeout:
        print("timeout occured")
        return None,None

def get_status_predict_auto():
#   """get status predict"""
    try:
        r = requests.get(f"http://{ip_server_web}/biofarma/api/info/status")
        r = r.json()
        status_predict = r['predict_auto']
        return status_predict
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def reset_predict_auto():
#   """ reset predict """
    try:
        r = requests.get(f"http://cpepolio.biofarma.co.id/biofarma/pages/reset_predict_auto")
    except requests.exceptions.ReadTimeout:
        print("training timeout occured")
        return None,None

def main():
    sct = mss()

    while True:
        status_start,status_metode = get_status_process_metode()
        status_run = get_status_run()                
        print(f"status start : {status_start}, status metode : {status_metode}, status run : {status_run}")

        # get status predict auto
        status_predict_auto = get_status_predict_auto()
        print(f"status predict : {status_predict_auto}")

        while status_metode == 0 and status_run == 0 and status_predict_auto == 1:
                print("predict auto is active")

                # get predict auto time
                timer_auto = get_time_predict()
                menit = timer_auto/60
                print(f"predict auto with {menit} minutes")

                # delay timer 
                time.sleep(timer_auto)

                # get ss
                get_screenshot = get_ss(sct)

                # rename file by up to date time and predict
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
                nama_file = f"{st}.png"      
                cv2.imwrite(nama_file,get_screenshot)

                # # send file to web server for only display
                # image = upload_predict_manual(str(nama_file))
                # os.remove(nama_file)

                # # uncomment if want predict to engine AI
                # image = upload_predict(str(nama_file))
                # os.remove(nama_file)
                # print(image)

                # # change status predict
                # reset_predict_auto()

                # get current predict status
                status_predict = get_status_predict_auto()
                print(f"status predict: {status_predict}")
        

                    
if __name__ == "__main__":
    main()