from matplotlib.pyplot import get
import requests


ip_server_web = "34.101.182.235"

def get_delay():
    """mendapatkan delay untuk setiap kelas dari database"""
    dict_time = dict()
    r_time = requests.get(f"http://{ip_server_web}/biofarma/api/info/setting")
    r_time = r_time.json()
    for i in range(len(r_time)):
      dict_time[r_time[i]["kategori"]] = r_time[i]["detik"]


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

# # get class paramater  
# r_time = get_delay()         
# print(f"r time   : {r_time}")                       
# list_kelas = list(r_time.keys())
# print(f"list class   : {list_kelas}")  

time = get_time_predict()
print(time)