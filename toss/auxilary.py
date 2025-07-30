import requests
import re
import os
from pymatgen.core import IStructure
#import pymysql
import pandas as pd
import numpy as np



def send_notice():
    event_name = "NOTICE:"
    key = "cbgG6OygzBbJGpXxeSbJgz"
    url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
    response = requests.request("POST", url)
    return None

def sent_message(value1 = "NOTICE", value2 = "Calculation Finished!!!", value3 = "Go to check what's new."):
    event_name = "Push"
    key = "cbgG6OygzBbJGpXxeSbJgz"
    url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
    payload = "{\n    \"value1\": \""+value1+"\",  \n  \"value2\": \""+value2+"\",  \n  \"value3\": \""+value3+"\"    \n}"
    headers = {
    'Content-Type': "application/json",
    'User-Agent': "PostmanRuntime/7.15.0",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
    'Host': "maker.ifttt.com",
    'accept-encoding': "gzip, deflate",
    'content-length': "63",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
    }
    response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
    return None


class one_file_lize():

    def __init__(self, num=1147168, one_file_name="all.cif", path="D:/share/TOSS/"):
        self.num = num
        self.path = path
        self.one_file_name = one_file_name

    def save(self):
        target_group = os.listdir(self.path + "structures/")
        assert len(target_group) == self.num
        
        with open(self.path + self.one_file_name, "w") as F:
            for single_file_name in target_group:
                with open(self.path + "structures/" + single_file_name, "r") as f:
                    str_ver = f.read()
                    without_n = re.sub("\n", "YUEYIN", str_ver)
                    the_line = "HEAD" + single_file_name + "TAIL" + without_n + "\n"
                    F.write(the_line)

    def get(self, mid, mid_line_dict):
        the_line = os.popen("awk 'NR=={}' {}".format(mid_line_dict[mid], self.path + self.one_file_name)).read()
        str_ver = re.sub("YUEYIN","\n", the_line)
        mid = re.findall(r'@Y@(.*)@Y@',str_ver)[0]
        struct = IStructure.from_str(str_ver, "cif")
        return struct