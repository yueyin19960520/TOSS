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
                    the_line = "HEAD" + file_name + "TAIL" + without_n + "\n"
                    F.write(the_line)

    def get(self, mid, mid_line_dict):
        the_line = os.popen("awk 'NR=={}' {}".format(mid_line_dict[mid], self.path + self.one_file_name)).read()
        str_ver = re.sub("YUEYIN","\n", the_line)
        mid = re.findall(r'@Y@(.*)@Y@',str_ver)[0]
        struct = IStructure.from_str(str_ver, "cif")
        return struct
"""END HERE"""




"""
class CIF_SQL():
    
    def __init__(self, host="localhost", user="root", password="tosstoss", database="cif_sql"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database)
    
    def add_cif(self, mid):
        cursor = self.db.cursor()
        struct_ori = IStructure.from_file("..structures/%s"%mid)
        dict_struct = struct_ori.as_dict()
        val_struct = self._dict_2_value(dict_struct)
    
        mid = mid[0:-4].replace("-","") if mid[0] == "m" else "OQMD" + mid[0:-4]
    
        cursor.execute("drop table if exists %s"%mid)
        self._create_mid_table(mid, val_struct, cursor)
        cursor.close()
        return None
        
    def check(self, mid):
        cursor = self.db.cursor()
        struct_ori = IStructure.from_file("D:/share/TOSS/structures/%s"%mid)
        
        mid = mid[0:-4].replace("-","") if mid[0] == "m" else "OQMD" + mid[0:-4]
        cursor.execute("SELECT * FROM %s"%mid)
        table = cursor.fetchall()
        dict_struct = self._table_2_dict(table)
        struct_sql = IStructure.from_dict(dict_struct)
        cursor.close()
        
        if struct_sql == struct_ori:
            return True
        else:
            return False

    def get_struct(self, mid):
        cursor = self.db.cursor()
        mid = mid[0:-4].replace("-","") if mid[0] == "m" else "OQMD" + mid[0:-4]
        cursor.execute("SELECT * FROM %s"%mid)
        table = cursor.fetchall()
        dict_struct = self._table_2_dict(table)
        cursor.close()
        return dict_struct 
    
    def _create_mid_table(self, mid, struct, cursor):
        sql_table = '''create table if not exists %s(
                    id int primary key auto_increment,
                    a varchar(20), b varchar(20), c varchar(20),
                    x varchar(20), y varchar(20), z varchar(20),
                    e varchar(20));'''%mid
        cursor.execute(sql_table)
        sql_data = "insert into %s"%mid + " (id, a, b, c, x, y, z, e) values (%s, %s, %s, %s, %s, %s, %s, %s);"
        cursor.executemany(sql_data, struct)
        self.db.commit()
        return None
    
    def _dict_2_value(self, struct):
        lattice = struct["lattice"]
        cell = lattice["matrix"]
        sites = struct["sites"]
        row_1 = (1,lattice["a"], lattice["alpha"], lattice["volume"], cell[0][0], cell[0][1], cell[0][2], "none")
        row_2 = (2,lattice["b"], lattice["beta"], lattice["volume"], cell[1][0], cell[1][1], cell[1][2], "none")
        row_3 = (3,lattice["c"], lattice["gamma"], lattice["volume"], cell[2][0], cell[2][1], cell[2][2], "none")
        list_ver = [row_1, row_2, row_3]
        idx = 4
        for site in sites:
            abc = site["abc"]
            xyz = site["xyz"]
            ele = site["label"]
            row = (idx, abc[0], abc[1], abc[2], xyz[0], xyz[1], xyz[2], ele)
            list_ver.append(row)
            idx += 1
        return list_ver
    
    def _table_2_dict(self, table):
        df = pd.DataFrame(table)
        dict_ver = {"@module":"pymatgen.core.structure","@class":"IStructure","@charge":None}
    
        lattice = {"matrix":np.array(df.iloc[0:3, 4:7], dtype="float32")}
        abc = {"a":np.array(df.iloc[0,1], dtype="float32"), 
               "b":np.array(df.iloc[1,1], dtype="float32"), 
               "c":np.array(df.iloc[2,1], dtype="float32")}
        xyz = {"alpha":np.array(df.iloc[0,2], dtype="float32"), 
               "beta":np.array(df.iloc[1,2], dtype="float32"), 
               "gamma":np.array(df.iloc[2,2], dtype="float32")}
        lattice.update(abc)
        lattice.update(xyz)
        lattice.update({"volume":np.array(df.iloc[2,3], dtype="float32")})
        dict_ver.update({"lattice":lattice})
    
        sites = []
        for i in range(3,df.shape[0]):
            temp = df.iloc[i,:]
            ele = temp.iloc[-1]
            temp_dict = {"species":[{"element":ele, "occu":1.0}],
                         "abc": np.array(temp.iloc[1:4], dtype="float32"),
                         "xyz": np.array(temp.iloc[4:7], dtype="float32"),
                         "label":"Ca",
                         "properties":{}}
            sites.append(temp_dict)
        dict_ver.update({"sites":sites})
        return dict_ver
    
    def __clear_db__(self):
        db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        cursor = self.db.cursor()
        cursor.execute("show tables;")
        for table in list(cursor):
            cursor.execute("drop table %s"%table[0])
        assert len(list(cursor)) == 0
        cursor.close()  
        return None
    
    def __len__(self):
        db = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database)
        cursor = self.db.cursor()
        cursor.execute("show tables;")
        return len(list(cursor))


    #usage
    cs = CIF_SQL()
    for mid in target_group:
        cs.add_cif(mid)
    print(cs.__len__())
    cs.__clear_db__()
    print(cs.__len__())
    """