import sqlite3
from xml.dom.minidom import parse
import os, random
from tqdm import tqdm, trange

def data_type():
    rand = random.random()
    if rand < 0.1:
        return 'valid'
    elif rand < 0.3:
        return 'test'
    else:
        return 'train'

def Read(dirname):
    sub_dir_list = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
    sub_dir_list = [fname for fname in sub_dir_list if os.path.isdir(fname)]
    for sub_dir_name in sub_dir_list:
        cur.execute("SELECT ID FROM Amazon")
        rst = cur.fetchall()
        cnt_ID = len(rst)
        domain_name = sub_dir_name.strip('/').rsplit('/')[-1]
        print("**** sub_dir_name >>> ", sub_dir_name)
        with open(os.path.join(sub_dir_name, "./all.txt"), 'r') as fr:
            for line in tqdm(fr):
                s = line.strip("\n").replace("'", "[pad]").split(' ', 1)
                d_type = data_type()
                try:
                    insert_cmd = f"INSERT INTO Amazon(ID, sentence, label, domain, data_type) \
                        VALUES ({cnt_ID}, '{s[1]}', {int(s[0])}, '{domain_name}', '{d_type}')"
                    cur.execute(insert_cmd)
                except Exception as e:
                    print("*** insert failed ==> ", str(e))
                    print("INSERT CMD : ", insert_cmd)
                    con.rollback()
                else:
                    cnt_ID += 1
                    con.commit()
        with open(os.path.join(sub_dir_name, "./unl.txt"), 'r') as fr:
            for line in tqdm(fr):
                s = line.strip("\n").replace("'", "[pad]").split(' ', 1)
                try:
                    insert_cmd = f"INSERT INTO Amazon(ID, sentence, label, domain, data_type) \
                        VALUES ({cnt_ID}, '{s[1]}', {int(s[0])}, '{domain_name}', 'unlabeled')"
                    cur.execute(insert_cmd)
                except Exception as e:
                    print("*** insert failed ==> ", str(e))
                    print("INSERT CMD : ", insert_cmd)
                    con.rollback()
                else:
                    cnt_ID += 1
                    con.commit()

def update(domain_list=None, dirname=None):
    assert domain_list is not  None or dirname is not  None
    if domain_list is None:
        sub_dir_list = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
        sub_dir_list = [fname for fname in sub_dir_list if os.path.isdir(fname)]
        domain_list = [sub_dir_name.strip('/').rsplit('/')[-1]
                                for sub_dir_name in sub_dir_list]
    for domain_name in domain_list:
        print("*** domain_name >>> ", domain_name)
        cur.execute(f"SELECT ID FROM Amazon WHERE domain == '{domain_name}' AND label != -1")
        IDs = cur.fetchall()
        new_IDs = random.sample(IDs, len(IDs))
        end_valid, end_te, end_tr = int(len(new_IDs)*0.1), int(len(new_IDs)*0.3), int(len(new_IDs)*1.0)
        try:
            for i in trange(end_valid):
                cur.execute(
                    f"UPDATE Amazon SET data_type='valid' WHERE ID =={new_IDs[i][0]}"
                )

            for i in trange(end_valid, end_te):
                cur.execute(
                    f"UPDATE Amazon SET data_type='test' WHERE ID =={new_IDs[i][0]}"
                )

            for i in trange(end_te, end_tr):
                cur.execute(
                    f"UPDATE Amazon SET data_type='train' WHERE ID =={new_IDs[i][0]}"
                )
        except Exception as e:
            print("*** update the table failed! >>> ", str(e))
            con.rollback()
        else:
            con.commit()


def CreateTable():
    cur.execute(
        """
        CREATE TABLE Amazon(
           ID INT PRIMARY KEY     NOT NULL,
           sentence       TEXT    NOT NULL,
           label          INT     NOT NULL,
           domain         TEXT    NOT NULL, 
           data_type      TEXT    NOT NULL
        );
        """
    )
    con.commit()

def DelTable():
    cur.execute(
        "DROP TABLE Amazon"
    )
    con.commit()

if __name__ == '__main__':
    con = sqlite3.connect("Sentiment.db")
    cur = con.cursor()
    DelTable()
    CreateTable()