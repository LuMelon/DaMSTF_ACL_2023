import sqlite3
from xml.dom.minidom import parse

label_dict = {
    "conflict" : 3,
    "positive" : 2,
    "neutral" : 1,
    "negative" : 0
}
def restaurant_Train():
    cur.execute(
    "SELECT ID FROM Train"
    )
    IDs = cur.fetchall()
    tree = parse("./Restaurants_Train.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO Train(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'restaurant')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def restaurant_Test():
    cur.execute(
    "SELECT ID FROM Test"
    )
    IDs = cur.fetchall()
    tree = parse("./ABSA_Gold_TestData/Restaurants_Test_Gold.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO Test(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'restaurant')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def restaurant_FewShot():
    cur.execute(
    "SELECT ID FROM FewShot"
    )
    IDs = cur.fetchall()
    tree = parse("./restaurants_Trial.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO FewShot(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'restaurant')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def Laptop_Train():
    cur.execute(
    "SELECT ID FROM Train"
    )
    IDs = cur.fetchall()
    tree = parse("./Laptops_Train.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO Train(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'laptop')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def Laptop_Test():
    cur.execute(
    "SELECT ID FROM Test"
    )
    IDs = cur.fetchall()
    tree = parse("./ABSA_Gold_TestData/Laptops_Test_Gold.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO Test(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'laptop')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def Laptop_FewShot():
    cur.execute(
    "SELECT ID FROM FewShot"
    )
    IDs = cur.fetchall()
    tree = parse("./laptops_Trial.xml")
    cnt_ID = len(IDs) + 1
    for ele in tree.getElementsByTagName('sentence'):
        text = ele.getElementsByTagName('text')[0].childNodes[0].data.strip("\n").replace("'", "[pad]")
        for ape in ele.getElementsByTagName('aspectTerm'):
            asp = ape.getAttribute('term').strip("\n").replace("'", "[pad]")
            polarity = ape.getAttribute('polarity')
            posi = label_dict[polarity]
            try:
                insert_cmd = f"INSERT INTO FewShot(ID, aspect, sentence, label, domain) \
                    VALUES ({cnt_ID}, '{asp}', '{text}', {posi}, 'laptop')"
                cur.execute(insert_cmd)
            except Exception as e:
                print("*** insert failed ==> ", str(e))
                print("INSERT CMD : ", insert_cmd)
                con.rollback()
            else:
                con.commit()
            item = []
            cnt_ID += 1

def TwitterTest():
    with open("./test.raw", 'r') as fr:
        line_idx, cnt_ID = 0, 1
        item = []
        for line in fr:
            item.append(line.strip("\n").replace("'", "[pad]"))
            if len(item) == 3:
                try:
                    insert_cmd = f"INSERT INTO Test(ID, aspect, sentence, label, domain) \
                        VALUES ({cnt_ID}, '{item[1]}', '{item[0]}', {int(item[2]) + 1}, 'twitter')"
                    cur.execute(insert_cmd)
                except Exception as e:
                    print("*** insert failed ==> ", str(e))
                    print("INSERT CMD : ", insert_cmd)
                    con.rollback()
                else:
                    con.commit()
                item = []
                cnt_ID += 1

def TwitterTrainFewShot():
    with open("./train.raw", 'r') as fr:
        line_idx, cnt_ID = 0, 1
        item = []
        for line in fr:
            item.append(line.strip("\n").replace("'", "[pad]"))
            if len(item) == 3:
                try:
                    if cnt_ID < 6000:
                        insert_cmd = f"INSERT INTO Train(ID, aspect, sentence, label, domain) \
                            VALUES ({cnt_ID}, '{item[1]}', '{item[0]}', {int(item[2]) + 1}, 'twitter')"
                    else:
                        insert_cmd = f"INSERT INTO FewShot(ID, aspect, sentence, label, domain) \
                            VALUES ({cnt_ID-6000}, '{item[1]}', '{item[0]}', {int(item[2]) + 1}, 'twitter')"
                    cur.execute(insert_cmd)
                except Exception as e:
                    print("*** insert failed ==> ", str(e))
                    print("INSERT CMD : ", insert_cmd)
                    con.rollback()
                else:
                    con.commit()
                item = []
                cnt_ID += 1

def CreateTable():
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Train(
           ID INT PRIMARY KEY     NOT NULL,
           aspect         TEXT    NOT NULL,
           sentence       TEXT    NOT NULL,
           label          INT     NOT NULL,
           domain         TEXT    NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS FewShot(
           ID INT PRIMARY KEY     NOT NULL,
           aspect         TEXT    NOT NULL,
           sentence       TEXT    NOT NULL,
           label          INT     NOT NULL,
           domain         TEXT    NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Test(
           ID INT PRIMARY KEY     NOT NULL,
           aspect         TEXT    NOT NULL,
           sentence       TEXT    NOT NULL,
           label          INT     NOT NULL,
           domain         TEXT    NOT NULL
        );
        """
    )
    con.commit()

def DelTable():
    cur.execute(
        "DROP TABLE IF EXISTS Test"
    )
    cur.execute(
        "DROP TABLE IF EXISTS Train"
    )
    cur.execute(
        "DROP TABLE IF EXISTS FewShot"
    )
    con.commit()

if __name__ == '__main__':
    con = sqlite3.connect("./DA_ASBA.db")
    cur = con.cursor()
    DelTable()
    CreateTable()
    TwitterTrainFewShot()
    TwitterTest()
    Laptop_Train()
    Laptop_Test()
    Laptop_FewShot()
    restaurant_FewShot()
    restaurant_Test()
    restaurant_Train()
