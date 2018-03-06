#################################################
#Programe Name  :RuleBased IE
#Author         :kpchen@nlg.csie.ntu.edu.tw
#Relative Files :./Character.py
#                ./data/train.txt
#                ./data/test.txt
#                ./data/Dream_of_the_Red_Chamber.txt             
#################################################
from Character import Character

characters = {}
relations = {}
character_identify = 0
main_role = ['賈','史','王','薛']
location_prefix = ['在','到','到了','來','至','去','往','從','向','住','住了']
location_postfix = ['來','內','中','去']
slave_prefix = ['丫鬟名','丫鬟','丫頭','命','喚','、']
slave_postfix = ['、']

#################################################
#判斷兩個 list 是否有重複項目
#################################################
def match(list1, list2):
    match = set(list1) & set(list2)
    return len(match)>=1

#################################################
#判斷兩個角色是否有已知的血緣關係
#################################################
def blood_rel(c1, c2):
    return match(entity1.sibling,entity2.sibling) or match(entity1.grand,entity2.grand)
           
#################################################
#根據 train.txt 及 test.txt 建立共用的角色字典
#若是紅樓夢四大家族:賈史王薛的成員標記為主要角色
################################################# 
with open('./data/train.txt', encoding='utf8') as f:
    next(f)
    for line in f:
        data = line.strip('\n').split('\t')
        if data[1] not in characters:
            characters[data[1]] = Character(character_identify)
            if any(x in data[1] for x in main_role):
                setattr(characters[data[1]], "main_role", True)
            character_identify += 1
        if data[2] not in characters: 
            characters[data[2]] = Character(character_identify)
            if any(x in data[2] for x in main_role):
                setattr(characters[data[2]], "main_role", True)
            character_identify += 1
            
with open('./data/test.txt', encoding='utf8') as f:
    next(f)
    for line in f:
        data = line.strip('\n').split('\t')
        if data[1] not in characters:
            characters[data[1]] = Character(character_identify)
            if any(x in data[1] for x in main_role):
                setattr(characters[data[1]], "main_role", True)
            character_identify += 1
        if data[2] not in characters: 
            characters[data[2]] = Character(character_identify)
            if any(x in data[2] for x in main_role):
                setattr(characters[data[2]], "main_role", True)
            character_identify += 1
            
#################################################
#根據 Dream_of_the_Red_Chamber.txt 計算角色詞頻
#加入預先定義的前綴、後綴詞來判斷主僕及地點關係
################################################# 
with open('./data/Dream_of_the_Red_Chamber.txt', encoding='utf8') as f:
    data = f.read()
    for character,attr in characters.items():
        if len(character)==3:
            search = character[1:]            
        else:
            search = character
        setattr(characters[character], "freq", data.count(search))  
        l_freq = 0
        for w in location_prefix:
            l_freq += data.count(w+character)
        for w in location_postfix:
            l_freq += data.count(character+w)
        setattr(characters[character], "l_freq", l_freq)
            
        slave_count = 0
        for w in slave_prefix:
            slave_count += data.count(w+character)
        for w in slave_postfix:
            slave_count += data.count(character+w)
        if slave_count > 0 and not attr.main_role and attr.freq <= 700:
            setattr(characters[character], "slave", True)

#################################################
#根據 train.txt 記錄角色性別及人物關係資訊
################################################# 
with open('./data/train.txt', encoding='utf8') as f:
    next(f)
    for line in f:
        data = line.strip('\n').split('\t')
        if data[3] == "母女":
            setattr(characters[data[1]], "gender", 0)
            setattr(characters[data[2]], "gender", 0)
        if data[3] == "父子":
            setattr(characters[data[1]], "gender", 1)
            setattr(characters[data[2]], "gender", 1)
                  
    f.seek(0)
    next(f)
    for line in f:
        data = line.strip('\n').split('\t')
        if data[3] == "母子":
            if characters[data[1]].gender == 0:
                setattr(characters[data[2]], "gender", 1-characters[data[1]].gender)
                setattr(characters[data[2]], "mother", characters[data[1]].identify)
            elif characters[data[2]].gender == 0:
                setattr(characters[data[1]], "gender", 1-characters[data[2]].gender)
                setattr(characters[data[1]], "mother", characters[data[2]].identify)
            elif characters[data[1]].gender == 1:
                setattr(characters[data[2]], "gender", 1-characters[data[1]].gender)
                setattr(characters[data[1]], "mother", characters[data[2]].identify)
            elif characters[data[2]].gender == 1:
                setattr(characters[data[1]], "gender", 1-characters[data[2]].gender)
                setattr(characters[data[2]], "mother", characters[data[1]].identify)
        elif data[3] == "父女":
            if characters[data[1]].gender == 1:
                setattr(characters[data[2]], "gender", 1-characters[data[1]].gender)
                setattr(characters[data[2]], "father", characters[data[1]].identify)
            elif characters[data[2]].gender == 1:
                setattr(characters[data[1]], "gender", 1-characters[data[2]].gender)
                setattr(characters[data[1]], "father", characters[data[2]].identify)
            elif characters[data[1]].gender == 0:
                setattr(characters[data[2]], "gender", 1-characters[data[1]].gender)
                setattr(characters[data[1]], "father", characters[data[2]].identify)
            elif characters[data[2]].gender == 0:
                setattr(characters[data[1]], "gender", 1-characters[data[2]].gender)
                setattr(characters[data[2]], "father", characters[data[1]].identify)
        elif data[3] == "祖孫":
            characters[data[1]].grand.append(characters[data[2]].identify)
            characters[data[2]].grand.append(characters[data[1]].identify)
        elif data[3] == "兄弟姊妹":
            characters[data[1]].sibling.append(characters[data[2]].identify)
            characters[data[2]].sibling.append(characters[data[1]].identify)

#################################################
#預測 test.txt 的每組人物關係
#姑叔舅姨甥侄 及 師徒 關係較難預測故放棄
################################################# 
with open('./data/test.txt', encoding='utf8') as f:
    next(f)
    count = 0
    hit = 0
    for line in f:
        count += 1
        data = line.strip('\n').split('\t')
        entity1,entity2 = Character(-1),Character(-1) #預設為未知人物
        if (data[1] in characters):
            entity1 = characters[data[1]]
        if (data[2] in characters):
            entity2 = characters[data[2]]
            
        if entity1.slave or entity2.slave: #判斷是否為主僕關係
            if (data[1][0] != data[2][0]):
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"主僕"))
                if data[3] == "主僕":
                    hit += 1
                continue
        if (not blood_rel(entity1,entity2)) and \
           (entity1.gender != entity2.gender) and \
           (entity1.main_role or entity2.main_role) and \
           (data[1][0] != data[2][0]): #判斷是否為夫妻關係
               print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"夫妻"))
               if data[3] == "夫妻":
                   hit += 1
               continue
            
        if (entity1.freq >= 1 and (entity1.l_freq/entity1.freq >= 0.5)) or (entity2.freq >= 1 and (entity2.l_freq/entity2.freq >= 0.5)):
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"居處"))
            if data[3] == "居處": #判斷是否為居處關係
                hit += 1
            continue
        
        if entity1.gender == 1 and entity2.gender == 1: #依照性別及父母資訊判斷是否為各種親子關係
            print(data[3])
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"父子"))
            if data[3] == "父子":
                hit += 1
            continue
        elif entity1.gender == 0 and entity2.gender != 0:
            if entity1.identify == entity2.mother:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"母子"))
                if data[3] == "母子":
                    hit += 1
                continue
            else:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"父女"))
                if data[3] == "父女":
                    hit += 1
                    continue
        elif entity2.gender == 0 and entity1.gender != 0:
            if entity2.identify == entity1.mother:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"母子"))
                if data[3] == "母子":
                    hit += 1
                    continue
            else:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"父女"))
                if data[3] == "父女":
                    hit += 1
                    continue
        elif entity1.gender == 0 and entity2.gender == 0:
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"母女"))
            if data[3] == "母女":
                hit += 1
                continue

        if (entity1.identify in entity2.grand) or (entity2.identify in entity1.grand): #判斷是否為祖孫關係
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"祖孫"))
            if data[3] == "祖孫":
                hit += 1
            continue
        
        if (entity1.identify in entity2.sibling) or (entity2.identify in entity1.sibling): #判斷是否為直接的兄弟姊妹關係
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"兄弟姊妹"))
            if data[3] == "兄弟姊妹":
                hit += 1
            continue
        sibling_joint = set(characters[data[1]].sibling) & set(characters[data[2]].sibling)
        if len(sibling_joint) >= 1:  #判斷是否為間接的兄弟姊妹關係
            print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"兄弟姊妹"))
            if data[3] == "兄弟姊妹":
                hit += 1
            continue
            
        if data[1][0] == data[2][0]: #判斷是否同姓氏
            if (entity1.freq<=3 or entity2.freq<=3): #判斷是否為遠親或祖孫關係
                if (entity1.freq>=100 or entity2.freq>=100):
                    print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"遠親"))
                    if data[3] == "遠親":
                        hit += 1
                    continue
                else:
                    print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"祖孫"))
                    if data[3] == "祖孫":
                        hit += 1
                    continue
            else: #判斷是否為兄弟姊妹關係
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"兄弟姊妹"))
                if data[3] == "兄弟姊妹":
                    hit += 1
                continue
        else: #判斷是否為夫妻或祖孫關係
            if entity1.freq+entity2.freq >= 5:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"夫妻"))
                if data[3] == "夫妻":
                    hit += 1
                continue
            else:
                print("{} {} {} answer:{} predict:{}".format(data[0],data[1],data[2],data[3],"祖孫"))
                if data[3] == "祖孫":
                    hit += 1
                continue

    print("Accuracy: {:.2f}%".format((hit/count)*100)) #計算預測準確率
