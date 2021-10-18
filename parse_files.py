from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random

def parse_files(dataset_name, classes_number):
    
    path1= "./Kismet_data/"
    path2= "./BabyEars_WAV/"
    valid_files=[]
    Name_label=[]

    if(dataset_name=="kismet"):
        path=path1
        files = [f for f in listdir(path) if isfile(join(path, f))]
    elif(dataset_name=="babyears"):
        path=path2
        files = [f for f in listdir(path) if isfile(join(path, f))]
    elif(dataset_name=="both"):
        path =[path1, path2]
        files1 = [f for f in listdir(path1) if isfile(join(path1, f))]
        files2 = [f for f in listdir(path2) if isfile(join(path2, f))]
        files = np.concatenate((files1,files2), axis = 0)

    else:
        print("Dataset name not recognized")
        return None, None
    
    print("Number of files :", len(files))

    for i in range(0, len(files)):
        name, ext = splitext(files[i])
        if(ext=='.en') or (ext=='.f0') : 
            valid_files.append(files[i])

    print("Number of valid files :", len(valid_files))

    if(dataset_name=="kismet"):
        classes_choice= random.sample(["at","pw","ap"], classes_number)
        print("Classes chosen :", classes_choice)
    elif(dataset_name=="babyears"):
        classes_choice= random.sample(["at","pr","ap"], classes_number)
        print("Classes chosen :", classes_choice)    
    #We consider the labels "pw" and "pr" the same
    elif(dataset_name=="both"):
        classes_choice= random.sample(["at","p","ap"], classes_number)
        print("Classes chosen :", classes_choice)

    for i in range(0, len(valid_files)):
        if(dataset_name=="kismet"):
            if(("at" in valid_files[i]) and ("at" in classes_choice)):
                Name_label.append([valid_files[i], "at"])
            elif(("pw" in valid_files[i]) and ("pw" in classes_choice)):
                Name_label.append([valid_files[i], "pw"])
            elif(("ap" in valid_files[i])and ("ap" in classes_choice)):
                Name_label.append([valid_files[i], "ap"])

        elif(dataset_name=="babyears"):
            if(("at" in valid_files[i]) and ("at" in classes_choice)):
                Name_label.append([valid_files[i], "at"])
            elif(("pr" in valid_files[i]) and ("pr" in classes_choice)):
                Name_label.append([valid_files[i], "pw"])
            elif(("ap" in valid_files[i])and ("ap" in classes_choice)):
                Name_label.append([valid_files[i], "ap"])

        elif(dataset_name=="both"):
            if(("at" in valid_files[i]) and ("at" in classes_choice)):
                Name_label.append([valid_files[i], "at"])
            elif((("pr" in valid_files[i]) or ("pw" in valid_files[i])) and ("p" in classes_choice)):
                Name_label.append([valid_files[i], "p"])
            elif(("ap" in valid_files[i])and ("ap" in classes_choice)):
                Name_label.append([valid_files[i], "ap"])
        
    return Name_label, path