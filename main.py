from os import listdir
from os.path import isfile, join, splitext
import numpy as np
from compute_functional import *
from voiced_unvoiced import *
from classification import *
from parse_files import *

en_files=[]
f0_files=[]

#replace with "babyears" to use the second dataset / "both" for the two datasets
# replace nb_classes (2 or 3) to change the number classes

dataset_name="babyears"
nb_classes=3

[Name_label, path] = parse_files(dataset_name, nb_classes)

en_elem=[]
f0_elem=[]
f0=[]
en=[]
labels=[]

for i in range(0,len(Name_label)):
    if(dataset_name=="kismet" or dataset_name=="babyears"):
        with open(path+Name_label[i][0], "r") as f:
            en_elem=[]
            f0_elem=[]

            if(i%2==0):
                #we only add 1 label for 2 files read because they both correspond to the same data
                labels.append(Name_label[i][1])

            if("en"in Name_label[i][0]):
                for x in f:
                    x = x.split(" ")
                    en_elem.append(int(x[1]))
                en.append(en_elem)

            elif("f0" in Name_label[i][0]):
                for x in f:
                    x = x.split(" ")
                    f0_elem.append(int(x[1]))
                f0.append(f0_elem)
    elif(dataset_name=="both"):
        for j in range(0, len(path)):
            path_a=path[j]

            if(isfile(path_a+Name_label[i][0])):
                with open(path_a+Name_label[i][0], "r") as f:
                    en_elem=[]
                    f0_elem=[]

                    if(i%2==0):
                        #we only add 1 label for 2 files read because they both correspond to the same data
                        labels.append(Name_label[i][1])

                    if("en"in Name_label[i][0]):
                        for x in f:
                            x = x.split(" ")
                            en_elem.append(int(x[1]))
                        en.append(en_elem)

                    elif("f0" in Name_label[i][0]):
                        for x in f:
                            x = x.split(" ")
                            f0_elem.append(int(x[1]))
                        f0.append(f0_elem)

f0 = np.array(f0 ,dtype=list)
en = np.array(en ,dtype=list)

# Compute the functionals for f0 and en
func_f0=compute_functionals(f0)
func_en=compute_functionals(en)

# We concatenate the functionals computed on the f0 data and the en data for each example
X_data=np.concatenate((func_f0,func_en), axis=1)

print("\nVoiced and unvoiced segments classification:")
classification("KNN", X_data, labels)
classification("SVM", X_data, labels)

#Split the data into voiced and unvoiced segments
[voiced_f0, voiced_en, unvoiced_f0, unvoiced_en ] = voiced_unvoiced(f0, en)

func_unv_f0=compute_functionals(unvoiced_f0)
func_unv_en=compute_functionals(unvoiced_en)

func_v_f0=compute_functionals(voiced_f0)
func_v_en=compute_functionals(voiced_en)


X_voiced=np.concatenate((func_v_f0, func_v_en), axis = 1)
X_unvoiced=np.concatenate((func_unv_f0, func_unv_en), axis = 1)

print("\nUnvoiced segment classification :")
classification("KNN", X_unvoiced, labels)
classification("SVM", X_unvoiced, labels)


"""
The unvoiced segments classification results are close to the "random guess", it can be explained by the fact
that we lose a lot of information, we only have the energy information to perform the classification.
With both dataset concatenated, we even get worse results
"""


print("\nVoiced segment classification :")
classification("KNN", X_voiced, labels)
classification("SVM", X_voiced, labels)

"""
The voiced segments classification results better than the unvoiced ones, we can explain it because it is in those
segments that we have the most information (f0 + energy)
"""

"""
When we compare the Multi corpus classification, we have significantly better results when we agregate the kismet 
and babyears dataset, this can be explained by the already good accuracy on the kismet dataset classification, which
increases logically the accuracy in both datasets classification.

When we compare 2 labels vs 3 labels classification, we have better accuracy for the 2 labels ones on the babyears dataset 
(0.6 average for 2 classes / 0.44 average for 3 classes). The difference on kismet isn't very significant, this can be explained
by the classes distribution in those datasets.
We only look at voiced segments because it is the most dense in information.
"""
