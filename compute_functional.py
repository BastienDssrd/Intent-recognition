import numpy as np

def compute_functionals(data):
    #Functionnals computed : -mean - max - variance - median - 1st quantile - 3rd quantile -mean absolute local derivate
    func2=[]
    for i in data:
        x=0
        func=[]
        i=np.array(i)
        func.append(np.mean(i))
        func.append(np.max(i))
        func.append(np.std(i))
        func.append(np.median(i))
        func.append(np.quantile(i, 0.25))
        func.append(np.quantile(i, 0.75))
        # x=[]
        # for j in range(0, len(i)-1):
        #     x.append(i[j+1]-i[j])
        # func.append(np.mean(x))
        # #We add each functionnal array to the output array
        func2.append(func)

    return func2