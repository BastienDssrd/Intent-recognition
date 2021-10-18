import numpy as np

def voiced_unvoiced(data_f0, data_en):
    v_f0 = []
    v_en = []
    unv_f0 = []
    unv_en = []
    
    voiced_f0, voiced_en, unvoiced_f0, unvoiced_en=[], [], [], []

    
    for i in range(0, len(data_f0)):
        v_f0 = []
        v_en = []
        unv_f0 = []
        unv_en = []

        f0=np.array(data_f0[i])
        en=np.array(data_en[i])

        val1=np.where(f0 == 0)[0]
        val2=np.where(f0 != 0)[0]
        #print(val1, val2)
        unv_f0.append(f0[val1])
        unv_en.append(en[val1])
        
        v_f0.append(f0[val2])
        v_en.append(en[val2])
       
        voiced_f0.append(v_f0)
        voiced_en.append(v_en)
        unvoiced_f0.append(unv_f0)
        unvoiced_en.append(unv_en)
        #input()
        
    return [voiced_f0, voiced_en, unvoiced_f0, unvoiced_en ]