'''
This scripts plots rmse graph for regular and meta model in a few shot testing scenario.
The results are taken from temp folder. It is developed for noise2true, noise2self, ssrl_noise2self model particularly.

'''

import matplotlib.pyplot as plt
import numpy as np

idx = 0

noise_lvl = []
rglr      = []
meta_mdl  = []

for num_of_updates in [1,5,10]:

    filename1 = "Test_Results/ours_ssrl_noise2self_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"      
    regular = np.load(filename1)

    filename2 = "Test_Results/ours_ssrl_noise2selfMTL_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"   
    meta_model = np.load(filename2)

    noise_lvl.append(regular['noise_lvl'])
    rglr.append(regular['rmse_b'][num_of_updates-1])
    meta_mdl.append(meta_model['rmse_b'][num_of_updates-1])

data = [noise_lvl,rglr,meta_mdl]
print(data)

X = np.arange(3)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
ax.legend(labels=['Noise Level','SSRL Noise2Self','SSRL Noise2SelfMTL'])

# ax.bar(X,data)
plt.title("Task "+str(idx))
figName = "temp_Figures/ours_ssrl_noise2self_task_"+str(idx)
plt.savefig(figName)
# plt.show()
