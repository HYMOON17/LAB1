'''
This scripts plots rmse graph for regular and meta model in a few shot testing scenario.
The results are taken from temp folder. It is developed for noise2true model particularly.

'''

import matplotlib.pyplot as plt
import numpy as np

idx = 19

noise_lvl = []
rglr      = []
meta_mdl  = []

for num_of_updates in [1,5,10]:

    filename1 = "temp/noise2true_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"      
    regular = np.load(filename1)

    filename2 = "temp/noise2trueMTL_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"   
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
ax.legend(labels=['Noise Level','Noise2True','Noise2TrueMTL'])
plt.title("Task "+str(idx))
figName = "temp_Figures/noise2true_task_"+str(idx)
# plt.savefig(figName)
plt.show()

'''
# data = [[1,2,3,4],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]
# X = np.arange(4)
# fig = plt.figure()
# ax = fig.add_axes([0.1,0.1,0.9,0.9])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
# ax.legend(labels=['tush','Gush','Mush'])
# plt.show()
'''




# print(data)
# # fig = plt.figure()
# # ax = fig.add_axes([0.1,0.1,0.8,0.8])
# # langs = ['Noise Level', 'denoised regular', 'denoise meta']
# # students = [70,60,50]
# # ax.bar(langs,students)
# # # plt.xlabel('Blah')
# # # plt.plot(students)
# # plt.show()
