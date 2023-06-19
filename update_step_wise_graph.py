'''
This scripts plots rmse graph for regular and meta model in a few shot testing scenario.
The results are taken from temp folder. It is developed for noise2true, noise2self, ssrl_noise2self model particularly.

'''

import matplotlib.pyplot as plt
import numpy as np

idx = 1

rmse_n2t          = []
rmse_n2t_mtl      = []
rmse_n2s          = []
rmse_n2s_mtl      = []
rmse_n2s_ssrl     = []
rmse_n2s_ssrl_mtl = []

for num_of_updates in [1,5,10]:

    filename1 = "temp/noise2true_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"      
    n2t = np.load(filename1)

    filename2 = "temp/revisited_noise2trueMTL_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"   
    n2t_mtl = np.load(filename2)

    filename3 = "Test_Results/noise2self_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"      
    n2s = np.load(filename3)

    filename4 = "Test_Results/noise2selfMTL_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"   
    n2s_mtl = np.load(filename4)

    filename5 = "Test_Results/ours_ssrl_noise2self_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"      
    n2s_ssrl = np.load(filename5)

    filename6 = "Test_Results/ours_ssrl_noise2selfMTL_task_"+str(idx)+"_updates_"+str(num_of_updates)+".npz"   
    n2s_ssrl_mtl = np.load(filename6)

    # noise_lvl.append(regular['noise_lvl'])
    # rglr.append(regular['rmse_b'][num_of_updates-1])
    # meta_mdl.append(meta_model['rmse_b'][num_of_updates-1])
    rmse_n2t.append(n2t['rmse_b'])
    rmse_n2t_mtl.append(n2t_mtl['rmse_b'])
    rmse_n2s.append(n2s['rmse_b'])
    rmse_n2s_mtl.append(n2s_mtl['rmse_b'])
    rmse_n2s_ssrl.append(n2s_ssrl['rmse_b'])
    rmse_n2s_ssrl_mtl.append(n2s_ssrl_mtl['rmse_b'])   

# data = [noise_lvl,rglr,meta_mdl]
# print(data)

# X = np.arange(3)
# fig = plt.figure()
# ax = fig.add_axes([0.1,0.1,0.8,0.8])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
# ax.legend(labels=['Noise Level','SSRL Noise2Self','SSRL Noise2SelfMTL'])

# # ax.bar(X,data)
# plt.title("Task "+str(idx))
# figName = "temp_Figures/ours_ssrl_noise2self_task_"+str(idx)
# plt.savefig(figName)
# # plt.show()

labels=['Noise2True','Noise2TrueMTL','Noise2Self','Noise2SelfMTL','SSRL Noise2Self','SSRL Noise2SelfMTL']
for i in [1,2]:
    plt.figure(i)
    plt.plot(rmse_n2t[i])
    plt.plot(rmse_n2t_mtl[i])
    plt.plot(rmse_n2s[i],'*')
    plt.plot(rmse_n2s_mtl[i],'*')
    plt.plot(rmse_n2s_ssrl[i],'--')
    plt.plot(rmse_n2s_ssrl_mtl[i],'--')
    plt.legend(labels)
    plt.title('Task '+str(idx))
    plt.xlabel('Num of updates')
    plt.ylabel('RMSE (HU)')
    # figName = "temp_Figures/task_"+str(idx)+"_updates_"+str(i*5)
    # plt.savefig(figName)
plt.show()
