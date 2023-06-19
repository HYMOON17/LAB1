import matplotlib.pyplot as plt
import numpy as np

filename ="Test_Results/lr_b_0.1/"+"/Meta_Batch_size_"+str(15)+"_lr_a_"+str(0.01) \
+"_updates_"+str(50)+".npz"     
data = np.load(filename)
losses_b = data['losses_b']
rmses_b  = data['rmses_b']
param_diff  = data['param_error']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle("lr_b_"+str(0.1))
ax1.plot(losses_b)
# ax1.xlabel("Epochs")
ax1.set_title("Meta Update Loss Curve")
ax2.plot(rmses_b)
# ax2.xlabel("Epochs")
ax2.set_title("Meta Update RMSE Curve")
ax3.plot(param_diff)
ax3.set_title("Param rel error Curve")
plt.show()