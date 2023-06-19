# Code for continuous Plot

import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata1 = []
ydata2 = [] 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line1, = axes.plot(xdata, ydata1, 'b-')
line2, = axes.plot(xdata, ydata2, 'r-')
 
for i in range(100):
    xdata.append(i)
    ydata1.append(ysample[i])
    ydata2.append(ysample[99-i])
    line1.set_xdata(xdata)
    line1.set_ydata(ydata1)
    line2.set_xdata(xdata)
    line2.set_ydata(ydata2)
    
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
 
# add this if you don't want the window to disappear at the end
plt.show()