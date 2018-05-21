from pylab import * 

labels = ['skin','HEPMASS','HIGGS'] # do HIGGS wide
x = 1 + arange(len(labels))

# y1 = [0.33,6.83,21.69,122.21] # base
# y2 = [0.07,0.32,0.28,0.23] # radon
# y3 = [0.07,0.18,0.20,0.23] # PCA



y1 = [0.33,6.83,21.69,122.21,19.04,74.72] # base
y2 = [0.07,0.32,0.28 ,0.23  ,0.48 ,0.52] # radon
y3 = [0.07,0.18,0.20 ,0.23  ,0.31 ,0.49] # PCA

# print(y1)
# print(y2)
# print(y3)
# exit()
y1 = array(y1)
y2 = array(y2)
y3 = array(y3)

# y1 /= y3 
# y2 /= y3 
# y3 /= y3

figure(dpi=300)
title('execution time across various datasets')
bar(x-0.2,y1,width=0.2,color='r', log=True, align='center')
bar(x,y2,width=0.2,color='b', log=True, align='center')
bar(x+0.2,y3,width=0.2,color='g', log=True, align='center')
xticks(x, labels)
ylabel('seconds')
legend(('base learner','radon machine','PCA'))

show()