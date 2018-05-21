from pylab import * 

labels = ['skin','HEPMASS','HIGGS'] # do HIGGS wide
x = 1 + arange(len(labels))


y1 = [93.77,54.03] # averaging
y2 = [94.73,87.21] # pca


y1 = array(y1)
y2 = array(y2)


figure(dpi=300)
title('Performance across various datasets')
bar(x-0.1,y1,width=0.2,color='r', log=True, align='center')
bar(x+0.1,y2,width=0.2,color='b', log=True, align='center')
xticks(x, labels)
ylabel('seconds')
legend(('averaging','PCA'))

show()