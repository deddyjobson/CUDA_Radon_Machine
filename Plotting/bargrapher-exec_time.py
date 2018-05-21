from pylab import * 

labels = ['skin','HEPMASS','HIGGS','SUSY']	
x = 1 + arange(len(labels))


y1 = [0.1621,0.0855,0.5130,0.3264] # radon
y2 = [0.1178,0.0752,0.3875,0.2493] # pca


y1 = array(y1)
y2 = array(y2)

y1 /= y2

figure(dpi=300)
# title('Train time across various datasets')
bar(x,y1,width=0.3,color='g', log=False, align='center')
xticks(x, labels)
ylabel('speed up')
# legend(('radon','PCA'))

show()