from pylab import * 

labels = ['skin','HEPMASS','HIGGS','SUSY'] # do HIGGS wide
x = 1 + arange(len(labels))


y1 = [93.77,77.09,51.59,50.85] # averaging
y2 = [94.73,88.28,63.19,70.83] # pca
y3 = [93.75,85.32,54.72,63.54] # just radon


y1 = array(y1)
y2 = array(y2)


figure(dpi=300)
# title('Performance across various datasets')
bar(x-0.2,y1,width=0.2,color='r', log=False, align='center')
bar(x,y2,width=0.2,color='b', log=False, align='center')
bar(x+0.2,y3,width=0.2,color='g', log=False, align='center')
xticks(x, labels)
ylabel('f-score(%)')
legend(('averaging','radon-pca','radon'))

show()