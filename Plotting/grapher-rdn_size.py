from pylab import * 


x = [3,7,11,15,19,23,27]
y = [0.07,0.24,0.35,0.66,1.00,1.30,2.06] # averaging


y = array(y)
y /= y[0]

figure(dpi=300)
# title('Train speed vs Number of features')
plot(x,y,'.-')
ylabel('execution time ratio')
xlabel('number of features')
show()