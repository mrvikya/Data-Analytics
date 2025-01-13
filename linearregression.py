import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

ID=random.sample(range(0,500),500)
TV=random.sample(range(200,800),500)
Radio=random.sample(range(100,900),500)
Newspaper=random.sample(range(100,600),500)
Sales=random.sample(range(100,900),500)
data=list(zip(ID,TV,Radio,Newspaper,Sales))
df=pd.DataFrame(data,columns=['ID','TV','Radio','Newspaper','Sales'])
print(df)

x=np.array(df[['TV']])
y=np.array(df[['Sales']])

print(x.shape)
print(y.shape)

plt.scatter(x,y,color="red")
plt.title("tv vs sales")
plt.xlabel('tv')
plt.ylabel('sales')
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=15)

regressor=LinearRegression()
regressor.fit(x_train,y_train)

plt.scatter(x_test,y_test,color="green")
plt.plot(x_train,regressor.predict(x_train),color="red",linewidth=3)
plt.title('Regressiopn(test set)')
plt.xlabel('tv')
plt.ylabel('sales')
plt.show()

plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red",linewidth=3)
plt.title('Regressiopn(train set)')
plt.xlabel('tv')
plt.ylabel('sales')
plt.show()

y_pred=regressor.predict(x_test)
print('R2 score:%.2f'%r2_score(y_test,y_pred))

print('Mean Error :',mean_squared_error(y_test,y_pred))
