import nummpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import PolynomialRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('data.csv')
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

lin = LinearRegression()
lin.fit(x, y)

poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

poly.fit(x_poly, y)
lin2 = LinearRegression()
lin2.fit(x_poly, y)


#linear regression graph

plt.scatter(x, y, color='red')

plt.plot(x, lin.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('pressure')
plt.ylabel('temperature')
plt.show()

#polynomial regression graph
plt.scatter(x, lin2.predict(poly.fit_transform(x)), color='red')
plt.title('Polynomial Regression')
plt.xlabel('pressure')
plt.ylabel('temperature')
plt.show()
