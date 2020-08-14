import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

table = pd.read_csv(r'C:\Users\Harmohan\Desktop\Record2.csv')
print(table)
X = table.Day.values
Y = table.Confirmed.values

print(X)
print(Y)

plt.scatter(X, Y)
plt.show()

X = X.reshape(len(X), 1)
Y = Y.reshape(len(Y), 1)

print(X)
print(Y)

model = LinearRegression()

model.fit(X, Y)

b0 = model.intercept_
b1 = model.coef_

Y1 = model.predict(X)

print(X[0], Y[0], Y1[0])

Confirmed = r2_score(Y, Y1)

predictionForDayNumber = 45
predicted_y = b0[0] + b1[0][0]*predictionForDayNumber
Y_pred = Y1
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
print("Predicted Confirmed cases in Punjab would be:", predicted_y)