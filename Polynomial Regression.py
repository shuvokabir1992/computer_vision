import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Load the dataset
url = "automobileEDA.csv"
df = pd.read_csv(url)

# Features for the first linear regression model
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Linear Regression Model 1
lm = LinearRegression()
lm.fit(Z, df['price'])
print('Intercept of lm:', lm.intercept_)
print('Coefficients of lm:', lm.coef_)

# Features for the second linear regression model
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses', 'highway-mpg']], df['price'])
print('Intercept of lm2:', lm2.intercept_)
print('Coefficients of lm2:', lm2.coef_)

# Regression plot for highway-mpg vs price
plt.figure(figsize=(6, 5))
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.show()

# Correlation matrix for peak-rpm, highway-mpg, and price
print(df[["peak-rpm", "highway-mpg", "price"]].corr())

# Residual plot for highway-mpg vs price
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()

# Predictions from the first linear regression model
Y_hat = lm.predict(Z)

# Distribution plot for actual vs fitted values
ax1 = sns.distplot(df['price'], hist=False, color='r', label='Actual Value')
sns.distplot(Y_hat, hist=False, color='b', label='Fitted Value', ax=ax1)
plt.title('Actual vs Fitted Value for the Price')
plt.xlabel('Price (in Dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
plt.close()

# Function to plot polynomial regression
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ ' + Name)
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

# Polynomial regression model for highway-mpg vs price
x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
PlotPolly(p, x, y, 'highway-mpg')
