#Indian Covid EDA Project Version 3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None) #Makes all columns visible in the terminal

covid_data = pd.read_csv("H:\EDA Covid India\India COVID data by state.csv") #file path to dataset

#Look at the shape, info and summary statistics of the dataset
print(covid_data.info())
print(covid_data.shape)
print(covid_data.describe())

#Comparing some barcharts to see what states should be the primary focus
#This will look at states that are above the 50th percentile for the columns of interest
covid_data['Cases in 1000s'] = covid_data['Total Cases'] / 1000
high_tc = covid_data[covid_data['Cases in 1000s'] >= 474.9]
high_tc.plot(x = 'State/UTs', y = 'Cases in 1000s', kind = 'bar', title = 'Total Cases Above 50th Percentile in 1000s', xlabel = 'State/UTs', ylabel = 'Covid Cases')

covid_data['Active Cases in 1000s'] = covid_data['Active'] / 1000
high_active_cases = covid_data[covid_data['Active Cases in 1000s'] >= 0.416]
high_active_cases.plot(x = 'State/UTs', y = 'Active Cases in 1000s', kind = 'bar', title ='Active Covid Cases Above 50th Percentile in 1000s', xlabel = 'State/UTs', ylabel = 'Active Cases')

covid_data['Deaths in 1000s'] = covid_data['Deaths'] / 1000
high_deaths50 = covid_data[covid_data['Deaths'] >= 5.502]
high_deaths50.plot(x = 'State/UTs', y = 'Deaths in 1000s', kind = 'bar', title ='Covid Deaths Above 50th Percentile in 1000s', xlabel = 'State/UTs', ylabel = 'Deaths')

high_death_ratio50 = covid_data[covid_data['Death Ratio'] >= 1.3] 
high_death_ratio50.plot(x = 'State/UTs', y = 'Death Ratio', kind = 'bar', title = 'Covid Death Ratio Above 50th Percentile', xlabel = 'State/UTs', ylabel = 'Death Ratio' )

plt.show()
high_total_cases = covid_data.loc[covid_data['Total Cases'] == covid_data['Total Cases'].max()]
high_active_infections = covid_data.loc[covid_data['Active'] == covid_data['Active'].max()]
high_deaths = covid_data.loc[covid_data['Deaths'] == covid_data['Deaths'].max()]
high_death_ratio =  covid_data.loc[covid_data['Death Ratio'] == covid_data['Death Ratio'].max()]
print(high_total_cases,high_active_infections,high_deaths,high_death_ratio)

#Based off the initial analysis, the states of interest are: Maharashtra, Kerala, and Punjab
#The next task will be looking at vaccines vs deaths and GDP vs deaths and making a linear regression
#Intial scatter plots and dropped any null values from the GDP column
covid_data.dropna()
sns.pairplot(covid_data, x_vars = ['Vaccine Doses (in 1000s)','GDP Cr (2018-2019)'], y_vars = 'Deaths', size = 4, aspect = 1, kind = 'scatter')
plt.show()
sns.heatmap(covid_data.corr(), cmap = 'Blues', annot = True) #Shows how related the variables are to each other
plt.show()
#Linear regression for vaccines vs deaths
X = covid_data['Vaccine Doses (in 1000s)']
y = covid_data['Deaths']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train_sm = sm.add_constant(X_train) #adds a c value based on data
lr = sm.OLS(y_train, X_train_sm).fit() #ordinary least squares method for model
print(lr.summary())
#Constant = -4460.4629, m(vaccines) = 0.826634
#Deaths = -4460.4629 + 0.826634 * Vaccine Doses
plt.scatter(X_train, y_train)
plt.title('Linear Regression of Covid Training Data')
plt.xlabel('Vaccination Training Data')
plt.ylabel('Deaths Training Data')
plt.plot(X_train, -4460.4629 + 0.826634*X_train, 'r')
plt.show()

#Residual analysis
#Error = Actual y value - y predicted value
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)
#plot the error to see if it normally distrubuted
fig = plt.figure()
sns.distplot(res, bins = 15)
plt.title('Error Terms', fontsize = 15)
plt.xlabel('y_train - y_train_pred', fontsize = 15)
plt.show()
#make a scatterplot of the res to look for patterns
plt.scatter(X_train, res)
plt.title('Scatterplot of Error Terms')
plt.show()

#Evaluating the model
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm) #predict y values based on X_test_sm
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_test_pred) #compares the actual and predicted y values
print(r_squared)

#Visualizing the test set 
plt.scatter(X_test, y_test)
plt.title('Covid Vaccination Regression on Test Data')
plt.plot(X_test, y_test_pred, 'r')
plt.show()