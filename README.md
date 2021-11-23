# EDA-Indian-Covid-Data: Project Overview
* Performed an EDA on some Covid data from India to generate insights and look at some trends. 

## Code and Resources Used
* Python version: 3.8
* Packages: pandas, numpy, matplotlib, seaborn, statsmodels

## Background
* SARS-CoV-2 is virus that is a member of the coronavirus family, and in early 2019 it was recognized as a pandemic. This pandemic has affected many aspects of society, and has exacerbated the need for health care workers for hospitals around the world. I approached this problem from the perspective of a data analyst working for a health staffing company in India, where the main business request was finding what states should be the main focus of company efforts. 

## Questions Explored
* What states should be the main focus of company resources?
* Are there any other trends that should be observed for future business decisions?

## Methods
### data collection
   * An initial dataset was provided as a csv file from Kaggle.com and featured columns such as total cases, deaths, and discharge rate. The columns containing data on vaccination, population and GDP per state were manually scraped from external sources and added to the csv file.
### data visualization
  * Data from the csv file were visualized with python 3.8 using the following libraries: pandas, matplotlib.pyplot, seaborn, and sklearn. These libraries were selected for the analysis because they are common analyst tools, and provide a means for robust graphs and charts.
### analysis 
   * The first question was answered by creating four bar charts where the total cases, active cases, deaths, and death ratio were compared among the states, given that their respective value was above the 50th percentile for that value. States were selected based on their presence throughout the graphs in addition to standing out as a high value within each graph. 
The second question was answered by first creating two scatter plots: vaccine doses versus deaths; and GPD versus deaths. This was used to quickly look for a correlation between variables in addition to using a heatmap that displayed the correlation coefficient between all variables present in the dataset. The next step was to develop a linear regression to further explore and visualize the relationship of vaccine doses and deaths. The hypothesis was that as the number of vaccine doses increased, the number of deaths would decrease, showing an inverse relationship between the two variables. Given the small size of the dataset, a linear regression was created using 70% of the data as the training set and ordinary least squares method (OLS) to formulate an equation that could be used to predict deaths given vaccine doses.  After the regression was created, a residual analysis was performed to check if the error was normally distributed, and that the regression could be used for predictions.

## Results
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Active%20Covid%20Cases%20Above%2050th%20Percentile.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Covid%20Death%20Ratio%20Above%2050th%20Percentile.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Covid%20Deaths%20Above%2050th%20Percentile.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Total%20Cases%20Above%2050th%20Percentile.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Correlation%20coefficient%20heatmap.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Training%20Linear%20Regression%20Plotted.png)
![alt text](https://github.com/lazyrandy/EDA-Indian-Covid-Data/blob/main/graphs/Error%20Terms%20Visualized.png)
## Conclusion
Based on the bar charts created, the three states of interest are Maharashtra, Kerala, Punjab. The direct correlation between vaccine doses and deaths is not very high, and the regression should not be used as an indicator of deaths within a state. This suggests there are several variables that affect the number of deaths within a single Indian state, and is better represented by a multivariable regression. Additionally, the error terms analysis shows that the error for this study is not normally distribtued further support a multivariable model for death prediction.

## Discussion
In future studies, additional environmental factors should be considered such as clean water availability, population density, and average age of the population. This would allow for a more robust model that could better capture the relationship between independent and dependent variables. 

## Sources
* Original Covid Dataset: https://www.kaggle.com/anandhuh/latest-covid19-india-statewise-data
* Vaccine Data: https://www.statista.com/statistics/1222266/india-cumulative-coverage-of-covid-19-vaccine-across-india/
* State population data: https://statisticstimes.com/demographics/india/indian-states-population.php
* State GDP (2018- 2019): https://statisticstimes.com/economy/india/indian-states-gdp.php
* Linear Regression: https://towardsdatascience.com/simple-linear-regression-model-using-python-machine-learning-eab7924d18b4
* Ordinary Least Squares: https://setosa.io/ev/ordinary-least-squares-regression/
* Normal Distribution of Error: https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd445.htm
