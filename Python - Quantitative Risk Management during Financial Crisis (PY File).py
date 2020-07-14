#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# In[364]:


import statsmodels.api as sm
from pypfopt import CLA
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from scipy.stats import norm
from scipy.stats import norm,anderson
from scipy.stats import skewnorm, skewtest


# ### Uploading data containing daily stock prices of 4 major banks(Morgan Stanley, Citi, JPMorgan Chase, Goldman Sachs) during period before, during and after Financial Crisis of 2008

# In[281]:


df1 = pd.read_csv("Financial_Stocks.csv")


# In[282]:


df = df1.copy()


# ### Data Preprocessing

# In[283]:


df.head(10)


# In[284]:


df.isna().sum()


# In[285]:


df['Date'] = pd.to_datetime(df['Date'])


# In[286]:


df = df.set_index('Date')


# In[287]:


df4 = df.copy()


# In[288]:


df.head(3)


# In[289]:


df.describe()


# In[290]:


# from 2007 - 2010
df.plot(legend = 'MS',figsize=(13,8))
plt.ylabel("Close Price")
plt.title('Close Price of 4 major price during financial crisis')


# ### Quantifying Return (Taking log returns in place of close prices due to high autocorrelation in prices)

# In[291]:


df['Lag_MS'] = df['Close_MS'].shift(1)
df['Return_MS'] = (np.log(df['Close_MS']/df['Lag_MS']))*100
df['Lag_Citi'] = df['Close_Citi'].shift(1)
df['Return_Citi'] = (np.log(df['Close_Citi']/df['Lag_Citi']))*100
df['Lag_JPM'] = df['JPM_Close'].shift(1)
df['Return_JPM'] = (np.log(df['JPM_Close']/df['Lag_JPM']))*100
df['Lag_GS'] = df['GS_Close'].shift(1)
df['Return_GS'] = (np.log(df['GS_Close']/df['Lag_GS']))*100


# In[292]:


df.head(10)


# In[293]:


df1 = df.drop(['Lag_Citi','Lag_MS','Lag_JPM','Lag_GS'],axis=1)


# In[294]:


df1.head(10)


# In[295]:


df2 = df1.drop(['Close_MS','Close_Citi','GS_Close','JPM_Close'],axis=1)


# In[296]:


df2.head(3)


# In[297]:


df3 = df2.copy()


# In[353]:


df3.head(10)


# In[299]:


df3.describe()


# ### Considering equal weightage to each asset in a portfolio

# In[354]:


returns = df3.dropna(axis=0)


# In[355]:


w = (0.25,0.25,0.25,0.25)


# In[356]:


# Multilying weight vector with returns vector to calculate portfolio returns
portfolio_returns = returns.dot(w)


# In[357]:


portfolio_returns.head(10)


# In[368]:


losses = -1*portfolio_returns


# In[369]:


losses.head(10)


# In[304]:


# -ve returns are losses and +ve returns a are profit


# In[305]:


type(portfolio_returns)


# ### Pandas Series Object to Pandas Dataframe

# In[306]:


portfolio_returns_new = pd.Series(portfolio_returns)
print (portfolio_returns_new)


# In[307]:


df = portfolio_returns_new.to_frame()


# In[308]:


df = df.rename(columns={0: "returns"})


# In[309]:


df.head(10)


# In[310]:


portfolio_returns.plot(color='red').set_ylabel("Daily Return, %")
plt.show()


# Above graph shows very high volatility from July 2008 to July 2009 

# In[311]:


# The asset prices plot shows how the global financial crisis created a loss in confidence in investment banks from September 2008
# There was an event during September that precipitated this decline. The 'spikiness' of portfolio returns indicates how 
# uncertain and volatile asset returns became.


# In[312]:


portfolio_returns_percent = portfolio_returns*100


# ## VaR using Variance Covariance (Parametric Estimation)

# In[313]:


covar = df3.cov()


# #### Correlation in Percentage

# In[314]:


print(covar)


# In[315]:


# Annualize the covariance using 252 trading days per year
covar_ann = covar*252


# In[316]:


print(covar_ann)


# In[317]:


portfolio_variance = np.transpose(w)@covar_ann@w


# In[318]:


portfolio_volatility = np.sqrt(portfolio_variance)


# #### Portfolio Volatliltiy

# In[319]:


print(portfolio_volatility)


# In[320]:


# Portfolio has 61% volatilily


# In[321]:


#The volatility of a portfolio of stocks is a measure of how wildly the total value of all the stocks in that portfolio
# appreciates or declines.


# # Rolling Volatility

# In[322]:


windowed = df.rolling(30)


# In[323]:


volatility = windowed.std()*np.sqrt(252)


# In[257]:


volatility.plot(color = 'green').set_ylabel("Annualized Volatility, 30-day Window")


# In[258]:


df.head(10)


# # Risk Factors

# ### Variables or events that drive portfolio return and volatility

# Two types of risk factors are:
#     1. Systematic Risk
#     2. Idisyncratic Risk
#     
# #### Systematic Risk
# Systematic risk is inherent to the market as a whole, reflecting the impact of economic, geo-political and financial factors.
# This type of risk is distinguished from unsystematic risk, which impacts a specific industry or security.
# Investors can somewhat mitigate the impact of systematic risk by building a diversified portfolio.
# Ex: interest rate changes, inflation, recessions, and wars, among other major changes.
#     
# #### Idiosyncratic Risk
# Idiosyncratic risk refers to the inherent factors that can negatively impact individual securities or a very specific group of assets.
# The opposite of Idiosyncratic risk is a systematic risk, which refers to broader trends that impact the overall financial system or a very broad market.
# Idiosyncratic risk can generally be mitigated in an investment portfolio through the use of diversification
# Idiosyncratic risk is a type of investment risk that is endemic to an individual asset (like a particular company's stock),
# or a group of assets (like a particular sector's stocks), or in some cases, a very specific asset class (like collateralized mortgage obligations). 
# 
# #### Idiosyncratic Risk vs. Systematic Risk
# While idiosyncratic risk is, by definition, irregular and unpredictable, studying a company or industry can help an 
# investor to identify and anticipate—in a general way—its idiosyncratic risks. Idiosyncratic risk is also highly individual, 
# even unique in some cases. It can, therefore, be substantially mitigated or eliminated from a portfolio by using adequate 
# diversification. Proper asset allocation, along with hedging strategies, can minimize its negative impact on an investment 
# portfolio by diversification or hedging.
# In contrast, systematic risk cannot be mitigated just by adding more assets to an investment portfolio. This market risk 
# cannot be eliminated by adding stocks of various sectors to one's holdings. These broader types of risk reflect the 
# macroeconomic factors that affect not just a single asset but other assets like it and greater markets and economies as well.

# ### Factor Models

# Factor models assess on which risk factors asset returns or volatility are mostly dependent.
# We can model theses factors using :
#     1. Ordinary Least Square - Regression Model - 
#        dependent variable - Asset returns/volatility
#        independent variable - risk factors
#     2. Fama French Model - combination of market risk and idiosyncratic risk (firm size and value)

# Considering MBS(Mortgage Backed Security) 90 days mortgage Delinquency as a risk factor which caused the bankcruptcy of
# Lehman Brothers. Risk factor delinquency rate was highly correlated with the returns.

# Risk factor models often rely upon data that is of different frequencies. A typical example is when using quarterly
# macroeconomic data, such as prices, unemployment rates.
# here also delinquency rate is taken for 90 days (1 Q) so sampling returns for quarter

# In[259]:


returns_avg = df.resample('Q').mean()


# In[115]:


returns_avg.tail()


# In[116]:


# Now convert daily returns to weekly minimum returns
returns_min = df.resample('Q').min()
returns_min.head()


# In[117]:


delin = pd.read_csv("Delinq_rate.csv")


# In[118]:


returns_avg.describe()


# In[119]:


delin.describe()


# In[120]:


plt.scatter(returns_avg,delin['Delinq_Rate'])
plt.xlabel("Quarterly Average Return")
plt.ylabel("Delinquency rate, decimal %")


# In[121]:


plt.scatter(returns_min,delin['Delinq_Rate'])
plt.xlabel("Quarterly Min Return")
plt.ylabel("Delinquency rate, decimal %")


# In[122]:


# # Initial assessment indicates that there is little correlation between average returns and mortgage delinquencies, 
# but a stronger negative correlation exists between minimum returns and delinquency. In the following exercises we'll
# quantify this using least-squares regression.


# In[123]:


delin.head()


# In[124]:


delin['Date'] = pd.to_datetime(delin['Date'])
delin = delin.set_index('Date')


# In[125]:


delin.head()


# In[127]:


# Crisis Factor Model (OLS) LEFT to study


# In[64]:


regression = sm.OLS(returns_avg,delin['Delinq_Rate']).fit()


# In[65]:


print(regression.summary())


# In[66]:


regression_qmin = sm.OLS(returns_min,delin['Delinq_Rate']).fit()


# In[67]:


print(regression_qmin.summary())


# In[68]:


# Now convert daily returns to weekly minimum returns
returns_vol = df.resample('Q').std()
returns_vol.head()


# In[69]:


regression_vol = sm.OLS(returns_vol,delin['Delinq_Rate']).fit()


# In[70]:


print(regression_vol.summary())


# ### As seen from the regressions, mortgage delinquencies are acting as a systematic risk factor for both minimum quarterly returns and average volatility of returns, but not for average quarterly returns. The R-squared goodness of fit isn't high in any case, but a model with more factors would likely generate greater explanatory power.

# In[71]:


##R-Squared is a statistical measure of fit that indicates how much variation of a dependent variable is explained by the 
##independent variable(s) in a regression model.


# # Modern Portfolio Theory

# What maximum return an investor can expect as per given risk apetite calculated from the portfolio volatility 
# 
# #### Eficient Portfolio
# portfolio with weights generating highest expected return for given level of risk
# 
# #### Efficient Frontier
# Locus of (risk,return) pairs created by efficient portfolio

# In[64]:


# pip install pyportfolioopt


# In[73]:



# Compute the annualized average historical return
mean_returns = mean_historical_return(df4, frequency = 252)

# Plot the annualized average historical return
plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.show()


# In[74]:


mean_returns.head()


# In[75]:


# The average historical return is usually available as a proxy for expected returns, but is not always accurate--a more 
# thorough estimate of expected returns requires an assumption about the return distribution, which we'll discuss in the context
# of Loss Distributions later in the course.


# In[76]:


df4.head(10)


# In[77]:


# Import the CovarianceShrinkage object, it reduces/shrinks the errors/residuals while calculating the covariance matrix

# Create the CovarianceShrinkage instance variable
cs = CovarianceShrinkage(df4)


# In[78]:


# Difference in calculating covariance matrix through covariance shrinkage and through sample cov() method
# Compute the sample covariance matrix of returns
sample_cov = df4.pct_change().cov() * 252

# Compute the efficient covariance matrix of returns
e_cov = cs.ledoit_wolf()

# Display both the sample covariance_matrix and the efficient e_cov estimate
print("Sample Covariance Matrix\n", sample_cov, "\n")
print("Efficient Covariance Matrix\n", e_cov, "\n")


# In[79]:


# Although the differences between the sample covariance and the efficient covariance (found by shrinking errors) 
# may seem small, they have a huge impact on estimation of the optimal portfolio weights and the generation of the efficient 
# frontier. Practitioners generally use some form of efficient covariance for Modern Portfolio Theory.


# In[80]:


df4.head()


# In[81]:


# Create a dictionary of time periods (or 'epochs')
epochs = { 'during' : {'start': '1-1-2007', 'end': '31-12-2008'},
           'after'  : {'start': '1-1-2009', 'end': '31-12-2010'}
         }

# Compute the efficient covariance for each epoch
e_cov = {}
for x in epochs.keys():
    sub_price = df4.loc[epochs[x]['start']:epochs[x]['end']]
    e_cov[x] = CovarianceShrinkage(sub_price).ledoit_wolf()

# Display the efficient covariance matrices for all epochs
print("Efficient Covariance Matrices\n", e_cov)


# In[128]:


df3.head()


# In[83]:


# Great! The breakdown of the 2007 - 2010 period into sub-periods shows how the portfolio's risk increased during the crisis
# , and this changed the risk-return trade-off after the crisis. For future reference, also note that although we used a loop
# in this exercise, a dictionary comprehension could also have been used to create the efficient covariance matrix.


# In[85]:


# Create a dictionary of time periods (or 'epochs')
epochs = { 'during' : {'start': '1-1-2007', 'end': '31-12-2008'}}

# Compute the efficient covariance for each epoch
e_cov_during = {}
for x in epochs.keys():
    sub_price = df4.loc[epochs[x]['start']:epochs[x]['end']]
    e_cov_during[x] = CovarianceShrinkage(sub_price).ledoit_wolf()

# Display the efficient covariance matrices for all epochs
print("Efficient Covariance Matrices\n", e_cov_during)


# ## Efficient Frontier Using CLA Algorithm

# In[260]:


df4.head()


# In[261]:


df3.head()


# In[262]:


df6=df3.loc['2007-03-01':'2008-12-31']


# In[263]:


df7=df4.loc['2007-03-01':'2008-12-31']


# In[264]:


e_cov_during = np.array(CovarianceShrinkage(df7).ledoit_wolf())


# In[265]:


type(e_cov_during)


# In[266]:


returns_during = np.array(df6.mean())


# In[267]:


efficient_portfolio_during = CLA(returns_during, e_cov_during)


# In[268]:


print(efficient_portfolio_during.min_volatility())


# In[269]:


# Compute the efficient frontier
(ret, vol, weights) = efficient_portfolio_during.efficient_frontier()


# In[278]:


plt.figure(figsize=(20,12))
plt.xlabel('Standard Deviation/Volatiltiy/Risk')
plt.ylabel('Return for period 2007-2008')
plt.title('Efficient Frontier during crisis',size=20)
plt.plot(vol,ret,c='r')


# In[271]:


df9=df4.loc['2009-01-01':'2010-12-31']  # for covariance matrix (prices)
df10=df3.loc['2009-01-01':'2010-12-31'] # returns


# In[272]:


returns_after = np.array(df10.mean())
print(returns_after)


# In[273]:


e_cov_after = np.array(CovarianceShrinkage(df9).ledoit_wolf())
efficient_portfolio_after = CLA(returns_after, e_cov_after)
(ret, vol, weights) = efficient_portfolio_after.efficient_frontier()
# Add the frontier to the plot showing the 'before' and 'after' frontiers


# In[279]:


plt.figure(figsize=(20,12))
plt.xlabel('Standard Deviation/Volatiltiy/Risk')
plt.ylabel('Return for period 2009-2010')
plt.title('Efficient Frontier after crisis',size=20)
plt.plot(vol,ret,c='g')


# In[118]:


## Risk reduced after crisis


# # Portfolio Optimization

# In[324]:


df_returns = df1[['Return_MS','Return_Citi','Return_JPM','Return_GS']]


# In[325]:


df_returns.head(10)


# In[326]:


df_returns.hist(bins=100,figsize=(12,8))
plt.tight_layout()


# In[327]:


df_returns.mean()


# In[328]:


df_returns.cov()*252


# ## Portfolio Optimization using Monte Carlo Simulation (Random Weights)

# In[144]:


np.random.seed(101)
print(df1.columns)
rand_weights = np.array(np.random.rand(4))
print('Random Weights : ',rand_weights)
## To make sum of random weights equal to 1 , divide each random generated weight by sum
print('Rebalance')
weights = rand_weights/np.sum(rand_weights)
print(weights)


# In[153]:


## Yearly portfolio expected return
exp_ret = np.sum(df_returns.mean()*weights*252)
exp_ret


# In[154]:


## Portfolio Volatility Yearly
exp_vol = np.sqrt(np.dot(weights.T,np.dot(df_returns.cov()*252,weights)))
exp_vol


# In[156]:


## Sharpe Ratio
sr = exp_ret/exp_vol
print('Sharpe Ratio :',sr)


# In[158]:


## Final code for monte carlo
np.random.seed(101)
num_ports = 10000
all_weights = np.zeros((num_ports,len(df_returns.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    rand_weights = np.array(np.random.rand(4))
    ## To make sum of random weights equal to 1 , divide each random generated weight by sum
    weights = rand_weights/np.sum(rand_weights)
    all_weights[ind,:] = weights
    ## Yearly portfolio expected return
    ret_arr[ind] = np.sum(df_returns.mean()*weights*252)
    ## Portfolio Volatility Yearly
    vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(df_returns.cov()*252,weights)))
    ## Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

sharpe_arr.max()
# In[160]:


sharpe_arr.argmax()


# In[161]:


all_weights[7872,:]


# In[163]:


max_sr_ret = ret_arr[7872]
max_sr_vol = vol_arr[7872]


# In[165]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')


# # Var(Value at Risk) of a Normal Distribution

# In[330]:


# Var of a Normal Distribution
# Create the VaR measure at the 95% confidence level using norm.ppf()
VaR_95 = norm.ppf(0.95)

# Create the VaR meaasure at the 5% significance level using numpy.quantile()
draws = norm.rvs(size = 100000)
VaR_99 = np.quantile(draws, 0.99)

# Compare the 95% and 99% VaR
print("95% VaR: ", VaR_95, "; 99% VaR: ", VaR_99)

# Plot the normal distribution histogram and 95% VaR measure
plt.hist(draws, bins = 100)
plt.axvline(x = VaR_95, c='r', label = "VaR at 95% Confidence Level")
plt.axvline(x = VaR_99, c='g', label = "VaR at 99% Confidence Level")

plt.legend(); plt.show()


# ## CVAR of a Normal Distribution

# In[405]:


losses.head(10)


# In[406]:


mean_loss = losses.mean()


# In[407]:


mean_loss


# In[408]:


std_loss = losses.std()


# In[409]:


std_loss


# In[410]:


# Compute the mean and variance of the portfolio losses
pm = mean_loss
ps = std_loss

# Compute the 95% VaR using the .ppf()
VaR_95 = norm.ppf(0.95, loc = pm, scale = ps)
# Compute the expected tail loss and the CVaR in the worst 5% of cases
tail_loss = norm.expect(lambda x: x, loc = pm, scale = ps, lb = VaR_95)
CVaR_95 = (1 / (1 - 0.95)) * tail_loss

# Plot the normal distribution histogram and add lines for the VaR and CVaR
plt.hist(norm.rvs(size = 100000, loc = pm, scale = ps), bins = 100)
plt.axvline(x = VaR_95, c='r', label = "VaR, 95% confidence level")
plt.axvline(x = CVaR_95, c='g', label = "CVaR, worst 5% of outcomes")
plt.legend(); plt.show()


# ## VaR of Student's t-distribution

# In[411]:


from scipy.stats import t


# In[413]:


mu = losses.rolling(30).mean()
sigma = losses.rolling(30).std()


# In[414]:


mu


# In[415]:


sigma


# In[416]:


rolling_parameters = [(29, mu[i], s) for i,s in enumerate(sigma)]
VaR_99 = np.array( [ t.ppf(0.99, *params) 
                    for params in rolling_parameters ] )

# Plot the minimum risk exposure over the 2005-2010 time period
plt.plot(losses.index, 0.01 * VaR_99 * 100000)
plt.show()


# In[427]:


# Fit the Student's t distribution to crisis losses
p = t.fit(losses)

# Compute the VaR_99 for the fitted distribution
VaR_99 = t.ppf(0.99, *p)

# Use the fitted parameters and VaR_99 to compute CVaR_99
tail_loss = t.expect( lambda y: y, args = (p[0],), loc = p[1], scale = p[2], lb = VaR_99 )
CVaR_99 = (1 / (1 - 0.99)) * tail_loss
print(CVaR_99)


# 26% Loss (CVaR) on a given portfolio investment during financial crisis

# ## Parametric Estimation VaR

# Parameter estimation is the strongest method of VaR estimation because it assumes that the loss distribution class is known. 
# Parameters are estimated to fit data to this distribution, and statistical inference is then made.

# ##### Finding best parameters (Theta - Mean and SD) given portfolio data is called Parametric Estimation

# In Parameter Estimation VaR, loss distribution is not given, thereby we fit different distribution and with the help of 
# Anderson Darling test we check goodness of fit.

# In[339]:


df.head(10)


# In[340]:


df_returns = df.dropna(axis=0)


# In[341]:


df_returns.head(3)


# In[348]:


df_returns.describe()


# In[370]:


params = norm.fit(losses)


# In[371]:


params


# In[372]:


VaR_95 = norm.ppf(0.95, *params)


# In[373]:


print("VaR_95, Normal distribution: ", VaR_95)


# In[374]:


print("Anderson-Darling test result: ", anderson(losses))


# ##### The Anderson-Darling test above value of 38.20 exceeds the 99% critical value of 1.088 by a large margin, indicating that the Normal distribution  may be a poor choice to represent portfolio losses

# In[ ]:


## Null Hypothesis - No Skewness


# In[375]:


# Test the data for skewness
print("Skewtest result: ", skewtest(losses))


# In[376]:


# Fit the portfolio loss data to the skew-normal distribution
params = skewnorm.fit(losses)


# In[377]:


# Compute the 95% VaR from the fitted distribution, using parameter estimates
VaR_95 = skewnorm.ppf(0.95, *params)
print("VaR_95 from skew-normal: ", VaR_95)


# Losses are not normally distributed as the critical value exceeeds the 99% conidence interval of test statistic value
# Losses can be skewed
# 
# Definition wiki - anderson
# In many cases (but not all), you can determine a p value for the Anderson-Darling statistic and use that value to help you 
# determine if the test is significant are not. Remember the p ("probability") value is the probability of getting a result 
# #that is more extreme if the null hypothesis is true. If the p value is low (e.g., <=0.05), you conclude that the data do 
# not follow the normal distribution. Remember that you chose the significance level even though many people just use 0.05 
# the vast majority of the time. We will look at two different data sets and apply the Anderson-Darling test to both sets.
# 
# 

# Note that although the VaR estimate for the 
# Normal distribution from the previous exercise is larger than the skewed Normal distribution estimate, the Anderson-Darling 
# and skewtest results show the Normal distribution estimates cannot be relied upon. Skewness matters for loss distributions, 
# and parameter estimation is one way to quantify this important feature of the financial crisis.

# # Historical Simulation
# # EXAMPLE
# #weights = [0.25, 0.25, 0.25, 0.25]
# #portfolio_returns = asset_returns.dot(weights)
# #losses = - portfolio_returns
# #VaR_95 = np.quantile(losses, 0.95)

# Historical simulation: use past to predict future
# No distributional assumption required
# Data about previous losses become simulated losses for tomorrow

# In[378]:


VaR_95_HS = np.quantile(losses,0.95)


# In[379]:


print(VaR_95_HS)


# In[380]:


## 5 % Loss with 95% confidence interval


# ## Historical with monte carlo simulation VaR 

# In[381]:


# Initialize daily cumulative loss for the 4 assets, across N runs
N=10000
daily_loss = np.zeros((4,N))


# In[383]:


returns.head()


# In[394]:


mu = np.array([returns['Return_MS'].mean(),returns['Return_Citi'].mean(),returns['Return_JPM'].mean(),
               returns['Return_GS'].mean()])


# In[395]:


mu


# In[401]:


mu = np.array([[-0.1087564],
      [-0.24369987],
      [-0.01287549],
      [-0.0179018]])


# In[402]:


type(mu)


# In[384]:


e_cov = returns.cov()


# In[386]:


e_cov_1 = np.array(e_cov)


# In[387]:


e_cov_1


# In[389]:


total_steps = 1440


# In[ ]:


# Create the Monte Carlo simulations for N runs
for n in range(N):
    # Compute simulated path of length total_steps for correlated returns
    correlated_randomness = e_cov @ norm.rvs(size = (4,total_steps))
    # Adjust simulated path by total_steps and mean of portfolio losses
    steps = 1/total_steps
    minute_losses = mu * steps + correlated_randomness * np.sqrt(steps)
    daily_loss[:, n] = minute_losses.sum(axis=1)


# In[ ]:


losses = weights @ daily_loss
print("Monte Carlo VaR_95 estimate: ", np.quantile(losses, 0.95))


# Ordinary Least Square
# Ordinary least squares (OLS) regression is a statistical method of analysis that estimates the relationship between one or more independent variables and a dependent variable; the method estimates the relationship by minimizing the sum of the squares in the difference between the observed and predicted values of the dependent variable configured as a straight line.

# ### Structural Breaks - Theory

# Chow Test = Whether or not a structural break has occured in the data
# Visualization cannot determine exact structural break in the data
# Alternative - Time of structural break concides with time of increasing volatility
# Stochastic Volatility Model : Volatility can be analyzed statistically through the random probability distribution but
# cannot be predicted precisely
# 

# To check if the volatility is non stationary rolling window volatility is calculated

# VaR and CVaR estimates that data distribution is same throughout (Stationarity Assumption) but there are structural breaks
# in between.
# 
# So Assume specific points in time for change
# Break up data into sub-periods
# Within each sub-period, assume stationarity
# 
# Chow TEST: Test for evidence of structural breaks
#     1. Null hypothesis - No break
#     2. Requires three OLS regressions
#     3. Regression for entire period
#     4. Two regressions, before and after break
#     5. Collect sum-of-squared residuals
#     6. Test statistic is distributed according to "F" distribution
#     
# Noe sometimes it is not easy to visualize the losses to detect the structural break
# Sometimes we can use Rolling window volatility to visualize the rolling volatity in the given time period
# 
# std() calculates a single value of volatility
# rolling.std calculates rolling volatility and you can plot and see the structural break
# 
# Backtesting
# Backtesting is the process of applying a trading strategy or analytical method to historical data to see how accurately
# the strategy or method would have predicted actual results.

# In[ ]:




