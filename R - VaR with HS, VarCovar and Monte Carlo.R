#Function to calculate historic simulation VaR
#The function reads in a set of market prices, a starting point, and end point and a confidence level.
#The function also reads in a delta vector showing the sensitivity of the portfolio to each of the 
#risk factors.

#To operate the function a data frame of market rates (first column being dates) must be supplied
#The test function below runs the function on a portfolio of 10 stocks 
#
#
###################################################################
HSVaR <- function(dataSet,startIndex,endIndex,deltaVector,alpha){
  numColumns = ncol(dataSet) 
  X = dataSet[startIndex:endIndex,2:numColumns]
  numRows = nrow(X)
  Y  <- matrix(0,numRows-1,numColumns-1)
  for (i in 1:(numColumns-1)){
    Y[,i]=diff(log(X[,i]))
  }
  #
  res <- vector() #Simulated P&L outcomes
  for (i in 1:numRows-1){
    res[i] = 0
    for (j in 1:(numColumns-1)){
      res[i] = res[i]+deltaVector[j]*Y[i,j]
    }
    
  }
  quantile(res,alpha) # We choose the alpha'th quantile (eg 5th or 1st percentile) as the VaR
}

VCVVaR <- function(dataSet,startIndex,endIndex,deltaVector,alpha){
  numColumns = ncol(dataSet) 
  sd_vect <- vector()
  w <- vector() # Will be set to StDev * Delta
  X = dataSet[startIndex:endIndex,2:numColumns]
  numRows = nrow(X)
  #Create a matrix of returns
  Y  <- matrix(0,numRows-1,numColumns-1)
  for (i in 1:(numColumns-1)){
    Y[,i]=diff(log(X[,i])) # Y is the matrix of returns 
    sd_vect[i]=sd(Y[,i]) # sd_vect is the standard deviation of these returns
  }
  #We compute the correlation matrix
  CM = cor(Y)
  #We compute the product of the factor standard deviations with their sensitivities
  for (i in (1:numColumns-1)){
    w[i] <- sd_vect[i]*deltaVector[i]
  }
  portfolio_standard_deviation <- sqrt(t(w) %*% (CM %*% w))
  portfolio_VaR <- portfolio_standard_deviation * qnorm(1-alpha)
  return(portfolio_VaR*-1)
}


MCVaR <- function(dataSet,startIndex,endIndex,deltaVector,alpha,numSims){
  numColumns = ncol(dataSet) 
  z <- vector()
  zStar <- vector()
  sd_vect <- vector()
  PnL <- vector() # We will use this to store the simulated P&L values
  X = dataSet[startIndex:endIndex,2:numColumns]
  numRows = nrow(X)
  #Create a matrix of returns
  Y  <- matrix(0,numRows-1,numColumns-1)
  for (i in 1:(numColumns-1)){
    Y[,i]=diff(log(X[,i])) # Y is the matrix of returns 
    sd_vect[i]=sd(Y[,i]) # sd_vect is the standard deviation of these returns
  }
  #We compute the correlation matrix
  CM = cor(Y)
  chol_mat <- chol(CM)
  #We compute the product of the factor standard deviations with their sensitivities
  set.seed(42)
  for (i in 1:numSims){
    my_PnL <- 0
    z <- rnorm(numColumns-1)
    zStar <- chol_mat %*% z
    for (j in 1:(numColumns-1)){
      zStar[j] <- zStar[j]*sd_vect[j]
      my_PnL <- my_PnL + zStar[j]*deltaVector[j]
    }
    
    PnL[i] <- my_PnL
    
  }
  portfolio_VaR <- quantile(PnL,alpha)
  return(portfolio_VaR)
}

##################################################################
#Test the functions
#Create some vectors

library(readxl)
StockPrices <- read_excel("C:/StockPrices.xlsx")
View(StockPrices)
S <- as.data.frame(StockPrices)

my_1y_HS_var <- vector()
my_2y_HS_var <- vector()
my_3y_HS_var <- vector()
my_1y_VCV_var <- vector()
my_2y_VCV_var <- vector()
my_3y_VCV_var <- vector()
my_1y_MC_var <- vector()
my_2y_MC_var <- vector()
my_3y_MC_var <- vector()
delta_vect <- vector()
N <- 10000

for (i in 1:10){
  delta_vect[i]=1e5 # Assume a holding of ???100k in each asset
}
for(i in 1:(length(S$Date)-260)){
  my_1y_HS_var[i]=HSVaR(S,i,i+260,delta_vect,0.05)
  my_1y_VCV_var[i]=VCVVaR(S,i,i+260,delta_vect,0.05)
  my_1y_MC_var[i]=MCVaR(S,i,i+260,delta_vect,0.05,N)
  
}
for(i in 1:(length(S$Date)-260*2)){
  my_2y_HS_var[i]=HSVaR(S,i,i+260*2,delta_vect,0.05)
  my_2y_VCV_var[i]=VCVVaR(S,i,i+260*2,delta_vect,0.05)
  my_2y_MC_var[i]=MCVaR(S,i,i+260*2,delta_vect,0.05,N)
  
  
}
for(i in 1:(length(S$Date)-260*3)){
  my_3y_HS_var[i]=HSVaR(S,i,i+260*3,delta_vect,0.05)
  my_3y_VCV_var[i]=VCVVaR(S,i,i+260*3,delta_vect,0.05)
  my_3y_MC_var[i]=MCVaR(S,i,i+260*3,delta_vect,0.05,N)
  
  
}

par(mfrow=c(3,1))

plot(S$Date[261:length(S$Date)],my_1y_HS_var,type='l',main='1 Year VaR',xlab='Date',ylab='VaR (€)',col='blue')
lines(S$Date[261:length(S$Date)],my_1y_VCV_var,col='red')
lines(S$Date[261:length(S$Date)],my_1y_MC_var,col='green')
legend("bottomleft",legend = c("Historic Simulation","Variance Covariance","Monte Carlo"),col=c("blue","red","green"),lty=1:1,box.lty=0,inset = 0.01)
grid()

plot(S$Date[521:length(S$Date)],my_2y_HS_var,type='l',main='2 Year VaR',xlab='Date',ylab='VaR (€)',col = 'blue')
lines(S$Date[521:length(S$Date)],my_2y_VCV_var,col='red')
lines(S$Date[521:length(S$Date)],my_2y_MC_var,col='green')
legend(legend = c('Historic Simulation','Variance Covariance'),col=c('blue','red'))
legend("bottomleft",legend = c("Historic Simulation","Variance Covariance","Monte Carlo"),col=c("blue","red","green"),lty=1:1,box.lty=0,inset = 0.01)
grid()

plot(S$Date[781:length(S$Date)],my_3y_HS_var,type='l',main='3 Year VaR',xlab='Date',ylab='VaR (€)',col = 'blue')
lines(S$Date[781:length(S$Date)],my_3y_VCV_var,col='red')
lines(S$Date[781:length(S$Date)],my_3y_MC_var,col='green')
legend("bottomleft",legend = c("Historic Simulation","Variance Covariance","Monte Carlo"),col=c("blue","red","green"),lty=1:1,box.lty=0,inset = 0.01)
grid()


