
# Correlated Errors

Today, we will consider another departure from our usual model for the variance
(i.e. equal variance $\sigma^2$ and independent).

Before we do this, let's refresh our memory on *weighted least squares.*


```R
options(repr.plot.width=5, repr.plot.height=5)
```

## Weighted least squares

In the last set of notes, we considered a model
$$
Y = X\beta + \epsilon, \qquad \epsilon \sim N(0, W^{-1})
$$
where
$$
W^{-1} = \sigma^2 \cdot \text{diag}(V_1, \dots, V_n).
$$

This model has independent errors, but of different variance: a *heteroscedastic* model.

## The fix

We saw that by defining
$$
\tilde{Y} = W^{1/2}Y, \qquad \tilde{X} = W^{1/2}X
$$
we transformed our original model to more familiar model:
$$
\tilde{Y} = \tilde{X}\beta + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2 I).
$$

The usual estimator in this model is the *WLS* estimator
$$
\hat{\beta}_W = (X^TWX)^{-1}X^TWY.
$$



## The implications

If we ignore *heteroscedasticity* then our *OLS* estimator has
distribution
$$
\hat{\beta} = (X^TX)^{-1}X^TY \sim N(\beta, \sigma^2 (X^TX)^{-1} X^TW^{-1}X (X^TX)^{-1}).
$$

This form of the variance matrix is called the *sandwich form*. 

**This means that our `Std. Error` column will be off! In other words, `R` will report $t$ statistics that are off by some multiplicative factor!**

Another reason to worry about $W$ is that if we use the correct $W$, we have
a more *efficient* unbiased estimator: smaller confidence intervals.

## The implications

Using the correct $W$ *proportional to inverse variance of the errors* and
form the WLS estimator we have
$$
\hat{\beta}_{WLS} \sim N(\beta, \sigma^2 (X^TWX)^{-1}).
$$


## Autocorrelation

- The model of the variance that we will consider today 
is a model where the errors are *correlated*.

- In the random effects model, outcomes within groups were correlated.
Other regression applications also have correlated outcomes (i.e.
errors).

- Common examples of this type of errors occur in time series data, a
common model for financial applications.

### Why should we worry?
Just as in the *heteroscedastic case*, ignoring autocorrelation can lead to underestimates of `Std. Error` $\rightarrow$ inflated $t$’s $\rightarrow$ false positives.

### What is autocorrelation?

* Suppose we plot Palo Alto’s daily average temperature – clearly we
  would see a pattern in the data.

* Sometimes, this pattern can be attributed to a deterministic
  phenomenon (i.e. predictable seasonal fluctuations).

* Other times, “patterns” are due to correlations in the noise, maybe
  small time fluctuations in the stock market, economy, etc.

* Example: financial time series: NASDAQ close prices.

* Example: residuals regressing consumer expenditure on money stock (this one is discussed in your textbook and used as an example below).

### Average Max Temp in Palo Alto

The daily max temperature shows clear seasonal fluctuations.


```R
PA.temp <- read.table('http://stats191.stanford.edu/data/paloaltoT.table', header=F, skip=2)
plot(PA.temp[,3], xlab='Day', ylab='Average Max Temp (F)', pch=23, bg='orange')
```

### NASDAQ daily close 2011

- Another example of a time series can be found from financial data. 
The price of many assets fluctuate from day to day. 

- Still, there is a *pattern* in this process. 

- Given enough information, we might try to also explain
this pattern as a deterministic model, like the temperature data. (This is, in some sense, what business news sites try to do on a daily basis).

- A simpler model for this pattern is that of some unexplainable noise...

- Below, we plot some
closing prices of NASDAQ for the year 2011. Data was obtained from
on [yahoo finance](http://finance.yahoo.com).



```R
fname = 'http://stats191.stanford.edu/data/nasdaq_2011.csv'
nasdaq.data = read.table(fname, header=TRUE, sep=',')
plot(nasdaq.data$Date, nasdaq.data$Close, xlab='Date', ylab='NASDAQ close')
abline(h=mean(nasdaq.data$Close))
```


```R
ndays = length(nasdaq.data$Date)
log_return = log(nasdaq.data$Close[2:ndays] / nasdaq.data$Close[1:(ndays-1)])
plot(nasdaq.data$Date[2:ndays], 
    log_return, xlab='Date', ylab='log(NASDAQ return)')

```

### NASDAQ daily close 2011, ACF

- One way this noise is measured is through the *ACF (Auto-Correlation Function)*, which we will define below.

- A time series with no auto-correlation (i.e. our usual multiple linear regression model) has an ACF that contains only a spike at 0.

- The NASDAQ close clearly has some auto-correlation.


```R
acf(nasdaq.data$Close)
```


```R
acf(log_return)
```

# ACF of independent noise


```R
acf(rnorm(length(nasdaq.data$Close)))
```

### Expenditure vs. stock

The example we will consider is that of *consumer expenditure* vs. *money stock*, the supply of available money in the economy.

Data is collected yearly, so perhaps there is autocorrelation in the model
$$
{\tt Expenditure}_t = \beta_0 + \beta_1 {\tt Stock}_t + \epsilon_t
$$


```R
fname = 'http://stats191.stanford.edu/data/expenditure.table'
expenditure.table =read.table(fname, header=T)
attach(expenditure.table)
plot(Stock, Expenditure, pch=23, bg='orange', cex=2)
```

### Expenditure vs. stock: residuals

A plot of residuals against `time`, i.e. their index may show 
evidence of autocorrelation.


```R
exp.lm = lm(Expenditure ~ Stock)
plot(resid(exp.lm), type='l', lwd=2, col='red')
```

### ACF of residuals

A plot of the ACF may also help. Since there seem to be some
points outside the confidence bands, this is some evidence that auto-correlation
is present in the errors.


```R
acf(resid(exp.lm))
```

## Models for correlated errors



### AR(1) noise

* Suppose that, instead of being independent, the errors in our model
  were $$\varepsilon_t = \rho \cdot \varepsilon_{t-1} + \omega_t, \qquad -1 <
  \rho < 1$$ with $\omega_t \sim N(0,\sigma^2)$ independent.
  
* If $\rho$ is close to 1, then errors are very correlated, $\rho=0$
  is independence.
  
* This is “Auto-Regressive Order (1)” noise (AR(1)). Many other models
  of correlation exist: ARMA, ARIMA, ARCH, GARCH, etc.

### AR(1) noise, $\rho=0.9$


```R
nsample = 200
rho = 0.95
mu = 1.0
plot(arima.sim(list(ar=rho), nsample), lwd=2, col='red')
```

## Autocorrelation function

* For a “stationary” time series $(Z_t)_{1 \leq t \leq \infty}$ define
  $$ACF(t) = \text{ Cor}(Z_s, Z_{s+t}).$$
* Stationary means that correlation above does not depend on $s$.
* For AR(1) model, $$ACF(t) = \rho^t.$$
* For a sample $(Z_1, \dots, Z_n)$ from a stationary time series
  $$\widehat{ACF}(t) = \frac{\sum_{j=1}^{n-t} (Z_j - \overline{Z})(Z_{t+j} - \overline{Z})}{\sum_{j=1}^n(Z_j - \overline{Z})^2}.$$

### ACF of AR(1) noise, $\rho=0.9$


```R
acf(arima.sim(list(ar=0.9), 100))
```

### Effects on inference

* So far, we have just mentioned that things *may* be correlated, but
  not thought about how it affects inference.
* Suppose we are in the “one sample problem” setting and we observe
  $$W_i  = Z_i + \mu, \qquad 1 \leq i \leq n$$ with the $Z_i$’s from
  an $AR(1)$ time series.
* It is easy to see that $$E(\overline{W}) = \mu$$ *BUT*, generally 
  $$\text{Var}(\overline{W}) >  \frac{\text{Var}(Z_1)}{n}$$ how much
  bigger depends on $\rho.$

## Misleading inference ignoring autocorrelation

Just as in weighted least squares, ignoring the autocorrelation
yields misleading `Std. Error` values.

Below, we show that ignoring autocorrelation will yield
incorrect confidence intervals. The red curve is (an estimate of) the true 
density of the sample mean, while the blue curve is what
we think it should be if the errors were independent. The blue
curve is way too optimistic.


```R
ntrial = 1000

sample.mean = numeric(ntrial)
sample.var = numeric(ntrial)

for (i in 1:ntrial) {
  cur.sample = arima.sim(list(ar=rho), nsample) + mu
  sample.mean[i] = mean(cur.sample)
  sample.var[i] = var(cur.sample)
}

data.frame(mean=mean(sample.mean), sd=sqrt(mean(sample.var)))

xval = seq(-5,5,0.05)
Y = c(density(sample.mean)$y, dnorm(xval, mean=mean(sample.mean),
                  sd=sqrt(mean(sample.var) / nsample)))
X = c(density(sample.mean)$x, xval)
```


```R
plot(X, Y, type='n', main='Actual and "naive" density of sample mean', xlim=c(-1,3))
lines(xval, dnorm(xval, mean=mean(sample.mean),
                  sd=sqrt(mean(sample.var) / nsample)), lwd=4, col='blue')
lines(density(sample.mean), lwd=4, col='red')
legend(-1,1, c('actual', 'naive'), col=c('red', 'blue'), lwd=rep(4,3))
```

## Regression model with auto-correlated errors (AR(1))

* Observations:
  $$Y_t = \beta_0 + \sum_{j=1}^p X_{tj} \beta_j + \varepsilon_t, \qquad 1 \leq t \leq n$$
* Errors:
  $$\varepsilon_t = \rho \cdot \varepsilon_{t-1} + \omega_t, \qquad -1 < \rho < 1$$
* Question: how do we determine if autocorrelation is present?
* Question: what do we do to correct for autocorrelation?

### Graphical checks for autocorrelation

* A plot of residuals vs. time is helpful.
* Residuals clustered above and below 0 line can indicate
  autocorrelation.


### Expenditure vs. stock: residuals


```R
exp.lm = lm(Expenditure ~ Stock)
plot(resid(exp.lm), type='l', lwd=2, col='red')
```

### Durbin-Watson test

* In regression setting, if noise is AR(1), a simple estimate of
  $\rho$ is obtained by (essentially) regressing $e_t$ onto $e_{t-1}$
  $$\widehat{\rho} = \frac{\sum_{t=2}^n \left(e_t e_{t-1}\right)}{\sum_{t=1}^n e_t^2}.$$
* To formally test $H_0:\rho=0$ (i.e. whether residuals are
  independent vs. they are AR(1)), use Durbin-Watson test, based on
  $$d \approx 2(1 - \widehat{\rho}).$$

### Correcting for AR(1)

* Suppose we know $\rho$, we can then “whiten” the data and regressors
  $$\begin{aligned}
     \tilde{Y}_{t+1} &= Y_{t+1} - \rho Y_t, t > 1   \\
     \tilde{X}_{(t+1),j} &= X_{(t+1),j} - \rho X_{t,j}, i > 1
     \end{aligned}$$ for $1 \leq t \leq n-1$. This model satisfies
  “usual” assumptions, i.e. the errors
  $$\tilde{\varepsilon}_t = \omega_{t+1} = \varepsilon_{t+1} - \rho \cdot \varepsilon_t$$
  are independent $N(0,\sigma^2)$.
* For coefficients in new model $\tilde{\beta}$,
  $\beta_0 = \tilde{\beta}_0 / (1 - \rho)$,
  $\beta_j = \tilde{\beta}_j.$
* Problem: in general, we don’t know $\rho$.

### Two-stage regression

As in weighted least squares, we will use a two-stage procedure.

* Step 1: Fit linear model to unwhitened data (OLS: ordinary least
  squares, i.e. no prewhitening).
* Step 2: Estimate $\rho$ with $\widehat{\rho}$.
* Step 3: Pre-whiten data using $\widehat{\rho}$ – refit the model.

## Whitening

Our solution in the weighted least squares and auto-correlated errors
examples were the same. This procedure is generally called *whitening*.

Consider a model
$$
Y = X\beta + \epsilon, \qquad \epsilon \sim N(0, \Sigma).
$$

If $\Sigma$ is invertible, then we can find a inverse square root of $\Sigma$:
$$
\Sigma^{-1/2}\Sigma (\Sigma^{-1/2})^T = I, \qquad (\Sigma^{-1/2})^T\Sigma^{-1/2} = \Sigma^{-1}.
$$

Define
$$
\tilde{Y} = \Sigma^{-1/2}Y, \qquad \tilde{X} = \Sigma^{-1/2}X.
$$
Then
$$
\tilde{Y} = \tilde{X}\beta + \tilde{\epsilon}, \qquad \tilde{\epsilon} \sim N(0, I).
$$



### Generalized least squares

- The OLS estimator based on $(\tilde{Y}, \tilde{X})$ is 
$$
\hat{\beta}_{\Sigma} = (X^T\Sigma^{-1}X)^{-1}X^T\Sigma^{-1}Y \sim N(\beta, (X^T\Sigma^{-1}X)^{-1})
$$

- It is often called the *GLS (Generalized Least Squares)* estimate based on
the covariance matrix $\Sigma$.

- The OLS estimator based on $(Y,X)$ has the sandwich form again:
$$
\hat{\beta} = (X^TX)^{-1}X^TY \sim N(\beta, (X^TX)^{-1}X^T\Sigma X (X^TX)^{-1}).
$$

- As in WLS, the GLS estimator with $\Sigma=\text{Var}(Y)$ will generally be a more efficient estimator.

- WLS is special case when $\Sigma$ is diagonal.

## Interpreting results of two-stage fit

* Basically, interpretation is unchanged, but the exact degrees of
  freedom in the error is not exactly clear.

* Commonly applied argument: 
  “this works for large degrees of freedom, so we
  hope we have enough degrees of freedom so this point is not
  important.”

* Can treat $t$-statistics as $Z$-statistics, $F$’s as $\chi^2$,
  appealing to asymptotics:

  * $t_{\nu}$, with $\nu$ large is like $N(0,1)$;
  * $F_{j,\nu}$, with $\nu$ large is like $\chi^2_j/j.$

### Expenditure vs. stock: Durbin-Watson


```R
library(car) # durbin.watson is in the "car" package
durbinWatsonTest(exp.lm)
rho.hat = durbinWatsonTest(exp.lm)$r

```

Given the value of $\rho$, we can apply our whitening procedure.




```R
wExp = numeric(length(Expenditure) - 1)
wStock = numeric(length(Expenditure) - 1)
for (i in 2:length(Expenditure)) {
  wExp[i-1] = Expenditure[i] - rho.hat * Expenditure[i-1]
  wStock[i-1] = Stock[i] - rho.hat * Stock[i-1]
}
```

After whitening, we refit the model.


```R
exp.whitened.lm = lm(wExp ~ wStock)
plot(resid(exp.whitened.lm), type='l', lwd=2, col='red')
#summary(exp.whitened.lm)
```

Comparing to our original fit, we see that our $t$ statistic has changed
by a factor of roughly 2.5 from 20 to 8.6!


```R
summary(exp.lm)
```


```R
summary(exp.whitened.lm)
```

Lastly, let's take a look at the residuals of the whitened data. If
our whitening has been successful, this should just be a spike at 0.


```R
acf(resid(exp.whitened.lm))
```


```R

```
