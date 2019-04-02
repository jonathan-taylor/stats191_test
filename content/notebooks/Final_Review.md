
### Topics covered

* Simple linear regression.
* Diagnostics for simple linear regression.
* Multiple linear regression.
* Diagnostics.
* Interactions and ANOVA.
* Weighted Least Squares.
* Autocorrelation.
* Bootstrapping `lm`.
* Model selection.
* Penalized regression.
* Logistic and Poisson regression.



```R
options(repr.plot.width=5, repr.plot.height=5)
```

### Simple linear regression


### Least squares

* We used "least squares" regression. This measures the goodness of fit of a line by the sum of squared errors, $SSE$.
* Least squares regression chooses the line that minimizes $SSE(\beta_0, \beta_1) = \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 \cdot X_i)^2.$

### Geometry of Least Squares: Simple Linear Model

<img src="figs/axes_simple.svg">


### What is a $t$-statistic?

* Start with $Z \sim N(0,1)$ is standard normal and $S^2 \sim \chi^2_{\nu}$, independent of $Z$.
* Compute $T = \frac{Z}{\sqrt{\frac{S^2}{\nu}}}.$
* Then, $T \sim t_{\nu}$ has a $t$-distribution with $\nu$ degrees of freedom.
* Generally, a $t$-statistic has the form $$ T = \frac{\hat{\theta} - \theta}{SE(\hat{\theta})}$$


### Interval for $\beta_1$

 A $(1-\alpha) \cdot 100 \%$ confidence interval: $\widehat{\beta}_1 \pm SE(\widehat{\beta}_1) \cdot t_{n-2, 1-\alpha/2}.$
Interval for regression line $\beta_0 + \beta_1 \cdot X$

* $(1-\alpha) \cdot 100 \%$ confidence interval for $\beta_0 + \beta_1 X$: $\widehat{\beta}_0 + \widehat{\beta}_1 X \pm SE(\widehat{\beta}_0 + \widehat{\beta}_1 X) \cdot t_{n-2, 1-\alpha/2}$ where $SE(a_0\widehat{\beta}_0 + a_1\widehat{\beta}_1) = \widehat{\sigma} \sqrt{\frac{a_0^2}{n} + \frac{(a_0\overline{X} - a_1)^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}$

### Prediction intervals for $\beta_0 +\beta_1 X_{new} + \epsilon_{new}$

* $SE(\widehat{\beta}_0 + \widehat{\beta}_1 X_{\text{new}} + \varepsilon_{\text{new}}) = \widehat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(\overline{X} - X_{\text{new}})^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}.$
* Prediction interval is $\widehat{\beta}_0 +  \widehat{\beta}_1 X_{\text{new}} \pm t_{n-2, 1-\alpha/2} \cdot SE(\widehat{\beta}_0 + \widehat{\beta}_1 X_{\text{new}} + \varepsilon_{\text{new}})$


### Sums of squares

 $$\begin{aligned}
   SSE &= \sum_{i=1}^n(Y_i - \widehat{Y}_i)^2 = \sum_{i=1}^n (Y_i - \widehat{\beta}_0 - \widehat{\beta}_1 X_i)^2 \\
   SSR &= \sum_{i=1}^n(\overline{Y} - \widehat{Y}_i)^2 = \sum_{i=1}^n (\overline{Y} - \widehat{\beta}_0 - \widehat{\beta}_1 X_i)^2 \\
   SST &= \sum_{i=1}^n(Y_i - \overline{Y})^2 = SSE + SSR \\
   R^2 &= \frac{SSR}{SST} = 1 - \frac{SSE}{SST} = \widehat{Cor}(\pmb{X},\pmb{Y})^2.
   \end{aligned}$$
   


### $F$-test in simple linear regression

* *Full (bigger) model :*
   $FM: \qquad Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i$
* *Reduced (smaller) model:*
   $RM: \qquad Y_i = \beta_0  + \varepsilon_i$
* The $F$-statistic has the form $F=\frac{(SSE(RM) - SSE(FM)) / (df_{RM} - df_{FM})}{SSE(FM) / df_{FM}}.$
* Reject $H_0: RM$ is correct, if $F > F_{1-\alpha, 1, n-2}$.

### Assumptions in the simple linear regression model

* $Y_i = \beta_0 + \beta_1 X_{i} + \varepsilon_i$
* Errors $\varepsilon_i$ are assumed independent $N(0,\sigma^2)$.

### Diagnostic plots


```R
simple.lm = lm(y2 ~ x2, data=anscombe)
plot(anscombe$x2, resid(simple.lm), ylab='Residual', xlab='X',
     pch=23, bg='orange', cex=1.2)
abline(h=0, lwd=2, col='red', lty=2)
```


```R
par(mfrow=c(2,2))
plot(simple.lm)
```

### Quadratic model


```R
quadratic.lm = lm(y2 ~ poly(x2, 2), data=anscombe)
Xsort = sort(anscombe$x2)
plot(anscombe$x2, anscombe$y2, pch=23, bg='orange', cex=1.2, ylab='Y', xlab='X')
lines(Xsort, predict(quadratic.lm, list(x2=Xsort)), col='red', lty=2, lwd=2)
```


```R
par(mfrow=c(2,2))
plot(quadratic.lm)
```

### Simple linear diagnostics

- Outliers

- Nonconstant variance


```R
url = 'http://stats191.stanford.edu/data/HIV.VL.table'
viral.load = read.table(url, header=T)
plot(viral.load$GSS, viral.load$VL, pch=23, bg='orange', cex=1.2)
viral.lm = lm(VL ~ GSS, data=viral.load)
abline(viral.lm, col='red', lwd=2)

```

### Multiple linear regression model

* Rather than one predictor, we have $p=6$ predictors.
* $Y_i = \beta_0 + \beta_1 X_{i1} + \dots + \beta_p X_{ip} + \varepsilon_i$
* Errors $\varepsilon$ are assumed independent $N(0,\sigma^2)$, as in simple linear regression.
* Coefficients are called (partial) regression coefficients because they "allow" for the effect of other variables.

### Geometry of Least Squares: Multiple Regression

<img src="figs/axes_multiple.svg">

### Overall $F$-test

* *Full (bigger) model :*
   $$Y_i = \beta_0 + \beta_1 X_{i1} + \dots \beta_p X_{ip} + \varepsilon_i$$
* *Reduced (smaller) model:*
   $$Y_i = \beta_0  + \varepsilon_i$$
* The $F$-statistic has the form $F=\frac{(SSE(R) - SSE(F)) / (df_R - df_F)}{SSE(F) / df_F}.$

### Matrix formulation

* ${ Y}_{n \times 1} = {X}_{n \times (p + 1)} {\beta}_{(p+1) \times 1} + {\varepsilon}_{n \times 1}$
* ${X}$ is called the *design matrix*
   of the model
* ${\varepsilon} \sim N(0, \sigma^2 I_{n \times n})$ is multivariate normal
$SSE$ in matrix form
$$SSE(\beta) = ({Y} - {X} {\beta})'({Y} - {X} {\beta})$$


### OLS estimators

* Normal equations yield
$$\widehat{\beta} = ({X}^T{X})^{-1}{X}^T{Y}
$$
* Properties: $$\hat{\beta}  \sim N(\beta, \sigma^2 (X^TX)^{-1} )$$


### Confidence interval for $\sum_{j=0}^p a_j \beta_j$

* Suppose we want a $(1-\alpha)\cdot 100\%$ CI for $\sum_{j=0}^p a_j\beta_j$.
* Just as in simple linear regression:
* $\sum_{j=0}^p a_j \widehat{\beta}_j \pm t_{1-\alpha/2, n-p-1} \cdot SE\left(\sum_{j=0}^p a_j\widehat{\beta}_j\right).$
* Standard error:
$$
SE\left(\sum_{j=0}^p a_j\widehat{\beta}_j\right) = \sqrt{\hat{\sigma}^2 a^T(X^TX)^{-1}a}
$$

### General $F$-tests

-   Given two models $R \subset F$ (i.e. $R$ is a subspace of $F$), we
    can consider testing $$ H_0:  \text{$R$ is adequate (i.e. $\mathbb{E}(Y) \in R$)} $$ vs. $$ H_a: \text{$F$ is adequate (i.e. $\mathbb{E}(Y) \in F$)}
    $$
    
    - The test statistic is $$ F = \frac{(SSE(R) - SSE(F)) / (df_R - df_F)}{SSE(F)/df_F} $$

-   If $H_0$ is true, $F \sim F_{df_R-df_F, df_F}$ so we reject $H_0$ at
    level $\alpha$ if $F > F_{df_R-df_F, df_F, 1-\alpha}$.


### Diagnostics: What can go wrong?

-   Regression function can be wrong: maybe regression function should
    have some other form (see diagnostics for simple linear regression).

-   Model for the errors may be incorrect:

    -   may not be normally distributed.

    -   may not be independent.

    -   may not have the same variance.

-   Detecting problems is more *art* then *science*, i.e. we cannot
    *test* for all possible problems in a regression model.

-   Basic idea of diagnostic measures: if model is correct then
    residuals $e_i = Y_i -\widehat{Y}_i, 1 \leq i \leq n$ should look
    like a sample of (not quite independent) $N(0, \sigma^2)$ random
    variables.


### Diagnostics


```R
url = 'http://stats191.stanford.edu/data/scottish_races.table'
races.table = read.table(url, header=T)
attach(races.table)
races.lm = lm(Time ~ Distance + Climb)
par(mfrow=c(2,2))
plot(races.lm, pch=23 ,bg='orange',cex=1.2)
```

### Diagnostics measures

* DFFITS: $$DFFITS_i = \frac{\widehat{Y}_i - \widehat{Y}_{i(i)}}{\widehat{\sigma}_{(i)} \sqrt{H_{ii}}}$$

* Cook's Distance: $$D_i = \frac{\sum_{j=1}^n(\widehat{Y}_j - \widehat{Y}_{j(i)})^2}{(p+1) \, \widehat{\sigma}^2}$$

* DFBETAS: $$DFBETAS_{j(i)} = \frac{\widehat{\beta}_j - \widehat{\beta}_{j(i)}}{\sqrt{\widehat{\sigma}^2_{(i)} (X^TX)^{-1}_{jj}}}.$$


```R
influence.measures(races.lm)
```

### Outliers

* Observations $(Y, X_1, \dots, X_p)$ that do not follow the model, while most other observations seem to follow the model.
* One solution: Bonferroni correction, threshold at $t_{1 - \alpha/(2*n), n-p-2}$.
* Bonferroni: if we are doing many $t$ (or other) tests, say $m >>1$ we can control overall false positive rate at $\alpha$ by testing each one at level $\alpha/m$.


```R
library(car)
outlierTest(races.lm)
```

### Qualitative variables and interactions


```R
url = 'http://stats191.stanford.edu/data/jobtest.table'
jobtest.table <- read.table(url, header=T)
jobtest.table$MINORITY <- factor(jobtest.table$MINORITY)
attach(jobtest.table)
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=1.2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=1.2, bg='green')
```


```R
jobtest.lm1 = lm(JPERF ~ TEST, jobtest.table)
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=1.2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=1.2, bg='green')
abline(jobtest.lm1$coef, lwd=3, col='blue')
```


```R
jobtest.lm4 = lm(JPERF ~ TEST * MINORITY)
print(summary(jobtest.lm4))
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=1.2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=1.2, bg='green')
abline(jobtest.lm4$coef['(Intercept)'], jobtest.lm4$coef['TEST'], lwd=3, col='purple')
abline(jobtest.lm4$coef['(Intercept)'] + jobtest.lm4$coef['MINORITY1'],
      jobtest.lm4$coef['TEST'] + jobtest.lm4$coef['TEST:MINORITY1'], lwd=3, col='green')

```

### ANOVA models: one-way

<table>
<tr><td>Source</td><td width="300">SS</td><td width="100">df</td><td width="100">$\mathbb{E}(MS)$</td></tr>
<tr><td>Treatment</td><td>$SSTR=\sum_{i=1}^r n_i \left(\overline{Y}_{i\cdot} - \overline{Y}_{\cdot\cdot}\right)^2$</td><td>r-1</td><td>$\sigma^2 + \frac{\sum_{i=1}^r n_i \alpha_i^2}{r-1}$</td></tr>
<tr><td>Error</td><td>$SSE=\sum_{i=1}^r \sum_{j=1}^{n_i}(Y_{ij} - \overline{Y}_{i\cdot})^2$</td>
<td>$\sum_{i=1}^r (n_i - 1)$</td><td>$\sigma^2$</td></tr>
</table>

### ANOVA models: two-way

<table>
<tr><td>Source</td><td width="400">SS</td><td width="100">df</td><td width="200">$\mathbb{E}(MS)$</td></tr>
<tr><td>A</td><td>$SSA=nm\sum_{i=1}^r  \left(\overline{Y}_{i\cdot\cdot} - \overline{Y}_{\cdot\cdot\cdot}\right)^2$</td><td>r-1</td><td>$\sigma^2 + nm\frac{\sum_{i=1}^r \alpha_i^2}{r-1}$</td></tr>
<tr><td>B</td><td>$SSB=nr\sum_{j=1}^m  \left(\overline{Y}_{\cdot j\cdot} - \overline{Y}_{\cdot\cdot\cdot}\right)^2$</td>
<td>m-1</td><td>$\sigma^2 + nr\frac{\sum_{j=1}^m \beta_j^2}{m-1}$</td></tr>
<tr><td>A:B</td><td>$SSAB = n\sum_{i=1}^r \sum_{j=1}^m  \left(\overline{Y}_{ij\cdot} - \overline{Y}_{i\cdot\cdot} - \overline{Y}_{\cdot j\cdot} + \overline{Y}_{\cdot\cdot\cdot}\right)^2$</td>
<td>(m-1)(r-1)</td><td>$\sigma^2 + n\frac{\sum_{i=1}^r\sum_{j=1}^m (\alpha\beta)_{ij}^2}{(r-1)(m-1)}$</td></tr>
<tr><td>Error</td><td>$SSE = \sum_{i=1}^r \sum_{j=1}^m \sum_{k=1}^{n}(Y_{ijk} - \overline{Y}_{ij\cdot})^2$</td>
<td>(n-1)mr</td><td>$\sigma^2$</td></tr>
</table>


### Weighted Least Squares

* A way to correct for errors with unequal variance (**but we need a model of the variance**).

* Weighted Least Squares $$SSE(\beta, w) = \sum_{i=1}^n w_i \left(Y_i - \beta_0 - \beta_1 X_i\right)^2.$$
* In general, weights should be like: $$w_i = \frac{1}{\text{Var}(\varepsilon_i)}.$$

* WLS estimator:
$$
\hat{\beta}_W = (X^TWX)^{-1}(X^TWY).
$$

* If weights are ignored standard errors are wrong! 

* Briefly talked about efficiency of estimators.

### Correlated errors: NASDAQ daily close 2011


```R
url = 'http://stats191.stanford.edu/data/nasdaq_2011.csv'
nasdaq.data = read.table(url, header=TRUE, sep=',')

plot(nasdaq.data$Date, nasdaq.data$Close, xlab='Date', ylab='NASDAQ close',
     pch=23, bg='red', cex=1.2)
```

### AR(1) noise

* Suppose that, instead of being independent, the errors in our model were $\varepsilon_t = \rho \cdot \varepsilon_{t-1} + \omega_t, \qquad -1 < \rho < 1$ with $\omega_t \sim N(0,\sigma^2)$ independent.
* If $\rho$ is close to 1, then errors are very correlated, $\rho=0$ is independence.
* This is "Auto-Regressive Order (1)" noise (AR(1)). Many other models of correlation exist: ARMA, ARIMA, ARCH, GARCH, etc.

### Correcting for AR(1) 

* Suppose we know $\rho$, if we "whiten" the data and regressors $$\begin{aligned}
     \tilde{Y}_{t+1} &= Y_{t+1} - \rho Y_t, t > 1   \\
     \tilde{X}_{(t+1)j} &= X_{(t+1)j} - \rho X_{tj}, i > 1
     \end{aligned}$$ for $1 \leq t \leq n-1$. This model satisfies "usual" assumptions, i.e. the errors $\tilde{\varepsilon}_t = \omega_{t+1} = \varepsilon_{t+1} - \rho \cdot \varepsilon_t$ are independent $N(0,\sigma^2)$.
* For coefficients in new model $\tilde{\beta}$, $\beta_0 = \tilde{\beta}_0 / (1 - \rho)$, $\beta_j = \tilde{\beta}_j.$
* Problem: in general, we don’t know $\rho$, but estimated it.

* If correlation structure is ignored standard errors are wrong! 

* Another example of **whitening when we can model the variance.**


```R
acf(nasdaq.data$Close)
```

### Bootstrapping `lm`

- Using WLS (weighted least squares) requires a model for the variance of $\epsilon$ given $X$.

- Ignoring this changing variance (heteroskedasticity) and using OLS leads to bad intervals, p-values, etc.
**because standard errors are incorrect.**

- The (pairs) bootstrap uses the OLS estimator but is able to get a **correct estimator of standard error**.


```R
library(car)
n = 50
X = rexp(n)
Y = 3 + 2.5 * X + X * (rexp(n) - 1) # our usual model is false here! W=X^{-2}
Y.lm = lm(Y ~ X)
pairs.Y.lm = Boot(Y.lm, coef, method='case', R=1000)
confint(pairs.Y.lm, type='norm') # using bootstrap SE
```

### Model selection criteria

* Mallow's $C_p$:
$$C_p({\cal M}) = \frac{SSE({\cal M})}{\widehat{\sigma}^2} + 2 \cdot p({\cal M}) - n.$$
* Akaike (AIC) defined as $$AIC({\cal M}) = - 2 \log L({\cal M}) + 2 p({\cal M})$$ where $L({\cal M})$ is the maximized likelihood of the model.
* Bayes (BIC) defined as $$BIC({\cal M}) = - 2 \log L({\cal M}) + \log n \cdot p({\cal M})$$
* Adjusted $R^2$
* Stepwise (`step`) vs. best subsets (`leaps`).
* **Beware of data snooping when reporting p-values, confidence intervals. Use data splitting!**

### $K$-fold cross-validation 

* Fix a model ${\cal M}$. Break data set into $K$ approximately equal sized groups $(G_1, \dots, G_K)$.
* for (i in 1:K)
   Use all groups except $G_i$ to fit model, predict outcome in group $G_i$ based on this model $\widehat{Y}_{j,{\cal M}, G_i}, j \in G_i$.
* Estimate $$CV({\cal M}) = \frac{1}{n}\sum_{i=1}^K \sum_{j \in G_i} (Y_j - \widehat{Y}_{j,{\cal M},-G_i})^2.$$

### Shrinkage estimator

* In one sample problem, when trying to estimate $\mu$ from $Y_i \sim N(\mu, \sigma^2)$ we looked at the estimator
$$
\hat{Y}_{\alpha} = \alpha \cdot \bar{Y}.
$$

* The "quality" of the estimator decomposed as
$$
E((\hat{Y}_{\alpha}-\mu)^2) = \text{Bias}(\hat{Y}_{\alpha})^2 + \text{Var}(\hat{Y}_{\alpha})
$$


```R
nsample = 40
ntrial = 500
mu = 0.5
sigma = 2.5
MSE = function(mu.hat, mu) {
  return(sum((mu.hat - mu)^2) / length(mu))
}

alpha = seq(0.0,1,length=20)

mse = numeric(length(alpha))

for (i in 1:ntrial) {
  Z = rnorm(nsample) * sigma + mu
  for (j in 1:length(alpha)) {
    mse[j] = mse[j] + MSE(alpha[j] * mean(Z) * rep(1, nsample), mu * rep(1, nsample)) / ntrial
  }
}
```


```R
plot(alpha, mse, type='l', lwd=2, col='red', ylim=c(0, max(mse)),
     xlab=expression(paste('Shrinkage parameter,', alpha)), 
     ylab=expression(paste('MSE(', alpha, ')')), 
     cex.lab=1.2)
```

## Ridge regression

$$
\hat{\beta}_{\lambda} = \text{argmin}_{\beta} \frac{1}{2n} \|Y-X\beta\|^2_2 + \lambda \|\beta\|_2^2
$$




```R
library(lars)
data(diabetes)
library(MASS)
diabetes.ridge <- lm.ridge(diabetes$y ~ diabetes$x, 
                           lambda=exp(seq(0,log(1e3),length=100)))
plot(diabetes.ridge, lwd=3)
```


```R
par(cex.lab=1.2)
plot(diabetes.ridge$lambda, diabetes.ridge$GCV, xlab='Lambda', ylab='GCV', type='l', lwd=3, col='orange')
select(diabetes.ridge)
```

## LASSO

$$
\hat{\beta}_{\lambda} = \text{argmin}_{\beta} \frac{1}{2n} \|Y-X\beta\|^2_2 + \lambda \|\beta\|_1$$


```R
library(lars)
data(diabetes)
diabetes.lasso = lars(diabetes$x, diabetes$y, type='lasso')
plot(diabetes.lasso)
```


```R
par(cex.lab=1.2)
cv.lars(diabetes$x, diabetes$y, K=10, type='lasso')
```

## LASSO with `glmnet`


```R
library(glmnet)
plot(glmnet(diabetes$x, diabetes$y))
```


```R
plot(cv.glmnet(diabetes$x, diabetes$y))
```

### Ridge with `glmnet`


```R
plot(glmnet(diabetes$x, diabetes$y, alpha=0))
```


```R
plot(cv.glmnet(diabetes$x, diabetes$y, alpha=0))
```

### Logistic regression model

* Logistic model $$E(Y|X) = \pi(X) = \frac{\exp(X^T\beta)}{1 + \exp(X^T\beta)}$$
* This automatically fixes $0 \leq E(Y) = \pi(X) \leq 1$.
* The logistic transform:
   $\text{logit}(\pi(X)) = \log\left(\frac{\pi(X)}{1 - \pi(X)}\right) = X^T\beta$
   
* An example of a *generalized linear model*
    - link function $\text{logit}(\pi(X)) = X^T\beta$
    - Variance function: $\text{Var}(Y|X) = \pi(X)(1 - \pi(X))$


### Odds Ratios

* One reason logistic models are popular is that the parameters have simple interpretations in terms of odds.
* Logistic model: $$OR_{X_j} = \frac{ODDS(Y=1|\dots, X_j=x_j+h, \dots)}{ODDS(Y=1|\dots, X_j=x_j, \dots)} = e^{h\beta_j}$$
* If $X_j \in {0, 1}$ is dichotomous, then odds for group with $X_j = 1$ are $e^{\beta_j}$ higher, other parameters being equal.

### Deviance

* For logistic regression model ${\cal M}$, $DEV({\cal M})$ replaces $SSE({\cal M})/\sigma^2$.
* In least squares regression, we use $$\frac{SSE({\cal M}_R) - SSE({\cal M}_F)}{\sigma^2} \sim  \chi^2_{df_R-df_F}$$
* This is replaced with $DEV({\cal M}_R) - DEV({\cal M}_F) \overset{n \rightarrow \infty}{\sim} \chi^2_{df_R-df_F}$
* For Poisson and binary regression, $\sigma^2=1$ (dispersion parameter of `glm`).

### Poisson log-linear regression model

* Log-linear model $$E(Y|X) = \exp(X^T\beta)$$
* This automatically fixes $ E(Y|X) \geq 0$.
   
* An example of a *generalized linear model*
    - link function $\text{log}(E(Y|X)) = X^T\beta$
    - Variance function: $\text{Var}(Y|X) = E(Y|X)$

* Interpretation:
$$
\frac{E(Y|\dots, X_j=x_j+h, \dots)}{E(Y|\dots, X_j=x_j, \dots)} = e^{h\beta_j}$$
