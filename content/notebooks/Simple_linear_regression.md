
# Simple linear regression 


```R
options(repr.plot.width=5, repr.plot.height=3)
set.seed(0)
```

The first type of model, which we will spend a lot of time on, is the *simple linear regresssion model*. One simple way to think of it
is via scatter plots. Below are [heights](http://www.stat.cmu.edu/~roeder/stat707/=data/=data/data/Rlibraries/alr3/html/heights.html) of mothers and daughters collected 
by Karl Pearson in the late 19th century. 


```R
library(alr3)
data(heights)
M = heights$Mheight
D = heights$Dheight
```


```R
library(ggplot2)
heights_fig = ggplot(heights, aes(Mheight, Dheight)) + geom_point() + theme_bw();
heights_fig

```

A simple linear regression model fits a line through the above scatter plot in a particular way. Specifically, it tries to estimate
the height of a new daughter in this population, say $D_{new}$, whose mother had height $H_{new}$. It does this by considering
each slice of the data. Here is a slice of the data near $M=66$, the slice is taken over a window of size 1 inch.


```R
X = 66
rect = data.frame(xmin=X-.5, xmax=X+.5, ymin=-Inf, ymax=Inf)
heights_fig + geom_rect(data=rect, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
                        color="grey20",
                        alpha=0.5,
                        inherit.aes = FALSE)
```


```R
selected_points = (M <= X+.5) & (M >= X-.5)
mean_within_slice = mean(D[selected_points])
mean_within_slice
```

We see that, in our sample, the average height of daughters whose height fell within our slice is about 65.2 inches. Of course this
height varies by slice. For instance, at 60 inches:


```R
X = 60
selected_points = (M <= X+.5) & (M >= X-.5)
mean_within_slice = mean(D[selected_points])
mean_within_slice
```


```R
X = 60
rect = data.frame(xmin=X-.5, xmax=X+.5, ymin=-Inf, ymax=Inf)
heights_fig + geom_rect(data=rect, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
                        color="grey20",
                        alpha=0.5,
                        inherit.aes = FALSE)
```

The regression model puts a line through this scatter plot in an *optimal* fashion.

To do this, it assumes that the mean in slice `M` lies on some line
$$
\beta_0+\beta_1 M.
$$

It then chooses $(\beta_0, \beta_1)$ based on the data.



```R
parameters = lm(D ~ M)$coef
print(parameters)
intercept = parameters[1]
slope = parameters[2]
```


```R
heights_fig + geom_abline(slope=slope, intercept=intercept, color='red', size=3)

```

### What is a "regression" model?

A regression model is a model of the relationships between some covariates (predictors) and an outcome. Specifically, regression is a model of the average outcome given the covariates.

### Mathematical formulation

For height of couples data: a mathematical model:
$$
{\tt Daughter} = f({\tt Mother}) + \varepsilon
$$
where $f$ gives the average height of the daughter of a mother of height Mother and $\varepsilon$ is the random variation within the slice.

### Linear regression models

* A *linear* regression model says that
the function $f$ is a sum (linear combination) of functions of ${\tt Mother}$.

* Simple linear regression model:
   $$f({\tt Mother}) = \beta_0 + \beta_1 \cdot {\tt Mother}$$
   for some unknown parameter vector $(\beta_0, \beta_1)$.

* Could also be a sum (linear combination) of *fixed* functions of `Mother`:
   $$f({\tt Mother}) = \beta_0 + \beta_1 \cdot {\tt Mother} + \beta_2 \cdot {\tt Mother}^2
   $$


### Simple linear regression model

 *  *Simple linear* regression is the case when there is only one predictor:
   $$
   f({\tt Mother}) = \beta_0 + \beta_1  \cdot {\tt Mother}.$$

* Let $Y_i$ be the height of the $i$-th daughter in the sample, $X_i$ be the height of the $i$-th mother.

* Model:
   $$
   Y_i = \underbrace{\beta_0 + \beta_1 X_i}_{\text{regression equation}} + \underbrace{\varepsilon_i}_{\text{error}}$$
   where $\varepsilon_i \sim N(0, \sigma^2)$ are independent.

* This specifies a *distribution* for the $Y$'s given the $X$'s, i.e.
   it is a *statistical model*.


### Fitting the model

* We will be using *least squares* regression. This measures
   the *goodness of fit* of a line by the sum of squared errors, $SSE$.
   
* Least squares regression chooses the line that minimizes
   $$
   SSE(\beta_0, \beta_1) = \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 \cdot X_i)^2.$$

* In principle, we might measure goodness of fit differently: 
   $$
   SAD(\beta_0, \beta_1) = \sum_{i=1}^n |Y_i - \beta_0 - \beta_1 \cdot X_i|.$$
   
* For some *loss function* $L$ we might try to minimize
    $$
    L(\beta_0,\beta_1) = \sum_{i=1}^n L(Y_i-\beta_0-\beta_1X_i) 
    $$


### Why least squares?

* With least squares, the minimizers have explicit formulae -- not so important with today's computer power -- especially when $L$ is convex.

* Resulting formulae are *linear* in the outcome $Y$. This is important
   for inferential reasons. For only predictive power, this is also not so important.
   
* If assumptions are correct, then this is *maximum likelihood estimation*.

* Statistical theory tells us the *maximum likelihood estimators (MLEs)* are generally good estimators.



### Choice of loss function

The choice of the function we use to measure goodness of fit, or the *loss* function, has an outcome on what
sort of estimates we get out of our procedure. For instance, if, instead of fitting a line to a scatterplot, we were
estimating a *center* of a distribution, which we denote by $\mu$, then we might consider minimizing several loss functions.

### Choice of loss function

* If we choose the sum of squared errors:
$$
SSE(\mu) = \sum_{i=1}^n (Y_i - \mu)^2.
$$
Then, we know that the minimizer of $SSE(\mu)$ is the sample mean.

* On the other hand, if we choose the sum of the absolute errors
 $$
   SAD(\mu) = \sum_{i=1}^n |Y_i - \mu|.$$
   Then, the resulting minimizer is the sample median.
   
* Both of these minimization problems also have *population* versions as well. For instance, the population mean
minimizes, as a function of $\mu$
$$
\mathbb{E}((Y-\mu)^2)
$$
while the population median minimizes
$$
\mathbb{E}(|Y-\mu|).
$$

### Visualizing the loss function

Let's take some a random scatter plot and view the loss function.



```R
X = rnorm(50)
Y = 1.5 + 0.1 * X + rnorm(50) * 2
parameters = lm(Y ~ X)$coef
intercept = parameters[1]
slope = parameters[2]
ggplot(data.frame(X, Y), aes(X, Y)) + geom_point() + geom_abline(slope=slope, intercept=intercept)
```

Let's plot the *loss* as a function of the parameters. Note that the
*true* intercept is 1.5 while the *true* slope is 0.1.


```R
grid_intercept = seq(intercept - 2, intercept + 2, length=100)
grid_slope = seq(slope - 2, slope + 2, length=100)
loss_data = expand.grid(intercept_=grid_intercept, slope_=grid_slope)
loss_data$squared_error = numeric(nrow(loss_data))
for (i in 1:nrow(loss_data)) {
    loss_data$squared_error[i] = sum((Y - X * loss_data$slope_[i] - loss_data$intercept_[i])^2)
}
squared_error_fig = (ggplot(loss_data, aes(intercept_, slope_, fill=squared_error)) + 
                     geom_raster() +
                     scale_fill_gradientn(colours=c("gray","yellow","blue")))
squared_error_fig
```

Let's contrast this with the sum of absolute errors.


```R
loss_data$absolute_error = numeric(nrow(loss_data))
for (i in 1:nrow(loss_data)) {
    loss_data$absolute_error[i] = sum(abs(Y - X * loss_data$slope_[i] - loss_data$intercept_[i]))
}
absolute_error_fig = (ggplot(loss_data, aes(intercept_, slope_, fill=absolute_error)) + 
                      geom_raster() +
                      scale_fill_gradientn(colours=c("gray","yellow","blue")))
absolute_error_fig
```

### Geometry of least squares

The following picture will be with us, in various guises, throughout much of the course. It depicts
the geometric picture involved in least squares regression.

<img src="http://stats191.stanford.edu/figs/axes_simple.svg" width="600">

It requires some imagination but the picture should be thought as representing vectors in $n$-dimensional space, l where $n$ is the number of points in the scatterplot. In our height data, $n=1375$. The bottom two axes should be thought of as 2-dimensional, while the axis marked "$\perp$" should be thought of as $(n-2)$ dimensional, or, 1373 in this case.



## Important lengths

The (squared) lengths of the above vectors are important quantities in what follows.

There are three to note:
$$
\begin{aligned}
   SSE &= \sum_{i=1}^n(Y_i - \widehat{Y}_i)^2 = \sum_{i=1}^n (Y_i - \widehat{\beta}_0 - \widehat{\beta}_1 X_i)^2 \\
   SSR &= \sum_{i=1}^n(\overline{Y} - \widehat{Y}_i)^2 = \sum_{i=1}^n (\overline{Y} - \widehat{\beta}_0 - \widehat{\beta}_1 X_i)^2 \\
   SST &= \sum_{i=1}^n(Y_i - \overline{Y})^2 = SSE + SSR \\
   R^2 &= \frac{SSR}{SST} = 1 - \frac{SSE}{SST} = \widehat{Cor}(\pmb{X},\pmb{Y})^2.
   \end{aligned}
$$

## Important lengths


An important summary of the fit is the ratio
$$
R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}
$$
which measures *how much variability in $Y$* is explained by $X$.

## Example: wages vs. education


In this example, we'll look at the output of *lm* for the wage
data and verify that some of the equations we present for the 
least squares solutions agree with the output.
The data was compiled from a study in econometrics [Learning about Heterogeneity in Returns to Schooling]( http://www.econ.queensu.ca/jae/2004-v19.7/koop-tobias/readme.kt.txt).



```R
url = 'http://www.stanford.edu/class/stats191/data/wage.csv'
wages = read.table(url, sep=',', header=TRUE)
print(head(wages))
```

In order to access the variables in `wages` we `attach` it so that the variables
are in the toplevel namespace.


```R
attach(wages)
mean(logwage)
```

Let's fit the linear regression model.


```R
wages.lm = lm(logwage ~ education)
print(wages.lm)
```

As in the mother-daughter data, we might want to plot the data and add the regression line.

Earlier, we used `ggplot2`, below we use base `R` instead. Typically `ggplot2` will
be more attractive, though its result are sometimes a little difficult to tweak (in my limited experience).


```R
logwage_fig = (ggplot(wages, aes(education, logwage)) + geom_point() + theme_bw() +
               geom_abline(slope=wages.lm$coef[2], 
                           intercept=wages.lm$coef[1], 
                           color='red', 
                           size=3))
logwage_fig

```

### Least squares estimators

There are explicit formulae for the least squares estimators, i.e. the minimizers of the error sum of squares.

For the slope, $\hat{\beta}_1$, it can be shown that 
$$
   \widehat{\beta}_1 = \frac{\sum_{i=1}^n(X_i - \overline{X})(Y_i - \overline{Y}
)}{\sum_{i=1}^n (X_i-\overline{X})^2} = \frac{\widehat{Cov}(X,Y)}{\widehat{Var}(
X)}.$$

Knowing the slope estimate, the intercept estimate can be found easily:
$$ \widehat{\beta}_0 = \overline{Y} - \widehat{\beta}_1 \cdot \overline{
X}.$$



#### Wages example


```R
beta.1.hat = cov(education, logwage) / var(education)
beta.0.hat = mean(logwage) - beta.1.hat * mean(education)
print(c(beta.0.hat, beta.1.hat))
print(coef(wages.lm))
```

### Estimate of $\sigma^2$

There is one final quantity needed to estimate all of our parameters in our (statistical) model for the scatterplot. This is $\sigma^2$,
the variance of the random variation within each slice (the regression model assumes this variance is constant within each slice...).

The estimate most commonly used is
$$
\hat{\sigma}^2 = \frac{1}{n-2} \sum_{i=1}^n (Y_i - \hat{\beta}_0 - \hat{\beta}_1 X_i)^2 = \frac{SSE}{n-2} = MSE
$$

Above, note the practice of replacing the quantity $SSE(\hat{\beta}_0,\hat{\beta}_1)$, i.e. the minimum of this function, with just $SSE$.

The term *MSE* above refers to mean squared error: a sum of squares divided by what we call its *degrees of freedom*. The degrees of freedom
of *SSE*, the *error sum of squares* is therefore $n-2$. Remember this $n-2$ corresponded to $\perp$ in the picture above...

Using some statistical calculations that we will not dwell on, if our simple linear regression model is correct, then we can see that
$$
\frac{\hat{\sigma}^2}{\sigma^2} \sim \frac{\chi^2_{n-2}}{n-2}
$$
where the right hand side denotes a *chi-squared* distribution with $n-2$ degrees of freedom.

(Note: our estimate of $\sigma^2$ *is not* the maximum likelihood estimate.)

### Wages example


```R
sigma.hat = sqrt(sum(resid(wages.lm)^2) / wages.lm$df.resid)
c(sigma.hat, sqrt(sum((logwage - predict(wages.lm))^2) / wages.lm$df.resid))
```

The summary from *R* also contains this estimate of $\sigma$:


```R
summary(wages.lm)
```

# Inference for the simple linear regression model

### What do we mean by inference?

* Generally, by inference, we mean "learning something about
   the relationship between the sample $(X_1, \dots, X_n)$ and $(Y_1, \dots, Y_n)$."

* In the simple linear regression model, this often means learning about $\beta_0, \beta_1$.
Particular forms of inference are **confidence intervals** or **hypothesis tests**. More on these later.

* Most of the questions of *inference* in this course
   can be answered in terms of $t$-statistics or $F$-statistics.

* First we will talk about $t$-statistics, later $F$-statistics.

### Examples of (statistical) hypotheses

* [One sample problem:](http://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test) given an independent sample $\pmb{X}=(X_1, \dots, X_n)$ where $X_i\sim N(\mu,\sigma^2)$, the *null hypothesis $H_0:\mu=\mu_0$*  says that in fact the population mean is some specified value $\mu_0$.

* [Two sample problem:](http://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test) given two independent samples $\pmb{Z}=(Z_1, \dots, Z_n)$, $\pmb{W}=(W_1, \dots, W_m)$  where $Z_i\sim N(\mu_1,\sigma^2)$ and $W_i \sim N(\mu_2, \sigma^2)$, the *null hypothesis $H_0:\mu_1=\mu_2$* says that in fact the population means from which the two samples are drawn are identical.

### Testing a hypothesis

We test a null hypothesis, $H_0$ based on some test statistic $T$ whose distribution is fully known when $H_0$ is true.

For example, in the one-sample problem, if $\bar{X}$ is the sample mean of our sample $(X_1, \dots, X_n)$ and
$$
S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i-\bar{X})^2
$$
is the sample variance. Then
$$
T = \frac{\bar{X}-\mu_0}{S/\sqrt{n}}
$$
has what is called a [Student's t](http://en.wikipedia.org/wiki/Student's_t-distribution) distribution with $n-1$ degrees of freedom *when $H_0:\mu=\mu_0$ is true.* 

**When the null
hypothesis is not true, it does not have this distribution!**

### General form of a (Student's) $T$ statistic

* A $t$ statistic with $k$ degrees of freedom, has a form that becomes easy to recognize after seeing it several times. 

* It has two main parts: a numerator and a denominator. The numerator $Z \sim N(0,1)$ while
$D \sim \sqrt{\chi^2_k/k}$ that is assumed *independent* of $Z$.

* The $t$-statistic has the form
$$
T = \frac{Z}{D}.
$$

### General form of a (Student's) $T$ statistic

* Another form of the $t$-statistic is
$$
T = \frac{\text{estimate of parameter} - \text{true parameter}}{\text{accuracy of the estimate}}.
$$

* In more formal terms, we write this as
$$
T = \frac{\hat{\theta} - \theta}{SE(\hat{\theta})}.
$$
Note that the denominator is the accuracy of the *estimate* and not the "accuracy" of the true parameter (which is usually assumed fixed, though not for Bayesians).

- The term $SE$ or *standard error* will, in this course, usually refer to an estimate of the accuracy of estimator. Therefore, it is the square root of an estimate of the variance of an estimator.

### General form of a (Student's) $T$ statistic


* In our simple linear regression model, a natural (**unobservable**) $t$-statistic is
$$
\frac{\hat{\beta}_1 - \beta_1}{SE(\hat{\beta}_1)}.
$$
We've seen how to compute $\hat{\beta}_1$, we never get to see the true $\beta_1$, so the only quantity we have anything left to say about is the standard error $SE(\hat{\beta}_1)$. 

* How many degrees of freedom would this $T$ have?

### Comparison of Student's $t$ to normal distribution

As the degrees of freedom increases, the population histogram, or density, of the $T_k$ distribution looks more and more
like the standard normal distribution usually denoted by $N(0,1)$.


```R
rejection_region = function(dens, q_lower, q_upper, xval) {
    fig = (ggplot(data.frame(x=xval), aes(x)) +
        stat_function(fun=dens, geom='line') +
        stat_function(fun=function(x) {ifelse(x > q_upper | x < q_lower, dens(x), NA)},
                      geom='area', fill='#CC7777') + 
            labs(y='Density', x='T') +
        theme_bw())
}

xval = seq(-4,4,length=101)
q = qnorm(0.975)
Z_fig = rejection_region(dnorm, -q, q, xval) + 
          annotate('text', x=2.5, y=dnorm(2)+0.3, label='Z statistic',
                  color='#CC7777')


```

This change in the density has an effect on the *rejection rule* for hypothesis tests based on the $T_k$ distribution.
For instance, for the standard normal, the 5% rejection rule is to reject if the so-called $Z$-score is larger than about 2 in absolute value.


```R
Z_fig
```

For the $T_{10}$ distribution, however, this rule must be modified.


```R
q10 = qt(0.975, 10)
T_fig = (Z_fig + stat_function(fun=function(x) {ifelse(x > q10 | x < -q10, dt(x, 10), NA)},
                      geom='area', fill='#7777CC', alpha=0.5) +
         stat_function(fun=function(x) {dt(x, 10)}, color='blue') + 
         annotate('text', x=2.5, y=dnorm(2)+0.27, label='T statistic, df=10',
                  color='#7777CC')
        );

```


```R
T_fig
```

### One sample problem revisited

Above, we used the one sample problem as an example of a $t$-statistic. Let's be a little more specific.

* Given an independent sample $\pmb{X}=(X_1, \dots, X_n)$ where $X_i\sim N(\mu,\sigma^2)$ we can test $H_0:\mu=0$ using a $T$-statistic.

* We can prove that the random variables
   $$\overline{X} \sim N(\mu, \sigma^2/n), \qquad \frac{S^2_X}{\sigma^2} \sim \frac{\chi^2_{n-1}}{n-1}$$
   are independent.

* Therefore, whatever the true $\mu$ is
   $$
   \frac{\overline{X} - \mu}{S_X / \sqrt{n}} = \frac{ (\overline{X}-\mu) / (\sigma/\sqrt{n})}{S_X / \sigma} \sim t_{n-1}.$$
  
* Our null hypothesis specifies a particular value for $\mu$, i.e. 0. Therefore, under $H_0:\mu=0$ (i.e. assuming that $H_0$ is true), $$\overline{X}/(S_X/\sqrt{n}) \sim t_{n-1}.$$


### Confidence interval

The following are examples of confidence intervals we saw in our review.

* One sample problem: instead of deciding whether $\mu=0$, we might want 
to come up with an (random) interval $[L,U]$ based on the sample $\pmb{X}$ such 
that the probability
   the true (nonrandom) $\mu$ is contained in $[L,U]$ equal to $1-\alpha$, i.e. 
95%.

*  Two sample problem: find a (random) interval $[L,U]$ based on the sampl
es $\pmb{Z}$ and $\pmb{W}$ such that
   the probability the true (nonrandom) $\mu_1-\mu_2$ is contained in $[L,U]$ is
 equal to $1-\alpha$, i.e. 95%.

### Confidence interval for one sample problem

* In the one sample problem, we might be interested in a confidence interval for the unknown $\mu$.

* Given an independent sample $(X_1, \dots, X_n)$ where 
   $X_i\sim N(\mu,\sigma^2)$ we can test construct 
   a $(1-\alpha)*100\%$ using the
   numerator and denominator of the $t$-statistic.
 

### Confidence interval for one sample problem

*   Let $q=t_{n-1,(1-\alpha/2)}$

   $$
   \begin{aligned}
   1 - \alpha &= P_{\mu}\left(-q \leq \frac{\mu - \overline{X}}
   {S_X / \sqrt{n}} \leq q \right) \\
   &= P_{\mu}\left(-q \cdot {S_X / \sqrt{n}} \leq {\mu - \overline{X}} 
   \leq q  \cdot {S_X / \sqrt{n}} \right) \\
   &= P_{\mu}\left(\overline{X} - q  \cdot {S_X / \sqrt{n}} 
   \leq {\mu} \leq \overline{X} + q  \cdot {S_X / \sqrt{n}} \right) \\
   \end{aligned}
   $$
   
* Therefore, the interval $\overline{X} \pm q \cdot {S_X / \sqrt{n}}$ is a $(1-\alpha)*100\%$ confidence interval for $\mu$.

### Inference for $\beta_0$ or $\beta_1$

* Recall our model $$
   Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i,$$
   errors $\varepsilon_i$ are independent $N(0, \sigma^2)$.
   
* In our heights example, we might want to now if there
   really is a linear association between ${\tt Daughter}=Y$
   and ${\tt Mother}=X$. This can be answered with a *hypothesis test* of the null hypothesis $H_0:\beta_1=0$.
   This assumes the model above is correct, but that $\beta_1=0$.
   
* Alternatively, we might want to have a range of values that we can be fairly certain $\beta_1$ lies within.
This is a *confidence interval* for $\beta_1$.

### Geometric picture of model

The hypothesis test has a geometric interpretation which we will revisit later for other models.
It is a comparison of two models. The first model is our original model.

<img src="http://stats191.stanford.edu/figs/axes_simple_full.svg" width="600">

### Setup for inference

* Let $L$ be the subspace of $\mathbb{R}^n$ spanned $\pmb{1}=(1, \dots, 1)$ and ${X}=(X_1, \dots, X\
_n)$.

* Then,
   $${Y} = P_L{Y} + ({Y} - P_L{Y}) = \widehat{{Y}} + (Y - \widehat{{Y}}) = \widehat{{Y}} + e$$

* In our model $\mu=\beta_0 \pmb{1} + \beta_1 {X} \in L$ so that
   $$
   \widehat{{Y}} = \mu + P_L{\varepsilon}, \qquad {e} = P_{L^{\perp}}{{Y}} = P_{L^{\perp}}{\varepsilon}$$
 
* Our assumption that $\varepsilon_i$'s are independent $N(0,\sigma^2)$ tells us that: ${e}$ and $\widehat{{Y}}$ are independent; $\widehat{\sigma}^2 = \|{e}\|^2 / (n-2) \sim \sigma^2 \cdot \chi^2_{n-2} / (n-2)$.



### Setup for inference

* In turn, this implies
$$
   \widehat{\beta}_1 \sim N\left(\beta_1, \frac{\sigma^2}{\sum_{i=1}^n(X_i-\overline{X})^2}\right).$$

* Therefore, $$\frac{\widehat{\beta}_1 - \beta_1}{\sigma \sqrt{\frac{1}{\sum_{i=1}^n(X_i-\overline{X})^2}}} \sim N(\
0,1).$$

* The other quantity we need is the *standard error* or SE of $\hat{\beta}_1$. This is
obtained from estimating the variance of $\widehat{\beta}_1$, which, in this case means simply
plugging in our estimate of $\sigma$, yielding
$$
   SE(\widehat{\beta}_1) = \widehat{\sigma} \sqrt{\frac{1}{\sum_{i=1}^n(X_i-\overline{X})^2}} \qquad 
   \text{independent of $\widehat{\beta}_1$}$$


### Testing $H_0:\beta_1=\beta_1^0$

* Suppose we want to test that $\beta_1$ is some pre-specified
   value, $\beta_1^0$ (this is often 0: i.e. is there a linear association)

* Under $H_0:\beta_1=\beta_1^0$
   $$\frac{\widehat{\beta}_1 - \beta^0_1}{\widehat{\sigma} \sqrt{\frac{1}{\sum_{i=1}^n(X_i-\overline{X})^2}}}
   = \frac{\widehat{\beta}_1 - \beta^0_1}{ \frac{\widehat{\sigma}}{\sigma}\cdot \sigma \sqrt{\frac{1}{
\sum_{i=1}^n(X_i-\overline{X})^2}}} \sim t_{n-2}.$$


* Reject $H_0:\beta_1=\beta_1^0$ if $|T| > t_{n-2, 1-\alpha/2}$.
   

#### Wage example

Let's perform this test for the wage data.


```R
SE.beta.1.hat = (sigma.hat * sqrt(1 / sum((education - mean(education))^2)))
Tstat = (beta.1.hat - 0) / SE.beta.1.hat
data.frame(beta.1.hat, SE.beta.1.hat, Tstat)

```

Let's look at the output of the `lm` function again.


```R
summary(wages.lm)
```

We see that *R* performs this test in the second row of the `Coefficients` table. It is clear that
wages are correlated with education.

### Why reject for large |T|?

* Observing a large $|T|$ is unlikely if $\beta_1 = \beta_1^0$: reasonable to conclude that $H_0$ 
is false.

* Common to report $p$-value:
$$\mathbb{P}(|T_{n-2}| > |T|_{obs}) = 2 \mathbb{P} (T_{n-2} > |T_{obs}|)$$


```R
2*(1 - pt(Tstat, wages.lm$df.resid))
```


```R
detach(wages)
```

### Confidence interval based on Student's $t$ distribution

*   Suppose we have a parameter estimate $\widehat{\theta} \sim N(\theta, {\sigma}_{\theta}^2)$, and standard error $SE(\widehat{\theta})$ such that
   $$
   \frac{\widehat{\theta}-\theta}{SE(\widehat{\theta})} \sim t_{\nu}.$$

* We can find a $(1-\alpha) \cdot 100 \%$ confidence interval by:
   $$
   \widehat{\theta} \pm SE(\widehat{\theta}) \cdot t_{\nu, 1-\alpha/2}.$$
   
* To prove this, expand the absolute value as we did for the one-sample CI
   $$
   1 - \alpha = \mathbb{P}_{\theta}\left(\left|\frac{\widehat{\theta} - \theta}{SE(\widehat{\theta})} \right| < t_{\nu, 1-\alpha/2}\right).$$

### Confidence interval for regression parameters

* Applying the above to the parameter $\beta_1$ yields a confidence interval of the form
$$
   \hat{\beta}_1 \pm SE(\hat{\beta}_1) \cdot t_{n-2, 1-\alpha/2}.$$
   
* We will need to compute $SE(\hat{\beta}_1)$. This can be computed using this formula
   $$
   SE(a_0\hat{\beta}_0 + a_1\hat{\beta}_1) = \hat{\sigma} \sqrt{\frac{a_0^2}{n} + \frac{(a_0\overline{X} - a_1)^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}$$
with $(a_0,a_1) = (0, 1)$.




We also need to find the quantity $t_{n-2,1-\alpha/2}$. This is defined by
$$
\mathbb{P}(T_{n-2} \geq t_{n-2,1-\alpha/2}) = \alpha/2.
$$
In *R*, this is computed by the function `qt`.



```R
alpha = 0.05
n = length(M)
qt(1-0.5*alpha, n-2)
```

Not surprisingly, this is close to that of the normal distribution, which is a Student's $t$ with $\infty$ for degrees of freedom.


```R
qnorm(1 - 0.5*alpha)
```

We will not need to use these explicit formulae all the time, as *R* has some built in functions
to compute confidence intervals.


```R
L = beta.1.hat - qt(0.975, wages.lm$df.resid) * SE.beta.1.hat
U = beta.1.hat + qt(0.975, wages.lm$df.resid) * SE.beta.1.hat
data.frame(L, U)
```


```R
confint(wages.lm)
```

### Predicting the mean

Once we have estimated a slope $(\hat{\beta}_1)$ and an intercept $(\hat{\beta}_0)$, we can predict the height
of the daughter born to a mother of any particular height by the plugging-in the height of the new mother, $M_{new}$ into
our regression equation:
$$
E[{D}_{new}] = {\beta}_0  +{\beta}_1 M_{new}.
$$
This equation says that our best guess at the height of the new daughter born to a mother of height $M_{new}$ is $\hat{D}_{new}$. 
Does this say that the height will be *exactly* this value? No, there is some random variation in each slice, and we would expect the same random variation for this new daughter's height as well.

 We might also want a confidence interval for the average height of daughters born to a mother of height $M_{new}=66$ inches:
$$
\hat{\beta}_0 + 66 \cdot \hat{\beta}_1 \pm SE(\hat{\beta}_0 + 66 \cdot \hat{\beta}_1) \cdot t_{n-2, 1-\alpha/2}.
$$

Recall that the parameter of interest is the average within the slice. Let's look at our picture again:


```R
height.lm = lm(D~M)
predict(height.lm, list(M=c(66, 60)), interval='confidence', level=0.90)
heights_fig
```

## Computing $SE(\hat{\beta}_0 + 66 \hat{\beta}_1)$

- We use the previous formula
  $$
   SE(a_0\hat{\beta}_0 + a_1\hat{\beta}_1) = \hat{\sigma} \sqrt{\frac{a_0^2}{n} + \frac{(a_0\overline{X} - a_1)^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}$$
   with $(a_0, a_1) = (1, 66)$.
   
- In particular,
$$
 SE(\hat{\beta}_0 + 66 \hat{\beta}_1) = \hat{\sigma} \sqrt{\frac{1}{n} + \frac{(\overline{X} - 66)^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}
 $$
  
- As $n$ grows, $SE(\hat{\beta}_0 + 66 \hat{\beta}_1)$ should shrink to 0. Why?

### Forecasting intervals

There is yet another type of interval we might consider: can we find an interval that covers the height of a 
particular daughter knowing only that her mother's height as 66 inches?

This interval has to cover the variability of the new random variation with our slice at 66 inches. So, it must be at least
as wide as $\sigma$, and we estimate its width to be at least as wide as $\hat{\sigma}$. 


```R
X = 66
selected_points = (M <= X+.5) & (M >= X-.5)
center = mean(D[selected_points])
sd_ = sd(D[selected_points])
L = center - qnorm(0.95) * sd_
U = center + qnorm(0.95) * sd_
data.frame(center, L, U)
```


```R
predict(height.lm, list(M=66), interval='prediction', level=0.90)
```


```R
(69.41-61.94)
```

With so much 
data in our heights example, this 90% interval will have width roughly `2 * qnorm(0.95) * sigma.hat.height`.


```R
sigma.hat.height = sqrt(sum(resid(height.lm)^2) / height.lm$df.resid)
2 * qnorm(0.95) * sigma.hat.height
```

The actual width will depend on how accurately we have estimated $(\beta_0, \beta_1)$ as well as $\hat{\sigma}$. Here is the
full formula. Again it is based on the $t$ distribution, the only thing we need to change is what we use for the SE.

$$
SE(\widehat{\beta}_0 + \widehat{\beta}_1 66 + \varepsilon_{\text{new}}) = \widehat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(\overline{X} - 66)^2}{\sum_{i=1}^n \left(X_i-\overline{X}\right)^2}}.
$$

The final interval is
$$ \hat{\beta}_0 +  \hat{\beta}_1 66 \pm t_{n-2, 1-\alpha/2} \cdot SE(\hat{\beta}_0 + \hat{\beta}_1 66 + \varepsilon_{\text{new}}).
   $$
