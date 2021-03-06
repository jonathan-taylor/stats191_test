
# Course Introduction and Review

## Outline

* What is a regression model?

* Descriptive statistics -- numerical

* Descriptive statistics -- graphical

* Inference about a population mean
  
* Difference between two population means


```R
options(repr.plot.width=5, repr.plot.height=3)
set.seed(0)
```

# What is course about?

* It is a course on applied statistics.

* Hands-on: we use [R](http://cran.r-project.org), an open-source statistics software environment.

* Course notes will be  [jupyter](http://jupyter.org) notebooks.

* We will start out with a review of introductory statistics to see `R` in action.
 
* Main topic is *(linear) regression models*: these are the *bread and butter* of applied statistics.



## What is a regression model? 


A regression model is a model of the relationships between some 
*covariates (predictors)* and an *outcome*.


Specifically, regression is a model of the *average* outcome *given or having fixed* the covariates. 
    
  

# Heights of mothers and daughters


      
* We will consider the [heights](http://www.stat.cmu.edu/~roeder/stat707/=data/=data/data/Rlibraries/alr3/html/heights.html) of mothers and daughters collected 
by Karl Pearson in the late 19th century.

* One of our goals is to understand height of the daughter, `D`, knowing the height of the
mother, `M`.


* A mathematical  model might look like
  $$
  D = f(M) + \varepsilon$$
  where $f$ gives the average height of the daughter
  of a mother of height `M` and
  $\varepsilon$ is *error*: not *every* daughter has the same height.

* A statistical question: is there *any*
relationship between covariates and outcomes -- is $f$ just a constant?



Let's create a plot of the heights of the mother/daughter pairs. The data is in an `R` package that can be downloaded
from [CRAN](http://cran.r-project.org/) with the command:

    install.packages("alr3")
    
If the package is not installed, then you will get an error message when calling `library(alr3)`.


```R
library(alr3)
data(heights)
M = heights$Mheight
D = heights$Dheight
plot(M, D, pch = 23, bg = "red", cex = 2)
```

In the first part of this course we'll talk about fitting a line to this data. Let's do that and remake the plot, including
this "best fitting line".


```R
plot(M, D, pch = 23, bg = "red", cex = 2)
height.lm = lm(D ~ M)
abline(height.lm, lwd = 3, col = "yellow")
```

# Linear regression model

* How do we find this line? With a model.

* We might model the data as
$$
D = \beta_0+ \beta_1 M + \varepsilon.
$$

* This model is *linear* in $(\beta_0, \beta_1)$, the intercept and the coefficient of  `M` (the mother's height), it is a 
*simple linear regression model*.

* Another model:
$$
D = \beta_0 + \beta_1 M + \beta_2 M^2  + \beta_3 F + \varepsilon
$$
where $F$ is the height of the daughter's father.

* Also linear (in $(\beta_0, \beta_1, \beta_2, \beta_3)$, the coefficients of  $1,M,M^2,F$).

* Which model is better? We will need a tool to compare models... more to come later.


# A more complex model

* Our example here was rather simple: we only had one independent variable.

* Independent variables are sometimes called *features* or *covariates*.

* In practice, we often have many more than one independent variable.

# Right-to-work

This example from the text considers the effect of right-to-work legislation (which varies by state) on various
factors. A [description](http://www.ilr.cornell.edu/~hadi/RABE4/Data4/P005.txt) of the data can be found here.

The variables are:

* Income: income for a four-person family

* COL: cost of living for a four-person family

* PD: Population density

* URate: rate of unionization in 1978

* Pop: Population

* Taxes: Property taxes in 1972

* RTWL: right-to-work indicator
   

In a study like this, there are many possible questions of interest. Our focus will be on the
relationship between `RTWL` and `Income`. However, we recognize that other variables
have an effect on `Income`. Let's look at some of these relationships.


```R
url = "http://www1.aucegypt.edu/faculty/hadi/RABE4/Data4/P005.txt"
rtw.table <- read.table(url, header=TRUE, sep='\t')
print(head(rtw.table))
```

A graphical way to 
visualize the relationship between `Income` and `RTWL`  is the *boxplot*.


```R
attach(rtw.table) # makes variables accessible in top namespace
boxplot(Income ~ RTWL, col='orange', pch=23, bg='red')
```

One variable that may have an important effect on the relationship between
 is the cost of living `COL`. It also varies between right-to-work states.


```R
boxplot(COL ~ RTWL, col='orange', pch=23, bg='red')
```

We may want to include more than one plot in a given display. The first line of the
code below achieves this.


```R
options(repr.plot.width=7, repr.plot.height=7)
```


```R
par(mfrow=c(2,2))
plot(URate, COL, pch=23, bg='red', main='COL vs URate')
plot(URate, Income, pch=23, bg='red')
plot(URate, Pop, pch=23, bg='red')
plot(COL, Income, pch=23, bg='red')
```

`R` has a builtin function that will try to display all pairwise relationships in a given dataset, the function `pairs`.


```R
pairs(rtw.table, pch=23, bg='red')
```

In looking at all the pairwise relationships. There is a point that stands out from all the rest.
This data point is New York City, the 27th row of the table. (Note that `R` uses 1-based instead of 0-based indexing for rows and columns of arrays.)


```R
print(rtw.table[27,])
pairs(rtw.table[-27,], pch=23, bg='red')
```


```R
options(repr.plot.width=5, repr.plot.height=3)
```

# Right-to-work example

## Building a model

Some of the main goals of this course:

* Build a statistical model describing the *effect* of `RTWL` on `Income`.

* This model should recognize that other variables also affect `Income`.

* What sort of *statistical confidence* do we have in our 
conclusion about `RTWL` and `Income`?

* Is the model adequate do describe this dataset?

* Are there other (simpler, more complicated) better models?
   


# Numerical descriptive statistics

## Mean of a sample

Given a sample of numbers $X=(X_1, \dots, X_n)$ the sample mean, 
$\overline{X}$ is
$$
\overline{X} = \frac1n \sum_{i=1}^n X_i.$$
   
There are many ways to compute this in `R`.


```R
X = c(1,3,5,7,8,12,19)
print(X)
print(mean(X))
print((X[1]+X[2]+X[3]+X[4]+X[5]+X[6]+X[7])/7)
print(sum(X)/length(X))
```

We'll also illustrate thes calculations with part of an example we consider below, on differences
in blood pressure between two groups.


```R
url = 'http://www.stanford.edu/class/stats191/data/Calcium.html' # from DASL
calcium.table = read.table(url, header=TRUE, skip=26, nrow=21)
attach(calcium.table)
treated = Decrease[(Treatment == 'Calcium')]
placebo = Decrease[(Treatment == 'Placebo')]
treated
mean(treated)
```

## Standard deviation of a sample

Given a sample of numbers $X=(X_1, \dots, X_n)$ the sample 
standard deviation $S_X$ is
$$
S^2_X = \frac{1}{n-1}  \sum_{i=1}^n (X_i-\overline{X})^2.$$


```R
S2 = sum((treated - mean(treated))^2) / (length(treated)-1)
print(sqrt(S2))
print(sd(treated))
```

## Median of a sample

   Given a sample of numbers $X=(X_1, \dots, X_n)$ the sample median is
   the `middle` of the sample:
   if $n$ is even, it is the average of the middle two points.
   If $n$ is odd, it is the midpoint.



```R
X
print(c(X, 13))
median(c(X, 13))
```

## Quantiles of a sample

Given a sample of numbers $X=(X_1, \dots, X_n)$ the  $q$-th quantile is
a point $x_q$ in the data such that $q \cdot 100\%$ of the data lie to the 
left of $x_q$.

### Example

The $0.5$-quantile is the median: half 
of the data lie to the right of the median.



```R
quantile(X, c(0.25, 0.75))
```

# Graphical statistical summaries

We've already seen a boxplot. Another common statistical summary is a 
histogram.


```R
hist(treated, main='Treated group', xlab='Decrease', col='orange')
```

# Inference about a population mean

## A testing scenario

* Suppose we want to determine the efficacy of a new drug on blood pressure.

* Our study design is: we will treat
a large patient population (maybe not so large: budget constraints limit it $n=20$) with the drug and measure their
blood pressure before and after taking the drug.

* We conclude that the drug is effective if the blood pressure has decreased on average. That is,
if the average difference between before and after is positive.



## Setting up the test



* The *null hypothesis*, <font color="red">$H_0$</font> is: <font color="red">*the average difference is less
than zero.*</font>

* The *alternative hypothesis*, <font color="green">$H_a$</font>, is: <font color="green">*the average difference 
is greater than zero.*</font>

* Sometimes (actually, often), people will test the alternative, <font color="green">$H_a$</font>: *the
average difference is not zero* vs. <font color="red">$H_0$</font>: *the average difference is zero.*

* The test is performed by estimating
the average difference and converting to standardized units.


## Drawing from a box

* Formally, could set up the above test as drawing from a box of *differences
in blood pressure*.

* A box model is a useful theoretical device that describes the experiment
under consideration. In our example, we can think of the sample of decreases
drawn 20 patients at random from a large population (box) containing all the possible
decreases in blood pressure.



## A simulated box model

* In our box model, we will assume that the decrease is an integer drawn at random from 
$-3$ to 6.

* We will draw 20 random integers from -3 to 6 with replacement and test whether the mean
of our "box" is 0 or not.


```R
mysample = sample(-3:6, 20, replace=TRUE)
mysample
```

The test is usually a $T$ test that uses the statistic
$$
   T = \frac{\overline{X}-0}{S_X/\sqrt{n}} 
    $$
    
The formula can be read in three parts:

- estimating the mean: $\overline{X}$;

- comparing to 0: subtracting 0 in the numerator;

- converting difference to standardized units: dividing by $S_X/\sqrt{n}$ our estimate of the variability of $\overline{X}$.


```R
T = (mean(mysample) - 0) / (sd(mysample) / sqrt(20))
T
```

This $T$ value is often compared to a table for the appropriate $T$ distribution (in this case there are 19 *degrees of freedom*) and the 5% cutoff is


```R
cutoff = qt(0.975, 19)
cutoff
```

Strictly speaking the $T$ distribution should be used when the values in the box
are spread similarly to a normal curve. This is not the case here, but if $n$ is large enough,
there is not a huge difference.


```R
qnorm(0.975)
```

The result of the two-sided test is


```R
reject = (abs(T) > cutoff)
reject
```

If `reject` is `TRUE`, then we reject $H_0$ the mean is 0 at a level of 5%, while if it is `FALSE` we do not reject. Of course, in this example we know the mean in our "box" is not 0, it is 1.5.


This rule can be visualized with the $T$ density. The total grey area is 0.05=5%, and the cutoff is chosen to be symmetric
around zero and such that this area is exactly 5%.

For a test of size $\alpha$ we write this cutoff $t_{n-1,1-\alpha/2}$.


```R
library(ggplot2)
alpha = 0.05
df = 19
xval = seq(-4,4,length=101)
q = qt(1-alpha/2, df)

rejection_region = function(dens, q_lower, q_upper, xval) {
    fig = (ggplot(data.frame(x=xval), aes(x)) +
        stat_function(fun=dens, geom='line') +
        stat_function(fun=function(x) {ifelse(x > q_upper | x < q_lower, dens(x), NA)},
                    geom='area', fill='#CC7777')  + 
        labs(y='Density', x='T') +
        theme_bw())
    return(fig)
}

T19_fig = rejection_region(function(x) { dt(x, df)}, -q, q, xval) + 
          annotate('text', x=2.5, y=dt(2,df)+0.3, label='Two sided rejection region, df=19')
```


```R
T19_fig
```

## Reasoning behind the test

Suppose $H_0$ was true -- say the mean of the box was zero.

For example, we might assume the difference is drawn at random from integers -5 to 5 inclusive.


```R
# Generate a sample from a box for which the null is true
null_sample = function(n) {
    return(sample(-5:5, n, replace=TRUE))
}

# Compute the T statistic
null_T = function(n) {
    cur_sample = null_sample(n) 
    return((mean(cur_sample) - 0) / (sd(cur_sample) / sqrt(n)))
}
```

## Type I error

When the null hypothesis is true, like in our simulation,
we expect that the $T$ statistic will exceed the cutoff only about 5% of the time.

If we use the cutoff $t_{19,0.975}$ to decide in favor or against $H_0$, rejecting
$H_0$ when the absolute value is larger than this value, then we have a test whose
**Type I error** is about 5%.

It is exactly 5% if the sample were drawn from a box whose values follow a normal curve...


```R
results = numeric(10000)
for (i in 1:10000) {
    results[i] = null_T(20)
}
mean(abs(results) >= qt(0.975, 19))
```

We use the $T$ curve (close to the normal curve) because when $H_0$
is true, the distribution of the T statistic is close to the $T$ curve


```R
plot(density(results), lwd=3)
xval = seq(-4,4,length=201)
lines(xval, dt(xval, 19), col='red', lwd=3) # T_19 density
lines(xval, dnorm(xval), col='blue', lwd=3) # Normal(0,1) density
```

`R` will compute this $T$ statistic for you, and many other things. `R` will use the $T$ distribution.


```R
t.test(mysample)
T
2 * pt(abs(T), 19, lower=FALSE)
```

As mentioned above, sometimes tests are one-sided. If the null hypothesis we tested was that the mean is less than 0, then we would reject this hypothesis if our observed mean was much larger than 0. This corresponds to a positive $T$ value.


```R
cutoff = qt(0.95, 19)
T19_pos = rejection_region(function(x) { dt(x, df)}, -Inf, cutoff, xval) + 
          annotate('text', x=2.5, y=dt(2,df)+0.3, label='One sided rejection region, df=19')

```


```R
T19_pos
```

The rejection rules are affected by the degrees of freedom. Here is the rejection region
when we only have 5 samples from our "box".


```R
df = 4
cutoff = qt(0.975, df)
T4_fig = rejection_region(function(x) { dt(x, df)}, -cutoff, cutoff, xval) + 
          annotate('text', x=2.5, y=dt(2,19)+0.3, label='Two sided rejection region, df=4')

```


```R
T4_fig
```

## Confidence intervals

* Instead of testing a particular hypothesis, we might be interested
in coming up with a reasonable range for the mean of our "box".

* Statistically, this is done via a *confidence interval*.

* If the 5% cutoff is $q$ for our test, then the 95% confidence interval is
$$
[\bar{X} - q S_X / \sqrt{n}, \bar{X} + q S_X / \sqrt{n}]
$$
where we recall $q=t_{n-1,0.975}$ with $n=20$. 

* If we wanted 90% confidence interval, we would use $q=t_{19,0.95}$. Why?


```R
cutoff = qt(0.975, 19)
L = mean(mysample) - cutoff*sd(mysample)/sqrt(20)
U = mean(mysample) + cutoff*sd(mysample)/sqrt(20)
data.frame(L, U) 
t.test(mysample)
```

Note that the endpoints above depend on the data. Not every interval will cover
the true mean of our "box" which is 1.5. Let's take a look at 100 intervals of size 90%. We would expect
that roughly 90 of them cover 1.5.


```R
cutoff = qt(0.975, 19)
L = c()
U = c()
covered = c()
box = -3:6
for (i in 1:100) {
   mysample = sample(box, 20, replace=TRUE)
   l = mean(mysample) - cutoff*sd(mysample)/sqrt(20)
   u = mean(mysample) + cutoff*sd(mysample)/sqrt(20)
   L = c(L, l)
   U = c(U, u)
   covered = c(covered, (l < mean(box)) * (u > mean(box)))
}
sum(covered)
```

A useful picture is to plot all these intervals so we can see the randomness
in the intervals, while the true mean of the box is unchanged.


```R
mu = 1.5
plot(c(1, 100), c(-2.5+mu,2.5+mu), type='n', ylab='Confidence Intervals', xlab='Sample')
for (i in 1:100) {
   if (covered[i] == TRUE) {
       lines(c(i,i), c(L[i],U[i]), col='green', lwd=2)
   }
   else {
      lines(c(i,i), c(L[i],U[i]), col='red', lwd=2)
   } 
}
abline(h=mu, lty=2, lwd=4)
```

# Blood pressure example

* A study was conducted to study the effect of calcium supplements
on blood pressure.



* We had loaded the data above, storing the two samples in the variables `treated` and `placebo`.

* Some questions might be:
    - What is the mean decrease in BP in the treated group? placebo group?
    - What is the median decrease in BP in the treated group? placebo group?
    -  What is the standard deviation of decrease in BP in the treated group? placebo group?
    - Is there a difference between the two groups? Did BP decrease more in the treated group?


```R
summary(treated)
summary(placebo)
boxplot(Decrease ~ Treatment, col='orange', pch=23, bg='red')
```

## A hypothesis test

In our setting, we have two groups that we have reason to believe are 
different.

* We have two samples:
   - $(X_1, \dots, X_{10})$ (`treated`)
   - $(Z_1, \dots, Z_{11})$ (`placebo`)
   
* We can answer this statistically by testing the null hypothesis 
$$H_0:\mu_X = \mu_Z.$$

* If variances are equal, the *pooled $t$-test* is appropriate.

## Pooled $t$ test

* The test statistic is $$ T = \frac{\overline{X} - \overline{Z} - 0}{S_P \sqrt{\frac{1}{10} + \frac{1}{11}}}, \qquad S^2_P = \frac{9 \cdot S^2_X + 10 \cdot S^2_Z}{19}.$$
   
*  For two-sided test at level $\alpha=0.05$, reject if $|T| > t_{19, 0.975}$.
   
* Confidence interval: for example, a $90\%$ confidence interval
for $\mu_X-\mu_Z$ is $$ \overline{X}-\overline{Z} \pm S_P \sqrt{\frac{1}{10} + \frac{1}{11}} \cdot  t_{19,0.95}.$$

* T statistic has the same form as before!


```R
sdP = sqrt((9*sd(treated)^2 + 10*sd(placebo)^2)/19)
T = (mean(treated)-mean(placebo)-0) / (sdP * sqrt(1/10+1/11))
c(T, cutoff)
```

`R` has a builtin function to perform such $t$-tests.


```R
t.test(treated, placebo, var.equal=TRUE)
```

If we don't make the assumption of equal variance, `R` will give a slightly different result.


```R
t.test(treated, placebo)
```

## Pooled estimate of variance

* The rule for the $SD$ of differences is
   $$
   SD(\overline{X}-\overline{Z}) = \sqrt{SD(\overline{X})^2+SD(\overline{Z})^2}$$
   
* By this rule, we might take our estimate to be
   $$
   \widehat{SD(\overline{X}-\overline{Z})} = \sqrt{\frac{S^2_X}{10} + \frac{S^2_Z}{11}}.
   $$
   
* The pooled estimate assumes $\mathbb{E}(S^2_X)=\mathbb{E}(S^2_Z)=\sigma^2$ and replaces
   the $S^2$'s above with $S^2_P$, a better estimate of
   $\sigma^2$ than either $S^2_X$ or $S^2_Z$.

## Where do we get $df=19$?

Well, the $X$  sample has $10-1=9$ degrees of freedom
   to estimate $\sigma^2$ while the $Z$  sample
   has $11-1=10$ degrees of freedom.
   
Therefore, the total degrees of freedom is $9+10=19$.

## Our first regression model

* We can put the two samples together:
   $$Y=(X_1,\dots, X_{10}, Z_1, \dots, Z_{11}).$$

*  Under the same assumptions as the pooled $t$-test:
   $$
   \begin{aligned}
   Y_i &\sim N(\mu_i, \sigma^2)\\
   \mu_i &=
   \begin{cases}
   \mu_X & 1 \leq i \leq 10 \\ \mu_Z & 11 \leq i \leq 21.
   \end{cases}
   \end{aligned}
   $$
   
* This is a (regression) model for the sample $Y$. The
   (qualitative) variable `Treatment` is
   called a *covariate* or *predictor*.
   
* The decrease in BP is the *outcome*.

* We assume that the relationship between treatment and average
   decrease in BP is simple: it depends only on which group a subject is in.
   
* This relationship is *modelled* through the mean
   vector $\mu=(\mu_1, \dots, \mu_{21})$.



```R
print(summary(lm(Decrease ~ Treatment)))
print(sdP*sqrt(1/10+1/11))
print(sdP)
```
