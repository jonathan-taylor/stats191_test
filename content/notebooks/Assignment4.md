

**You may discuss homework problems with other students, but you have to prepare the written assignments yourself.**

**Please combine all your answers, the computer code and the figures into one file.**

**Grading scheme: 10 points per question, total of 50.**
 
**Due date: March 6, 2019, 11:59PM.**

# Question 1


The data set [http://stats191.stanford.edu/data/asthma.table](http://stats191.stanford.edu/data/asthma.table) contains data on the number of admittances, `Y` to an emergency room for asthma-related problems in a hospital for several days. On each day, researchers also recorded the daily high temperature `T`, and the level of some atmospheric pollutants `P`.

1. Fit a linear regression model to the observed counts as a linear function of `T` and `P`.

2. Looking at the usual diagnostics plots, does the constant variance assumption seem justified?

3. The outcomes are counts, for which a common model is the so-called Poisson model which says that
$\text{Var}(Y) = E(Y)$. In words,  this says that the variance of the outcome is equal to the expected value of the outcome. Using a two-stage procedure, fit a weighted least squares regression to `Y` as a function of `T` and `P` with weights being inversely proportional to the fitted values of the initial model in 1.

4. Looking at the usual diagnostics plots of this model (which takes the weights into account), does the constant variance assumption seem more reasonable? (The change may not be astonishing -- the point of the problem is to try using weighted least squares.)

5. Using the weighted least squares fit, test the hypotheses at level $\alpha = 0.05$ that 

    * the number of asthma cases is uncorrelated to the temperature allowing for pollutants;
    * the number of asthma cases is uncorrelated to the atmospheric pollutants allowing for temperature. 
   
   

# Question 2 (Based on RABE 8.4-8.6)

The file 

    http://www1.aucegypt.edu/faculty/hadi/RABE5/Data5/P229-30.txt

contains the values of the daily DJIA (Dow Jones Industrial Average)
for all the trading days in 1996. The variable `Time` denotes the trading day of the year. There were 262 trading days in 1996.

1. Fit a linear regression model connecting `DJIA` with `Time` using all 262 trading days in 1996. Is the linear trend model adequate? Examine the residuals for time dependencies, including a plot of the autocorrelation function.

2. Regress `DJIA[t]` against its lagged by one version `DJIA[t-1]`. Is this an adequate model? Are there any evidences of autocorrelation in the residuals?

3. The variability (volatility) of the daily DJIA is large, and to accomodate this phenomenon the analysis is crried out on the logarithm of the DJIA. Repeat 2. above using  `log(DJIA)` instead of `DJIA`.

4.  A simplified version of the random walk model of stock prices states that the best prediction of the stock index at `Time=t` is the value of the index at `Time=t − 1`. Show that this corresponds to a simple linear regression model for 2. with an intercept of 0 and a slope of 1.

5. Carry out the the appropriate tests of significance at level `α = 0.05` for 4. Test each coefficient separately ($t$-tests) , then test both simultaneously (i.e. an F test). 

6. The random walk theory implies that the first differences of the index (the difference between successive values) should be independently normally distributed with mean zero and constant variance. What kind of plot can be used to visually assess this hypothesis? Provide the plot.

## Question 3

We revisit Q. 4 from Assignment 2, using weighted least squares to estimate standard error.
When generating data below, set 
$X$ to be sampled as 100 points drawn randomly on the interval (0,1): `runif(100)`. Use the regression function
$$
f(X) = 1 + 2 \cdot X.
$$

1. Write a function with the same regression function but errors that are not normally distributed using, say, `rexp(n)-1`. Use weighted least squares and construct the $Z$ statistic (i.e. ignore degrees of freedom, pretending they are `Inf`) to test $H_0:\beta_1=2$. Does the $Z$-statistic have close to a $N(0,1)$ distribution? How often is your $Z$ statistic larger than the usual 5% threshold? 

2. Write a new function with the same regression function but multiply the errors by `sqrt(X+1)` so the $i$-th variance (given `X[i]`) is $1+X[i]$. Repeat the experiment in part 1. 

3. Redo part 2. replacing `sqrt(X+1)` by `exp(0.5 * (1 + 5 * X))`.

## Question 4

We revisit Q. 4 from Assignment 2, using the bootstrap to estimate standard error which does not need access to the weights we had in question 3. The bootstrap should be able to give correct inference without knowing the weights we used to make non-constant variance, at least for parts 1 to 3.

When generating data below, set 
$X$ to be sampled as 100 points drawn randomly on the interval (0,1): `runif(100)`. Use the regression function
$$
f(X) = 1 + 2 \cdot X.
$$

1. Repeat 1. of Question 3 above using the bootstrap estimate of standard error to form the $Z$ statistic testing $H_0:\beta_1=2$.

2. Repeat 2. of Question 3 above using the bootstrap for the two different data generating mechanisms for the variance `1+X`.

3. Repeat 2. of Question 3 above using the bootstrap for the two different data generating mechanisms for the variance `exp(1+5X)`.

4. Compare the intervals formed using weighted least squares and those formed using the bootstrap estimate of standard error. Are one method's intervals shorter than the other?

5. Write a new function with same regression function but errors that are not independent.Do this by first generating a vector of errors `error` and then returning a new vector whose first entry is `error[1]` but for $i>1$ the $i$-th entry is `error[i-1] + error[i]`. ** This is the same data generating mechanism we saw in Question 4 from Assignment 2 so you can reuse your code.** Repeat the experiment in part 1. Does the $Z$-statistic approximately have the distribution it should have? Comment on whether the bootstrap can fix this problem of correlated errors.

## Question 5

In this question we will look at inference ($t$-tests and confidence
intervals) after model selection. We are going to repeatedly make
new data frames with junk features and look at p-values and intervals for those
features behave.  We will use the `prostate` data from `ElemStatLearn`, with our response being
a slightly noisy version of our original response `lpsa`.

<h4>
The different parts of the question below need not be separated exactly in assignment. Ultimately, we want:

<ul>
<li>To see how $p$-values and intervals of features we know to be junk behave when they are selected using
something like `step`. All of the null hypotheses for these junk features are true. All of these intervals are trying
to cover 0. Do they behave as we want?

<li> To see whether data splitting "fixes" the behavior.
</ul>

</h4>


```R
library(ElemStatLearn)
data(prostate)
```

1. Write a function in `R` that takes a regression model output by `lm` and creates a new design matrix
by adding some number `k` columns to the model's design matrix whose entries are filled with `rnorm`. The functions
`cbind`, `matrix`, `rnorm` and `as.data.frame` will be helpful here. The function should return a new `data.frame` with all the original variables as well as these `k` new ones.

2. Try adding `k=20` columns to the design matrix. **After adding some additional `rnorm` noise to `lpsa` of variance about half the estimated variance from the model with all the original variables excluding `train`** run `step` in a forward direction starting with just an intercept and the largest model being the model including all variables including the noise. If `noisy.prostate` is one of these data frames, a simple way to do this is along the lines of: `step(lm(lpsa.noisy ~ 1, data=new.prostate), list(upper=lm(lpsa.noisy ~ ., data=new.prostate)))`.
 
3. We know that all of the new variables have nothing to do with the real data so they must have coefficient 0
in these regression models we've created. Do the $p$-values you see after running `step` look as if they
come from features in regressions with 0 (i.e. null) coefficients? 

4. Write a function that repeats steps 2. returning a list with one entry `pvalues` for the null $p$-values and the other entry `intervals` for 95% confidence intervals for these null variables. Repeatedly call this function, storing the $p$-values until you have 1000 $p$-values and plot a histogram. Do they look as you'd expect? 

5. Repeat 4., this time checking which of the confidence intervals cover 0 (the true coefficient in this case). Is it roughly 95%?

6. Modify your functions above but randomly split the data each time, using 50 of the cases to run `step` and the remaining ones to compute `pvalues` and `intervals`. Do the intervals behave better than without splitting? Is there a downside to splitting the data?


```R

```
