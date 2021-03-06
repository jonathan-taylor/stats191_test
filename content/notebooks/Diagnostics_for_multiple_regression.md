

```R
options(repr.plot.width=5, repr.plot.height=5)
```

## Diagnostics in multiple linear regression

### Outline

-   Diagnostics – again

-   Different types of residuals

-   Influence

-   Outlier detection

-   Residual plots:

    -   partial regression (added variable) plot,

    -   partial residual (residual plus component) plot.


## Scottish hill races data

The dataset we will use is based on record times on [Scottish hill races](http://www.statsci.org/data/general/hills.html).

<table>
<tr><td><b>Variable</b></td><td><b>Description</b></td></tr>
<tr><td>Time</td><td>Record time to complete course</td></tr>
<tr><td>Distance</td><td>Distance in the course</td></tr>
<tr><td>Climb</td><td>Vertical climb in the course</td></tr>
</table>


```R
url = 'http://www.statsci.org/data/general/hills.txt' 
races.table = read.table(url, header=TRUE, sep='\t')
head(races.table)
```

As we'd expect, the time increases both with `Distance` and `Climb`.


```R
plot(races.table[,2:4], pch=23, bg='orange', cex=2)
```

Let's look at our multiple regression model.


```R
races.lm = lm(Time ~ Distance + Climb, data=races.table)
summary(races.lm)
```

But is this a good model? 

## Diagnostics

### What can go wrong?

-   Regression function can be wrong: maybe regression function should
    have some other form (see diagnostics for simple linear regression).

-   Model for the errors may be incorrect:

    -   may not be normally distributed.

    -   may not be independent.

    -   may not have the same variance.

-   Detecting problems is more *art* then *science*, i.e. we cannot
    *test* for all possible problems in a regression model.

## Diagnostics

-   Basic idea of diagnostic measures: if model is correct then
    residuals $e_i = Y_i -\widehat{Y}_i, 1 \leq i \leq n$ should look
    like a sample of (not quite independent) $N(0, \sigma^2)$ random
    variables.

## Standard diagnostic plots

`R` produces a set of standard plots for `lm` that help us assess whether our assumptions are reasonable or not. We will go through each in some, but not too much, detail.

As we see below, there are some quantities which we need to define in order to read these plots. We will define these first.


```R
par(mfrow=c(2,2))
plot(races.lm, pch=23 ,bg='orange',cex=2)
```

## Problems with the errors

### Possible problems & diagnostic checks

-   Errors may not be normally distributed or may not have the same
    variance – qqnorm can help with this. This may not be too important
    in large samples.

-   Variance may not be constant. Can also be addressed in a plot of $X$ vs. $e$
: *fan shape* or other trend indicate
    non-constant variance.

-   Influential observations. Which points “affect” the regression line
    the most?

-   Outliers: points where the model really does not fit! Possibly
    mistakes in data transcription, lab errors, who knows? Should be
    recognized and (hopefully) explained.

## Types of residuals

-   Ordinary residuals: $e_i = Y_i - \widehat{Y}_i$. These measure the
    deviation of predicted value from observed value, but their
    distribution depends on unknown scale, $\sigma$.

-   Internally studentized residuals (`rstandard` in R):
    $$r_i = e_i / SE(e_i) = \frac{e_i}{\widehat{\sigma} \sqrt{1 - H_{ii}}}$$
    
- Above, $H$ is the “hat” matrix $H=X(X^TX)^{-1}X^T$. These are almost $t$-distributed, except
    $\widehat{\sigma}$ depends on $e_i$.

## Types of residuals

- Externally studentized residuals (`rstudent` in R):
    $$t_i = \frac{e_i}{\widehat{\sigma_{(i)}} \sqrt{1 - H_{ii}}} \sim t_{n-p-2}.$$
    These are exactly $t$ distributed so we know their distribution and
    can use them for tests, if desired.
    
- The quantity $\hat{\sigma}^2_{(i)}$ is the MSE of the model fit to all data except case $i$ (i.e. it has $n-1$ observations and $p$ features).

- Numerically, these residuals are highly correlated, as we would expect.


```R
plot(resid(races.lm), rstudent(races.lm), pch=23, bg='blue', cex=3)
```


```R
plot(rstandard(races.lm), rstudent(races.lm), pch=23, bg='blue', cex=3)
```

## Standard diagnostic plots

The first plot is the quantile plot for the residuals, that compares their distribution
to that of a sample of independent normals.



```R
qqnorm(rstandard(races.lm), pch=23, bg='red', cex=2)
```

If the residuals were really normal we'd expect this plot to be roughly on the diagonal.


```R
qqnorm(rnorm(500), pch=23, bg='red', cex=2)
abline(0, 1)
```

Two other plots try address the constant variance assumptions. If these plots
have a particular shape (maybe the spread increases with $\hat{Y}$) then maybe the variance is not constant.


```R
plot(fitted(races.lm), sqrt(abs(rstandard(races.lm))), pch=23, bg='red', ylim=c(0,1))
```


```R
plot(fitted(races.lm), resid(races.lm), pch=23, bg='red', cex=2)
abline(h=0, lty=2)
```

## Influence of an observation

Other plots provide an assessment of the `influence` of each observation.
Usually, this is done by dropping an entire case $(y_i, x_i)$ from the dataset and
refitting the model.

-   In this setting, a $\cdot_{(i)}$ indicates $i$-th observation was
    not used in fitting the model.

-   For example: $\widehat{Y}_{j(i)}$ is the regression function
    evaluated at the $j$-th observation predictors BUT the coefficients
    $(\widehat{\beta}_{0(i)}, \dots, \widehat{\beta}_{p(i)})$ were fit
    after deleting $i$-th case from the data.

## Influence of an observation

-   Idea: if $\widehat{Y}_{j(i)}$ is very different than $\widehat{Y}_j$
    (using all the data) then $i$ is an influential point, at least for
    estimating the regression function at $(X_{1,j}, \dots, X_{p,j})$.
    
- Could also look at difference between $\widehat{Y}_{i(i)} - \widehat{Y}_i$, or any other measure.
    
-  There are various standard measures of influence.

## DFFITS

-   $$DFFITS_i = \frac{\widehat{Y}_i - \widehat{Y}_{i(i)}}{\widehat{\sigma}_{(i)} \sqrt{H_{ii}}}$$

-   This quantity measures how much the regression function changes at
    the $i$-th case / observation when the $i$-th case / observation is
    deleted.

-   For small/medium datasets: value of 1 or greater is “suspicious” (RABE).
    For large dataset: value of $2 \sqrt{(p+1)/n}$.
    
- `R` has its own standard rules similar to the above for marking an observation
as influential.



```R
plot(dffits(races.lm), pch=23, bg='orange', cex=2, ylab="DFFITS")
```

It seems that some observations had a high influence measured by $DFFITS$:


```R
races.table[which(dffits(races.lm) > 0.5),]
```

It is perhaps not surprising that the longest course and the course with the most elevation gain seemed to have a strong effect on the fitted values. What about `Knock Hill`? We'll come back to this later.

## Cook’s distance

Cook’s distance measures how much the entire regression function
    changes when the $i$-th case is deleted.

-   $$D_i = \frac{\sum_{j=1}^n(\widehat{Y}_j - \widehat{Y}_{j(i)})^2}{(p+1) \, \widehat{\sigma}^2}$$

-   Should be comparable to $F_{p+1,n-p-1}$: if the “$p$-value” of $D_i$
    is 50 percent or more, then the $i$-th case is likely influential:
    investigate further. (RABE)
    
- Again, `R` has its own rules similar to the above for marking an observation
as influential.

- What to do after investigation? No easy answer.
    


```R
plot(cooks.distance(races.lm), pch=23, bg='orange', cex=2, ylab="Cook's distance")
```


```R
races.table[which(cooks.distance(races.lm) > 0.1),]
```

Again, the same 3 races. This is not surprising as both $DFFITS$ and Cook's distance measure changes in fitted values. The difference is that one measures the influence on one fitted value, while the other measures the influence on the entire vector of fitted values.

## DFBETAS

This quantity measures how much the coefficients change when the
    $i$-th case is deleted.


-   $$DFBETAS_{j(i)} = \frac{\widehat{\beta}_j - \widehat{\beta}_{j(i)}}{\sqrt{\widehat{\sigma}^2_{(i)} (X^TX)^{-1}_{jj}}}.$$

   
-   For small/medium datasets: absolute value of 1 or greater is
    “suspicious”. For large dataset: absolute value of $2 /  \sqrt{n}$.




```R
plot(dfbetas(races.lm)[,'Climb'], pch=23, bg='orange', cex=2, ylab="DFBETA (Climb)")
races.table[which(abs(dfbetas(races.lm)[,'Climb']) > 1),]
```


```R
plot(dfbetas(races.lm)[,'Distance'], pch=23, bg='orange', cex=2, ylab="DFBETA (Climb)")
races.table[which(abs(dfbetas(races.lm)[,'Distance']) > 0.5),]
```

## Outliers

The essential definition of an *outlier* is an observation pair $(Y, X_1, \dots, X_p)$ that does not follow the model, while most other observations seem to follow the model.

-   Outlier in *predictors*: the $X$ values of the observation may lie
    outside the “cloud” of other $X$ values. This means you may be
    extrapolating your model inappropriately. The values $H_{ii}$ can be
    used to measure how “outlying” the $X$ values are.

-   Outlier in *response*: the $Y$ value of the observation may lie very
    far from the fitted model. If the studentized residuals are large:
    observation may be an outlier.
    
- The races at `Bens of Jura` and `Lairig Ghru` seem to be outliers in *predictors*
as they were the highest and longest races, respectively.

- How can we tell if the `Knock Hill` result is an outlier? It seems to have taken much
longer than it should have so maybe it is an outlier in the *response*.


## Outlying $X$ values

One way to detect outliers in the *predictors*, besides just looking at the actual values themselves, is through their leverage values, defined by
$$
\text{leverage}_i = H_{ii} = (X(X^TX)^{-1}X^T)_{ii}.
$$

Not surprisingly, our longest and highest courses show up again. This at least
reassures us that the leverage is capturing some of this "outlying in $X$ space".


```R
plot(hatvalues(races.lm), pch=23, bg='orange', cex=2, ylab='Hat values')
races.table[which(hatvalues(races.lm) > 0.3),]
```

## Outliers in the response

We will consider a crude outlier test that tries to find residuals that are
"larger" than they should be.

- Since `rstudent` are $t$ distributed, we could just compare them to the $T$ distribution and reject if their absolute value is too large.

- Doing this for every observation results in $n$ different hypothesis tests.

-   This causes a problem: if $n$ is large, if we “threshold” at
    $t_{1-\alpha/2, n-p-2}$ we will get many outliers by chance even if
    model is correct. 
    
- In fact, we expect to see $n \cdot \alpha$
    “outliers” by this test. Every large data set would have outliers in
    it, even if model was entirely correct!
    
    

Let's sample some data from our model to convince ourselves that this is a real problem.


```R
X = rnorm(100)
Y = 2 * X + 0.5 + rnorm(100)
alpha = 0.1
cutoff = qt(1 - alpha / 2, 97)
sum(abs(rstudent(lm(Y~X))) > cutoff)
```


```R
# Bonferroni correction
X = rnorm(100)
Y = 2 * X + 0.5 + rnorm(100)
cutoff = qt(1 - (alpha / 100) / 2, 97)
sum(abs(rstudent(lm(Y~X))) > cutoff)
```

### Multiple comparisons

-   This problem we identified is known as *multiple comparisons* or *simultaneous inference.* 

- When performing many tests (say $m$) each at level $\alpha$, we expect at least $\alpha m$ rejections
even when *all* null hypotheses are true!

- In outlier detection, we are performing $m=n$ hypothesis tests, but might still
like to control the probability of making *any* false positive
errors.
    
- The reason we don't want to make errors here is that we don't
want to throw away data unnecessarily.

- One solution: Bonferroni correction, threshold at
$t_{1 - \alpha/(2*n), n-p-2}$.
    

### Bonferroni correction

- Dividing $\alpha$ by $n$, the number of tests, is known as a *Bonferroni* correction.

-  If we are doing many $t$ (or other) tests, say $m \gg 1$ we can
  control overall false positive rate at $\alpha$ by testing each one
  at level $\alpha/m$. 
  
- In this case $m=n$, but other times we might look at a different number of tests.

### Bonferroni correction

- Essentially the *union bound* for probability.

- **Proof:** when the model is correct, with studentized residuals $T_i$:

    $$\begin{aligned}
        P\left( \text{at least one false positive} \right)
        &  = P \left(\cup_{i=1}^m |T_i| \geq t_{1 - \alpha/(2*m), n-p-2} \right) \\
        & \leq \sum_{i=1}^m P \left( |T_i| \geq t_{1 - \alpha/(2*m), n-p-2} \right) \\
        &  = \sum_{i=1}^m  \frac{\alpha}{m} = \alpha. \\
       \end{aligned}$$

Let's apply this to our data. It turns out that `KnockHill` is a [known error](http://www.statsci.org/data/general/hills.html).


```R
n = nrow(races.table)
cutoff = qt(1 - 0.05 / (2*n), (n - 4))
races.table[which(abs(rstudent(races.lm)) > cutoff),]
```

The package `car` has a built in function to do this test.


```R
library(car)
outlierTest(races.lm)
```

### Final plot

The last plot that `R` produces is a plot of residuals against leverage. Points that have
high leverage and large residuals are particularly influential.


```R
plot(hatvalues(races.lm), rstandard(races.lm), pch=23, bg='red', cex=2)
```

`R` will put the IDs of cases that seem to be influential in these (and other plots). Not surprisingly, we see our usual three suspects.


```R
plot(races.lm, which=5)
```

## Influence measures

As mentioned above, `R` has its own rules for flagging points as being influential. To
see a summary of these, one can use the `influence.measures` function.


```R
influence.measures(races.lm)
```

While not specified in the documentation, the meaning of the asterisks can be found
by reading the code. The function `is.influential` makes the decisions
to flag cases as influential or not. 

- We see that the `DFBETAS` are thresholded at 1.

- We see that `DFFITS` is thresholded at `3 * sqrt((p+1)/(n-p-1))`.

- Etc. 


```R
influence.measures
```

## Problems in the regression function

-   True regression function may have higher-order non-linear terms,
    polynomial or otherwise.

-   We may be missing terms involving more than one ${X}_{(\cdot)}$,
    i.e. ${X}_i \cdot {X}_j$ (called an *interaction*).

-   Some simple plots: *added-variable* and *component plus residual*
    plots can help to find nonlinear functions of *one variable*.
    
- I find these plots of somewhat limited use in practice, but we will go over them as
possibly useful diagnostic tools.


### Added variable plots

- The plots can be helpful for finding influential points, outliers. The functions can 
be found in the `car` package.

-   Procedure:

    -   Let $\tilde{e}_{X_j,i}, 1\leq i \leq n$ be the residuals after
        regressing $X_j$ onto all columns of $X$ except
        $X_j$;

    -   Let $e_{X_j,i}$ be the residuals after regressing ${Y}$ onto
        all columns of ${X}$ except ${X}_j$;

    -   Plot $\tilde{e}_{X_j}$ against $e_{X_j}$.
    
    - If the (partial regression) relationship is linear this plot should look linear.



```R
avPlots(races.lm, 'Distance')
```


```R
avPlots(races.lm, 'Climb')
```

### Component + residual plots

-   Similar to added variable, but may be more helpful in identifying nonlinear relationships.

-   Procedure: plot $X_{ij}, 1 \leq i \leq n$ vs.
    $e_i + \widehat{\beta}_j \cdot X_{ij} , 1 \leq i \leq n$.
   
- The green line is a non-parametric smooth of the scatter plot that may suggest
relationships other than linear.



```R
crPlots(races.lm, 'Distance')
```


```R
crPlots(races.lm, 'Climb')
```
