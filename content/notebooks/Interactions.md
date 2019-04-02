
## Interactions and qualitative variables 

### Chapter 5, RABE

- Most variables we have looked at so far were continuous: `height`,
`rating`, etc.

- In many situations, we record a categorical variable: `sex` or `gender`, `state`, `country`, etc.

- We call these variables *categorical* or *qualtitative* variables.
In `R`, these are referred to as `factors`.

- For our purposes, we want to answer: **How do we include this in our model?**

This will eventually lead us to the notion of *interactions* and some special regression models called
*ANOVA* (analysis of variance) models.



```R
options(repr.plot.width=4, repr.plot.height=4)
```

### Two-sample problem

In some sense, we have already seen a regression model with categorical variables:
the two-sample model.

* Two sample problem with equal variances: suppose
$Z_j \sim N(\mu_1, \sigma^2), 1 \leq j \leq m$ and
$W_j \sim N(\mu_2, \sigma^2), 1 \leq j \leq n $.

* For $1 \leq i \leq n$, let 
$$X_i =
\begin{cases}
1 & 1 \leq i \leq m \\
0 & \text{otherwise.}
\end{cases}$$


The design matrix and response look like
$$ Y_{(n+m) \times 1} = 
\begin{pmatrix}
Z_1 \\
\vdots \\
Z_m \\
W_1 \\
\vdots \\
W_n \\
\end{pmatrix}, \qquad
X_{(n+m) \times 2} =
 \begin{pmatrix}
1 & 1 \\
 \vdots & \vdots \\
1 & 1 \\
1 & 0 \\
\vdots & \vdots \\
1 & 0
\end{pmatrix}$$

### Salary example

In this example, we have data on salaries of employees in IT (several years ago?) based on their years of experience, their
education level and whether or not they are management.

-   Outcome: `S`, salaries for IT staff in a corporation.

-   Predictors: 
    * `X`, experience (years)
    * `E`, education (1=Bachelor’s, 2=Master’s, 3=Ph.D)
    * `M`, management (1=management, 0=not management)


```R
url = 'http://stats191.stanford.edu/data/salary.table'
salary.table <- read.table(url, header=T)
salary.table$E <- factor(salary.table$E)
salary.table$M <- factor(salary.table$M)
```

Let's take a quick look at how `R` treats a `factor`


```R
str(salary.table$E)
```

Let's take a look at the data. We will use triangles for management, diamonds for non-management
red for education=1, green for education=2 and blue for education=3.



```R
plot(salary.table$X, salary.table$S, type='n', xlab='Experience', ylab='Salary')
colors <- c('red', 'green', 'blue')
symbols <- c(23,24)
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((salary.table$E == i) * (salary.table$M == j))
        points(salary.table$X[subset], salary.table$S[subset], pch=symbols[j+1], bg=colors[i], cex=2)
    }
}
```

## Effect of experience

In these pictures, the slope of each line seems to be about the same. How might
we estimate it?

### One solution is *stratification*.

* Make six separate models (one for each combination of `E` and `M`) and estimate the slope.

* Combining them: we could average them?

* We have few degrees of freedom in each group.


### Or, use *qualitative* variables

-   IF it is reasonable to assume that $\sigma^2$ is constant for each
    observation.

-   THEN, we can incorporate all observations into 1 model.

$$S_i = \beta_0 + \beta_1 X_i + \beta_2 E_{i2} + \beta_3 E_{i3} + \beta_4 M_i + \varepsilon_i$$

Above, the variables are:

* $$
E_{i2} = \begin{cases}
1 & \text{if $E_i$=2} \\
0 & \text{otherwise.}
\end{cases}
$$

* $$
E_{i3} = \begin{cases}
1 & \text{if $E_i$=3} \\
0 & \text{otherwise.}
\end{cases}
$$

### Notes

-   Although $E$ has 3 levels, we only added 2 variables to the model.
    In a sense, this is because `(Intercept)` (i.e. $\beta_0$) absorbs one level.

-   If we added three variables then the columns of design matrix would
    be linearly dependent so we would not have a unique least squares solution.

-   Assumes $\beta_1$ – effect of experience is the same in all groups,
    unlike when we fit the model separately. This may or may not be
    reasonable.


```R
salary.lm <- lm(S ~ E + M + X, salary.table)
summary(salary.lm)
```

Now, let's take a look at our design matrix


```R
head(model.matrix(salary.lm))
```

Comparing to our actual data, we can understand how the columns above were formed. They were formed
just as we had defined them above.


```R
head(model.frame(salary.lm))
```

### Effect of experience

-   Our model has enforced the constraint the $\beta_1$ is the same
    within each group.

-   Graphically, this seems OK, but how can we test this?

-   We could fit a model with different slopes in each group, but
    keeping as many degrees of freedom as we can.

-   This model has *interactions* in it: the effect of experience
    depends on what level of education you have.


### Interaction between experience and education

-   Model: $$\begin{aligned}
       S_i &= \beta_0 + \beta_1 X_i + \beta_2 E_{i2} + \beta_3 E_{i3} +\
 \beta_4 M_i \\
       & \qquad  + \beta_5 E_{i2} X_i + \beta_6 E_{i3} X_i + \varepsilon_i.
       \end{aligned}$$
       
- What is the regression function within each group?

-   Note that we took each column corresponding to education and
    multiplied it by the column for experience to get two new
    predictors.

-   To test whether the slope is the same in each group we would just
    test $H_0:\beta_5 = \beta_6=0$.

-   Based on figure, we expect not to reject $H_0$.


```R
model_XE = lm(S~ E + M + X + X:E, salary.table)
summary(model_XE)
```


```R
anova(salary.lm, model_XE)
```

The notation `X:E` denotes an *interaction*. Generally, `R` will take the columns added for `E` and the columns added
for `X` and add their elementwise product (Hadamard product) to the design matr.x

Let's look at our design matrix again to be sure we understand what model was fit.


```R
model.matrix(model_XE)[10:20,]
```

## Remember, it's still a model (i.e. a plane)

<img src="http://stats191.stanford.edu/figs/axes_multiple_full.svg" width="700">


### Interaction between management and education

* We can also test for interactions between qualitative variables.

* In our plot, note that Master's in management make more than PhD's in management, but this difference disappears in 
non-management.

* This means the effect of education is different in the two management levels. This is evidence of
an *interaction*.

* To see this, we plot the residuals within groups separately.


```R
plot(salary.table$X, salary.table$S, type='n', xlab='Experience', ylab='Salary')
colors <- c('red', 'green', 'blue')
symbols <- c(23,24)
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((salary.table$E == i) * (salary.table$M == j))
        points(salary.table$X[subset], salary.table$S[subset], pch=symbols[j+1], bg=colors[i], cex=2)
    }
}
```


```R
r = resid(salary.lm)
k = 1
plot(salary.table$X, r, xlim=c(1,6), type='n', xlab='Group', ylab='Residuals')
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((salary.table$E == i) * (salary.table$M == j))
        points(rep(k, length(r[subset])), r[subset], pch=symbols[j+1], bg=colors[i], cex=2)
        k = k+1
    }
}
```

`R` has a special plot that can help visualize this effect, called an `interaction.plot`.


```R
interaction.plot(salary.table$E, salary.table$M, r, type='b', col=c('red',
                'blue'), lwd=2, pch=c(23,24))
```

### Interaction between management and education

-   Based on figure, we expect an interaction effect.

-   Fit model $$\begin{aligned}
       S_i &= \beta_0 + \beta_1 X_i + \beta_2 E_{i2} + \beta_3 E_{i3} +\
 \beta_4 M_i \\
       & \qquad  + \beta_5 E_{i2} M_i + \beta_6 E_{i3} M_i + \varepsilon_i.
       \end{aligned}$$

-   Again, testing for interaction is testing $H_0:\beta_5=\beta_6=0.$

- What is the regression function within each group?



```R
model_EM = lm(S ~ X + E:M + E + M, salary.table)
summary(model_EM)
```


```R
anova(salary.lm, model_EM)
```

Let's look at our design matrix again to be sure we understand what model was fit.


```R
head(model.matrix(model_EM))
```

We will plot the residuals as functions of experience
with each *experience* and *management* having a 
different symbol/color.


```R
r = rstandard(model_EM)
plot(salary.table$X, r, type='n')
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((salary.table$E == i) * (salary.table$M == j))
        points(salary.table$X[subset], r[subset], pch=symbols[j+1], bg=colors[i], cex=2)
    }
}
```

One observation seems to be an outlier.


```R
library(car)
outlierTest(model_EM)
```

Let's refit our model to see that our conclusions are not vastly different.


```R
subs33 = c(1:length(salary.table$S))[-33]
salary.lm33 = lm(S ~ E + X + M, data=salary.table, subset=subs33)
model_EM33 = lm(S ~ E + X + E:M + M, data=salary.table, subset=subs33)
anova(salary.lm33, model_EM33)

```

Let's replot the residuals


```R
r = rstandard(model_EM33)
mf = model.frame(model_EM33)
plot(mf$X, r, type='n')
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((mf$E == i) * (mf$M == j)) 
        points(mf$X[subset], r[subset], pch=symbols[j+1], bg=colors[i], cex=2)
    }
}
```

Let's make a final plot of the fitted values.


```R
salaryfinal.lm = lm(S ~ X + E * M, salary.table, subset=subs33)
mf = model.frame(salaryfinal.lm)
plot(mf$X, mf$S, type='n', xlab='Experience', ylab='Salary')
colors <- c('red', 'green', 'blue')
ltys <- c(2,3)
symbols <- c(23,24)
for (i in 1:3) {
    for (j in 0:1) {
        subset <- as.logical((mf$E == i) * (mf$M == j))
        points(mf$X[subset], mf$S[subset], pch=symbols[j+1], bg=colors[i], cex=2)
        lines(mf$X[subset], fitted(salaryfinal.lm)[subset], lwd=2, lty=ltys[j], col=colors[i])
    }
}
```

### Visualizing an interaction

From our first look at the data, the difference between 
Master's and PhD in the
management group is different than in the non-management
group. This is an interaction between the two qualitative
variables
*management,M* and *education,E*. We can visualize this
by first removing the effect of experience, then plotting
the means within each of the 6 groups using *interaction.plot*.


```R
U = salary.table$S - salary.table$X * model_EM$coef['X']
interaction.plot(salary.table$E, salary.table$M, U, type='b', col=c('red',
                'blue'), lwd=2, pch=c(23,24))

```

### Jobtest employment data (RABE)

<table>
<tr><td><b>Variable</b></td><td><b>Description</b></td></tr>
<tr><td>TEST</td><td>Job aptitude test score</td></tr>
<tr><td>MINORITY</td><td>1 if applicant could be considered minority, 0 otherwise</td></tr>
<tr><td>PERF</td><td>Job performance evaluation</td></tr>
</table>


```R
url = 'http://stats191.stanford.edu/data/jobtest.table'
jobtest.table <- read.table(url, header=T)
jobtest.table$MINORITY <- factor(jobtest.table$MINORITY)
```

Since I will be making several plots, it will be easiest to attach `jobtest.table` though I will detach it later.

**These plots would be easier with `ggplot`.**


```R
attach(jobtest.table)
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=2, 
       bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=2, bg='green')
```

### General model

-   In theory, there may be a linear relationship between $JPERF$ and
    $TEST$ but it could be different by group.

-   Model:
    $$JPERF_i = \beta_0 + \beta_1 TEST_i + \beta_2 MINORITY_i + \beta_3 MINORITY_i * TEST_i + \varepsilon_i.$$

-   Regression functions:
   $$
   Y_i =
   \begin{cases}
   \beta_0 + \beta_1 TEST_i + \varepsilon_i & \text{if $MINORITY_i$=0} \\
   (\beta_0 + \beta_2) + (\beta_1 + \beta_3) TEST_i + \varepsilon_i & \text{if 
$MINORITY_i=1$.} \\
   \end{cases}
   $$





### Our first model: ($\beta_2=\beta_3=0$)

This has no effect for `MINORITY`.


```R
jobtest.lm1 <- lm(JPERF ~ TEST, jobtest.table)
print(summary(jobtest.lm1))
```


```R
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=2, bg='green')
abline(jobtest.lm1$coef, lwd=3, col='blue')
```

### Our second model ($\beta_3=0$)

This model allows for an effect of `MINORITY` but no interaction between `MINORITY` and `TEST`.


```R
jobtest.lm2 = lm(JPERF ~ TEST + MINORITY)
print(summary(jobtest.lm2))
```


```R
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=2, bg='green')
abline(jobtest.lm2$coef['(Intercept)'], jobtest.lm2$coef['TEST'], lwd=3, col='purple')
abline(jobtest.lm2$coef['(Intercept)'] + jobtest.lm2$coef['MINORITY1'], jobtest.lm2$coef['TEST'], lwd=3, col='green')
```

### Our third model $(\beta_2=0)$:

This model includes an interaction between `TEST` and `MINORITY`. These lines have the same intercept but possibly different slopes within the `MINORITY` groups.


```R
jobtest.lm3 = lm(JPERF ~ TEST + TEST:MINORITY, jobtest.table)
summary(jobtest.lm3)
```


```R
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=2, bg='green')
abline(jobtest.lm3$coef['(Intercept)'], jobtest.lm3$coef['TEST'], lwd=3, col='purple')
abline(jobtest.lm3$coef['(Intercept)'], jobtest.lm3$coef['TEST'] + jobtest.lm3$coef['TEST:MINORITY1'], lwd=3, col='green')
```

Let's look at our design matrix again to be sure we understand which model was fit.


```R
head(model.matrix(jobtest.lm3))
```

### Our final model: no constraints

This model allows for different intercepts and different slopes.


```R
jobtest.lm4 = lm(JPERF ~ TEST * MINORITY, jobtest.table)
summary(jobtest.lm4)
```


```R
plot(TEST, JPERF, type='n')
points(TEST[(MINORITY == 0)], JPERF[(MINORITY == 0)], pch=21, cex=2, bg='purple')
points(TEST[(MINORITY == 1)], JPERF[(MINORITY == 1)], pch=25, cex=2, bg='green')
abline(jobtest.lm4$coef['(Intercept)'], jobtest.lm4$coef['TEST'], lwd=3, col='purple')
abline(jobtest.lm4$coef['(Intercept)'] + jobtest.lm4$coef['MINORITY1'],
      jobtest.lm4$coef['TEST'] + jobtest.lm4$coef['TEST:MINORITY1'], lwd=3, col='green')
```

The expression `TEST*MINORITY` is shorthand for `TEST + MINORITY + TEST:MINORITY`.


```R
head(model.matrix(jobtest.lm4))
```

### Comparing models

Is there any effect of MINORITY on slope or intercept?


```R
anova(jobtest.lm1, jobtest.lm4) # ~ TEST to ~ TEST * MINORITY
```

Is there any effect of MINORITY on intercept? (Assuming we have accepted the hypothesis that the slope is the same within
each group).


```R
anova(jobtest.lm1, jobtest.lm2) # ~ TEST to ~ TEST + MINORITY
```

We could also have allowed for the possiblity that the slope is different within each group and still check for a different intercept.


```R
anova(jobtest.lm3, jobtest.lm4) # ~ TEST + TEST:MINORITY to 
                                # ~ TEST * MINORITY
```

Is there any effect of `MINORITY` on slope?  (Assuming we have accepted the hypothesis that the intercept is the same within each
group).


```R
anova(jobtest.lm1, jobtest.lm3) # ~ TEST vs. ~ TEST + TEST:MINORITY
```

Again, we could have allowed for the possibility that the intercept is different within each group.


```R
anova(jobtest.lm2, jobtest.lm4) # ~ TEST + MINORITY vs. 
                                # ~ TEST * MINORITY
```

In summary, without taking the several tests into account here, there does seem to be some evidence
that the  slope is different within the two groups.


```R
detach(jobtest.table)
```

## Model selection

Already with this simple dataset (simpler than the IT salary data) we have 4 competing models. How are we going
to arrive at a final model? 

This highlights the need for *model selection*. We will come to this topic shortly.
