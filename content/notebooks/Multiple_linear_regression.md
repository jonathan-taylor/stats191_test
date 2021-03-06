
# Multiple linear regression

## Outline

-   Specifying the model.

-   Fitting the model: least squares.

-   Interpretation of the coefficients.

-   More on $F$-statistics.

-   Matrix approach to linear regression.

-   $T$-statistics revisited.

-   More $F$ statistics.

-   Tests involving more than one $\beta$.

## Prostate data

For more information on the [Gleason score](http://en.wikipedia.org/wiki/Gleason_Grading_System).

<table>
<tr><td><b>Variable</b></td><td><b>Description</b></td></tr>
<tr><td>lcavol</td><td>(log) Cancer Volume</td></tr>
<tr><td>lweight</td><td>(log) Weight</td></tr>
<tr><td>age</td><td>Patient age</td></tr>
<tr><td>lbph</td><td>(log) Vening Prostatic Hyperplasia</td></tr>
<tr><td>svi</td><td>Seminal Vesicle Invasion</td></tr>
<tr><td>lcp</td><td>(log) Capsular Penetration</td></tr>
<tr><td>gleason</td><td>Gleason score</td></tr>
<tr><td>pgg45</td><td>Percent of Gleason score 4 or 5</td></tr>
<tr><td>lpsa</td><td>(log) Prostate Specific Antigen</td></tr>
<tr><td>train</td><td>Label for test / training split</td></tr>
</table>


```R
library(ElemStatLearn)
data(prostate)
pairs(prostate, pch=23, bg='orange',                                          
      cex.labels=1.5) 
```

## Specifying the model

- We will use variables `lcavol, lweight, age, lbph, svi, lcp` and `pgg45` to predict `lpsa`.

-   Rather than one predictor, we have $p=7$ predictors.

### Model

-   $$Y_i = \beta_0 + \beta_1 X_{i1} + \dots + \beta_p X_{ip} + \varepsilon_i$$

-   Errors $\varepsilon$ are assumed independent $N(0,\sigma^2)$, as in
    simple linear regression.

-   Coefficients are called (partial) regression coefficients because
    they “allow” for the effect of other variables.

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_multiple_full.svg">


## Fitting the model


-   Just as in simple linear regression, model is fit by minimizing
    $$\begin{aligned}
       SSE(\beta_0, \dots, \beta_p) &= \sum_{i=1}^n\left(Y_i - \left(\beta_0 + \sum_{j=1}^p \beta_j \
X_{ij} \right) \right)^2 \\
       &= \|Y - \widehat{Y}(\beta)\|^2
       \end{aligned}$$

-   Minimizers:
    $\widehat{\beta} = (\widehat{\beta}_0, \dots, \widehat{\beta}_p)$
    are the “least squares estimates”: are also normally distributed as
    in simple linear regression.


```R
prostate.lm = lm(lpsa ~ lcavol + lweight + age + lbph + svi + 
                 lcp + pgg45, data=prostate)
prostate.lm
```


## Estimating $\sigma^2$

-   As in simple regression
    $$\widehat{\sigma}^2 = \frac{SSE}{n-p-1} \sim \sigma^2 \cdot \frac{\chi^2_{n-p-1}}{n-p\
-1}$$
    independent of $\widehat{\beta}$.

-   Why $\chi^2_{n-p-1}$? Typically, the degrees of freedom in the
    estimate of $\sigma^2$ is
    $n-\# \text{number of parameters in regression function}$.



```R
prostate.lm$df.resid
sigma.hat = sqrt(sum(resid(prostate.lm)^2) / prostate.lm$df.resid)
sigma.hat
```

## Interpretation of $\beta_j$’s

-   Take $\beta_1=\beta_{\tt{lcavol}}$ for example. This is the amount the average `lpsa`
    rating increases for one “unit” of increase in `lcavol`, keeping
    everything else constant.

-   We refer to this as the effect of `lcavol` *allowing
    for* or *controlling for* the other variables.
    
- For example, let's take the 10th case in our data and change `lcavol` by 1 unit.



```R
case1 = prostate[10,]
case2 = case1
case2['lcavol'] = case2['lcavol'] + 1
Yhat = predict(prostate.lm, rbind(case1, case2))
Yhat
```

Our regression model says that this difference should be $\hat{\beta}_{\tt lcavol}$.


```R
c(Yhat[2]-Yhat[1], coef(prostate.lm)['lcavol'])
```

## Partial regression coefficients

-   The term *partial* refers to the fact that the coefficient $\beta_j$
    represent the partial effect of ${X}_j$ on ${Y}$, i.e. after
    the effect of all other variables have been removed.

-   Specifically,
    $$Y_i - \sum_{l=1, l \neq j}^k X_{il} \beta_l = \beta_0 + \beta_j X_{ij} + \varepsilon_i.$$

-   Let $e_{i,(j)}$ be the residuals from regressing ${Y}$ onto all
    ${X}_{\cdot}$’s except ${X}_j$, and let $X_{i,(j)}$ be the
    residuals from regressing ${X}_j$ onto all ${X}_{\cdot}$’s
    except ${X}_j$, and let $X_{i,(j)}$.

-   If we regress $e_{i,(j)}$ against $X_{i,(j)}$, the coefficient is
    *exactly* the same as in the original model.


Let's verify this interpretation of regression coefficients.


```R
partial_resid_lcavol = resid(lm(lcavol ~  lweight + age + lbph + svi + 
                                lcp + pgg45, data=prostate))
partial_resid_lpsa = resid(lm(lpsa ~  lweight + age + lbph + svi + 
                              lcp + pgg45, data=prostate))
summary(lm(partial_resid_lpsa ~ partial_resid_lcavol))
```

## Goodness of fit for multiple regression



$$\begin{aligned}
   SSE &= \sum_{i=1}^n(Y_i - \widehat{Y}_i)^2 \\
   SSR &= \sum_{i=1}^n(\overline{Y} - \widehat{Y}_i)^2 \\
   SST &= \sum_{i=1}^n(Y_i - \overline{Y})^2 = SSE + SSR \\
   R^2 &= \frac{SSR}{SST}
   \end{aligned}$$ 
   
   $R^2$ is called the *multiple correlation
coefficient* of the model, or the *coefficient of multiple
determination*.


The sums of squares and $R^2$ are defined analogously
to those in simple linear regression.



```R
Y = prostate$lpsa
n = length(Y)
SST = sum((Y - mean(Y))^2)
MST = SST / (n - 1)
SSE = sum(resid(prostate.lm)^2)
MSE = SSE / prostate.lm$df.residual
SSR = SST - SSE
MSR = SSR / (n - 1 - prostate.lm$df.residual)
print(c(MST,MSE,MSR))
```

## Adjusted $R^2$

-   As we add more and more variables to the model – even random ones,
    $R^2$ will increase to 1.

-   Adjusted $R^2$ tries to take this into account by replacing sums of
    squares by *mean squares*
    $$R^2_a = 1 - \frac{SSE/(n-p-1)}{SST/(n-1)} = 1 - \frac{MSE}{MST}.$$



```R
summary(prostate.lm)
```


## Goodness of fit test


-   As in simple linear regression, we measure the goodness of fit of
    the regression model by
    $$F = \frac{MSR}{MSE} = \frac{\|\overline{Y}\cdot {1} - \widehat{{Y}}\|^2/p}{\\
\|Y - \widehat{{Y}}\|^2/(n-p-1)}.$$

-   Under $H_0:\beta_1 = \dots = \beta_p=0$, $$F \sim F_{p, n-p-1}$$ so
    reject $H_0$ at level $\alpha$ if $F > F_{p,n-p-1,1-\alpha}.$



```R
summary(prostate.lm)
F = MSR / MSE
F

```

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_multiple_full.svg">

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_multiple_reduced.svg">

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_multiple.svg">


## Intuition behind the $F$ test (?)

-   The $F$ statistic is a ratio of lengths of orthogonal vectors
    (divided by degrees of freedom).

- Let $\mu=E(Y)=X\beta$ be the true mean vector for $Y$:
$$
\mu_i = \mathbb{E}(Y_i) = \beta_0 + X_{i1} \beta_1  + \dots +  X_{ip} \beta_p
$$

-   We can prove that our model implies (whether $H_0$ is true or not) $$\begin{aligned}
       \mathbb{E}\left(MSR\right) &= \sigma^2 + \underbrace{\|{\mu} - \overline{\mu} \cdot {1}\|^2 / p}_{(*)} \\
       \mathbb{E}\left(MSE\right) &= \sigma^2 \\
       \end{aligned}$$ 
       
- If $H_0$ is true, then $(*)=0$ and $\mathbb{E}(MSR)=\mathbb{E}(MSE)=\sigma^2$ so the $F$ should not be
too different from 1.

-   If $F$ is large, it is evidence that $\mathbb{E}\left(MSR\right) \neq \sigma^2$, i.e. $H_0$ is
    false.


### Where does (*) come from?

Least squares regression can be expressed in terms of orthogonal projections. 
That is, 
$$
\hat{Y} = PY
$$
for some $P_{n \times n}$ where $P=P^T$ and $P^2=P$ (this makes it an orthogonal
projection matrix). We will call this $P_F$ where $F$ stands for "full". Recall that for
any projection matrix  and any vector $y$
$$
\begin{aligned}
\|Py\|^2 &= (Py)^T(Py) \\
&= y^TP^TPy  \\
& = y^TP^2y \\
&= y^TPy.
\end{aligned}
$$

Let $P_R$ denote projection onto the 1-dimensional model determined by the 1 vector. 
Note that $P_F-P_R$ is again a projection: it projects onto the 
orthogonal complement of the 1 vector within the $(p+1)$-dimensional full model. So, it
is a projection onto a $p$ dimensional space.

We see that 
$$
\begin{aligned}
SSR &= \|(P_F-P_R)Y\|^2 \\
&= Y^T(P_F-P_R)Y \\
&= (\mu+\epsilon)^T(P_F-P_R)(\mu+\epsilon) \\
&= \mu^T(P_F-P_R)\mu + 2 \mu^T(P_F-P_R)\epsilon + \epsilon^T(P_F-P_R)\epsilon \\
&= \|\mu - \bar{\mu} \cdot 1\|^2 + 2 \mu^T(P_F-P_R)\epsilon + \epsilon^T(P_F-P_R)\epsilon \\
\end{aligned}
$$

Now, let's take expectations. The first term is a constant, the cross term
has expected value zero and the expected value of the final term is
$$
p \cdot \sigma^2.
$$
This comes from the fact that
$$
\epsilon^T(P_F-P_R)\epsilon = \|(P_F-P_R)\epsilon\|^2 \sim \sigma^2 \chi^2_p.
$$



## $F$-test revisited

The $F$ test can be thought of as comparing two models:

-   *Full (bigger) model :*

    $$Y_i = \beta_0 + \beta_1 X_{i1} + \dots \beta_p X_{ip} + \varepsilon_i$$

-   *Reduced (smaller) model:*

    $$Y_i = \beta_0  + \varepsilon_i$$

-   The $F$-statistic has the form
    $$F=\frac{(SSE(R) - SSE(F)) / (df_R - df_F)}{SSE(F) / df_F}.$$

- **Note: the smaller model should be nested within the bigger model.**

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_general.svg">


## Matrix formulation


$${ Y}_{n \times 1} = {X}_{n \times (p + 1)} {\beta}_{(p+1) \times 1} + {\varepsilon}_{n \times 1}$$

-   ${X}$ is called the *design matrix* of the model

-   ${\varepsilon} \sim N(0, \sigma^2 I_{n \times n})$ is
    multivariate normal

## $SSE$ in matrix form

$$SSE(\beta) = ({Y} - {X} {\beta})'({Y} - {X} {\beta}) = \|Y-X\beta\|^2_2$$

## Design matrix

Design matrix

-   The design matrix is the $n \times (p+1)$ matrix with entries
    $$X =
       \begin{pmatrix}
       1 & X_{11} & X_{12} & \dots & X_{1,p} \\
       \vdots &   \vdots & \ddots & \vdots \\
       1 & X_{n1} & X_{n2} &\dots & X_{n,p} \\
       \end{pmatrix}$$



```R
n = length(Y)
attach(prostate)
X = cbind(rep(1,n), lcavol, lweight, age, lbph, svi, lcp, pgg45)
detach(prostate)
colnames(X)[1] = '(Intercept)'
head(X)

```

The matrix X is the same as formed by `R`


```R
head(model.matrix(prostate.lm))
```

## Least squares solution


-   Normal equations
    $$\frac{\partial}{\partial \beta_j} SSE \biggl|_{\beta = \widehat{\beta}_{}} = -2 \left({Y\
} - {X} \widehat{\beta}_{} \right)^T {X}_j = 0, \qquad 0 \leq j \leq p.$$

-   Equivalent to $$\begin{aligned}
       ({Y} - {X}{\widehat{\beta}_{}})^T{X} &= 0 \\
       {\widehat{\beta}} &= ({X}^T{X})^{-1}{X}^T{Y}
       \end{aligned}$$

-   Properties: $$\widehat{\beta} \sim N(\beta, \sigma^2 (X^TX)^{-1}).$$

## Multivariate Normal

To obtain the distribution of $\hat{\beta}$ we used the following fact about the 
multivariate Normal.

** Suppose $Z \sim N(\mu,\Sigma)$. Then, for any fixed matrix $A$
$$
AZ \sim N(A\mu, A\Sigma A^T).
$$**

(It goes without saying that the dimensions of the matrix must agree with those of $Z$.)

### How did we derive the distribution of $\hat{\beta}$?

Above, we saw that $\hat{\beta}$ is equal to a matrix times $Y$. The matrix form of our
model is
$$
Y \sim N(X\beta, \sigma^2 I).
$$

Therefore,
$$
\begin{aligned}
\hat{\beta} &\sim N\left((X^TX)^{-1}X^T (X\beta), (X^TX)^{-1}X^T X (\sigma^2 I) (X^TX)^{-1}\right) \\
&\sim N(\beta, \sigma^2 (X^TX)^{-1}).
\end{aligned}
$$

## Least squares solution

Let's verify our equations for $\hat{\beta}$.



```R
beta = as.numeric(solve(t(X) %*% X) %*% t(X) %*% Y)
names(beta) = colnames(X)
print(beta)
print(coef(prostate.lm))
```

# Inference for multiple regression

## Regression function at one point

  One thing one might want to *learn* about the regression function in
    the prostate example is something about the regression function at
    some fixed values of ${X}_{1}, \dots, {X}_{7}$, i.e. what
    can be said about
    $$
 \begin{aligned}
  \beta_0 + 1.3 \cdot \beta_1  &+ 3.6 \cdot \beta_2  + 64 \cdot \beta_3 + \\
    0.1 \cdot \beta_4 &+ 0.2 \cdot \beta_5 - 0.2 \cdot \beta_6 + 25 \cdot \beta_7  
   \end{aligned}$$
    roughly the regression function at “typical” values of the
    predictors.

   The expression above is equivalent to
    $$\sum_{j=0}^7 a_j \beta_j, \qquad a=(1,1.3,3.6,64,0.1,0.2,-0.2,25).$$

## Confidence interval for $\sum_{j=0}^p a_j \beta_j$

-   Suppose we want a $(1-\alpha)\cdot 100\%$ CI for
    $\sum_{j=0}^p a_j\beta_j$.

-   Just as in simple linear regression:

    $$\sum_{j=0}^p a_j \widehat{\beta}_j \pm t_{1-\alpha/2, n-p-1} \cdot SE\left(\sum_{j=0}^p a_j\widehat{\beta}_j\right).$$

`R` will form these coefficients for each coefficient separately when using the `confint` function. These linear combinations are of the form
$$
a_{\tt lcavol} = (0,1,0,0,0,0,0,0)
$$
so that
$$
a_{\tt lcavol}^T\widehat{\beta} = \widehat{\beta}_1 = {\tt coef(prostate.lm)[2]}
$$


```R
confint(prostate.lm, level=0.90)
```


```R
predict(prostate.lm, list(lcavol=1.3, lweight=3.6, age=64, 
                         lbph=0.1, svi=0.2, lcp=-.2, pgg45=25), 
       interval='confidence', level=0.90)
```


## $T$-statistics revisited

Of course, these confidence intervals are based on the standard ingredients of a
$T$-statistic.

-   Suppose we want to test $$H_0:\sum_{j=0}^p a_j\beta_j= h.$$ As in
    simple linear regression, it is based on
    $$T = \frac{\sum_{j=0}^p a_j \widehat{\beta}_j - h}{SE(\sum_{j=0}^p a_j \widehat{\beta\
}_j)}.$$

-   If $H_0$ is true, then $T \sim t_{n-p-1}$, so we reject $H_0$ at
    level $\alpha$ if $$\begin{aligned}
       |T| &\geq t_{1-\alpha/2,n-p-1}, \qquad \text{ OR} \\
       p-\text{value} &= {\tt 2*(1-pt(|T|, n-p-1))} \leq \alpha.
       \end{aligned}$$


`R` produces these in the `coef` table `summary` of the linear regression model. Again, each of these 
linear combinations is a vector $a$ with only one non-zero entry like $a_{\tt lcavol}$ above.


```R
summary(prostate.lm)$coef
```

Let's do a quick calculation to remind ourselves the relationships of the variables
in the table above.


```R
T1 = 0.56954 / 0.08584
P1 = 2 * (1 - pt(abs(T1), 89))
print(c(T1,P1))
```

These were indeed the values for `lcavol` in the `summary` table.

## One-sided tests

-   Suppose, instead, we wanted to test the one-sided hypothesis
    $$H_0:\sum_{j=0}^p a_j\beta_j \leq  h, \  \text{vs.} \ H_a: \sum_{j=0}^p a_j\beta_j > \
 h$$

-   If $H_0$ is true, then $T$ is no longer exactly $t_{n-p-1}$ but we still have
    $$\mathbb{P}\left(T > t_{1-\alpha, n-p-1}\right) \leq 1 - \alpha$$
    so we reject $H_0$ at level $\alpha$ if $$\begin{aligned}
       T &\geq t_{1-\alpha,n-p-1}, \qquad \text{ OR} \\
       p-\text{value} &= {\tt (1-pt(T, n-p-1))} \leq \alpha.
       \end{aligned}$$
       
- **Note: the decision to do a one-sided $T$ test should be made *before* looking at the $T$ statistic. Otherwise, the probability of a false positive is doubled!**

## Standard error of $\sum_{j=0}^p a_j \widehat{\beta}_j$

- In order to form these $T$ statistics, we need the $SE$ of our estimate $\sum_{j=0}^p a_j \widehat{\beta}_j$.

-   Based on matrix approach to regression
    $$SE\left(\sum_{j=0}^p a_j\widehat{\beta}_j \right) = SE\left(a^T\widehat{\beta} \right) = \sqrt{\widehat{\sigma}^2 a^T (X^TX\
)^{-1} a}.$$

-   Don’t worry too much about specific implementation – for much of the effects
    we want `R` will do this for you in
    general.



```R
Y.hat = X %*% beta
sigma.hat = sqrt(sum((Y - Y.hat)^2) / (n - ncol(X)))
cov.beta = sigma.hat^2 * solve(t(X) %*% X)
cov.beta
```

The standard errors of each coefficient estimate are the square root of the diagonal entries. They appear as the
`Std. Error` column in the `coef` table.


```R
sqrt(diag(cov.beta))
```

Generally, we can find our estimate of the covariance function as follows:


```R
vcov(prostate.lm)
```

## Prediction interval

-   Basically identical to simple linear regression.

-   Prediction interval at $X_{1,new}, \dots, X_{p,new}$:
    $$\begin{aligned}
       \widehat{\beta}_0 + \sum_{j=1}^p X_{j,new} \widehat{\beta}_j\pm t_{1-\alpha/2, n-p-1} \times \ \sqrt{\widehat{\sigma}^2 + SE\left(\widehat{\beta}_0 + \sum_{j=1}^p X_{j,new}\widehat{\beta}_j\right)^2}.
       \end{aligned}$$


## Forming intervals by hand

While `R` computes most of the intervals we need,
we could write a function that
explicitly computes a confidence interval
(and can be used for prediction intervals 
with the "extra" argument).

This exercise shows the calculations that R 
is doing under the hood: the function *predict*
is generally going to be fine for our
purposes.


```R
CI.lm = function(cur.lm, a, level=0.95, extra=0) {

     # the center of the confidence interval
     center = sum(a*cur.lm$coef)

     # the estimate of sigma^2
     sigma.hat.sq = sum(resid(cur.lm)^2) / cur.lm$df.resid

     # the standard error of sum(a*cur.lm$coef)
     se = sqrt(extra * sigma.hat.sq + sum((a %*% vcov(cur.lm)) * a))

     # the degrees of freedom for the t-statistic
     df = cur.lm$df
     
     # the quantile used in the confidence interval

     q = qt((1 - level)/2, df, lower.tail=FALSE)

     # upper, lower limits
     upper = center + se * q
     lower = center - se * q
     return(data.frame(center, lower, upper))
}
```


```R
print(CI.lm(prostate.lm, c(1, 1.3, 3.6, 64, 0.1, 0.2, -0.2, 25)))
predict(prostate.lm,
        list(lcavol=1.3,
             lweight=3.6,age=64,lbph=0.1,svi=0.2,lcp=-0.2,pgg45=25), 
        interval='confidence')
```

### Prediction intervals

By using the *extra* argument, we can make
prediction intervals.


```R
print(CI.lm(prostate.lm, c(1, 1.3, 3.6, 64, 0.1, 0.2, -0.2, 25), extra=1))
predict(prostate.lm,
        list(lcavol=1.3,lweight=3.6,age=64,lbph=0.1,
             svi=0.2,lcp=-0.2,pgg45=25), 
        interval='prediction')

```

## Arbitrary contrasts

If we want, we can set the intercept term to 0. This allows us to construct confidence interval for, say, how much the `lpsa` score will change will increase if we change `age` by 2 years and `svi` by 0.5 units, leaving everything else unchanged. 

Therefore, what we want is a confidence interval for 2 times the coefficient of `age` + 0.5 times the coefficient of `lbph`:
$$
2 \cdot \beta_{\tt age} + 0.5  \cdot \beta_{\tt svi}
$$

Most of the time, *predict* will do what you want so this 
won't be used too often.



```R
CI.lm(prostate.lm, c(0,0,0,2,0,0.5,0,0))
```

## Questions about many (combinations) of $\beta_j$’s

-   In multiple regression we can ask more complicated questions than in
    simple regression.

-   For instance, we could ask whether `lcp` and `pgg45` 
explains little of the variability in the data, and might be dropped
from the regression model.

-   These questions can be answered answered by $F$-statistics.

- **Note: This hypothesis should really be formed *before* looking at the output of
`summary`.**

- Later we'll see some examples of the messiness when forming a hypothesis after seeing
the `summary`...


## Dropping one or more variables

-   Suppose we wanted to test the above hypothesis
    Formally, the null hypothesis is: $$ H_0: \beta_{\tt lcp} (=\beta_6) =\beta_{\tt pgg45} (=\beta_7) =0$$
    and the alternative is
    $$
    H_a = \text{one of $ \beta_{\tt lcp},\beta_{\tt pgg45}$ is not 0}. 
    $$

-   This test is an $F$-test based on two models $$\begin{aligned}
       Full: Y_i &= \beta_0 + \sum_{j=1}^7  X_{ij} \beta_j + \varepsilon_i \\
       Reduced: Y_i &= \beta_0 + \sum_{j=1}^5 \beta_j X_{ij} + \varepsilon_i \\
       \end{aligned}$$

-   **Note:    The reduced model $R$ must be a special case of the full model $F$
    to use the $F$-test. **

## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_general.svg">


## $SSE$ of a model

-   In the graphic, a “model”, ${\cal M}$ is a subspace of
    $\mathbb{R}^n$ (e.g. column space of ${X}$).

-   Least squares fit = projection onto the subspace of ${\cal M}$,
    yielding predicted values $\widehat{Y}_{{\cal M}}$

-   Error sum of squares:
    $$SSE({\cal M}) = \|Y - \widehat{Y}_{{\cal M}}\|^2.$$



## $F$-statistic for $H_0:\beta_{\tt lcp}=\beta_{\tt pgg45}=0$

-   We compute the $F$ statistic the same to compare any models
     $$\begin{aligned}
       F &=\frac{\frac{SSE(R) - SSE(F)}{2}}{\frac{SSE(F)}{n-1-p}} \\
       & \sim F_{2, n-p-1}       \qquad   (\text{if $H_0$ is true})
       \end{aligned}$$

-   Reject $H_0$ at level $\alpha$ if $F > F_{1-\alpha, 2, n-1-p}$.


When comparing two models, one a special case of the other (i.e. 
one nested in the other), we can test if the smaller
model (the special case) is roughly as good as the 
larger model in describing our outcome. This is typically
tested using an *F* test based on comparing
the two models. The following function does this.
 
 


```R
f.test.lm = function(R.lm, F.lm) {
    SSE.R = sum(resid(R.lm)^2)
    SSE.F = sum(resid(F.lm)^2)
    df.num = R.lm$df - F.lm$df
    df.den = F.lm$df
    F = ((SSE.R - SSE.F) / df.num) / (SSE.F / df.den)
    p.value = 1 - pf(F, df.num, df.den)
    return(data.frame(F, df.num, df.den, p.value))
}
```

`R` has a function that does essentially the same thing as `f.test.lm`: the function is 
`anova`. It can be used several ways, but it can be used to compare two models.


```R
reduced.lm = lm(lpsa ~ lcavol + lbph + lweight + age + svi, data=prostate)
print(f.test.lm(reduced.lm, prostate.lm))
anova(reduced.lm, prostate.lm)
```

## Dropping an arbitrary subset

- For an arbitrary model, suppose we want to test
   $$    \begin{aligned}
   H_0:&\beta_{i_1}=\dots=\beta_{i_j}=0 \\
   H_a:& \text{one or more of $\beta_{i_1}, \dots \beta_{i_j} \neq 0$}
   \end{aligned}
   $$
   for some subset $\{i_1, \dots, i_j\} \subset \{0, \dots, p\}$.
   
-   You guessed it: it is based on the two models: $$\begin{aligned}
       R: Y_i &= \sum_{l=0, l \not \in \{i_1, \dots, i_j\}}^p \beta_l X_{il} + \varepsilon_i \\
       F: Y_i &=  \sum_{l=0}^p \beta_l X_{il} + \varepsilon_i \\
       \end{aligned}$$ where $X_{i0}=1$ for all $i$.

- **Note: This hypothesis should really be formed *before* looking at the output of
`summary`. Looking at `summary` before deciding which to drop is problematic!**

 ## Dropping an arbitrary subset
 
 - Statistic: $$
   \begin{aligned}
   F &=\frac{\frac{SSE(R) - SSE(F)}{j}}{\frac{SSE(F)}{n-p-1}} \\
   & \sim F_{j, n-p-1}     \qquad    (\text{if $H_0$ is true})
   \end{aligned}
   $$
   
 - Reject $H_0$ at level $\alpha$ if $F > F_{1-\alpha, j, n-1-p}$.



## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_general_full.svg">


## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_general_reduced.svg">


## Geometry of Least Squares

<img src="http://stats191.stanford.edu/figs/axes_general.svg">


## General $F$-tests

-   Given two models $R \subset F$ (i.e. $R$ is a subspace of $F$), we
    can consider testing $$  H_0: \text{$R$ is adequate (i.e. $\mathbb{E}(Y) \in R$)} $$ vs. $$ H_a: \text{$F$ is adequate (i.e. $\mathbb{E}(Y) \in F$)}
    $$
    
    - The test statistic is $$  F = \frac{(SSE(R) - SSE(F)) / (df_R - df_F)}{SSE(F)/df_F} $$

-   If $H_0$ is true, $F \sim F_{df_R-df_F, df_F}$ so we reject $H_0$ at
    level $\alpha$ if $F > F_{1-\alpha, df_R-df_F, df_F}$.


## Constraining $\beta_{\tt lcavol}=\beta_{\tt svi}$ 

In this example, we might suppose that the
coefficients for `lcavol` and `svi` are the same
and want to test this. We do this, again, by
comparing a "full model" and a "reduced model".



-   Full model:
    $$\begin{aligned}
    Y_i &= \beta_0 + \beta_1 X_{i,{\tt lcavol}}  + \beta_2 X_{i,{\tt lweight}} + \beta_3 X_{i, {\tt age}} \\
    & \qquad+ \beta_4 X_{i,{\tt lbph}} + \beta_5 X_{i, {\tt svi}} + \beta_6  X_{i, {\tt lcp}} + \beta_7 X_{i, {\tt pgg45}} + \varepsilon_i
    \end{aligned}
    $$

-   Reduced model: 
$$\begin{aligned}
    Y_i &= \beta_0 + \tilde{\beta}_1 X_{i,{\tt lcavol}}  + \beta_2 X_{i,{\tt lweight}} + \beta_3 X_{i, {\tt age}} \\
    & \qquad+ \beta_4 X_{i,{\tt lbph}} + \tilde{\beta}_1 X_{i, {\tt svi}} + \beta_6  X_{i, {\tt lcp}} + \beta_7 X_{i, {\tt pgg45}} + \varepsilon_i
    \end{aligned}
    $$
    


```R
prostate$Z = prostate$lcavol + prostate$svi
equal.lm = lm(Y ~ Z + lweight + age + lbph + lcp + pgg45, data=prostate)
f.test.lm(equal.lm, prostate.lm)
```

## Constraining $\beta_{\tt lcavol}+\beta_{\tt svi}=1$ 

-   Full model:
$$\begin{aligned}
    Y_i &= \beta_0 + \beta_1 X_{i,{\tt lcavol}}  + \beta_2 X_{i,{\tt lweight}} + \beta_3 X_{i, {\tt age}} \\
    & \qquad+ \beta_4 X_{i,{\tt lbph}} + \beta_5 X_{i, {\tt svi}} + \beta_6  X_{i, {\tt lcp}} + \beta_7 X_{i, {\tt pgg45}} + \varepsilon_i
    \end{aligned}
    $$

-   Reduced model: 
    $$\begin{aligned}
    Y_i &= \beta_0 + \tilde{\beta}_1 X_{i,{\tt lcavol}}  + \beta_2 X_{i,{\tt lweight}} + \beta_3 X_{i, {\tt age}} \\
    & \qquad+ \beta_4 X_{i,{\tt lbph}} + (1 - \tilde{\beta}_1) X_{i, {\tt svi}} + \beta_6  X_{i, {\tt lcp}} + \beta_7 X_{i, {\tt pgg45}} + \varepsilon_i
    \end{aligned}
    $$
    


```R
prostate$Z2 = prostate$lcavol - prostate$svi
constrained.lm = lm(lpsa ~ Z2 + lweight + age + lbph + lcp + pgg45, 
                    data=prostate, 
                    offset=svi)
anova(constrained.lm, prostate.lm)
f.test.lm(constrained.lm, prostate.lm)
```

What we had to do above was subtract *X3* from *Y* on the right hand
side of the formula. R has a way to do this called using an *offset*. What this does
is it subtracts this vector from $Y$ before fitting.

## General linear hypothesis

An alternative version of the $F$ test can be derived that does not require
refitting a model.

Suppose we want to test $$H_0:C_{q \times (p+1)}\beta_{(p+1) \times 1} = h$$ vs.
$$
H_a :C_{q \times (p+1)}\beta_{(p+1) \times 1} \neq h.
$$

This can be tested via an $F$ test:
$$
F = \frac{(C\hat{\beta}-h)^T \left(C(X^TX)^{-1}C^T \right)^{-1} (C\hat{\beta}-h) / q}{SSE(F) / df_F} \overset{H_0}{\sim} F_{q, n-p-1}.
$$

**Note: we are assuming that $\text{rank}(C(X^TX)^{-1}C^T)=q$.**


Here's a function that implements this computation.


```R
general.linear = function(model.lm, linear_part, null_value=0) {
    # shorthand
    
    C = linear_part
    h = null_value
    
    beta.hat = coef(model.lm)
    V = as.numeric(C %*% beta.hat - null_value)
    invcovV = solve(C %*% vcov(model.lm) %*% t(C)) # the MSE is included in vcov
    
    df.num = nrow(C)
    df.den = model.lm$df.resid
    F = t(V) %*% invcovV %*% V / df.num
    p.value = 1 - pf(F, df.num, df.den)
    return(data.frame(F, df.num, df.den, p.value))
}
```

Let's test verify this work with our test for testing $\beta_{\tt lcp}=\beta_{\tt pgg45}=0$.


```R
A = matrix(0, 2, 8)
A[1,7] = 1
A[2,8] = 1
print(A)
```


```R
general.linear(prostate.lm, A)
f.test.lm(reduced.lm, prostate.lm)
```
