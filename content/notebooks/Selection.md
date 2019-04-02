
## Model selection

In a given regression situation, there are often many choices to
be made. Recall our usual setup
$$
Y_{n \times 1} = X_{n \times p} \beta_{p \times 1} + \epsilon_{n \times 1}.
$$

Any *subset $A \subset \{1, \dots, p\}$* yields a new regression model
$$
{\cal M}(A): Y_{n \times 1} = X[,A] \beta[A] + \epsilon_{n \times 1}
$$
by setting $\beta[A^c]=0$.

**Model selection** is, roughly speaking, how to choose $A$ among the
$2^p$ possible choices.



```R
options(repr.plot.width=5, repr.plot.height=5)
```

### Election data

Here is a dataset from the book that we will use to explore different model selection approaches.

Variable | Description
--- | ---
$V$ | votes for a presidential candidate
$I$ | are they incumbent?
$D$ | Democrat or Republican incumbent?
$W$ | wartime election?
$G$ | GDP growth rate in election year
$P$ | (absolute) GDP deflator growth rate
$N$ | number of quarters in which GDP growth rate $> 3.2\%$


```R
url = 'http://stats191.stanford.edu/data/election.table'
election.table = read.table(url, header=T)
pairs(election.table[,2:ncol(election.table)], cex.labels=3, pch=23,
      bg='orange', cex=2)
```

## Problem & Goals

* When we have many predictors (with many possible interactions), it can be difficult to find a good model.
* Which main effects do we include?
* Which interactions do we include?
* Model selection procedures try to *simplify / automate* this task.
* Election data has $2^6=64$ different models with just main effects!

## General comments

- This is generally an "unsolved" problem in statistics: there are no magic procedures to get you the "best model."

- Many machine learning methods look for good "sparse" models: selecting a "sparse" model.

- "Machine learning" often work with very many predictors.

- Our model selection problem is generally at a much smaller scale than "data mining" problems.

- Still, it is a hard problem.

- **Inference after selection is full of pitfalls!** 



## Hypothetical example
* Suppose we fit a a model $F: \quad Y_{n \times 1} = X_{n \times p} \beta_{p \times 1} + \varepsilon_{n \times 1}$ with predictors ${ X}_1, \dots, { X}_p$.
* In reality, some of the $\beta$’s may be zero. Let’s suppose that $\beta_{j+1}= \dots= \beta_{p}=0$.
* Then, any model that includes $\beta_0, \dots, \beta_j$ is *correct*: which model gives the *best* estimates of $\beta_0, \dots, \beta_j$?
* Principle of *parsimony* (i.e. Occam’s razor) says that the model with *only* ${X}_1, \dots, {X}_j$ is "best".

## Justifying parsimony

- For simplicity, let’s assume that $j=1$ so there is only one coefficient to estimate.
- Then, because each model gives an *unbiased* estimate of $\beta_1$ we can compare models based on $\text{Var}(\widehat{\beta}_1).$
- The best model, in terms of this variance, is the one containing only ${ X}_1$.
- What if we didn’t know that only $\beta_1$ was non-zero (which we don’t know in general)?
- In this situation, we must choose a set of variables.

## Model selection: choosing a subset of variables

* To "implement" a model selection procedure, we first need a criterion or benchmark to compare two models.
* Given a criterion, we also need a search strategy.
* With a limited number of predictors, it is possible to search all possible models (`leaps` in `R`).

## Candidate criteria

Possible criteria:

* $R^2$: not a good criterion. Always increase with model size $\implies$ "optimum" is to take the biggest model.
* Adjusted $R^2$: better. It "penalized" bigger models. Follows principle of parsimony / Occam’s razor.
* Mallow’s $C_p$ – attempts to estimate a model’s predictive power, i.e. the power to predict a new observation.

### Best subsets, $R^2$

Leaps takes a design matrix as argument: throw away the intercept
column or leaps will complain.


```R
election.lm = lm(V ~ I + D + W + G:I + P + N, election.table)
election.lm
```


```R
X = model.matrix(election.lm)[,-1]
library(leaps)
election.leaps = leaps(X, election.table$V, nbest=3, method='r2')
best.model.r2 = election.leaps$which[which((election.leaps$r2 == 
                                            max(election.leaps$r2))),]
best.model.r2
```

Let's plot the $R^2$ as a function of the model size. We see that the
full model does include all variables. 


```R
plot(election.leaps$size, election.leaps$r2, pch=23, bg='orange', cex=2)
```

## Best subsets, adjusted $R^2$

-   As we add more and more variables to the model – even random ones,
    $R^2$ will increase to 1.

-   Adjusted $R^2$ tries to take this into account by replacing sums of squares by *mean squares*
    $$R^2_a = 1 - \frac{SSE/(n-p-1)}{SST/(n-1)} = 1 - \frac{MSE}{MST}.$$


```R
election.leaps = leaps(X, election.table$V, nbest=3, method='adjr2')
best.model.adjr2 = election.leaps$which[which((election.leaps$adjr2 == max(election.leaps$adjr2))),]
best.model.adjr2
```


```R
plot(election.leaps$size, election.leaps$adjr2, pch=23, bg='orange', 
     cex=2)
```

### Mallow’s $C_p$

- $C_p({\cal M}) = \frac{SSE({\cal M})}{\widehat{\sigma}^2} + 2 \cdot p({\cal M}) - n.$
- $\widehat{\sigma}^2=SSE(F)/df_F$ is the "best" estimate of $\sigma^2$ we have (use the fullest model), i.e. in the election data it uses all 6 main effects.
- $SSE({\cal M})$ is the $SSE$ of the model ${\cal M}$.
- $p({\cal M})$ is the number of predictors in ${\cal M}$.
- This is an estimate of the expected mean-squared error of $\widehat{Y}({\cal M})$, it takes *bias* and *variance* into account.


```R
election.leaps = leaps(X, election.table$V, nbest=3, method='Cp')
best.model.Cp = election.leaps$which[which((election.leaps$Cp == 
                                            min(election.leaps$Cp))),]
best.model.Cp
```


```R
plot(election.leaps$size, election.leaps$Cp, pch=23, bg='orange', cex=2)
```

## Search strategies 

* Given a criterion, we now have to decide how we are going to search through the possible models.

* "Best subset": search all possible models and take the one with highest $R^2_a$ or lowest $C_p$ leaps. Such searches are typically
feasible only up to $p=30$ or $40$ at the very most.

* Stepwise (forward, backward or both): useful when the number of predictors is large. Choose an initial model and be "greedy".

* "Greedy" means always take the biggest jump (up or down) in your selected criterion.

### Implementations in `R`

* "Best subset": use the function `leaps`. Works only for multiple linear regression models.
* Stepwise: use the function `step`. Works for any model with Akaike Information Criterion (AIC). In multiple linear regression, AIC is (almost) a linear function of $C_p$.

### Akaike / Bayes Information Criterion

* Akaike (AIC) defined as $$AIC({\cal M}) = - 2 \log L({\cal M}) + 2 \cdot p({\cal M})$$ where $L({\cal M})$ is the maximized likelihood of the model.
* Bayes (BIC) defined as $$BIC({\cal M}) = - 2 \log L({\cal M}) + \log n \cdot p({\cal M})$$
* Strategy can be used for whenever we have a likelihood, so this generalizes to many statistical models.

### AIC for regression

* In linear regression with unknown $\sigma^2$ $$-2 \log L({\cal M}) = n \log(2\pi \widehat{\sigma}^2_{MLE}) + n$$ where $\widehat{\sigma}^2_{MLE} = \frac{1}{n} SSE(\widehat{\beta})$
* In linear regression with known $\sigma^2$ $$-2 \log L({\cal M}) = n \log(2\pi \sigma^2) + \frac{1}{\sigma^2} SSE(\widehat{\beta})$$ so AIC is very much like Mallow’s $C_p$ in this case.


```R
n = nrow(X)
p = 7 + 1 
c(n * log(2*pi*sum(resid(election.lm)^2)/n) + n + 2*p, AIC(election.lm))
```

### Properties of AIC / BIC

* BIC will typically choose a model as small or smaller than AIC (if using the same search direction).

* As our sample size grows, under some assumptions,
it can be shown that
     - AIC will (asymptotically) always choose a model that contains the true model, i.e. it won’t leave any variables out.
     - BIC will (asymptotically) choose exactly the right model.

### Election example

Let's take a look at `step` in action. Probably the simplest
strategy is *forward stepwise* which tries to add one variable at a time, 
as long as it can find a resulting model whose AIC is better than 
its current position. 

When it can make no further additions, it terminates.


```R
election.step.forward = step(lm(V ~ 1, election.table), 
                             list(upper = ~ I + D + W + G + G:I + P + N), 
                             direction='forward', 
                             k=2, trace=FALSE)
election.step.forward
```


```R
summary(election.step.forward)
```

## Interactions and hierarchy

We notice that although the *full* model we gave it had the interaction `I:G`, the function `step` never tried to use it. This is 
due to some rules implemented in `step` that do not include an interaction unless both main effects are already in the model. In this case, because neither $I$ nor $G$ were added, the interaction was never considered.

In the `leaps` example, we gave the function the design matrix
and it did not have to consider interactions: they were already encoded in the design matrix.

### BIC example

The only difference between AIC and BIC is the price paid
per variable. This is the argument `k` to `step`. By default `k=2` and for BIC
we set `k=log(n)`. If we set `k=0` it will always add variables.


```R
election.step.forward.BIC = step(lm(V ~ 1, election.table), 
                                 list(upper = ~ I + D + W +G:I + P + N), 
                                 direction='forward', k=log(nrow(X)))
```


```R
summary(election.step.forward.BIC)
```

### Backward selection

Just for fun, let's consider backwards stepwise. This starts at a full
model and tries to delete variables.

There is also a `direction="both"` option.


```R
election.step.backward = step(election.lm, direction='backward')
```


```R
summary(election.step.backward)
```

## Cross-validation

Yet another model selection criterion is 
$K$-fold cross-validation.

- Fix a model ${\cal M}$. Break data set into $K$ approximately equal sized groups $(G_1, \dots, G_K)$.
- For (i in 1:K) Use all groups except $G_i$ to fit model, predict outcome in group $G_i$ based on this model $\widehat{Y}_{j,{\cal M}, G_i}, j \in G_i$.
- Similar to what we saw in Cook's distance / DFFITS.
- Estimate $CV({\cal M}) = \frac{1}{n}\sum_{i=1}^K \sum_{j \in G_i} (Y_j - \widehat{Y}_{j,{\cal M},G_i})^2.$

### Comments about cross-validation.

* It is a general principle that can be used in other situations to "choose parameters."
* Pros (partial list): "objective" measure of a model's predictive power.
* Cons (partial list): all we know about inference is *usually* "out the window" (also true for other model selection procedures).
* If goal is not really inference about certain specific parameters, it is a reasonable way to compare models.


```R
library(boot)
election.glm = glm(V ~ ., data=election.table)
cv.glm(model.frame(election.glm), election.glm, K=5)$delta
```

### $C_p$ versus 5-fold cross-validation

- Let's plot our $C_p$ versus the $CV$ score.

- Keep in mind that there is additional randomness in the $CV$ score
due to the random assignments to groups. 



```R
election.leaps = leaps(X, election.table$V, nbest=3, method='Cp')
V = election.table$V
election.leaps$cv = 0 * election.leaps$Cp
for (i in 1:nrow(election.leaps$which)) {
    subset = c(1:ncol(X))[election.leaps$which[i,]]
    if (length(subset) > 1) {
       Xw = X[,subset]
       wlm = glm(V ~ Xw)
       election.leaps$CV[i] = cv.glm(model.frame(wlm), wlm, K=5)$delta[1]
    }
    else {
       Xw = X[,subset[1]]
       wlm = glm(V ~ Xw)
       election.leaps$CV[i] = cv.glm(model.frame(wlm), wlm, K=5)$delta[1]
    }
}
```


```R
plot(election.leaps$Cp, election.leaps$CV, pch=23, bg='orange', cex=2)
```


```R
plot(election.leaps$size, election.leaps$CV, pch=23, bg='orange', cex=2)
best.model.Cv = election.leaps$which[which((election.leaps$CV 
                                            == min(election.leaps$CV))),]
best.model.Cv
```

## Summarizing results

The model selected depends on the criterion used.

Criterion | Model
--- | ---
$R^2$ | ~ $ I + D + W +G:I + P + N$
$R^2_a$ | ~ $ I + D + P + N$
$C_p$ | ~ $D+P+N$
AIC forward | ~ $D+P$
BIC forward | ~ $D$
AIC backward | ~ $I + D + N + I:G$
5-fold CV | ~ $ I+W$

**The selected model is random and depends on which method we use!**

## Where we are so far

- Many other "criteria" have been proposed.
- Some work well for some types of data, others for different data.
- Check diagnostics!
- These criteria (except cross-validation) are not "direct measures" of predictive power, though Mallow’s $C_p$ is a step in this direction.
- $C_p$ measures the quality of a model based on both *bias* and *variance* of the model. Why is this important?
- *Bias-variance* tradeoff is ubiquitous in statistics. More soon.

## A larger example

- Resistance of $n=633$ different HIV+ viruses to drug 3TC.

- Features $p=91$ are mutations in a part of the HIV virus, response is log fold change
in vitro.




```R
X_HIV = read.table('http://stats191.stanford.edu/data/NRTI_X.csv', header=FALSE, sep=',')
Y_HIV = read.table('http://stats191.stanford.edu/data/NRTI_Y.txt', header=FALSE, sep=',')
set.seed(0)
Y_HIV = as.matrix(Y_HIV)[,1]
X_HIV = as.matrix(X_HIV)
nrow(X_HIV)
```

## Forward stepwise


```R
D = data.frame(X_HIV, Y_HIV)
M = lm(Y_HIV ~ ., data=D)
M_forward = step(lm(Y_HIV ~ 1, data=D), list(upper=M), 
                 trace=FALSE, direction='forward')
M_forward
```

## Backward stepwise


```R
M_backward = step(M, list(lower= ~  1), trace=FALSE, 
                  direction='backward')
M_backward
```

## Both directions


```R
M_both1 = step(M, list(lower= ~  1, upper=M), trace=FALSE, 
               direction='both')
M_both1
```


```R
M_both2 = step(lm(Y_HIV ~ 1, data=D), list(lower= ~  1, upper=M), trace=FALSE, direction='both')
M_both2
```

## Compare selected models


```R
sort(names(coef(M_forward)))
sort(names(coef(M_backward)))
sort(names(coef(M_both1)))
sort(names(coef(M_both2)))
```

## BIC vs AIC


```R
M_backward_BIC = step(M, list(lower= ~  1), trace=FALSE, 
                      direction='backward', k=log(633))
M_forward_BIC = step(lm(Y_HIV ~ 1, data=D), list(upper=M), trace=FALSE, 
                     direction='forward', k=log(633))
M_both1_BIC = step(M, list(upper=M, lower=~1), trace=FALSE, 
                     direction='both', k=log(633))
M_both2_BIC = step(lm(Y_HIV ~ 1, data=D), list(upper=M, lower=~1), trace=FALSE, 
                     direction='both', k=log(633))

sort(names(coef(M_backward_BIC)))
sort(names(coef(M_forward_BIC)))
sort(names(coef(M_both1_BIC)))
sort(names(coef(M_both2_BIC)))
```

## Inference after selection: data snooping and splitting

Each of the above criteria return a model. The `summary` provides
$p$-values.


```R
summary(election.step.forward)
```

We can also form confidence intervals. **But, can we trust these intervals or tests? No!**


```R
confint(election.step.forward)
```

## How bad could it really be?

To illustrate the "dangers" of trusting the above $p$-values, I will
use the `selectiveInference` package which has a variant of forward stepwise.



```R
library(selectiveInference)
plot(fs(X, V))
```


```R
fsInf(fs(X, V))
```

Let's generate data for which we know all coefficients
are zero and look at the $p$-values. 

We will look at the *naive* p-values which ignore selection, as well
as a certain kind of corrected $p$-values from `selectiveInference`.


```R
X_fake = matrix(rnorm(4000), 100, 40)
nsim = 1000
naiveP = c()
for (i in 1:nsim) {
    Z = rnorm(nrow(X_fake))
    fsfit = fs(X_fake, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    naive.lm = lm(Z ~ X_fake[,fsinf$vars[1]])
    naiveP = c(naiveP, summary(naive.lm)$coef[2,4])
}
```

What is the type I error of the naive test?


```R
ecdf(naiveP)(0.05)
```

## Data splitting

- A simple fix for this problem is to randomly split the data into two groups (often but not always of equal size).

- This gives valid inference under some assumptions.

- Which variable is chosen is random as it depends on the split. *Selected model might not be as "good".*

- Also, we have access to less data to form confidence intervals ($\implies$ wider), test hypotheses ($\implies$ less power).

## When is data splitting OK?

- Just as when we did not select a model, when computing $p$-values and
confidence intervals we use a model.

- If we are happy with these assumptions for a model selected from the data, then
data splitting is OK.


```R
splitP = c()
for (i in 1:nsim) {
    Z = rnorm(nrow(X_fake))
    subset_s = sample(1:nrow(X_fake), nrow(X_fake)/2, replace=FALSE) 
    #_s for selection
    subset_i = rep(TRUE, nrow(X_fake)) # _i for inference
    subset_i[subset_s] = FALSE
    
    # Step 1: choose your model
    fsfit = fs(X_fake[subset_s,], Z[subset_s], maxsteps=1)
    
    # Step 2: compute p-values
    split.lm = lm(Z ~ X_fake[,fsinf$vars[1]], subset=subset_i)
    splitP = c(splitP, summary(split.lm)$coef[2,4])
}
```


```R
plot(ecdf(splitP), lwd=3, col='red')
abline(0,1, lwd=2, lty=2)
```

It turns out that it is possible to modify the
usual $t$-statistic so that we still get valid tests
after the first step. Its description is a little beyond the level of this course.


```R
exactP = c()
for (i in 1:nsim) {
    Z = rnorm(nrow(X_fake))
    fsfit = fs(X_fake, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    exactP = c(exactP, fsinf$pv[1])
}
```


```R
plot(ecdf(exactP), lwd=3, col='blue', main='Uncorrected vs. corrected null p-values')
plot(ecdf(naiveP), lwd=3, col='red', add=TRUE)
abline(0,1, lwd=2, lty=2)
```

## Power

- Both the exact and data splitting give valid p-values. 

- Which is more powerful?

- Let's put some signal in and see.

- **Note, we can't be sure we are selecting the "right" variable each time so model might
not be "correct". But this can happen even when we write down a model without looking at the data.**



```R
exactP_A = c()
splitP_A = c()
correct_model = c()

beta1 = .3 # signal size

for (i in 1:nsim) {
    Z = rnorm(nrow(X_fake)) + X_fake[,1] * beta1 
    fsfit = fs(X_fake, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    exactP_A = c(exactP_A, fsinf$pv[1])
    
    correct_model = c(correct_model, fsinf$vars[1] == 1)
    
    # data splitting pvalue
    
    subset = sample(1:nrow(X_fake), nrow(X_fake)/2, replace=FALSE)
    subset_c = rep(TRUE, nrow(X_fake))
    subset_c[subset] = FALSE
    fsfit = fs(X_fake[subset,], Z[subset], maxsteps=1)
    split.lm = lm(Z ~ X_fake[,fsinf$vars[1]], subset=subset_c)
    splitP_A = c(splitP_A, summary(split.lm)$coef[2,4])

}
```


```R
plot(ecdf(exactP_A), lwd=3, col='blue', xlim=c(0,1), main='Power comparison (all pvalues)')
plot(ecdf(splitP_A), lwd=3, col='red', add=TRUE)
abline(0,1, lwd=2, lty=2)
```


```R
plot(ecdf(exactP_A[correct_model]), lwd=3, col='blue', xlim=c(0,1), 
     main='Power comparison (correct model)')
plot(ecdf(splitP_A[correct_model]), lwd=3, col='red', add=TRUE)
abline(0,1, lwd=2, lty=2)
```

## Confidence intervals

- Which has shorter confidence intervals?


```R
exactL = c()
splitL = c()
correct_model = c()

beta1 = 0.3
for (i in 1:nsim) {
    Z = rnorm(nrow(X_fake)) + X_fake[,1] * beta1 
    fsfit = fs(X_fake, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    exactL = c(exactL, fsinf$ci[1,2] - fsinf$ci[1,1])

    correct_model = c(correct_model, fsinf$vars[1] == 1)
    
    # data splitting pvalue
    
    subset = sample(1:nrow(X_fake), nrow(X_fake)/2, replace=FALSE)
    subset_c = rep(TRUE, nrow(X)) #_c for confirmatory
    subset_c[subset] = FALSE
    fsfit = fs(X_fake[subset,], Z[subset], maxsteps=1)
    split.lm = lm(Z ~ X_fake[,fsinf$vars[1]], subset=subset_c)
    conf = confint(split.lm, level=0.9)
    splitL = c(splitL, conf[2,2] - conf[2,1])

}
```


```R
data.frame(median(exactL), median(splitL))
```


```R
data.frame(median(exactL[correct_model]), median(splitL[correct_model]))
```

### An exciting area of research

This `exactP` was only discovered [recently](http://amstat.tandfonline.com/doi/abs/10.1080/01621459.2015.1108848)!

For a long time, I have been saying things along the lines of

    Inference after model selection 
    is basically out the window. 
    
    Forget all we taught you about t and 
    F distributions as it is no longer true...
    
    Use data splitting!
    
It turns out that inference after selection is possible, and it
doesn't force us to throw away all of our tools for inference.

*But, it is a little more complicated to describe.*

