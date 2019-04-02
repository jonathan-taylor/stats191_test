

```R
library(selectiveInference)
```


```R
X = matrix(rnorm(2000), 100, 20)
nsim = 1000
```


```R
exactP_A = c()
splitP_A = c()
exactL = c()
splitL = c()

beta1 = .3 # signal size

for (i in 1:nsim) {
    Z = rnorm(nrow(X)) + X[,1] * beta1 
    
    # selectiveInference
    
    fsfit = fs(X, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    exactP_A = c(exactP_A, fsinf$pv[1])
    exactL = c(exactL, fsinf$ci[1,2] - fsinf$ci[1,1])

    # data splitting pvalue
    
    split_ = c(rep(FALSE, nrow(X)/2), rep(TRUE, nrow(X)/2))[sample(1:nrow(X), nrow(X), replace=FALSE)]
    fsfit = fs(X[split_,], Z[split_], maxsteps=1)
    split.lm = lm(Z ~ X[,fsinf$vars[1]], subset=!split_)
    splitP_A = c(splitP_A, summary(split.lm)$coef[2,4])
    conf = confint(split.lm, level=0.9)
    splitL = c(splitL, conf[2,2] - conf[2,1])

}
```


```R
plot(ecdf(exactP_A), lwd=3, col='blue', xlim=c(0,1))
plot(ecdf(splitP_A), lwd=3, col='red', add=TRUE)
abline(0,1, lwd=2, lty=2)
```


```R
data.frame(median(exactL), median(splitL))
```

## $\beta=0.5$


```R
exactP_A = c()
splitP_A = c()
exactL = c()
splitL = c()

beta1 = .5 # signal size

for (i in 1:nsim) {
    Z = rnorm(nrow(X)) + X[,1] * beta1 
    
    # selectiveInference
    
    fsfit = fs(X, Z, maxsteps=1)
    fsinf = fsInf(fsfit, sigma=1)
    exactP_A = c(exactP_A, fsinf$pv[1])
    exactL = c(exactL, fsinf$ci[1,2] - fsinf$ci[1,1])

    # data splitting pvalue
    
    split_ = c(rep(FALSE, nrow(X)/2), rep(TRUE, nrow(X)/2))[sample(1:nrow(X), nrow(X), replace=FALSE)]
    fsfit = fs(X[split_,], Z[split_], maxsteps=1)
    split.lm = lm(Z ~ X[,fsinf$vars[1]], subset=!split_)
    splitP_A = c(splitP_A, summary(split.lm)$coef[2,4])
    conf = confint(split.lm, level=0.9)
    splitL = c(splitL, conf[2,2] - conf[2,1])

}
```


```R
plot(ecdf(exactP_A), lwd=3, col='blue', xlim=c(0,1))
plot(ecdf(splitP_A), lwd=3, col='red', add=TRUE)
abline(0,1, lwd=2, lty=2)
```


```R
data.frame(median(exactL), median(splitL))
```
