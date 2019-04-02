

**You may discuss homework problems with other students, but you have to prepare the written assignments yourself.**

**Please combine all your answers, the computer code and the figures into one file, and submit a copy in your dropbox on coursework.**

**Due date: February 1, 2019.**

**Grading scheme: 10 points per question, total of 60.**

# Question 1


In a recent, exciting, but also controversial Science article, [Tomasetti and Vogelstein](http://science.sciencemag.org/content/347/6217/78.full) attempt to explain why cancer incidence varies drastically across tissues (e.g. why one is much more likely to develop lung cancer rather than pelvic bone cancer). The authors show that a higher average lifetime risk for a cancer in a given tissue correlates with the rate of replication of stem cells in that tissue. The main inferential tool for their statistical analysis was a simple linear regression, which we will replicate here. 

You can download the dataset as follows:




```R
tomasetti = read.csv("https://stats191.stanford.edu/data/Tomasetti.csv")
```

The dataset contains information about 31 tumour types. The `Lscd` (Lifetime stem cell divisions) column refers to the total number of stem cell divisions during the average lifetime, while `Risk` refers to the lifetime risk for
cancer of that tissue type.

1.  Fit a simple linear regression model to the data with `log(Risk)` as the dependent variable and `log(Lscd)` as the independent variable. Plot the estimated regression line. 

2. Add upper and lower 95% prediction bands for the regression line on the plot, using `predict`. That is, produce one line for the upper limit of each interval over a sequence of densities, and one line for the lower limits of the intervals. Interpret these bands at a `Lscd` of $10^{10}$.

3. Add upper and lower 95% confidence bands for the regression line on the plot, using `predict`. That is, produce one line for the upper limit of each interval over a sequence of densities, and one line for the lower limits of the intervals. Interpret these bands at a `Lscd` of $10^{10}$.

4. Test whether the slope in this regression is equal to 0 at level $\alpha=0.05$. State the null hypothesis, the alternative, the conclusion and the $p$-value.

5. Give a 95% confidence interval for the slope of the regression line. Interpret your interval.

6. Report the $R^2$ and the adjusted $R^2$ of the model, as well as an estimate of the variance of the errors in the model.

7. Provide an interpretation of the $R^2$ you calculated above. According to a [Reuters article](http://www.reuters.com/article/health-cancer-luck-idUSL1N0UE0VF20150101) "Plain old bad luck plays a major role in determining who gets cancer and who does not, according to researchers who found that two-thirds of cancer incidence of various types can be blamed on random mutations and not heredity or risky habits like smoking." Is this interpretation of $R^2$ correct?


# Question 2 

Let $Y$ and $X$ denote variables in a simple linear regression of median home prices versus median income in state in the US. Suppose that the model
$$
Y = \beta_0 + \beta_1 X + \epsilon
$$
satisfies the usual regression assumptions.

The table below is a table similar to the output of `anova` when passed a simple linear regression model.


    Response: Y
              Df Sum Sq Mean Sq F value    Pr(>F)    
    X          1     NA    5291      NA        NA
    Residuals 48 181289      NA

1. Compute the missing values of in the above table.

2. Test the null hypothesis $H_0 : \beta_1 = 0$ at level $\alpha = 0.05$ using the above table.
Can you test the hypothesis $H_0 : \beta_1 < 0$ using Table 1?

3. Compute the $R^2$ for this simple linear regression.

4. If $Y$ and $X$ were reversed in the above regression, what would you
expect $R^2$ to be?



# Question 3

Power is an important quantity in many applications of statistics. This question investigates the power of a test in simple linear regression. In a simple linear regression setting, suppose the true slope of the regression line is $\beta_1$ and the true intercept is $\beta_0$.
If we assume  $\sigma$ is known, then we can test $H_0: \beta_1 =0$ using
$$
Z = \frac{\hat{\beta}_1 - 0}{SD(\hat{\beta}_1)}
$$
where
$SD(\hat{\beta}_1)$ is the standard deviation of our estimator $\hat{\beta}_1$. (We are using $Z$ here instead of $T$ to avoid complication of the degrees of freedom -- just imagine our sample size $n$ was really large. In this case our estimate of variability $SE(\hat{\beta}_1)$ is replaced by the true standard deviation $SD(\hat{\beta}_1)$, i.e. 
we have swapped $\hat{\sigma}^2$ with $\sigma^2$.)

The power of this test is a function of the true value $\beta_1$ as well as
the accuracy of our estimate $SD(\hat{\beta}_1)$. The power is defined as
$$
P_{(\beta_0,\beta_1)}(\text{$H_0$ is rejected}).
$$
That is, the probability we reject the null hypothesis as a function of $(\beta_0, \beta_1)$. Actually, the power will generally not depend on $\beta_0$ in this model, so it is really a function of $\beta_1$ (and $SD(\hat{\beta}_1)$).

As we change the true $\beta_1$, the probability we reject $H_0$ changes: if the true value of $\beta_1$ is much larger than 0 relative to $SD(\hat{\beta}_1)$ then
we are very likely to reject $H_0$.

1. What rule would you use to determine whether or not you reject $H_0$ at level $\alpha=0.05$.

2. What is the distribution of our test statistic $Z$?
Show that the distribution depends only on  the value $\beta_1 / SD(\hat{\beta}_1)$.
We call this quantity the non-centrality parameter or signal to noise ratio (SNR).

3. Plot the power of your test as your function of the SNR.

4. Roughly how large does the non-centrality parameter have to be in order to achieve
power of 85%?

# Question 4

In this problem, we will investigate what happens when the assumptions of the 
simple linear regression model do not hold. When generating data below, set 
$X$ to be equally spaced between 0 and 1 (i.e. `X = seq(0, 1, by=0.01)`) and use the regression function
$$
f(X) = 1 + 2 \cdot X.
$$

1. Write a function to generate data from the simple linear regression model with regression function as above and normally distributed errors, returning the $T$-statistic for testing whether the slope of the regression line is equal to 2.

2. Using your function, run a simulation with 5000 repetitions to see if the $T$-statistic has distribution close to a $T$ distribution. How many degrees of freedom should it have? How often is your $T$ statistic larger than the usual 5% threshold?

3. Write a new function with the same regression function but errors that are normally distributed using, say, `rt` with 5 degrees of freedom to generate errors. Repeat 2. Does the $T$-statistic still have close to a $T$ distribution? How often is your $T$ statistic larger than the usual 5% threshold? Try a few of your own different ways of generating errors as well.

4. Write a new function with same regression function but errors that do not have the same variance though they are normally distributed. Construct errors such that the variance of the $i$-th error is `1+X[i]` (recall that `X` is equally spaced over interval 0 to 1). Repeat 2. Exaggerate the effect of non-constant variance by making the variance `exp(1 + 5 * X[i])`. For both choices of variance, plot the variance as a function of `X`. Does the $T$-statistic still have close to a $T$ distribution in both coses? How often are your $T$ statistics larger than the usual 5% threshold?

5. Write a new function with same regression function but errors that are not independent. Do this by first generating a vector of errors `error` and then returning a new vector whose first entry is `error[1]` but for $i>1$ the $i$-th entry is `error[i-1] + error[i]`. Repeat 2. Does the $T$-statistic still have close to a $T$ distribution? How often is your $T$ statistic larger than the usual 5% threshold?

6. Summarize your findings. Which of the departures from the assumptions for the error term of the simple linear regression model seem important?



# Question 5

The tables below show the regression output of a multiple regression model relating `Salary`, the beginning salaries in dollars of employees in a given company to the following predictor variables: `Education, Experience` and a variable `STEM` indicating whether or not they have an undergraduate degree in a STEM field or not. (The units of both `Education` and `Experience` are years.)

    ANOVA table:

    Response: Salary
                     Df   Sum Sq   Mean Sq  F value   Pr(>F)    
        Regression   NA  2216338        NA       NA       NA 
        Residuals    62  8913083        NA      

    Coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)   3316.4      937.7      NA       NA
    Education      850.0         NA   3.646       NA
    Experience     932.4      260.1      NA       NA
    STEM              NA      330.1   1.675       NA


Below, specify the null and alternative hypotheses, the test used, and your conclusion using $\alpha=0.05$ throughout. You may not necessarily be able to compute everything, but be as explicit as possible.

1. Fill in the missing values in the above table.

2. Test whether or not the linear regression model explains significantly more variability in `Salary` than a model with no explanatory variables. What assumptions are you making?

3. Is there a positive linear relationship between `Salary` and `Experience`, after accounting for the effect of the variables `STEM` and  `Education`? (Hint: one-sided test)

4. What salary interval would you forecast for an electrical engineer with 10 years of education and 5 years working in a related field?

5. What salary interval would you forecast, on average, for english majors with 10 years of education and 6 years in a related field?


# Question 6 (Based on RABE 3.15)

A national insurance organization wanted to study the consumption pattern of cigarettes in all 50 states and the District of Columbia. The variables chosen for the study are:

* Age: Median age of a person living in a state.

* HS: Percentage of people over 25 years of age in a state who had completed high school.

* Income: Per capita personal income for a state (income in dollars).

* Black: Percentage of blacks living in a state.

* Female: Percentage of females living in a state.

* Price: Weighted average price (in cents) of a pack ofcigarettes in a state.

* Sales: Number of packs of cigarettes sold in a state on a per capita basis.

The data can be found at [http://www1.aucegypt.edu/faculty/hadi/RABE5/Data5/P088.txt](http://www1.aucegypt.edu/faculty/hadi/RABE5/Data5/P088.txt).

Below, specify the null and alternative hypotheses, the test used, and your conclusion using a 5% level of significance.

1. Test the hypothesis that the variable `Female` is not needed in the regression equation relating Sales to the six predictor variables.

2. Test the hypothesis that the variables `Female` and `HS` are not needed in the above regression equation.

3. Compute a 95% confidence interval for the true regression coefficient of the variable `Income`.

4. What percentage of the variation in `Sales` can be accounted for when `Income` is removed from the above regression equation? Which model did you use?
