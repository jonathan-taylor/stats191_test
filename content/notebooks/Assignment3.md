

**You may discuss homework problems with other students, but you have to prepare the written assignments yourself. **

**Please combine all your answers, the computer code and the figures into one file, and submit a copy in your dropbox on canvas.**

**Due date: 11:59 PM, February 15, 2017.**



# Question 1 (ALSM, 6.18)

A researcher in a scientific foundation wished to evaluate the relation between intermediate and senior level annuals salaries of bachelor’s and master’s level mathematicians (`Y`, in thousand dollars) and an index of work quality (`X1`), number of years of experience (`X2`), and an index of publication success (`X3`). The data for a sample of 24 bachelor’s and master’s level mathematicians can be found at [http://www.stanford.edu/class/stats191/data/math-salaries.table](http://www.stanford.edu/class/stats191/data/math-salaries.table).

1. Make the scatter plot matrix and the correlation matrix of the table. Summarize the results.

2. Fit a linear regression model for salary based on `X1, X2, X3`. Report the fitted regression function.

3. Test the overall goodness of fit for the regression model at level $\alpha = 0.10$. Specify the null and alternative hypotheses, as well as the test used.

4. Give Bonferroni corrected simultaneous 90 % confidence intervals for $\beta_1, \beta_2, \beta_3$.

5. What is the $R^2$ of the model? How is the $R^2$ interpreted? What is the adjusted $R^2$?

6. The researcher wishes to find confidence interval estimates at certain levels of the `X` variables found in [http://stats191.stanford.edu/data/salary_levels.table](http://stats191.stanford.edu/data/salary_levels.table). Construct Bonferonni corrected simultaneous 95% confidence intervals at each of the columns of the above table.

# Question 2

The dataset `state.x77` in R contains the following statistics (among others) related to the 50 states
of the United States of America:

* `Population`: population estimate (1975)

* `Income`: per capita income (1974)

* `Illiteracy`: illiteracy (1970, percent of population)

* `HS.Grad`: percent high school graduates (1970)


```R
state.data = data.frame(state.x77)
```

We are interested in the relation between Income and other 3 variables.

1. Produce a 4 by 4 scatter plot of the variables above.

2. Fit a multiple linear regression model to the data with Income as the
dependent variable, and Population, Illiteracy, HS.Grad as the independent
variables. Comment on the significance of the variables in the model using the
result of summary.

3. Produce standard diagnostic plots of the multiple regression fit in part 2.

4. Plot dffits of the observations and find observations which have high influence,
using critical value 0.5.

5. Plot Cook's distance of the observations and find observations which have high
influence, using critical value 0.1. Compare with the result of part 4.

6. Find states with outlying predictors by looking at the leverage values. Use
critical value 0.3.

7. Find outliers, if any, in the response. Remove them from the data and refit a
multiple linear regression model and compare the result with the previous fit.

8. As a summary, find all the influential states using `influence.measures` function.

# Question 3

The	dataset	`iris`	in `R`	gives the measurements in centimeters of the
variables sepal length and width and petal length and width, respectively, for 
50 flowers from each of 3 species of iris. 


```R
data(iris)
```

 

1. Fit a multiple linear regression model to the data with sepal length as 
the dependent variable and sepal width, petal length and petal width 
as the independent variables. 

2. Test the reduced model of $H_0: \beta_{\tt sepal width}=\beta_{\tt petal length} = 0$
with an
F-test at level $\alpha=0.05$

3. Test $H_0: \beta_{\tt sepal width} = \beta_{\tt petal length}$ at level $\alpha=0.05$

4. Test $H_0: \beta_{\tt sepal width} < \beta_{\tt petal length}$ at level $\alpha=0.05$.

# Question 4 

We revisit Tomasetti's and Vogelstein's study on cancer incidence across tissues from Assignment 2. The second part of their paper deals with the existence of two clusters in the dataset: According to the authors, D-tumours (D for deterministic) can be attributed to some degree to environmental and genetic factors, while the risk of R-tumours (R for replicative) is affected mainly by random mutations occuring during replication of stem cells.

1. The dataset also includes a column `Cluster` according to the classification of that tumour as Deterministic or Replicative. Fit a linear model as in Assignment 2, but with a different slope for D- and R-tumours. 

2. Draw a scatterplot, as well as the two regression lines.

3. Conduct a F-test to compare the regression model above to the regression model which does not account for this classification. What is the p-value?

4. Given that in the study the two clusters were assigned based on the dataset (i.e. based on `Lscd` and `Risk`), do you think the logic behind the p-value from part 3 is OK?


(Remark: The authors did not actually conduct the F-test from part 1; they only argued that the two "clusters" are meaningful.)

# Question 5

When running $F$ or $T$ tests we've noted that the null hypothesis we're looking to test should be specified before
looking at the data. Similarly, parameter for which we want 
confidence intervals should be specified before hand. This question will explore what happens when we choose
our parameters after looking at the data with a seemingly natural way of choosing which intervals to report.

1. Write a function that generates a sample $(X_i,Y_i)_{1 \leq i \leq n}$ with $X_i$ a vector of 
independent standard normals of length 10 and $n=100$. Generate $Y$ using the regression function
$$
f(X) = 1 + 0.1 \cdot X_1
$$
and $N(0,1)$ random errors.

2. Fit a model `lm(Y ~ X)`, computing the features for which the p-value is less than 10% and returning
95% confidence intervals for those selected coefficients. What number should each of these numbers cover? That is, if we form a 95% confidence interval for the effect of $X_3$ what should the interval cover? (Note that there are 11 coefficients so we want 11 different numbers.) How often do your intervals cover what they should? A hint for computation: write a function that returns a vector of length 11 as follows: if a feature is selected return 1 if the interval covers and 0 otherwise; if the feature is not covered set the value to be `NA`. Store these results as rows  in a matrix and compute the mean of each column (removing `NA`).

3. Using the same model, test the overall goodness of fit of the model using the $F$ test against the model `lm(Y ~ 1)` at level 10%. If we reject this null hypothesis, return the 90% confidence intervals for the effect of all the features. How often do these intervals cover what they should?



# Question 6 (ALSM 19.14)

A research laboratory was developing a new compound for the relief of severe cases of hay fever. In an experiment with 36 volunteers, the amounts of the two active ingredients (factors `A` and `B`) in the compound were varied at three levels each. Randomization was used in assigning four volunteers to each of the nine treatments. The data can be found at [http://stats191.stanford.edu/data/hayfever.table](http://stats191.stanford.edu/data/hayfever.table).


1.  Fit the two-way ANOVA model, including interactions. What is the
estimated mean when Factor `A` is 2 and Factor `B` is 1?

2. Using `R`’s standard regression plots, plot the `qqplot` of the residuals. Is there any serious violation of normality?

3. This question asks you to graphically summarize the data. Create a plot with Factor `A` on the x-axis, and, using 3 different plotting symbols, the mean for each level of Factor `B` above each level of Factor `A` (see kidney data example). Does there appear to be any interactions?

4. Test for an interaction at level $\alpha = 0.05$.

5. Test for main effects of Factors A and B at level $\alpha = 0.05$.
