

**You may discuss homework problems with other students, but you have to prepare the written assignments yourself. Late homework will be penalized 10% per day.**

**Please combine all your answers, the computer code and the figures into one file, and submit a copy in your dropbox on coursework.**

** Due March 15, 11:59PM.**

 **Grading scheme: 10 points per question, total of 40.**

# Question 1

Fot this problem, use the HIV resistance data in the penalized regression slides.

1. Randomly split the data in half. 

2. Using the first half of the data, fit the LASSO with cross-validation using `cv.glmnet`. Extract the coefficients at `lambda.min` and `lambda.1se`. Are the estimates sparse or are all coefficients non-zero? (Answer will depend somewhat on the seed you use -- set an integer seed and save it.)

3. Using the variables selected on the first half of the data, fit a model using `lm` on the second half of the data and report confidence intervals for the regression parameters in the model with the selected features. You can find the mutation names identified by position and amino acid here: [http://stats191.stanford.edu/data/NRTI_muts.txt](http://stats191.stanford.edu/data/NRTI_muts.txt).


# Question 2

In this question we will use the same data generating function from Q.5 of Assignment 4, i.e. a noisy version of `lpsa` of `data(prostate)` with `k=20` junk features. Below we will ask for `k=50` junk features as well. 

1. Generate noise as in Q.5 of Assignment 4 with 20 junk features. Randomly split the data in half.

2. Using the first half of the data: fit the LASSO with parameter `lambda.1se` as selected by `cv.glmnet`, store the coefficients in a vector `beta.lasso`; do the same but for ridge regression storing the result in `beta.ridge`.

3. Evaluate how well `beta.lasso` and `beta.ridge` predict on the second half of the data using mean squared error. Which one has smaller mean-squared error? (Answer will depend somewhat on the seed you use.)

4. Repeat steps 1.-3. using `k=50` junk features. 

# Question 3 (Based on RABE 12.3)

The O-rings in the booster rockets used in space launching play an important part in preventing rockets from exploding. Probabilities of O-ring failures are thought to be related to temperature. The data from 23 flights are given in [this file](http://stats191.stanford.edu/data/Orings.table)

For each flight we have an indicator of whether or not any O-rings were damaged and the temperature of the launch.

1. Fit a logistic regression, modeling the probability of having any O-ring failures based on the temperature of the launch. Interpret the coefficients in terms of odds ratios.

2. From the fitted model, find the probability of an O-ring failure when the temperature at launch was 31 degrees. This was the temperature forecast for the day of the launching of the fatal Challenger flight on January 20, 1986.

3. Find an approximate 95% confidence interval for the coefficient of temperature in the logistic regression using both the `summary` and `confint`. Are the confidence intervals the same? Why or why not?

# Question 4 (Based on RABE 12.5)


[Table 1.12](http://www1.aucegypt.edu/faculty/hadi/RABE5/Data5/P014.txt) of the textbook 
describes variables in a study of health care in 52 health care facilities in New Mexico 
in the year 1988. The variables collected are:

Variable | Description
---------|-------------
RURAL    | Is hospital in a rural or non-rural area?
BED      | Number of beds in facility.
MCDAYS   | Annual medical in-patient days (hundreds).
TDAYS    | Annual total patient days (hundreds).
PCREV    | Annual total patient care revenue (\$100).
NSAL     | Annual nursing salaries (\$100).
FEXP     | Annual facilities expenditures (\$100).
NETREV   | PCREV - NSAL - FEXP

1. Using a logistic regression model, test the null hypothesis that the measured covariates have no power to distinguish between rural facilities and than non-rural facilities. Use level $\alpha=0.05$.

2. Use a model selection technique based on AIC to choose a model that seems to best describe
the outcome `RURAL` based on the measured covariates. 

3. Repeat 2. but using BIC instead. Is the model the same?

4. Report estimates of the parameters for the variables in your final model. How are these
to be interpreted? 

5. Report confidence intervals for the parameters in 4. Do you think you can trust these intervals?
