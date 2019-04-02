
## Some help for R

In this short notebook, I will go through a few basic 
examples in `R` that you may find useful for the course.

These are just some of the things I find useful. Feel free to search
around for others. 

For those of you who have done some programming before, you will notice
that `R` is a functional programming language.

### Functions

You might get tired of always typing http://stats191.stanford.edu/data.
You could make a small function


```R
useful_function = function(dataname) {
    return(paste("http://stats191.stanford.edu/data/", dataname, sep=''))
}

useful_function("groundhog.table")
```

Let's load the heights data with less code


```R
h.table = read.table(useful_function("groundhog.table"), header=TRUE, sep=',')   
head(h.table)
```

Or, for all data sets in the course directory, we might try


```R
course_dataset = function(dataname, sep='', header=TRUE) {
    read.table(useful_function(dataname), header=header, sep=sep)
}
head(course_dataset('groundhog.table', sep=','))
```

Note that I didn't use `return` in the function above. By default, `R` will return the
object in the last line of the function code.


```R
test_func = function(x) {
    x^2
    3
}
test_func(4)
```

### Source

When working on a particular project or assignment, it is often
easiest to type commands in a text editor and rerun them several times.
The command *source* is an easy way to do this, and it takes either
the name of a file or a URL as argument. 
Suppose we have a
webpage http://stats191.stanford.edu/R/helper_code.R


Then, we can execute this as follows


```R
source("http://stats191.stanford.edu/R/helper_code.R")
head(course_dataset("groundhog.table", sep=','))
```

As you go through the course, you might copy this file to a your computer
and add some other useful functions
to this file. 

For larger collections of functions, `R` allows the creation of packages that can be 
installed and loaded with a call to the `library` function. Documentation on packages can be found [here](http://cran.r-project.org/doc/manuals/R-exts.html).



### Concatenation, sequences

Many tasks involving sequences of numbers.  Here are some basic examples on 
how to manipulate and create sequences.

The function `c`, concatenation, is used often in R, as are
`rep` and `seq`


```R
X = 3
Y = 4
c(X,Y)
```

The function `rep` denotes *repeat*.


```R
print(rep(1,4))
print(rep(2,3))
c(rep(1,4), rep(2,3))
```

The function `seq` denotes sequence. There are various ways of specifying the sequence.


```R
seq(0,10,length=11)
```


```R
seq(0,10,by=2)
```

You can sort and order sequences


```R
X = c(4,6,2,9)
sort(X)
```

Use an ordering of X to sort a list of Y in the same order


```R
Y = c(1,2,3,4)
o = order(X)
X[o]
Y[o]
```

A word of caution. In `R` you can overwrite builtin functions so try not to call variables `c`:


```R
c = 3
c
```

However, this has not overwritten the function `c`.


```R
c(3,4,5)
```

Other variables to be careful are the aliases `T` for `TRUE` and `F` for `FALSE`. Since we compute $t$ and $F$ statistics it is natural to also have variables named `T` so when you are expecting `T` to be `TRUE` you might get a surprise.


```R
c(T,F)
```

For other style advice, try reading Hadley Wickham's [style guide](http://adv-r.had.co.nz/Style.html). This is part of a fairly extensive online [book](http://adv-r.had.co.nz). Google also has a [style guide](http://google-styleguide.googlecode.com/svn/trunk/Rguide.xml).

### Indexing

Often times, we will want to extract a subset of rows (or columns) of a vector (or matrix). `R` supports using logical vectors as index objects.


```R
X = c(4,5,3,6,7,9)
Y = c(4,2,65,3,5,9)
X[Y>=5]
```

Suppose we have a `data.frame` and want to extract from rows or columns. Rows are the first of two indexing objects while columns correspond to the second indexing object. Suppose we want to find take the mother and daughter heights where the daughter's height is less than or equal to 62 inches. Note the "," in the square brackets below: this tells `R` that it is looking for a subset of *rows* of the `data.frame`.


```R
library(alr3)
data(heights)
head(heights)
subset_heights = heights[heights$Dheight <= 62,]
print(c(nrow(heights), nrow(subset_heights)))
```

### Plotting

`R` has a very rich plotting library. Most of our plots will be
fairly straightforward, "scatter plots".



```R
X = c(1:40)
Y = 2 + 3 * X + rnorm(40) * 10
plot(X, Y)

```

The plots can be made nicer by adding colors and using different symbols.
See the help for function *par*.



```R
plot(X, Y, pch=21, bg='red')
```

You can add titles, as well as change the axis labels.


```R
plot(X, Y, pch=23, bg='red', main='A simulated data set', xlab='Predictor', ylab='Outcome')
```

Lines are added with *abline*. We'll add some lines to our previous plot: a yellow line with
intercept 2, slope 3, width 3, type 2, as well as a vertical line at x=20 and horizontal line at y=60.


```R
plot(X, Y, pch=23, bg='red', main='A simulated data set', xlab='Predictor', ylab='Outcome')
abline(2, 3, lwd=3, lty=2, col='yellow') 
abline(h=60, col='green')   
abline(v=20, col='red')
```

You can add points and lines to existing plots. In this example, we plot
the first 20 points in red in one call to `plot`, then add the rest in blue with
an orange line connecting them.


```R
plot(X[1:20], Y[1:20], pch=21, bg='red', xlim=c(min(X),max(X)), ylim=c(min(Y),max(Y)))
points(X[21:40], Y[21:40], pch=21, bg='blue')        
lines(X[21:40], Y[21:40], lwd=2, lty=3, col='orange')   
```

You can put more than one plot on each device. Here we create
a 2-by-1 grid of plots


```R
par(mfrow=c(2,1))
plot(X, Y, pch=21, bg='red')
plot(Y, X, pch=23, bg='blue')
par(mfrow=c(1,1))
```

### Saving plots

Plots can be saved as *pdf*, *png*, *jpg* among other formats.
Let's save a plot in a file called "myplot.jpg"


```R
jpeg("myplot.jpg")
plot(X, Y, pch=21, bg='red')
dev.off()
```

Several plots can be saved using *pdf* files. This example has
two plots in it.


```R
pdf("myplots.pdf")
# make whatever plot you want
# first page
plot(X, Y, pch=21, bg='red')

# a new call to plot will make a new page
plot(Y, X, pch=23, bg='blue')

# close the current "device" which is this pdf file
dev.off()


```

### Loops

It is easy to use *for* loops in R


```R
for (i in 1:10) {
    print(i^2)
}
```


```R
for (w in c('red', 'blue', 'green')) {
    print(w)
}
```

Note that big loops can get really slow, a
drawback of many high-level languages.

### Builtin help

R has a builtin help system, which can be accessed and searched as follows

    > help(t.test)
    > help.search('t.test')

Many objects also have examples that show you their usage

    > example(lm)


```R
help(t.test)
```


```R
example(lm)
```

### Distributions in R

In practice, we will often be using the distribution (CDF), quantile (inverse
CDF) of standard random variables like the *T*, *F*, chi-squared and normal.


The standard 1.96 (about 2) standard deviation rule for $\alpha=0.05$:
(note that 1-0.05/2=0.975)



```R
qnorm(0.975)
```

We might want the $\alpha=0.05$ upper quantile for an F with 2,40 degrees of 
freedom:


```R
qf(0.95, 2, 40)
```

So, any observed F greater than 3.23 will get rejected at the $\alpha=0.05$ level.
Alternatively, we might have observed an F of 5 with
2, 40 degrees of freedom, and want the p-value


```R
1 - pf(5, 2, 40)
```

Let's compare this p-value with a 
chi-squared with 2 degrees of freedom,
which is like an F with infinite
degrees of freedom in the denominator (send 40 to infinity).
We also should multiply the 5 by 2
because it's divided by 2 (numerator
degrees of freedom) in the F.


```R
c(1 - pchisq(5*2, 2), 1 - pf(5, 2, 4000))
```

Other common distributions used in applied statistics are `norm` and `t`.

### Other references

* [An Introduction to R](http://cran.r-project.org/doc/manuals/R-intro.pdf)
  
* [R for Beginners](http://cran.r-project.org/doc/contrib/Paradis-rdebuts_en.pdf)

* [Modern Applied Statistics with S](http://www.stats.ox.ac.uk/pub/MASS4/)

* [Practical  ANOVA and Regression in R](http://cran.r-project.org/doc/contrib/Faraway-PRA.pdf)

* [simpleR](http://cran.r-project.org/doc/contrib/Verzani-SimpleR.pdf)

* [R Reference Card](http://cran.r-project.org/doc/contrib/Short-refcard.pdf)

* [R Manuals](http://cran.r-project.org/manuals.html)

* [R Wiki](http://wiki.r-project.org/)


