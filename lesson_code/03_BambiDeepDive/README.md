# Bambi Deep Dive

## Lesson References

### Methodological/Technical

* [Bayesian Analysis with Python](https://www.amazon.com/Bayesian-Analysis-Python-Practical-probabilistic/dp/1805127160)
    * Chapter 2 for probabilistic programming in PyMC
    * Chapter 4 for linear regression
* [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)
    * Chapter 4
* [Doing Bayesian Data Analysis](https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial-dp-0124058884/dp/0124058884/)
    * Chapter 17: Simple linear regression
    * Chapter 18: Multiple linear regression
* [Regression and Other Stories](https://www.cambridge.org/highereducation/books/regression-and-other-stories/DD20DD6C9057118581076E54E40C372C)
    * From Chapter 6 to Chapter 10
* [Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/welcome.html)
    * Chapter 3
* [Bayesian Workflow](https://arxiv.org/abs/2011.01808)

### Data

* Puranen, J., "Fish Catch data set", Journal of Statistics Education Data Archive (1917).
* I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

### Resources

* [Multiple linear regression in Bambi](https://bambinos.github.io/bambi/notebooks/ESCS_multiple_regression.html)
* [Plot conditional adjusted predictions in Bambi](https://bambinos.github.io/bambi/notebooks/plot_predictions.html)
* [Plot comparisons in Bambi](https://bambinos.github.io/bambi/notebooks/plot_comparisons.html)
* [Plot slopes in Bambi](https://bambinos.github.io/bambi/notebooks/plot_slopes.html)
* [Advanced usage of the interpret sub-module in Bambi](https://bambinos.github.io/bambi/notebooks/interpret_advanced_usage.html)


## Section Summaries

### 10. Introduction

In this section we'll introduce the topics we'll learn in this lesson. Primarily:

* Bambi
* Identifiability

### 20. The World's Simplest Model, Now Simpler

* Learn by doing: we create our first Bambi model
* See how that to create a Bambi model we need: 
    * A model formula
    * A data frame
* See how Bambi creates a PyMC model under the hood

### 30. Peeking Under The Hood

What's in a Bambi model?

* A formula: Describes the response and the predictors
* A family: Determines the likelihood function
* A link: Transforms the mean of the likelihood function
* Data: Where variable values are taken from
* Priors: The prior distributions for the parameters in the model

### 40. Sloping Up

In this section we show that

* Adding predictors is very easy 
    * Just add the name on the right side of the model formula
* Bambi also provides great utilities such as
    * Plot priors
    * Get a graph of the model
    * Plot predictions with `plot_predictions()`

### 50. Transformations in Bambi

Here we see that

* Model formulas support in-line transformations
* There are transformations of two kinds: 
    * Built-in Bambi: Common transformations are already implemented
    * External: Defined by you or a third-party library

### 60. Modeling Categories

In this section we learn

* Not all variables are numeric
* With Bambi, adding a categorical predictor is as easy as adding a numerical one
* Some results are surprising if we don't know what's happening internally

### 70. Parameter Identifiability

We cover a challenging topic: parameter identifiability. We'll see how

* Parameter non-identifiability is a common problem in statistics 
    * There is no unique solution for model parameters
    * The value of one or more parameters is a function of the others
* Bambi handles non-identifiabilities automatically 
    * It explains why some results may be surprising if we don't know it

### 80. Understanding Encodings

You'll learn that

* It's easy to introduce non-identifiabilities when adding categorical variables
* Parameter restrictions are used to make the model identifiable
* Alternative parametrizations or encodings can represent the same model

### 90. The Full Model

In this section we show

* Complex models are still a one-liner in Bambi 
    * We create a regression with varying intercepts and slopes for the species
* We introduce the interaction : operator 
    * The meaning depends on the types of the variables
* `plot_predictions()` can handle multiple covariates


### 100. Predictions

You'll see how

* Predictions were never easier 
    * Both in-sample and out-of-sample data!
* All you need is `.predict()`


### 110. An End to End Trip with Bambi

We show

* Prior elicitation is critical in any serious work
* It's important to consider 
    * Domain knowledge
    * Common sense
    * Data (last resort)
* Model evaluation is valuable and needed 
    * Visualizations
    * Numerical summaries