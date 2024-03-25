# Overdispersion

## Lesson References

### Methodological/Technical

- [Statistical Rethinking](https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-STAN/McElreath/p/book/9780367139919)
    - Chapter 12
- [Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/)
    - Check pages 114-117 
- [Foundations of Linear and Generalized Linear Models](https://www.wiley.com/en-us/Foundations+of+Linear+and+Generalized+Linear+Models-p-9781118730034)
    - Chapter 8
- [Categorical Data Analysis](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
    - Several chapters talk about Overdispersion
- [Extending the Linear Model with R](https://www.taylorfrancis.com/books/mono/10.1201/9781315382722/extending-linear-model-julian-faraway)
    - Chapter 3
- [Some Models for Overdispersed Binomial Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-842X.1988.tb00844.x)

### Data

- [Fish Species Diversity in Lakes](https://www.journals.uchicago.edu/doi/abs/10.1086/282927)
- [Sex Bias in Graduate Admissions: Data From Berkeley](https://www.science.org/doi/10.1126/science.187.4175.398)
- [Polymorphism in the rates of meiotic drive acting on the B-chromosome of Myrmeleotettix macu/a tus](https://www.nature.com/articles/hdy198572)
- Manly B. (1978) Regression models for proportions with extraneous variance. Biometrie-Praximetrie, 18, 1-18.

### Resources

- [ArviZ documentation on R-hat](https://python.arviz.org/en/stable/api/generated/arviz.rhat.html)
    - See papers linked in there
- [A Poisson–Gamma Mixture Is Negative-Binomially Distributed](https://gregorygundersen.com/blog/2019/09/16/poisson-gamma-nb/)
- [Gamma, Poisson, and negative binomial distributions](https://timothy-barry.github.io/posts/2020-06-16-gamma-poisson-nb/)
- [Gamma Poisson reference](https://const-ae.name/post/2021-01-24-gamma-poisson-distribution/gamma-poisson-reference/)
- [Probability Playground: The Beta-Binomial Distribution](https://www.acsu.buffalo.edu/~adamcunn/probability/betabinomial.html)
    - Also check the same tool for other distributions
- [Beta-Binomial Wikipedia page](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
    - [In Bayesian statistics](https://en.wikipedia.org/wiki/Beta-binomial_distribution#Beta-binomial_in_Bayesian_statistics)
    - [Related distributions](https://en.wikipedia.org/wiki/Beta-binomial_distribution#Related_distributions)


## Section Summaries

### 10. 12 segundos de oscuridad

This first section is a motivational one:

* First, we accept that learning complicated things is hard
* We assume we'll get right into the overdispersion darkness

### 20. Unanticipated consequences

In this section we:

* Introduce and exlpore a new dataset -- fish species diversity
* Create a Poisson regression model to...
    * Understand association between lake size and number of species
    * Predict number of species given lake size
* But then, we encounter a problem...
    * Overdispersion comes into scene

Remember: things can go wrong even in simple scenarios!

### 30. What's the deal with overdispersion

This section answers the following two questions:

* What is overdispersion?
* Why is it a problem?

And remember, the Poisson distribution can be too simple!

### 40. When the whole is more than the sum of the parts

Here we learn how to create new distributions from existing distributions
and show one case that will be very useful in the future.

* Engineering new distributions
* The Gamma-Poisson distribution
* The negative binomial distribution

### 50. Negative binomial regression

We're going to use what we just learned about the negative binomial distribution with a new Bambi model.

* A new model family: negative binomial
* Perform osterior predictive checks
* Explore the conditional posterior predictive distribution

### 60. Overdispersion for all

In this lesson we learn that overdispersion appears in another very common scenario: logistic regression.

* We introduce the student's admissions dataset
* Create a binomial model
* Face overdispersion issues, again
* Realize that overdispersion can affect estimates we care about

### 70. Beta-binomial to the rescue

Previously, we "created" a new distribution to deal with overdispersion with Poisson regression. Here, we do the same with the binomial distribution.

* Handcrafting another distribution
* The beta-binomial distribution
* Understanding distribution parameters

### 80. The beta-bnomial model in action

In this section we learn how to deal with overdispersion using the beta-binomial distributon.

* Set priors that make sense
* Create a beta-binomial model in Bambi
* Compare the estimated effect with the one obtained with the binomial model

### 90. Counting grasshoppers

In the previous section we solved an "artificial" problem, the data was too simple.

Here we create a binomial regression on another real dataset.

* Yet another dataset: Grasshoppers
* Create the binomial regression model
* Analyze the output with forest plots
* Explore the posterior predictive distibutions

The overdispersion problem can be subtle, too!

### 100. Beta-binomial for grasshoppers

Finally, we show the beta-binomial model in action with the recently introduced (and more realistic) problem using the grasshoppers dataset.

* Create a beta-binomial model with more predictors
* Specify sensible priors
* Compare inferences: binomial vs beta-binomial
* Perform posterior predictive checks