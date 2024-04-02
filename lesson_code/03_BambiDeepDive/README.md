# Bambi Deep Dive

## Concepts Introduced

- Bambi
    - Formula syntax
    - Automatic priors
- Transofrmations revisited
- Interaction effects
- Parameters identifiability
- Encoding categorical predictors
    - Restrictions 
    - Alternative parametrizations 
- Prior elicitation

### 20. The world's simplest model, now simpler

- Learn by doing: we create our first Bambi model
- To create a Bambi model we need:
    - A model formula
    - A data frame
- Bambi creates a PyMC model under the hood

### 30. Peeking under the hood

- What's in a Bambi model?
    - A formula: Describes the response and the predictors
    - A family: Determines the likelihood function
    - A link: Transforms the mean of the likelihood function
    - Data: Where variable values are taken from
    - Priors: The prior distributions for the parameters in the model

### 40. Sloping up

- Adding predictors is very easy
    - Just add the name on the right side of the model formula
- Bambi also provides utilities
    - Plot priors
    - Get a graph of the model
    - Plot predictions with `plot_cap()`

### 50. Transformations in Bambi

- Model formulas support in-line transformations
- There are transformations of two kinds:
    - Built-in Bambi: Common transformations are already implemented
    - External: Defined by you or a third-part library

### 60. Modeling categories

- Not all variables are numeric
- With Bambi, adding a categorical predictor is as easy as adding a numerical one
- Some results are surpising if we don't know what's happening internally

### 70. Parameter identifiability

- Parameter non-identifiability is a common problem in statistics
    - There is no unique solution for model parameters
    - The value of one or more parameters is a function of the others
- Bambi handles non-identifiabilities automatically
    - It explains why some results may be surprising if we don't know it

### 80. Understanding encodings

- It's easy to introduce non-identifiabilities when adding categorical variables
- Parameter restrictions are used to make the model identifiable
- Alternative parametrizations or encodings can represent the same model

### 90. The full model

- Complex models are still a one-liner in Bambi
    - We create a regression with varying intercepts and slopes for the species
- We introduce the interaction `:` operator
    - The meaning depends on the types of the variables
- `plot_cap()` can handle multiple covariates

### 100. Predictions

- Predictions were never easier
    - Both in-sample and out-of-sample data
- All you need is `.predict()`

### 110. An end to end trip with Bambi

- Prior elicitation is critical in any serious work
- It's important to consider
    - Domain knowledge
    - Common sense
    - Data (last resort)
- Model evaluation is valuable and needed
    - Visualizations
    - Numerical summaries