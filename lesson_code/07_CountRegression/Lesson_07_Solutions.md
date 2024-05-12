---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: ib_advanced_regression
    language: python
    name: ib_advanced_regression
---

<!-- #region slideshow={"slide_type": "slide"} -->
# Count Model Regression Exercises
<!-- #endregion -->

```python hideCode=false hidePrompt=false slideshow={"slide_type": "skip"}
import arviz as az
import bambi as bmb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
```

```python hideCode=false hidePrompt=false slideshow={"slide_type": "skip"}
plt.style.use("intuitivebayes.mplstyle")

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.spines.left"] = False
```

# Exercise 1: Transformation and Model Intuition [Easy]

Messing up the order of non linear transformations can lead to wrong answers all too easily. I know because I did this during writing this.

In this exercise were going to work through transformations outside of a model to ensure the difference is clear. we then are going to run a parameter recovery study, implementing models in both PyMC and Bambi to ensure we have an end to end understanding of everything that is going on.


## 1a: Show that the for a random vector show that `np.exp(mean(x)) != mean(np.exp))`

```python
x = np.random.normal(size=1000)
```

```python
x_mean = np.mean(x)

y = np.exp(x)

print("Calculation 1", np.exp(x_mean))
print("Calculation 2", np.mean(y))
```



```python
x = np.random.normal(size=1000)
y = 2
```

```python
print("Calculation 1", np.exp(x + y).mean())
print("Calculation 2", (np.exp(x) + 2).mean())
```

## 1b: Build a Poisson model in PyMC to estimate our known parameters

We're going to do a parameter recovery and prediction verification. Doing this end to end will ensure you're getting every step correct, and especially the transformation steps above correct as it can be easy or tempting to make a mistake.

Specifically we want two things an az.summary table showing the estimated parameters, which we than can compare with the fixed parameters we used for data generation

```python
# Parameters to recover
intercept = 0.3
slope = 0.7
categorical_effect = 0.5

# Random Observations. We create many so out sampler has the best chance of recovering the final value
x = stats.uniform(-1, 1).rvs(22345)
categorical_indicator = stats.bernoulli(p=0.7).rvs(22345)

# Data Generating Process
mu = slope * x + categorical_indicator * categorical_effect + intercept
lamda = np.exp(mu)
y = stats.poisson(lamda).rvs()
```

```python
data = pd.DataFrame({"x": x, "categorical": categorical_indicator, "y": y})
data["categorical"] = pd.Categorical(data["categorical"])
data.head()
```

With the data loaded let's write a PyMC model that takes into account the X input, the categorical indicator, and includes an intercept as well.

```python
with pm.Model() as model:
    slope_var = pm.Normal("slope", 0, 50)
    intercept_var = pm.Normal("intercept", 0, 50)
    categorical_effect_var = pm.Normal("categorical", 0, 50)

    μ = intercept_var + slope_var * x + categorical_effect_var * categorical_indicator
    λ = np.exp(μ)

    rooms = pm.Poisson("y", λ, observed=y)
    idata = pm.sample()
```

```python
pm.model_to_graphviz(model)
```

```python
az.plot_trace(idata);
```

```python
az.summary(idata)
```

Great! We got a model that largely estimates the input parameters.


## 1c: Build a Poisson model in Bambi to estimate our known parameters
Let's now do the same with Bambi to see what we get.

```python
# Build your bambi model here
```

```python
model = bmb.Model("y ~ x + categorical", family="poisson", data=data)
model.build()
model
```

```python
model.graph()
```

```python
idata_bmb = model.fit()
```

```python
az.summary(idata_bmb)
```

After inference we get results similar to the PyMC model. As expected Bambi automatically adds the intercept detects the categorical variable correctly. estimating the "1" level.


## 1d: Estimate the Poisson distribution at a fixed inputs value "by hand", with Bambi and posterior samples
We'll used the fixed values, where x is set a particular float value, and we include the categorical effect as well

You'll need to do three things here
* Simulate the expected Poisson distributions at fixed values of x and categorical indicators by hand
* Use PyMC posterior predictive functionality to show what the same estimate would be
* Use the Bambi predict functionality to generate a posterior predictive to show what the same estimate would be

The output were looking for is three plots, one that shows "true" Poisson distribution, one showing the PyMC estimate, and one showing the Bambi estimate

```python
# Here we provide the values for you
x_new = 5
categorical_indicator_new = 1
data_new = pd.DataFrame({"x": [x_new], "categorical": [categorical_indicator_new]})
data_new["categorical"] = pd.Categorical(data_new["categorical"])

data_new
```

```python
# Simulate the data manually
mu = slope * x_new + categorical_indicator_new * categorical_effect + intercept
_lambda = np.exp(mu)
y_manual = stats.poisson(_lambda).rvs(20000)
```

Calculate mu and lambda here. These are our "true deterministics" that we will be estimating later with PyMC and bambi. After getting. those plot the observed distribution of y_manual as well.

```python
mu, _lambda, y_manual.mean()
```

```python
#az.plot_dist(y_manual);
```

This is our "true" expected Poisson distribution from the data generating process. Note that it roughly ranges from 119 to 183, a range we'll reference later.


Let's now estimate our expected Try using `az.summary`, then try from the samples directly`

```python
summary = az.summary(idata_bmb)
summary
```

```python
mu_new_mean = (
    summary.loc["x", "mean"] * x_new
    + summary.loc["categorical[1]", "mean"] * categorical_indicator_new
    + summary.loc["Intercept", "mean"]
)

lambda_new_mean = np.exp(mu_new_mean)
mu_new_mean, lambda_new_mean
```

```python
# Get the samplpes 
slope_estimate = idata_bmb.posterior["x"].to_numpy().reshape(-1)
categorical_estimate = idata_bmb.posterior["categorical"].to_numpy().reshape(-1)
intercept_estimate = idata_bmb.posterior["Intercept"].to_numpy().reshape(-1)
```

```python
# Mean, Addition, Transform: This also should be wrong
MU_new_mean = (
    slope_estimate.mean() * x_new
    + categorical_estimate.mean() * categorical_indicator_new
    + intercept_estimate.mean()
)

MU_new_mean, np.exp(MU_new_mean)
```

```python
# Vector Add, Transform, Mean: This should be right
mu_new = (
    slope_estimate * x_new
    + categorical_estimate * categorical_indicator_new
    + intercept_estimate
)

# print(mu_new.shape)
lambda_new = np.exp(mu_new)
mu_new.mean(), lambda_new.mean()
```

Let's now estimate the same using Bambi's predict functionality.  We wont inspect the samples for the parameters again as part of this exercise, though you can do so yourself to verify they match the samples from PyMC.

```python
new_predictions = model.predict(
    idata_bmb,
    kind="pps",
    data=data_new,
    inplace=False,
)

new_predictions
```

```python
az.plot_dist(new_predictions.posterior_predictive["y"].values.reshape(-1));
```

# Exercise 2: More football analytics
We're going to reanalyze the football data. We've already looked at the data a couple of ways but perhaps there's more. Let's load the data first before we move onto the questions.

```python
df = pd.read_csv("data/season-1718.csv")
df.head()
```

```python
df.rename({"FTHG": "GoalsHome", "FTAG": "GoalsAway"}, axis=1, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
first_game = df["Date"].min()
df["Days_Since_First_Game"] = (df["Date"] - first_game).dt.days

df = df.iloc[:, [2, 3, 4, 5, -1]]
```

```python
home_goals = (
    df[["HomeTeam", "GoalsHome", "Days_Since_First_Game"]]
    .assign(GameType="Home")
    .rename({"HomeTeam": "Team", "GoalsHome": "Goals"}, axis=1)
)
away_goals = (
    df[["AwayTeam", "GoalsAway", "Days_Since_First_Game"]]
    .assign(GameType="Away")
    .rename({"AwayTeam": "Team", "GoalsAway": "Goals"}, axis=1)
)

long_df = pd.concat([home_goals, away_goals], axis=0)
long_df.head()
```

## 2a: Revisiting home versus away effect
Look back at Premiere League data. Let's answer the following questions.
* Did we use a shared parameter for the effect for all teams when estimating the home and away effect?
* Was the estimated effect the same size for all teams?

This exercise does not require any code, just referencing the lesson and thinking through the answer


**Answer**: 
This is a bit of a trick question. 
We do use a shared parameter for all teams. That's easiest seen using the model graph functionality from Bambi. You can see `Team_dim (20)` but `GameType` is 1, indicating one parameter.
However the estimated effect is not the same. That's because once we transform the sum of the individual team effect and the shared goal effect, the difference in the number of goals from home and away is different in the "observed goal" space.


## 2b: Estimating a home vs away effect per team

Now, actually extend the Bambi model from the lesson, but with one home effect _per_ team, not one home effect for all the teams.

```python
goals_model_bambi = bmb.Model(
    "Goals ~ 0 + Team + GameType:Team", long_df, family="poisson"
)
goals_model_bambi.build()
goals_model_bambi.graph()
```

```python
goals_model_bambi_data = goals_model_bambi.fit()
az.summary(goals_model_bambi_data)
```

## 2c: Team performance evolution

Let's now ask another question: **Does team performance drop over a season?** For this model we want a `Days_Since_First_Game` effect per team and an intercept per team.

Start with Manchester City and estimate the slope and intercept for that first. Check for converge and parameter estimations there first.
Then expand to all teams. Use Bambi for both models.

_Hint: if you run into any issues for the all-teams model, think about the scale of `Days_Since_First_Game`. Anything there that might cause issues?_


### Man City Model

```python
mancity_model = bmb.Model(
    "Goals ~ Days_Since_First_Game",
    long_df.query(f"Team == 'Man City'"),
    family="poisson",
)
mancity_idata = mancity_model.fit()
```

```python
mancity_model.graph()
```

Worked all good! No need to interpret now, let's go directly to the expanded model:


### All teams

```python
all_teams_model = bmb.Model(
    "Goals ~ Days_Since_First_Game:Team + Team + 0", long_df, family="poisson"
)
all_teams_idata = all_teams_model.fit()
```

```python
all_teams_model.graph()
```

```python
az.summary(all_teams_idata)
```

The model structure looks good, but convergence was really bad and it look a long time to sample. This is especically suspicious because the Manchester City model worked, even though it had an identical structure. 

We know the `Team` variable is not an issue. What about `Days_Since_First_Game`?

```python
az.plot_dist(long_df["Days_Since_First_Game"]);
```

Well, well, well! It's clear now: the issue is numerical overflow because `Days_Since_First_Game` can get so large. We need to scale its values. We could do this manually but instead we decide to use Bambi built in scaling function:

```python
all_teams_scaled_model = bmb.Model(
    "Goals ~ scale(Days_Since_First_Game):Team + Team + 0", long_df, family="poisson"
)
all_teams_scaled_idata = all_teams_scaled_model.fit()
```

```python
az.summary(all_teams_scaled_idata)
```

Convergence is all good now! Let's use a forest plot to see this effect:

```python
ax = az.plot_forest(all_teams_scaled_idata, var_names ="scale(Days_Since_First_Game)", filter_vars="like", combined=True)
ax[0].axvline(ls="--", lw=3, color="grey");
```

For some teams it does seem plausible that the performance changed over time. However, for most teams, it does not seem to be the case.


# Exercise 3: Rewriting the fishing model [Hard]
We're going to extend the fishing data. Let's start by loading in the data again.

```python
fish_data = pd.read_stata(
    "data/fish.dta",
    columns=["count", "livebait", "camper", "persons", "child"],
)

fish_data["livebait"] = pd.Categorical(fish_data["livebait"])
fish_data["camper"] = pd.Categorical(fish_data["camper"])
fish_data
```

## 3a Bambi scaled

Write this model again in Bambi, but this time standardizing the persons column (explain why that would even make sense to standardize that column). Think about why it makes sense to scale this for the purposes 



*Why would you want to scale, write your answer here*


Because a group of 0 person doesn't mean anything, so the intercept doesn't mean anything. 0 people catch zero fish. But when we standardize the persons column, the intercept becomes the average number of caught fish, for a group with mean number of persons, 0 children, and 0 whatever the other predictors are.

```python
fish_data["persons"].plot(kind="hist");
```

```python
fish_data["camper"].astype(float).plot(kind="hist");
```

```python
common_priors = {"common": bmb.Prior("Normal", mu=0, sigma=0.5)}

zip_fish_simple = bmb.Model(
    "count ~ 0 + camper + scale(persons) + child", 
    fish_data, 
    family='zero_inflated_poisson',
    priors=common_priors,
)
zip_fish_simple.build()
zip_fish_simple
```

```python
zip_fish_simple_idata = zip_fish_simple.fit()
```

```python
zip_fish_simple_idata
```

```python
az.summary(zip_fish_simple_idata)
```

<!-- #region -->
## 3b: Write the same model in PyMC
Standardize the persons column (this time you can't use Bambi to do it, you have to do it yourself).


We've copied the Bambi model for you for reference:

```python
formula = bmb.Formula(
    "count ~ 0 + camper + persons + child", # Poisson rate, mu
    "psi ~ child"    # Probability of entering Poisson process, psi
)
common_priors = {
    "common": bmb.Prior("Normal", mu=0, sigma=0.5),
    "psi": {"common": bmb.Prior("Normal", mu=0, sigma=0.5)},
}

zip_fish_complex = bmb.Model(
    formula, 
    fish_data, 
    family='zero_inflated_poisson',
    priors=common_priors,
)
```
<!-- #endregion -->

```python
# Google easiest way to normalize python. This also needs some changes to model structure
persons_standardized = (fish_data["persons"] - fish_data["persons"].mean()) / fish_data["persons"].std()
```

```python
az.plot_dist(persons_standardized);
```

```python
with pm.Model() as zip_fish_complex:
    # Data containers
    camper_idx = pm.MutableData("camper_idx", fish_data["camper"].astype(int).to_numpy())
    persons_data = pm.MutableData("persons_data", persons_standardized)
    child_data = pm.MutableData("child_data", fish_data["child"].to_numpy())
    
    ## Poisson model
    # Priors
    camper = pm.Normal("camper", 0, .5, shape=2)
    persons = pm.Normal("persons", 0, .5)
    child = pm.Normal("child", 0, .5)

    # Poisson rate
    μ = pm.Deterministic(
        "μ",
        pm.math.exp(camper[camper_idx] + persons * persons_data + child * child_data),
    )

    ## Zero-inflation process
    # Priors
    psi_intercept = pm.Normal("psi_intercept", 0, .5)
    psi_child = pm.Normal("psi_child", 0, .5)
    
    # Probability of entering Poisson process
    ψ = pm.Deterministic(
        "ψ",
        pm.math.sigmoid(psi_intercept + child_data * psi_child)
    )

    goals = pm.ZeroInflatedPoisson(
        "goals", psi=ψ, mu=μ, observed=fish_data["count"]
    )

    idata = pm.sample()
```

```python
az.plot_trace(idata, var_names=["~μ", "~ψ"]);
```

```python
az.summary(idata, var_names=["~μ", "~ψ"])
```

## 3c: Remake some plots!


Remake these plots using your PyMC model from the previous question and only use ArviZ or Matplotlib to recreate the graphs BAMBI gives you. To recreate the Bambi plots, we need to compute predictions conditional on the covariate of interest, while keeping the other covariates constant. The way Bambi does that under the hood, in the way we've asked it in the lesson, is to __keep covariates _not present_ in `conditional` at their mean (if numbers) or mode (if categories)__. 

For the covariate(s) we want to condition on, Bambi creates a grid of equally spaced values between the minimum and maximum values of the specified explanatory variable.

This exercise is a bit challenging, but it really forces you to think about the samples and how to summarsize them visually into something interesting. It'll also ensure you fully understand what is being plotted by Bambi.

```python
# Replace values of covariates
counterfact_persons = np.linspace(persons_standardized.min(), persons_standardized.max())
m_persons = pm.do(
    zip_fish_complex, 
    {
        "camper_idx": np.full_like(counterfact_persons, fish_data["camper"].mode()[0], dtype="int32"),
        "persons_data": counterfact_persons,
        "child_data": np.full_like(counterfact_persons, fish_data["child"].mean()),
    }
)
# get predictions
with m_persons:
    preds_persons = pm.sample_posterior_predictive(idata, var_names=["μ"])
```

```python
counterfact_child = np.linspace(fish_data["child"].min(), fish_data["child"].max())
m_child = pm.do(
    zip_fish_complex,
    {
        "camper_idx": np.full_like(counterfact_child, fish_data["camper"].mode()[0], dtype="int32"),
        "persons_data": np.full_like(counterfact_child, persons_standardized.mean()),
        "child_data": counterfact_child,
    }
)

with m_child:
    preds_child = pm.sample_posterior_predictive(idata, var_names=["μ", "ψ"])
```

```python
m_camper = pm.do(
    zip_fish_complex, 
    {
        "camper_idx": [0, 1],
        "persons_data": [persons_standardized.mean(), persons_standardized.mean()],
        "child_data": [fish_data["child"].mean(), fish_data["child"].mean()],
    }
)

with m_camper:
    preds_camper = pm.sample_posterior_predictive(idata, var_names=["μ"])
```

```python
_, (left, mid, right) = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

preds = preds_persons.posterior_predictive["μ"]
az.plot_hdi(counterfact_persons, preds, ax=left, color="C0")
left.plot(counterfact_persons, preds.mean(("chain", "draw")), lw="3", color="C0")
left.set(xlabel="persons (stdz)", ylabel="Mean fish count ($\mu$)")

preds = preds_child.posterior_predictive["μ"]
az.plot_hdi(counterfact_child, preds, ax=mid, color="C0")
mid.plot(counterfact_child, preds.mean(("chain", "draw")), lw="3", color="C0")
mid.set(xlabel="child", ylabel="")

hdi = az.hdi(preds_camper.posterior_predictive)["μ"]
right.vlines(
    [0, 1],
    hdi.sel(hdi="lower"),
    hdi.sel(hdi="higher"),
    lw=3,
    colors="C0",
)
right.set(xlabel="camper", ylabel="")
plt.suptitle("$\mu$ as a function of the predictors", fontsize=24)

_, ax = plt.subplots(1, 1, figsize=(14, 6))
preds = preds_child.posterior_predictive["ψ"]
az.plot_hdi(counterfact_child, preds, ax=ax, color="C0")
ax.plot(counterfact_child, preds.mean(("chain", "draw")), lw="3", color="C0")
ax.set(xlabel="child", ylabel="$\psi$", title="$\psi$ as a function of children");
```
