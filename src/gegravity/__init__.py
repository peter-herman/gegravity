from .OneSectorGE import *
from .BaselineData import *
from .MonteCarloGE import * # move these to after comments to produce documentation. Not sure if that affects packaging.

"""
# Documentation
--------------------
gegravity is a Python package containing tools used to estimate general equilibrium (GE) structural gravity models and simulate counterfactual experiments. The package is based on the well established version of the gravity model described by Yotov, Piermartini, Monteiro, and Larch (2016) *An Advanced Guide to Trade  Policy  Analysis:  The  Structural  Gravity  Model*. It implements the structural GE gravity model in a general, robust, and easy to use manner in an effort to make GE gravity modeling more accessible for researchers and policy analysts.

The package provides several useful tools for structural gravity modeling.

1. It computes theory consistent estimates of the structural multilateral resistance terms of Anderson and van Wincoop (2003) "Gravity with Gravitas" from standard econometric gravity results.

2. It simulates GE effects from counterfactual experiments such as new trade agreements or changes to other trade costs. The model can be flexibly used to study aggregate or sector level trade as well as many different types of trade costs and policies.

3. It conducts Monte Carlo simulations that provide a means to compute standard errors and other measures of statistical precision for the GE model results.

For more information about the GE gravity model, its implementation, and the various components of the package, see the companion paper ["gegravity: General Equilibrium Gravity Modeling in Python"](https://usitc.gov/sites/default/files/publications/332/working_papers/herman_2021_gegravity_modeling_in_python.pdf).


## Citation and license
The package is publicly available and free to use under the MIT license. Users are asked to please cite the following,

Herman, Peter (2021) "gegravity: General Equilibrium Gravity Modeling in Python." USITC Economics Working Paper 2021-04-B. 


## 0.3 Updates
* **Version 0.2** (Dec. 2022): Improved OneSectorGE speed and updated compatibility with dependencies.
* **Version 0.1** (Apr. 2021): Initial release.

## Installation
The package can be installed via pip with the following command.

>>> pip install gegravity



## Example
The following examples demonstrate how to perform a typical GE gravity analysis using the gegravity package. The code files for these examples as well as the sample data set used in them can be found at the project's github page or at the following gist locations.

* **Sample data:** https://gist.github.com/peter-herman/13b056e52105008c53faa482db67ed4a
* **Basic GE analysis script:** https://gist.github.com/peter-herman/faeea8ec032c4c2c13bcbc9c400cca9b
* **Monte Carlo GE analysis:** https://gist.github.com/peter-herman/a2ebf3997bfd6e9cb3268298d49b64b5


### Prepare data inputs
Begin by loading some needed packages
>>> import gegravity as ge
>>> import pandas as pd
>>> import gme as gme

Next, load some input data
>>> gravity_data_location = "https://gist.githubusercontent.com/peter-herman/13b056e52105008c53faa482db67ed4a/raw/83898713b8c695fc4c293eaa78eaf44f8e880a85/sample_gravity_data.csv"
>>> grav_data = pd.read_csv(gravity_data_location)
>>> print(grav_data.head())
  exporter importer  year  trade        Y       E  pta  contiguity  common_language  lndist  international
0      GBR      AUS  2006   4310   925638  362227    0           0                1  9.7126              1
1      FIN      AUS  2006    514   142759  362227    0           0                0  9.5997              1
2      USA      AUS  2006  16619  5019964  362227    1           0                1  9.5963              1
3      IND      AUS  2006    763   548517  362227    0           0                1  9.1455              1
4      SGP      AUS  2006   8756   329817  362227    1           0                1  8.6732              1



Prepare data and econometric inputs for the GE Model using the tools in the gme package. First, Define a gme EstimationData object.
>>> gme_data = gme.EstimationData(grav_data, # Dataset
...                               imp_var_name="importer", # Importer column name
...                               exp_var_name="exporter", # Exporter column name
...                               year_var_name = "year",  # Year column name
...                               trade_var_name="trade")  # Trade column name

Second, create and estimate a gme EstimationModel gravity model to derive trade cost parameter estimates.
>>> gme_model = gme.EstimationModel(gme_data, # Specify data to use
...                                 lhs_var="trade",                               # dependent, "left hand side" variable
...                                 rhs_var=["pta","contiguity","common_language", # independent variables
...                                          "lndist","international"],
...                                 fixed_effects=[["exporter"],["importer"]])     # Fixed effects to use

Estimate the gravity model with PPML.
>>> gme_model.estimate()
select specification variables: ['pta', 'contiguity', 'common_language', 'lndist', 'international', 'trade', 'importer', 'exporter', 'year'], Observations excluded by user: {'rows': 0, 'columns': 2}
drop_intratrade: no, Observations excluded by user: {'rows': 0, 'columns': 0}
drop_imp: none, Observations excluded by user: {'rows': 0, 'columns': 0}
drop_exp: none, Observations excluded by user: {'rows': 0, 'columns': 0}
keep_imp: all available, Observations excluded by user: {'rows': 0, 'columns': 0}
keep_exp: all available, Observations excluded by user: {'rows': 0, 'columns': 0}
drop_years: none, Observations excluded by user: {'rows': 0, 'columns': 0}
keep_years: all available, Observations excluded by user: {'rows': 0, 'columns': 0}
drop_missing: yes, Observations excluded by user: {'rows': 0, 'columns': 0}
Estimation began at 09:27 AM  on Mar 29, 2021
Omitted Columns: ['importer_fe_ZAF', 'importer_fe_USA']
Estimation completed at 09:27 AM  on Mar 29, 2021

To examine the econometric estimates, we can print a table of results.
>>> print(gme_model.results_dict['all'].summary())
                 Generalized Linear Model Regression Results
==============================================================================
Dep. Variable:                  trade   No. Iterations:                      8
Model:                            GLM   Df Residuals:                      807
Model Family:                 Poisson   Df Model:                           62
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:            -1.6147e+06
Covariance Type:                  HC1   Deviance:                   3.2210e+06
No. Observations:                 870   Pearson chi2:                 4.07e+06
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
pta                 0.4823      0.109      4.427      0.000       0.268       0.696
contiguity          0.8850      0.131      6.762      0.000       0.628       1.142
common_language     0.0392      0.084      0.464      0.643      -0.126       0.205
lndist             -0.3764      0.072     -5.227      0.000      -0.518      -0.235
international      -3.4224      0.215    -15.940      0.000      -3.844      -3.001
(truncated for brevity)





### Conduct a basic GE analysis


With the data entered into the gme EstimationStructure and cost estimates derrived, we can create the GE model. Define the GE model using the OneSectorGE class, which is the package's primary model.
>>> ge_model = ge.OneSectorGE(gme_model,                   # gme gravity model
...                        year = "2006",               # Year to use for model
...                        expend_var_name = "E",       # Expenditure column name
...                        output_var_name = "Y",       # Output column name
...                        reference_importer = "DEU",  # Reference importer
...                        sigma = 5)                   # Elasticity of substitution





#### Diagnose model solver issues

The following commands are not required to define or solve the GE model but can help diagnose issues that arise if
the model fails to solve.

**Examine input parameters:** The first thing that we can do is examine the parameters that are constructed for the model solvers. In particular, we can test if the model's system of equations is computable from the supplied data and parameters.
>>> test_diagnostics = ge_model.test_baseline_mr_function()
# See what is returned:
>>> print(test_diagnostics.keys())
dict_keys(['initial values', 'mr_params', 'function_value'])

Check the values of the model parameters computed from the baseline data, which should be numeric with no missing values.
>>> input_params = test_diagnostics['mr_params']
# Check one set of parameters, for example:
>>> print(input_params['cost_exp_shr'])
[[1.06731975e-03 6.11822268e-06 2.61950076e-05 2.01489708e-05
  1.81180910e-05 1.34706132e-05 1.49136923e-04 4.34060154e-06
  2.32380420e-05 4.76381695e-06 3.79942724e-05 3.65156972e-05
  1.67484536e-05 1.23685767e-05 2.44640604e-05 4.36190753e-06
  3.96578076e-05 1.09631645e-04 4.44143262e-05 1.40691044e-05
(truncated for brevity)

**Find a OMR rescale factor:** The second thing that can be tested is the scaling of the outward multilateral resistance (OMR) terms, which are some of the variables solved for using non-linear solvers as part of the baseline and counterfactual model construction. Rescaling the OMR terms can help when the magnitude of the OMRs differ significantly from the other variables being solved for. It rescales them within the sovlvers, which can improve convergence. The following method can be used to test multiple different potential scales and identify ones that result in convergence.


>>> rescale_eval = ge_model.check_omr_rescale(omr_rescale_range=3)
>>> print(rescale_eval)
   omr_rescale omr_rescale (alt format)  solved                                            message  max_func_value  mean_func_value  reference_importer_omr
0        0.001                    10^-3   False  The iteration is not making good progress, as ...    8.774878e-02     4.441303e-04                2.339813
1        0.010                    10^-2    True                            The solution converged.    3.683065e-11    -2.652545e-12                2.918339
2        0.100                    10^-1    True                            The solution converged.    2.610248e-09     4.552991e-11                2.920591
3        1.000                     10^0    True                            The solution converged.    7.409855e-10    -1.980349e-11                2.967636
4       10.000                     10^1    True                            The solution converged.    9.853662e-10    -2.213563e-12                2.918339
5      100.000                     10^2    True                            The solution converged.    3.629199e-10     2.458433e-11                2.918339
6     1000.000                     10^3    True                            The solution converged.    3.392378e-09    -3.910916e-11                2.918339

From the tests, it looks like 10, 100, and 1000 are good candidate rescale factors based on the fact that
the model solves (i.e. converges) and all three produce consistent solutions for the reference importer's
outward multilateral resistance (OMR) terms (2.918).




#### Solve baseline and counterfactual experiment GE model


Having found a rescale factor that successfully solves the model, we can construct the actual baseline model. Once complete, we can access the theoretical multilateral resistance terms of the model.
>>> ge_model.build_baseline(omr_rescale=100)
# Examine the solutions for the baseline multilateral resistances
>>> print(ge_model.baseline_mr.head())
         baseline omr  baseline imr
country
AUS          3.577130      1.421059
AUT          3.408633      1.224844
BEL          2.925592      1.050865
BRA          3.590866      1.292782
CAN          3.313605      1.338893

From this point, we can try a counterfactual experiment. For this example, let us consider a hypothetical experiment in which Canada (CAN) and Japan (JPN) sign a preferential trade agreement (pta). Begin by creating a copy of the baseline data. Here, it is important that we create a "deep" copy so as to avoid modifying the baseline data too.
>>> exp_data = ge_model.baseline_data.copy()

Next, we modify the copied data to reflect the hypothetical policy change.
>>> exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
>>> exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

Now, define the experiment by suppling the counterfactual data to the GE model.
>>> ge_model.define_experiment(exp_data)

At this point, we can examine the baseline and the newly created counterfactual trade costs.
>>> print(ge_model.bilateral_costs.head())
                   baseline trade cost  experiment trade cost  trade cost change (%)
exporter importer
AUS      AUS                  0.072546               0.072546                    0.0
         AUT                  0.000863               0.000863                    0.0
         BEL                  0.000848               0.000848                    0.0
         BRA                  0.000931               0.000931                    0.0
         CAN                  0.000902               0.000902                    0.0

With the experiment defined, the counterfactual model can be estimated. As the model solves, some diagnostic information will print to the console indicating if the first and second stages of the solution were successful.
>>> ge_model.simulate()




#### Access and Export Results

With the model estimated, we can retrieve many of the different sets of model results that are produced. The following are some of the more prominent collections of results.

**Country results:** A collection of many of the key country-level results (prices, total imports/exports, GDP, welfare, etc.)
>>> country_results = ge_model.country_results
# Print the first few rows of country-level estimated change in factory prices, GDP, and foreign exports
print(country_results[['factory gate price change (percent)', 'GDP change (percent)',
                       'foreign exports change (percent)']].head())
         factory gate price change (percent)  GDP change (percent)  foreign exports change (percent)
country
AUS                                 0.003621             -0.007297                         -0.087975
AUT                                -0.002883             -0.005345                         -0.034996
BEL                                -0.002154             -0.002516                         -0.038307
BRA                                -0.005570             -0.005659                         -0.080171
CAN                                -0.133574              0.435343                          1.634825

** Bilateral trade results:** Baseline and counterfactual trade between each pair of countries.
>>> bilateral_results = ge_model.bilateral_trade_results
>>> print(bilateral_results.head())
                   baseline modeled trade  experiment trade  trade change (percent)
exporter importer
AUS      AUS                216157.106997     216199.213723                0.019480
         AUT                   683.873129        683.730549               -0.020849
         BEL                  1586.476404       1586.023933               -0.028520
         BRA                  2794.995080       2794.072041               -0.033025
         CAN                  2891.501312       2821.979450               -2.404352

**Aggregate Trade:** Total imports, exports, intranational trade, output, and shipments for each country.
>>> agg_trade = ge_model.aggregate_trade_results

**Multilateral Resistances:** Country multilateral resistance (MR) terms
>>> mr_terms = ge_model.country_mr_terms

**Solver Diagnostics** A dictionary containing many types of solver diagnostic info.
>>> solver_diagnostics = ge_model.solver_diagnostics


It is also possible to export the results to a collection of spreadsheet (.csv) files and add trade values in levels to the outputs.
>>> ge_model.export_results(directory="C://examples//",name="CAN_JPN_PTA_experiment")



#### Post estimation analysis
There are several tools that allow for post-estimation analysis. These can help to better understand the effects of the analysis or to example particular countries of interest.

For example, we can examine how the counterfactual experiment affected Canadian imports from NAFTA members USA and Mexico.
>>> nafta_share = ge_model.trade_share(importers = ['CAN'],exporters = ['USA','MEX'])
>>> print(nafta_share)
                            description baseline modeled trade experiment trade change (percentage point) change (%)
0  Percent of CAN imports from USA, MEX                 21.948          21.4935                 -0.454491   -2.07077
1    Percent of USA, MEX exports to CAN                2.02794          1.98363                -0.0443055   -2.18475



We can calculate counterfactual trade values based on observed levels and the estimated changes in trade.
>>> levels = ge_model.calculate_levels()
>>> print(levels.head())
         baseline observed foreign exports  experiment observed foreign exports  baseline observed foreign imports  experiment observed foreign imports  baseline observed intranational trade  experiment observed intranational trade
exporter
AUS                                   42485                         42447.623705                              98938                         98903.447553                                 261365                            261415.913081
AUT                                   87153                         87122.499771                              96165                         96139.268310                                  73142                             73141.020472
BEL                                  258238                        258139.077301                             262743                        262661.185598                                 486707                            486652.495570
BRA                                   61501                         61451.693962                              56294                         56256.521333                                 465995                            465969.574790
CAN                                  256829                        261027.705514                             266512                        270678.983401                                 223583                            219107.825989

Finally, we can calculate trade weighted shocks to see which countries or country-pairs were most/least affected by the counterfactual experiment. Start with bilateral level shocks
>>> bilat_cost_shock = ge_model.trade_weighted_shock(how = 'bilateral')
>>> print(bilat_cost_shock.head())
  exporter importer  baseline modeled trade  trade cost change (%)  weighted_cost_change
0      AUS      AUS           216157.106997                    0.0                   0.0
1      AUS      AUT              683.873129                    0.0                   0.0
2      AUS      BEL             1586.476404                    0.0                   0.0
3      AUS      BRA             2794.995080                    0.0                   0.0
4      AUS      CAN             2891.501312                    0.0                   0.0

In this experiment, only Canada--Japan trade should be affected so all other shocks are zero.

Alternatively, we can also do this at country-level, which summarizes the bilateral shocks.
>>> country_cost_shock  = ge_model.trade_weighted_shock(how='country', aggregations = ['mean', 'sum', 'max'])
>>> print(country_cost_shock.head())
                    mean       sum       max      mean      sum      max
                exporter  exporter  exporter  importer importer importer
AUS             0.000000  0.000000  0.000000  0.000000      0.0      0.0
AUT             0.000000  0.000000  0.000000  0.000000      0.0      0.0
BEL             0.000000  0.000000  0.000000  0.000000      0.0      0.0
BRA             0.000000  0.000000  0.000000  0.000000      0.0      0.0
CAN             0.010171  0.305121  0.305121  0.033333      1.0      1.0

The maximum shock possible is 1 so we can see Canada's imports from Japan were the largest affected trade flow and likely the most influential factor underlying the counterfactual results.


#### Create model with alternative cost estimates

It is possible to supply cost estimates that were not derived using the gme.Estimation model. For example, this may be desireable if you'd like to use estimates derived with high dimensional fixed effects from a different package. To do so, prepare a DataFrame with the desired cost parameter values.
>>> coeff_data = [{'var':"lndist", 'coeff':-0.4, 'ste':0.05},
...               {'var':"contiguity", 'coeff':0.9, 'ste':0.10},
...               {'var':"pta", 'coeff':0.5, 'ste':0.02},
...               {'var':"common_language", 'coeff':0.04, 'ste':0.06},
...               {'var':"international", 'coeff':-3.2, 'ste':0.3}]
>>> coeff_df = pd.DataFrame(coeff_data)
>>> print(coeff_df)
               var  coeff   ste
0           lndist  -0.40  0.05
1       contiguity   0.90  0.10
2              pta   0.50  0.02
3  common_language   0.04  0.06
4    international  -3.20  0.30

Next, create a create a gegravity CostCoeff object from those values
>>> cost_params = ge.CostCoeffs(estimates = coeff_df, identifier_col = 'var', coeff_col = 'coeff', stderr_col = 'ste')

Finally, we can define a new OneSector GE model using those alternative cost parameters
>>> alternate_costs = ge.OneSectorGE(gme_model, year = "2006",
...                         expend_var_name = "E",  # Expenditure column name
...                         output_var_name = "Y",  # Output column name
...                         reference_importer = "DEU",  # Reference importer
...                         sigma = 5,
...                         cost_coeff_values=cost_params)






### Monte Carlo analysis
This example demonstrates a basic Monte Carlo GE analysis. It closely follows the previous example in terms of inputs and the counterfactual experiment.



Begin by loading some baseline data.
>>> raw_data = pd.read_csv("https://gist.githubusercontent.com/peter-herman/13b056e52105008c53faa482db67ed4a/raw/83898713b8c695fc4c293eaa78eaf44f8e880a85/sample_gravity_data.csv")

Define the cost variables to use in the analysis.
>>> cost_variables = ["pta", "contiguity", "common_language", "lndist", "international"]

Define the gme EstimationData object.
>>> est_data = gme.EstimationData(raw_data,
...                               imp_var_name='importer',
...                               exp_var_name='exporter',
...                               year_var_name='year',
...                               trade_var_name='trade')

Define and estimate the gme EstimationModel. Note that the argument full_results is set to True, which is required for MonteCarloGE. This argument insures that the model retains the estimated variance/covariance matrix, which is used by the MonteCarloGE model to randomly draw trade cost estimates.
>>> est_model = gme.EstimationModel(estimation_data=est_data,
...                                 lhs_var='trade',
...                                 rhs_var=cost_variables,
...                                 fixed_effects=[['importer'], ['exporter']],
...                                 omit_fixed_effect=[['importer']], retain_modified_data=True,
...                                 full_results=True)
>>> est_model.estimate()




With the econometric model estimated, we can define the MonteCarloGE model. In this example, we'll perform a small Monte Carlo experiment using 10 trials. The model will randomly draw 10 sets of cost coefficients from their joint normal distribution and simulate 10 OneSectorGE models corresponding to each draw. Most of the other MonteCarloGE arguments follow those from the OneSectorGE model. The two main exceptions are the "trials" and "seed" arguements. "trials" sets the number of Monte Carlo simulations to perform and "seed" sets the seed for the random draw. Setting a seed allows for repeatable/reproducible simulations.

>>> monte_model = ge.MonteCarloGE(est_model,
...                               year='2006',
...                               trials=10,
...                               reference_importer='DEU',
...                               sigma=5,
...                               expend_var_name='E',
...                               output_var_name='Y',
...                               cost_variables=cost_variables,
...                               results_key='all',
...                               seed=0)


When the model is defined, it creates the 10 draws of cost coefficients. We can examine those random draws in the following way.
>>> print(monte_model.coeff_sample)
             index         0         1         2         3         4         5         6         7         8         9
0              pta  0.481542  0.353868  0.378325  0.472573  0.501786  0.348512  0.428324  0.393075  0.541573  0.484753
1       contiguity  0.590650  1.081871  1.071498  0.879770  0.871200  0.938144  1.018024  0.997490  0.838174  1.088999
2  common_language  0.125048 -0.023092  0.035609  0.109166 -0.030624  0.084380  0.091459  0.076088  0.131230  0.014039
3           lndist -0.537955 -0.362250 -0.418252 -0.438092 -0.330030 -0.376829 -0.417805 -0.308843 -0.328604 -0.335478
4    international -2.995594 -3.509961 -3.332177 -3.176423 -3.626659 -3.309972 -3.375912 -3.513574 -3.539888 -3.592729

Similarly, we can examine summary information about the random draws. The first two columns report the point estimates from the econometric model (est_model) while the remaining report summary information about the Monte Carlo sample.

>>> print(monte_model.sample_stats)
                 beta_estimate  stderr_estimate  sample_count  sample_mean  sample_std  sample_min  sample_25%  sample_50%  sample_75%  sample_max
index
pta                   0.471138         0.107598          10.0     0.438433    0.067355    0.348512    0.382012    0.450448    0.483951    0.541573
contiguity            0.891577         0.132662          10.0     0.937582    0.152655    0.590650    0.873343    0.967817    1.058129    1.088999
common_language       0.032625         0.084023          10.0     0.061330    0.059040   -0.030624    0.019431    0.080234    0.104739    0.131230
lndist               -0.389862         0.072951          10.0    -0.385414    0.069469   -0.537955   -0.418140   -0.369540   -0.331392   -0.308843
international        -3.412584         0.215004          10.0    -3.397289    0.199950   -3.626659   -3.533309   -3.442937   -3.315523   -2.995594


Next, we can prepare the counterfactual experiment. As before, we'll consider a hypothetical Canada--Japan trade agreement.
>>> exp_data = monte_model.baseline_data.copy()
>>> exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
>>> exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

We can conduct the Monte Carlo analysis and simulate the trials by supplying the counterfactual experiment and using the following method. As before, most inputs follow from the OneSectroGE class. The two most notable exceptions are "results_stats" and "all_results", which determine what types of information are returned after running the model. "result_stats" determines the types of summary results that are produced across the different trials. For example, the list supplied below will produce the mean, standard deviation, standard error, and median values for each type of result, respectively. "all_results" determines if the model results for each individual trieal are retained and returned after the summary statistics are computed. If True, all that information is retained. Setting it to False disposes of these results after summary stats are computed and frees up any memory needed to store them, which can be sizable for models with a large number of trials and/or a large number of countries.
>>> monte_model.run_trials(experiment_data=exp_data,
...                        omr_rescale=100,
...                        result_stats = ['mean', 'std', 'sem', 'median'],
...                        all_results = True)

After the model finishes solving all ten trials, we can examine the MonteCarloGE results. The MonteCarloGE model populates with the same set of results attributes as the OneSectorGE model. For example, we can retrieve to main country level results.
>>> country_results = monte_model.country_results
>>> print(country_results.head())
        statistic  GDP change (percent)  expenditure change (percent)  factory gate price change (percent)  foreign exports change (percent)  foreign imports change (percent)  imr change (percent)  intranational trade change (percent)  omr change (percent)  output change (percent)  terms of trade change (percent)  welfare statistic
country
AUS          mean             -0.006570                      0.003496                             0.003496                         -0.080711                         -0.031464              0.010066                              0.017896             -0.003496                 0.003496                        -0.006570           1.000066
AUS           std              0.001230                      0.001413                             0.001413                          0.017219                          0.007057              0.002066                              0.003349              0.001412                 0.001413                         0.001230           0.000012
AUS           sem              0.000389                      0.000447                             0.000447                          0.005445                          0.002232              0.000653                              0.001059              0.000447                 0.000447                         0.000389           0.000004
AUS        median             -0.006898                      0.003563                             0.003563                         -0.086745                         -0.033234              0.010345                              0.018416             -0.003563                 0.003563                        -0.006898           1.000069
AUT          mean             -0.004718                     -0.002526                            -0.002526                         -0.031660                         -0.024115              0.002192                             -0.001555              0.002526                -0.002526                        -0.004718           1.000047
AUT           std              0.000834                      0.000435                             0.000435                          0.006080                          0.004579              0.000411                              0.001962              0.000435                 0.000435                         0.000834           0.000008
AUT           sem              0.000264                      0.000138                             0.000138                          0.001923                          0.001448              0.000130                              0.000620              0.000138                 0.000138                         0.000264           0.000003
AUT        median             -0.004525                     -0.002358                            -0.002358                         -0.032039                         -0.024445              0.002119                             -0.001043              0.002358                -0.002358                        -0.004525           1.000045


We can also check to see if any trials failed to solve.
>>> print(monte_model.num_failed_trials)
0

In this case, it is zero. However, it is possible that certain draws of coefficients may not yield a solution. In many cases, these issues can be resolved by adjusting the omr_rescale_factor. For example, using omr_"rescale=10" in this example tends to result in some trials failing to solve, depending on the seed.

If the set of all results was selected via "all_results=True", these results can be accessed from attributes labeled using the standard label with "all_" as a prefix. For example, the the "country_results" for each trial will populate in the attribute "monte_model.all_country_results".
>>> all_country_results = monte_model.all_country_results

Finally we can export the summary results to a series of .csv files.
>>> monte_model.export_results(directory="examples//", name = 'monte')


"""
