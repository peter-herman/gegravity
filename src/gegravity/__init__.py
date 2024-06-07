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


## Versions
* **Version 0.3** (Jun. 2024): Package reworked to no longer rely on the GME package for data and parameter inputs (may no longer be compatible with v0.2 models), MonteCarloGE features added to help identify and resolve issues with trials that fail to solve; and minor bug fixes and improvements to reliability. 
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

### OneSectorGE gravity model
#### Prepare data inputs
Begin by loading some needed packages
>>> import gegravity as ge
>>> import pandas as pd

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

Add a constant to the data
>>> grav_data['constant'] = 1

Create a new BaselineData object to manage the input trade, output, expenditure, and cost variable data for the GE model.
>>> baseline = ge.BaselineData(grav_data,
...                            imp_var_name='importer',  # Columns with importer identifiers
...                            exp_var_name='exporter',  # Column with exporter identifiers
...                            year_var_name='year',     # Column with year (time) identifier
...                            trade_var_name='trade',   # Column with trade values
...                            expend_var_name='E',      # Column with importer total expenditure values
...                            output_var_name='Y')      # Column with exporter total output values


#### Prepare the trade cost parameters
Define the parameters that will be used to construct trade costs. These values were estimated separately via PPML. 
>>> ests = [
...    #            'var',      'beta',   'stderr'
...    (         'lndist',  -0.3898623,   0.0729915),
...    (     'contiguity',    0.891577,   0.1327354),
...    ('common_language',   0.0326249,   0.0840702),
...    (            'pta',   0.4711383,   0.1076578),
...    (  'international',   -3.412584,   0.2151235),
...    (       'constant',    16.32434,   0.4844137)]
>>> ests = pd.DataFrame(ests, columns = ['var', 'beta', 'stderr'])

Use the ests dataframe to define a CostCoeff object for the gegravity model, specifying the columns containing the variable identifiers, coefficient estimates, and standard errors.
>>> cost_params = ge.CostCoeffs(estimates = ests,          # Dataframe with estimate values
...                             identifier_col ='var',     # Column with variable identifier
...                             coeff_col ='beta',         # Column with estimated coefficient values
...                             stderr_col ='stderr')      # Column with estimated standard errors


#### Conduct a basic GE analysis

With the data entered into the two input objects (CostCoeff and BaselineData), we can create the GE model. Define the GE model using the OneSectorGE class, which is the package's main model.
>>> ge_model = ge.OneSectorGE(baseline = baseline,                           # BaselineData input
...                           cost_coeff_values = cost_params,               # CostCoeff input
...                           cost_variables = ['lndist', 'contiguity',      # Variables to use to construct trade costs
...                                             'common_language', 'pta',
...                                             'international', 'constant'],
...                           year = "2006",                                 # Year to use for model
...                           reference_importer = "DEU",                    # Reference importer to use (normalizes IMRs to DEU's IMR)
...                           sigma = 5)                                     # Elasticity of substitution


Before solving the model, it is worth discussing one of the most important factors when it comes to finding a solution to the GE model. This factor is the scaling of the outward multilateral resistance terms, which can potentially cause issues for the solver if they are not scaled in a way that is numerically compatible with the other variables. Rescaling these terms can mitigate this numerical issue. The following method helps identify rescale factors that are likely to result in the model being robustly solveable.
>>> potential_factors = ge_model.check_omr_rescale(omr_rescale_range=4)
>>> print(potential_factors)
   omr_rescale omr_rescale (alt format)  solved                                            message  max_func_value  mean_func_value  reference_importer_omr
0       0.0001                    10^-4    True                            The solution converged.    1.706264e-08    -2.081753e-10                0.050346
1       0.0010                    10^-3    True                            The solution converged.    9.829577e-10     5.869294e-11                0.050346
2       0.0100                    10^-2    True                            The solution converged.    1.566544e-09     1.595307e-10                0.050346
3       0.1000                    10^-1    True                            The solution converged.    3.543053e-08    -1.622280e-11                0.050346
4       1.0000                     10^0    True                            The solution converged.    1.663875e-09    -1.471645e-11                0.050346
5      10.0000                     10^1    True                            The solution converged.    1.648595e-09     1.652183e-10                0.050346
6     100.0000                     10^2    True                            The solution converged.    1.260880e-12     6.217249e-15                0.050346
7    1000.0000                     10^3   False  The iteration is not making good progress, as ...    4.666270e-01    -3.827271e-02                0.053669
8   10000.0000                     10^4   False  The iteration is not making good progress, as ...    4.664453e-01    -3.825787e-02                0.053667

It looks like rescale factors between 0.0001 and 100 all yield a solveable model, produce similar solutions to the baseline model (reference importer OMR = 0.050346), and result in function values that are consistently close to zero, which are all good signs. Going forward, we'll use a factor of 1, which is the default (i.e. OMRs will not be rescaled in this case).

With an appropriate rescale factor in hand, we can build the baseline model. This process solves for baseline multialteral resistances and calibrates other model parameters.
>>> ge_model.build_baseline(omr_rescale = 1)
Solving for baseline MRs...
The solution converged.

We can examine the solutions for the baseline multilateral resistances
print(ge_model.baseline_mr.head(12))



#### Define counterfactual experiment and solve the counterfactual GE model

Next, we can try a counterfactual experiment. For this example, let us consider a hypothetical experiment in which Canada (CAN) and Japan (JPN) sign a preferential trade agreement (pta). Begin by creating a copy of the baseline data. Here, it is important that we create a "deep" copy so as to avoid modifying the baseline data too.
>>> exp_data = ge_model.baseline_data.copy()

Next, we modify the copied data to reflect the hypothetical policy change.
>>> exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
>>> exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

Now, define the experiment by supplying the counterfactual data to the GE model.
>>> ge_model.define_experiment(exp_data)

Counterfactual experiment trade costs are constructed when defining the experiment and can be examined. Trade costs in the GE model are represented similarly to trade costs estimated in the econometric model (trade = exp(BX), where BX is the trade cost estimate as captured by the model covariates). As a result, the cost values are generally positive and higher values imply more trade.
>>> print(ge_model.bilateral_costs.head())

We can also check the costs of Canadian exports to Japan, which were subject to the policy change
>>> print(ge_model.bilateral_costs.loc[('CAN','JPN'),:])
baseline trade cost      11305.165421
experiment trade cost    18108.800547
trade cost change (%)       60.181650
Name: (CAN, JPN), dtype: float64


With the experiment defined, the counterfactual model can be simulated. As the model solves, some diagnostic information will print to the console indicating if the first and second stages of the solution were successful.
>>> ge_model.simulate()




#### Access and Export Results

With the model estimated, we can retrieve many of the different sets of model results that are produced. The following are some of the more prominent collections of results.

**Country results:** A collection of many of the key country-level results (prices, total imports/exports, GDP, welfare, etc.)
>>> country_results = ge_model.country_results

Print the first few rows of country-level estimated change in factory prices, GDP, and foreign exports
>>> print(country_results.head())
       factory gate price change (percent)  omr change (percent)  imr change (percent)  GDP change (percent)  welfare statistic  terms of trade change (percent)  output change (percent)  expenditure change (percent)  foreign exports change (percent)  foreign imports change (percent)  intranational trade change (percent)
country                                                                                                                                                                                                                                                                                                                            
AUS                                 0.004143             -0.004143              0.011249             -0.007106           1.000071                        -0.007106                 0.004143                      0.004143                         -0.087678                         -0.033760                              0.019940
AUT                                -0.002774              0.002775              0.002405             -0.005179           1.000052                        -0.005179                -0.002774                     -0.002774                         -0.034691                         -0.026447                             -0.001602
BEL                                -0.002052              0.002052              0.000357             -0.002409           1.000024                        -0.002409                -0.002052                     -0.002052                         -0.037698                         -0.030564                             -0.011239
BRA                                -0.005588              0.005588             -0.000092             -0.005496           1.000055                        -0.005496                -0.005588                     -0.005588                         -0.080228                         -0.066497                             -0.005961
CAN                                -0.135770              0.135955             -0.551724              0.418261           0.995835                         0.418261                -0.135770                     -0.135770                          1.573180                          1.504744                             -1.939006


** Bilateral trade results:** Baseline and counterfactual trade between each pair of countries.
>>> bilateral_results = ge_model.bilateral_trade_results
>>> print(bilateral_results.head())
                   baseline modeled trade  experiment trade  trade change (percent)
exporter importer                                                                  
AUS      AUS                218129.130005     218172.625325                0.019940
         AUT                   666.648700        666.499697               -0.022351
         BEL                  1532.878160       1532.421087               -0.029818
         BRA                  2747.265374       2746.299784               -0.035147
         CAN                  2847.023843       2780.118420               -2.350013

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
0  Percent of CAN imports from USA, MEX              22.021883        21.578895                 -0.442988   -2.01158
1    Percent of USA, MEX exports to CAN               2.034768         1.991437                 -0.043331  -2.129523



We can calculate counterfactual trade values based on observed levels and the estimated changes in trade.
>>> levels = ge_model.calculate_levels()
>>> print(levels.head())
          baseline observed foreign exports  experiment observed foreign exports  baseline observed foreign imports  experiment observed foreign imports  baseline observed intranational trade  experiment observed intranational trade
exporter                                                                                                                                                                                                                                
AUS                                   42485                         42447.749958                              98938                         98904.598242                                 261365                            261417.116627
AUT                                   87153                         87122.765692                              96165                         96139.566802                                  73142                             73140.828196
BEL                                  258238                        258140.649294                             262743                        262662.696177                                 486707                            486652.299609
BRA                                   61501                         61451.658710                              56294                         56256.566037                                 465995                            465967.223949
CAN                                  256829                        260869.383269                             266512                        270522.322380                                 223583                            219247.712294


Finally, we can calculate trade weighted shocks to see which countries or country-pairs were most/least affected by the counterfactual experiment. Start with bilateral level shocks
>>> bilat_cost_shock = ge_model.trade_weighted_shock(how = 'bilateral')
>>> print(bilat_cost_shock.head())
  exporter importer  baseline modeled trade  trade cost change (%)  weighted_cost_change
0      AUS      AUS           218129.130005                    0.0                   0.0
1      AUS      AUT              666.648700                    0.0                   0.0
2      AUS      BEL             1532.878160                    0.0                   0.0
3      AUS      BRA             2747.265374                    0.0                   0.0
4      AUS      CAN             2847.023843                    0.0                   0.0

In this experiment, only Canada--Japan trade should be affected so all other shocks are zero.

Alternatively, we can also do this at country-level, which summarizes the bilateral shocks.
>>> country_cost_shock  = ge_model.trade_weighted_shock(how='country', aggregations = ['mean', 'sum', 'max'])
>>> print(country_cost_shock.head())
    weighted_cost_change                                                
                    mean       sum       max      mean      sum      max
                exporter  exporter  exporter  importer importer importer
AUS             0.000000  0.000000  0.000000  0.000000      0.0      0.0
AUT             0.000000  0.000000  0.000000  0.000000      0.0      0.0
BEL             0.000000  0.000000  0.000000  0.000000      0.0      0.0
BRA             0.000000  0.000000  0.000000  0.000000      0.0      0.0
CAN             0.009956  0.298669  0.298669  0.033333      1.0      1.0

The maximum shock possible is 1 so we can see Canada's imports from Japan were the largest affected trade flow and likely the most influential factor underlying the counterfactual results.




### Monte Carlo analysis
This example demonstrates a basic MonteCarloGE analysis. The Monte Carlo version of the model randomly draws a collection of cost parameter estimates a a simulates a collection of OneSectorGE models using thos parameters. The randomly drawn parameters are based on the varriance/covariances of each of the parameters as estimated by an econometric gravity model. 

#### Create the baseline data inputs
>>> import gegravity as ge
>>> import pandas as pd

Begin by loading some baseline data.
>>> raw_data = pd.read_csv("https://gist.githubusercontent.com/peter-herman/13b056e52105008c53faa482db67ed4a/raw/83898713b8c695fc4c293eaa78eaf44f8e880a85/sample_gravity_data.csv")
>>> raw_data['constant'] = 1

Define baseline data structure
>>> baseline = ge.BaselineData(raw_data, trade_var_name='trade', imp_var_name='importer', exp_var_name='exporter',
...                            year_var_name='year', output_var_name='Y', expend_var_name='E')


#### Setup the cost parameters



Create DataFrame of coefficient estimates and standard errors. These values were estimated separately using the same data and PPML

>>> ests = [
...     #            'var',      'beta',   'stderr'
...     (         'lndist',  -0.3898623,   0.0729915),
...     (     'contiguity',    0.891577,   0.1327354),
...     ('common_language',   0.0326249,   0.0840702),
...     (            'pta',   0.4711383,   0.1076578),
...     (  'international',   -3.412584,   0.2151235),
...     (       'constant',    16.32434,   0.4844137)]
>>> ests = pd.DataFrame(ests, columns = ['var', 'beta', 'stderr'])

Create dataframe of Variance/Covariance matrix and set row variable labels as DataFrame index
>>> var_covar = [
...    #            'var',     'lndist',  'contiguity',   'common_language',   'pta',    'international', 'constant'
...    (         'lndist',     .00532776,     .00504029,     .00093778,     .00473823,    -.01440259,    -.03476407),
...    (     'contiguity',     .00504029,     .01761868,    -.00175404,    -.00030881,    -.01492965,    -.03036031),
...    ('common_language',     .00093778,    -.00175404,     .00706779,     .00249937,     .00136678,    -.01312498),
...    (            'pta',     .00473823,    -.00030881,     .00249937,     .01159021,    -.01538926,    -.03255428),
...    (  'international',    -.01440259,    -.01492965,     .00136678,    -.01538926,     .04627812,     .08911527),
...    (       'constant',    -.03476407,    -.03036031,    -.01312498,    -.03255428,     .08911527,     .23465662)]
>>> var_covar = pd.DataFrame(var_covar, columns = ['var', 'lndist', 'contiguity', 'common_language', 'pta',
...                                                'international', 'constant'])
>>> var_covar.set_index('var', inplace = True)


Define gegravity CostValues object to organize this info as inputs for the GE model
>>> cost_params = ge.CostCoeffs(estimates = ests, identifier_col='var', coeff_col='beta', stderr_col='stderr',
...                             covar_matrix=var_covar)

Note: These estimates were produced using Stata and the following code
>>> import delimited "D:\work\Peter_Herman\projects\gegravity\examples\sample_data_set.dlm"
>>> encode importer, gen(imp_fe)
>>> encode exporter, gen(exp_fe)
>>> ppmlhdfe trade lndist  contiguity  common_language pta international, absorb(i.exp_fe i.imp_fe)
>>> matrix list e(V)



#### Defining and simulating the MonteCarloGE model

With the econometric model estimated, we can define the MonteCarloGE model. In this example, we'll perform a small Monte Carlo experiment using 10 trials. The model will randomly draw 10 sets of cost coefficients from their joint normal distribution and simulate 10 OneSectorGE models corresponding to each draw. Most of the other MonteCarloGE arguments follow those from the OneSectorGE model. The two main exceptions are the "trials" and "seed" arguements. "trials" sets the number of Monte Carlo simulations to perform and "seed" sets the seed for the random draw. Setting a seed allows for repeatable/reproducible simulations.

>>> mc_model = ge.MonteCarloGE(baseline,
...                            year = '2006',
...                            trials = 10,
...                            reference_importer='DEU',
...                            sigma=7,
...                            cost_variables=['lndist', 'contiguity', 'common_language', 'pta', 'international', 'constant'],
...                            cost_coeff_values=cost_params,
...                            allow_singular_covar=True,
...                            seed = 1)


When the model is defined, it creates the 10 draws of cost coefficients. We can examine those random draws in the following way.
>>> print(mc_model.coeff_sample)
               var          0          1          2          3          4          5          6          7          8          9
0           lndist  -0.520534  -0.521725  -0.371895  -0.390785  -0.455785  -0.343360  -0.304334  -0.343764  -0.402571  -0.368141
1       contiguity   0.900885   0.830745   0.922490   0.853686   0.934542   1.043870   0.828643   0.797255   0.824927   0.771972
2  common_language   0.029730  -0.022368   0.011377   0.086987  -0.001718   0.057161  -0.009925   0.179760   0.043748   0.023636
3              pta   0.336452   0.331530   0.599859   0.356130   0.422941   0.510970   0.630710   0.602066   0.533864   0.512185
4    international  -3.118628  -3.174684  -3.573262  -3.310206  -3.267814  -3.510545  -3.723681  -3.457308  -3.348660  -3.479166
5         constant  17.128512  17.209114  16.222708  16.288851  16.786156  15.989807  15.813891  15.920395  16.368214  16.219798



Similarly, we can examine summary information about the random draws. The first two columns report the point estimates from the econometric model (est_model) while the remaining report summary information about the Monte Carlo sample.

>>> print(mc_model.sample_stats)
                 beta_estimate  stderr_estimate  sample_count  sample_mean  sample_std  sample_min  sample_25%  sample_50%  sample_75%  sample_max
var                                                                                                                                               
lndist               -0.389862         0.072992          10.0    -0.402289    0.074404   -0.521725   -0.442482   -0.381340   -0.349858   -0.304334
contiguity            0.891577         0.132735          10.0     0.870902    0.080656    0.771972    0.825856    0.842216    0.917089    1.043870
common_language       0.032625         0.084070          10.0     0.039839    0.059033   -0.022368    0.001555    0.026683    0.053808    0.179760
pta                   0.471138         0.107658          10.0     0.483671    0.114499    0.331530    0.372833    0.511578    0.583360    0.630710
international        -3.412584         0.215123          10.0    -3.396395    0.186946   -3.723681   -3.502700   -3.402984   -3.278412   -3.118628
constant             16.324340         0.484414          10.0    16.394745    0.489138   15.813891   16.047305   16.255780   16.681671   17.209114



Next, we can prepare the counterfactual experiment. As before, we'll consider a hypothetical Canada--Japan trade agreement.
>>> exp_data = mc_model.baseline_data.copy()
>>> exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
>>> exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

We can conduct the Monte Carlo analysis and simulate the trials by supplying the counterfactual experiment and using the following method. As before, most inputs follow from the OneSectroGE class. The two most notable exceptions are "results_stats" and "all_results", which determine what types of information are returned after running the model. "result_stats" determines the types of summary results that are produced across the different trials. For example, the list supplied below will produce the mean, standard deviation, standard error, and median values for each type of result, respectively. "all_results" determines if the model results for each individual trieal are retained and returned after the summary statistics are computed. If True, all that information is retained. Setting it to False disposes of these results after summary stats are computed and frees up any memory needed to store them, which can be sizable for models with a large number of trials and/or a large number of countries.
>>> mc_model.run_trials(experiment_data=exp_data,
...                     omr_rescale=1,
...                     result_stats = ['mean', 'std', 'sem', 'median'],
...                     all_results = True)


#### Examining the Monte Carlo results

After the model finishes solving all ten trials, we can examine the MonteCarloGE results. The MonteCarloGE model populates with the same set of results attributes as the OneSectorGE model. For example, we can retrieve to main country level results.
>>> mc_country_results = mc_model.country_results
>>> print(mc_country_results.head(8))
      statistic  factory gate price change (percent)  omr change (percent)  imr change (percent)  GDP change (percent)  welfare statistic  terms of trade change (percent)  output change (percent)  expenditure change (percent)  foreign exports change (percent)  foreign imports change (percent)  intranational trade change (percent)
country                                                                                                                                                                                                                                                                                                                                      
AUS          mean                             0.002680             -0.002680              0.007648             -0.004968           1.000050                        -0.004968                 0.002680                      0.002680                         -0.088476                         -0.039323                              0.023454
AUS           std                             0.001212              0.001212              0.001252              0.001369           0.000014                         0.001369                 0.001212                      0.001212                          0.025571                          0.012610                              0.004520
AUS           sem                             0.000383              0.000383              0.000396              0.000433           0.000004                         0.000433                 0.000383                      0.000383                          0.008086                          0.003988                              0.001429
AUS        median                             0.002719             -0.002719              0.007692             -0.005198           1.000052                        -0.005198                 0.002719                      0.002719                         -0.090336                         -0.041185                              0.023691
AUT          mean                            -0.001898              0.001898              0.001662             -0.003560           1.000036                        -0.003560                -0.001898                     -0.001898                         -0.028666                         -0.022628                              0.005849
AUT           std                             0.000577              0.000577              0.000468              0.001041           0.000010                         0.001041                 0.000577                      0.000577                          0.007434                          0.005717                              0.002428
AUT           sem                             0.000182              0.000182              0.000148              0.000329           0.000003                         0.000329                 0.000182                      0.000182                          0.002351                          0.001808                              0.000768
AUT        median                            -0.002071              0.002071              0.001740             -0.003838           1.000038                        -0.003838                -0.002071                     -0.002071                         -0.029426                         -0.023267                              0.006439



If all_results = True, the model retains information about all of the trial runs. These can be accessed in a variety of ways. One option is to view all country-level results for each trial. Columns reflect trial numer and result type. (E.g. (0, 'welfare statistic') for the value of the Trial 0 welfare statistic).
>>> all_mc_country_results = mc_model.all_country_results


Alternatively, MonteCarloGE is based on running a series of OneSectorGE models using different cost parameters for each trial. These individual OneSectorGE models are stored in a list that can be accessed via the attribute trial_models. For example, we can examine the results from trial 2:
>>> trial_2_model = mc_model.trial_models[2]
>>> print(trial_2_model.country_results.head())
         factory gate price change (percent)  omr change (percent)  imr change (percent)  GDP change (percent)  welfare statistic  terms of trade change (percent)  output change (percent)  expenditure change (percent)  foreign exports change (percent)  foreign imports change (percent)  intranational trade change (percent)
country                                                                                                                                                                                                                                                                                                                            
AUS                                 0.003306             -0.003306              0.009805             -0.006498           1.000065                        -0.006498                 0.003306                      0.003306                         -0.126374                         -0.054366                              0.030599
AUT                                -0.002510              0.002510              0.002141             -0.004651           1.000047                        -0.004651                -0.002510                     -0.002510                         -0.037593                         -0.029239                              0.007880
BEL                                -0.001742              0.001742              0.000400             -0.002142           1.000021                        -0.002142                -0.001742                     -0.001742                         -0.041347                         -0.034214                             -0.005636
BRA                                -0.004893              0.004893             -0.000103             -0.004789           1.000048                        -0.004789                -0.004893                     -0.004893                         -0.100750                         -0.089339                              0.003946
CAN                                -0.147532              0.147750             -0.502212              0.356469           0.996448                         0.356469                -0.147532                     -0.147532                          2.148502                          2.047754                             -2.415646



Finally we can export the summary results to a series of .csv files.
>>> mc_model.export_results(directory="examples//", name = 'monte')





#### Dealing with Failed Trials

Some GE gravity models are more difficult to solve than others. The repeated sampling and resolving of the model can result in cases where the model solves for some trials but not others. There are a few tools available to identify and mitigate these issues. For simplicity, the methods are demonstrated using the above example model, which does not exhibit any failed trials within the 10 specified trials, but can still illustrate the methods.

We can check for failed trials, which are listed by their trial number (0 to N). In this case, there are none.
>>> print(mc_model.failed_trials)
[]

One likely source of solver issues is the OMR rescale factor. There is a version of the check_omr_rescale method for the MonteCarloGE model that will check some or all of the trials for feasible OMR scaling factors. Here, we will redefine the model from scratch and check the values for trials 1, 4, and 5. Omitting that argument will check all trials.
>>> mc_model_2 = ge.MonteCarloGE(baseline,
...                              year = '2006',
...                              trials = 10,
...                              reference_importer='DEU',
...                              sigma=7,
...                              cost_variables=['lndist', 'contiguity', 'common_language', 'pta', 'international', 'constant'],
...                              cost_coeff_values=cost_params,
...                              allow_singular_covar=True,
...                              seed = 1)
>>> mc_omr_checks = mc_model_2.check_omr_rescale(omr_rescale_range=5, trials = [1, 4, 5])

One potential option for dealing with a failed trial is to specify a different omr_rescale_factor to use for that specific trial. FOr example, we could specify that trials 4 and 5 use a rescale factor of 100 instead of 1, which is used for all other trials.
>>> mc_model_2.run_trials(experiment_data=exp_data,
...                       omr_rescale=1,
...                       trial_omr_rescale={4:100, 5:100})

If the source of the failure cannot be overcome by adjusting the rescale factor, another option is to draw additional trials to replace those that failed. This has the advantage of insuring that a desired number of trials is run in total. However, it should also be noted that if certain parameter draws are failing for systemitic reasons, then redrawing failed trials could introduce an unwanted bias in the results. Therefore, this option should be used with some caution.
>>> mc_model_2.run_trials(experiment_data=exp_data,
...                       omr_rescale=1,
...                       redraw_failed_trials=True)


"""
