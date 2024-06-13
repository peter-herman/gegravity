__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "March 01, 2021"
__Updated__ = "June 06, 2024"
__Description__ = """A demonstration of the OneSectorGE model and module"""

# ---
# Load Packages
# ---
# Note: The import statements are set to run from the github repository. To run the example code using the packaged
# version of gegravity, remove "src." from the gegravity import statement.
import src.gegravity as ge
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)


# ----
# Load some gravity data (data available on github: https://gist.github.com/peter-herman/13b056e52105008c53faa482db67ed4a)
# ----
gravity_data_location = "examples/sample_data_set.dlm"
grav_data = pd.read_csv(gravity_data_location)
print(grav_data.head())
'''
  exporter importer  year  ...  common_language  lndist  international
0      GBR      AUS  2006  ...                1  9.7126              1
1      FIN      AUS  2006  ...                0  9.5997              1
2      USA      AUS  2006  ...                1  9.5963              1
3      IND      AUS  2006  ...                1  9.1455              1
4      SGP      AUS  2006  ...                1  8.6732              1
'''

# Add a constant to the data
grav_data['constant'] = 1

# ----
# Prepare data and parameter inputs for GE Model
# ----
# The model requires two main types of inputs: baseline data and cost parameters. Data frames contianing both sets of
# information are used to create two special objects in the gegravity package, which organize the data and ensure that
# all needed pieces can be consistently found and drawn on while solving the model. Creating the objects largely
# requires identifying the columns in which certain specific pieces of information can be found


# 1. Baseline data:
# Create a new BaselineData object to manage the input trade, output, expenditure, and cost variable data for the
# GE model.
baseline = ge.BaselineData(grav_data,
                           imp_var_name='importer',  # Columns with importer identifiers
                           exp_var_name='exporter',  # Column with exporter identifiers
                           year_var_name='year',     # Column with year (time) identifier
                           trade_var_name='trade',   # Column with trade values
                           expend_var_name='E',      # Column with importer total expenditure values
                           output_var_name='Y')      # Column with exporter total output values

# 2. Cost parameters:
# Define the parameters that will be used to construct trade costs. These values were estimated separately via PPML.
# These estimates were produced with Stata via the following code, although any appropriate software could be used to
# derive these values.
'''
import delimited "D:\work\Peter_Herman\projects\gegravity\examples\sample_data_set.dlm"
encode importer, gen(imp_fe)
encode exporter, gen(exp_fe)
ppmlhdfe trade lndist  contiguity  common_language pta international, absorb(i.exp_fe i.imp_fe)
'''

# Create DataFrame of coefficient estimates and standard errors. Variable names ('var') should match the column
# names in the baseline data above. Standard errors are used in the MonteCarloGE model but are not used by OneSectorGE
# model (but can still be provided)
ests = [
    #            'var',      'beta',   'stderr'
    (         'lndist',  -0.3898623,   0.0729915),
    (     'contiguity',    0.891577,   0.1327354),
    ('common_language',   0.0326249,   0.0840702),
    (            'pta',   0.4711383,   0.1076578),
    (  'international',   -3.412584,   0.2151235),
    (       'constant',    16.32434,   0.4844137)]
ests = pd.DataFrame(ests, columns = ['var', 'beta', 'stderr'])

# Use the ests dataframe to define a CostCoeff object for the gegravity model, specifying the columns containing the variable identifiers, coefficient estimates, and standard errors
cost_params = ge.CostCoeffs(estimates = ests,          # Dataframe with estimate values
                            identifier_col ='var',     # Column with variable identifier
                            coeff_col ='beta',         # Column with estimated coefficient values
                            stderr_col ='stderr')      # Column with estimated standard errors


# ----
# Define a GE model and calibrate the baseline version of the model
# ----

# Define OneSectorGE model
ge_model = ge.OneSectorGE(baseline = baseline,                 # BaselineData input
                          cost_coeff_values = cost_params,     # CostCoeff input
                          cost_variables = ['lndist', 'contiguity',       # Variables to use to construct trade costs
                                            'common_language', 'pta',
                                            'international', 'constant'],
                          year = "2006",                       # Year to use for model
                          reference_importer = "DEU",          # Reference importer to use (normalizes IMRs to DEU's IMR)
                          sigma = 5)                            # Elasticity of substitution


# The scaling of the outward multilateral resistance term can potentially cause issues when solving the model. Rescaling
# the terms can mitigate this numerical issue. The following method helps identify rescale factors that are likely to
# result in the model being solveable.

potential_factors = ge_model.check_omr_rescale(omr_rescale_range=4)
print(potential_factors)
'''
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
'''
# It looks like rescale factors between 0.0001 and 100 all yield a solveable model, produce similar solutions to the
# baseline model (reference importer OMR = 0.050346), and result in function values that are consistently close to zero,
#  which are all good signs. Going forward, we'll use a factor of 1, which is the default (i.e. OMRs will not be
# rescaled)



# Build baseline (solves for baseline multilateral resistance trms and calibrates other model parameters)
ge_model.build_baseline(omr_rescale = 1)
'''
Solving for baseline MRs...
The solution converged.
'''
# Examine the solutions for the baseline multilateral resistances
print(ge_model.baseline_mr.head(12))


# ---
# Define counterfactual experiment
# ---

# Create a copy of the baseline data
exp_data = ge_model.baseline_data.copy()

# Modify the copied data to reflect a counterfactual experiment in which Canada (CAN) and Japan (JPN) sign a
# preferential trade agreement (pta) by setting the 'pta' variable to one for these countries
exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

# Define the model experiment by inputting this data into the GE model
ge_model.define_experiment(exp_data)

# Counterfactual experiment trade costs are constructed when defining the experiment and can be examined. Trade costs
# in the GE model are represented similarly to trade costs estimated in the econometric model (trade = exp(BX), where
# BX is the trade cost estimate as captured by the model covariates). As a result, the cost values are generally
# positive and higher values imply more trade.
print(ge_model.bilateral_costs.head())
# Check the costs of Canadian exports to Japan
print(ge_model.bilateral_costs.loc[('CAN','JPN'),:])
'''
baseline trade cost      11305.165421
experiment trade cost    18108.800547
trade cost change (%)       60.181650
Name: (CAN, JPN), dtype: float64
'''

##
# Simulate the counterfactual model
##
# Finally, we can simulate the effects of the counterfactual experiment
ge_model.simulate()



# -----
# Access Results
# -----

##
# Retrieve many of the different sets of model results
##
# A collection of many of the key country-level results (prices, total imports/exports, GDP, welfare, etc.)
country_results = ge_model.country_results
# The bilateral trade results
bilateral_results = ge_model.bilateral_trade_results
# A wider selection of aggregate, country-level trade results
agg_trade = ge_model.aggregate_trade_results
# country multilateral resistance (MR) terms
mr_terms = ge_model.country_mr_terms
# Get the solver diaganoistics, which is a dictionary containing many types of solver diagnostic info
solver_diagnostics = ge_model.solver_diagnostics

# ----
# Export results
# ----
# Export the results to a collection of spreadsheet (.csv) files and add trade values in levels to the outputs.
ge_model.export_results(directory="examples//",name="CAN_JPN_PTA_experiment", include_levels = True)
# It is also possible to add alternative country identifies such as full country names using the country_names argument.
# See the technical documentation for details



# -------
# Post Estimation Analysis
# -------

# Examine how the counterfactual experiment affected Canadian imports from NAFTA members USA and Mexico
nafta_share = ge_model.trade_share(importers = ['CAN'],exporters = ['USA','MEX'])
print(nafta_share)

# Calculate counterfactual experiment trade based on observed levels and estimated changes in trade
levels = ge_model.calculate_levels()
print(levels.head())

# Calculate trade weighted shocks
# Start with bilateral level shocks
bilat_cost_shock = ge_model.trade_weighted_shock(how = 'bilateral')
print(bilat_cost_shock.head())

# Now at country-level, which summarizes the bilateral shocks
country_cost_shock  = ge_model.trade_weighted_shock(how='country', aggregations = ['mean', 'sum', 'max'])
print(country_cost_shock.head())



# ----
# Diagnose model solver problems
# ----
# Define a new model to work with
new_model = ge.OneSectorGE(baseline = baseline,
                          cost_coeff_values = cost_params,
                          cost_variables = ['lndist', 'contiguity',
                                            'common_language', 'pta',
                                            'international', 'constant'],
                          year = "2006",
                          reference_importer = "DEU",
                          sigma = 5)


# In addition to the check_omr_rescale method, There are additional tools that can be used to help diagnose issues when
# a model fails to solve. One option is to test that the model system of equations is computable from the supplied data
# and parameters using the following method.
test_diagnostics = new_model.test_baseline_mr_function()

# This method computes the values of the MR system of equation at the initial values (not the solution) to insure
# that the functions are fully parameterized and can be computed. The various vectors and matrices should be filled
# completely with numeric values. The presence of any missing values (nan) is an indication that the baseline data
# and/or trade cost parameters are not complete or defined incorrectly.

# Check one set of parameters (trade cost x expenditure share), for example:
input_params = test_diagnostics['mr_params']
print(input_params['cost_exp_shr'])

