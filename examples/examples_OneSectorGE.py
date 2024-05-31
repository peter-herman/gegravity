__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "March 01, 2021"
__Description__ = """A demonstration of the OneSectorGE model and module"""

# ---
# Load Packages
# ---
from src.gegravity.OneSectorGE import OneSectorGE, CostValues
import pandas as pd
# Increase number of columns printed for a pandas DataFrame
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)
import gme as gme


# ----
# Load some gravity data
# ----
gravity_data_location = "examples/sample_data_set.dlm"
grav_data = pd.read_csv(gravity_data_location)
print(grav_data.head())


# ----
# Prepare data and econometric inputs for GE Model
# ----

# Define GME Estimation Data
gme_data = gme.EstimationData(grav_data, # Dataset
                              imp_var_name="importer", # Importer column name
                              exp_var_name="exporter", # Exporter column name
                              year_var_name = "year",  # Year column name
                              trade_var_name="trade")  # Trade column name
# Create Gravity Model
gme_model = gme.EstimationModel(gme_data, # Specify data to use
                                lhs_var="trade",                               # dependent, "left hand side" variable
                                rhs_var=["pta","contiguity","common_language", # independent variables
                                         "lndist","international"],
                                fixed_effects=[["exporter"],["importer"]])     # Fixed effects to use
# Estimate gravity model with PPML
gme_model.estimate()
# Print econometric results table
print(gme_model.results_dict['all'].summary())



# ----
# Define a GE model
# ----

# Define GE model
ge_model = OneSectorGE(gme_model,                   # gme gravity model
                       year = "2006",               # Year to use for model
                       expend_var_name = "E",       # Expenditure column name
                       output_var_name = "Y",       # Output column name
                       reference_importer = "DEU",  # Reference importer
                       sigma = 5)                   # Elasticity of substitution




# ----
# Diagnose model solvability
# ----

# The following commands are not required to define or solve the GE model but can help diagnose issues that arise if
# the model fails to solve.

##
# Check inputs
##

# Test that the model system of equations is computable from the supplied data and parameters
test_diagnostics = ge_model.test_baseline_mr_function()
# See what is returned:
print(test_diagnostics.keys())
# Check the values of the model parameters computed from the baseline data, which should be numeric with no missing
# values
input_params = test_diagnostics['mr_params']
# Check one set of parameters, for example:
print(input_params['cost_exp_shr'])


##
# Check scaling of outward multilateral resistances (OMRs)
##

# Check for OMR rescale factors that results in convergence
rescale_eval = ge_model.check_omr_rescale(omr_rescale_range=3)
print(rescale_eval)
# From the tests, it looks like 10, 100, and 1000 are good candidate rescale factors based on the fact that
# the model solves (i.e. converges) and all three produce consistent solutions for the reference importer's
# outward multilateral resistance (OMR) terms (2.918).



# ---
# Solve baseline and experiment GE model
# ---

##
# Solve the baseline model
##
ge_model.build_baseline(omr_rescale=100)
# Examine the solutions for the baselin multilateral resistances
print(ge_model.baseline_mr.head(12))

##
# Define the counterfactual experiment
##
# Create a copy of the baseline data
exp_data = ge_model.baseline_data.copy()

# Modify the copied data to reflect a counterfactual experiment in which Canada (CAN) and Japan (JPN) sign a
# preferential trade agreement (pta)
exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1
# Define the experiment within the GE model
ge_model.define_experiment(exp_data)
# Examine the baseline and counterfactual trade costs
print(ge_model.bilateral_costs.head())

##
# Simulate the counterfactual model
##
ge_model.simulate()
# Examine the counterfactual trade flows predicted by the model.
print(ge_model.bilateral_trade_results.head())


# -----
# Access and Export Results
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

##
# Export results
##
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

# Print the first few rows of country-level estimated change in factory prices, GDP, and foreign exports
print(country_results[['factory gate price change (percent)', 'GDP change (percent)',
                       'foreign exports change (percent)']].head())

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
# Create model with alternative cost estimates
# ----

# Prepare a DataFrame with the desired cost parameter values.
coeff_data = [{'var':"lndist", 'coeff':-0.4, 'ste':0.05},
              {'var':"contiguity", 'coeff':0.9, 'ste':0.10},
              {'var':"pta", 'coeff':0.5, 'ste':0.02},
              {'var':"common_language", 'coeff':0.04, 'ste':0.06},
              {'var':"international", 'coeff':-3.2, 'ste':0.3}]
coeff_df = pd.DataFrame(coeff_data)
print(coeff_df)

# Create a CostCoeff object from those values
cost_params = CostValues(estimates = coeff_df, identifier_col ='var', coeff_col ='coeff', stderr_col ='ste')

# Define a new OneSector GE model using those cost parameters
alternate_costs = OneSectorGE(gme_model, year = "2006",
                       expend_var_name = "E",  # Expenditure column name
                       output_var_name = "Y",  # Output column name
                       reference_importer = "DEU",  # Reference importer
                       sigma = 5,
                       cost_coeff_values=cost_params)
alternate_costs.build_baseline(omr_rescale=10)




