__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "March 01, 2021"
__Description__ = """A basic demonstration of the package to help users get started doing GE gravity modeling 
in Python"""

# Load Packages
from models.OneSectorGE import OneSectorGE
import pandas as pd # Data manipulation package
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)

import gme as gme # Gravity econometrics package

# Load data
gravity_data_location = "examples/sample_data_set.csv"
grav_data = pd.read_csv(gravity_data_location)
print(grav_data.head())

# ----
# Prepare inputs for GE Model
# ----

# Define GME Estimation Data
gme_data = gme.EstimationData(grav_data, # Dataset
                              imp_var_name="importer", # Importer column name
                              exp_var_name="exporter", # Exporter column name
                              year_var_name = "year", # Year column name
                              trade_var_name="trade")# Trade column name
# Create Gravity Model
gme_model = gme.EstimationModel(gme_data, # Specify data to use
                                lhs_var="trade", # dependent, "left hand side" variable
                                rhs_var=["pta","contiguity","common_language",
                                         "lndist","international"], # independent variables
                                fixed_effects=[["exporter"],["importer"]]) # Fixed effects to use
# Estimate gravity model with PPML
gme_model.estimate()
# Print results table
print(gme_model.format_regression_table(omit_fe_prefix=["imp","exp"], significance_levels=[0.01,0.05,0.1]))
gme_model.format_regression_table(format="tex",path = "examples\\example_econometric_ests.tex", omit_fe_prefix=["imp","exp"], significance_levels=[0.01,0.05,0.1])


# ----
# Create and Run GE model
# ----

# Define GE model
ge_model = OneSectorGE(gme_model, year = "2006",
                       expend_var_name = "E",  # Expenditure column name
                       output_var_name = "Y",  # Output column name
                       reference_importer = "DEU",  # Reference importer
                       sigma = 5)  # Elasticity of substitution

# Test that the model system of equations is computable from the supplied data and parameters
test_diagnostics = ge_model.test_baseline_mr_function()
# See what is returned:
print(test_diagnostics.keys())
# Check the values of the model parameters computed from the baseline data
input_params = test_diagnostics['mr_params']


# Check for OMR rescale factor that results in convergence
# rescale_eval = ge_model.check_omr_rescale()

# Solve for model baseline
ge_model.build_baseline(omr_rescale=10)
print(ge_model.baseline_mr.head())
# Define the counterfactual experiment (CAN-JPN PTA)
exp_data = ge_model.baseline_data.copy()
exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1
ge_model.define_experiment(exp_data)
print(ge_model.bilateral_costs.head())


ge_model.simulate()
print(ge_model.bilateral_trade_results.head())
# -----
# View Results
# -----

country_results = ge_model.country_results
bilateral_results = ge_model.bilateral_trade_results
ge_model.export_results(directory="examples//",name="CAN_JPN_PTA_experiment")


# -------
# Post Estimation Analysis
# -------
nafta_share = ge_model.trade_share(importers = ['CAN'],exporters = ['USA','MEX'])

# Print the first few rows of country-level estimated change in factory prices, GDP, and foreign exports
print(country_results[['factory gate price change (percent)', 'GDP change (percent)', 'foreign exports change (percent)']].head())

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
