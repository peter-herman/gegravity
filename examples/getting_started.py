__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "March 01, 2021"
__Description__ = """A basic demonstration of the package to help users get started doing GE gravity modeling 
in Python"""

# Load Packages
from models.OneSectorGE import OneSectorGE
import pandas as pd # Data manipulation package
import gme as gme # Gravity econometrics package

# Load data
gravity_data_location = "examples/sample_data_set.csv"
grav_data = pd.read_csv(gravity_data_location)


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
                                lhs_var="trade", # dependant, "left hand side" variable
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


ge_model.simulate()

# -----
# View Results
# -----

country_results = ge_model.country_results
bilateral_results = ge_model.bilateral_trade_results
ge_model.export_results(directory="examples//",name="CAN_JPN_PTA_experiment")

print(country_results[['factory gate price change (percent)', 'GDP change (percent)', 'foreign exports change (percent)']].head(3))
