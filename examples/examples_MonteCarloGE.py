__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "March 17, 2021"
__Description__ = '''A demonstartion of the MonteCarloGE model/module'''

# ----
# Import some packages
# ----

import pandas as pd
from gme.construct_data.EstimationData import EstimationData
from gme.estimate.EstimationModel import EstimationModel
from models.OneSectorGE import OneSectorGE
from models.MonteCarloGE import MonteCarloGE
import numpy as np


# ---
# Prepare model data and economteric inputs
# ---
# Load sample data.
raw_data = pd.read_csv("D:\work\Peter_Herman\projects\gegravity\examples\sample_data_set.dlm")

# Define cost variables to use in analysis.
cost_variables = ["pta", "contiguity", "common_language", "lndist", "international"]

# Define gme EstimationData.
est_data = EstimationData(raw_data,
                          imp_var_name='importer',
                          exp_var_name='exporter',
                          year_var_name='year',
                          trade_var_name='trade')

# Define and estimate the gme EstimationModel.
# Note that the argument full_results is set to True, which is required for MonteCarloGE. This produces a
# coefficient variance/covariance matrix for use in the MonteCarloGE model.
est_model = EstimationModel(estimation_data=est_data,
                            lhs_var='trade',
                            rhs_var=cost_variables,
                            fixed_effects=[['importer'], ['exporter']],
                            omit_fixed_effect=[['importer']], retain_modified_data=True,
                            full_results=True)
est_model.estimate()
print(est_model.results_dict['all'].summary())


# ----
# Define the MonetCarloGE model
# ----

# Here we demonstrate a small Monte Carlo experiment using 10 trials. The model will randomly draw 10 sets of cost
# coefficients from their joint normal distribution and simulate 10 OneSectorGE models corresponding to each draw.
# Most of the other MonteCarloGE arguments follow those from the OneSectorGE model. See that model for details.
monte_model = MonteCarloGE(est_model,
                           year='2006',
                           trials=10,
                           reference_importer='DEU',
                           sigma=5,
                           expend_var_name='E',
                           output_var_name='Y',
                           cost_variables=cost_variables,
                           results_key='all',
                           seed=0)
# The seed argument sets the seed for the random draw. Setting a seed allows for repeatable/reproducible simulations.

# Examine the random parameter draws
print(monte_model.coeff_sample)
# Examine summary information about the random draws
print(monte_model.sample_stats)

# ----
# Define the counterfactual experiment and run the model trials
# ----

exp_data = monte_model.baseline_data.copy()
exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

# Run the trials by supplying the counterfactual experiment and (optionally) setting certain solver setting. For details
# on these settings, see OneSectorGE.build_baseline() and OneSectorGE.simulate().
monte_model.run_trials(experiment_data=exp_data,
                       omr_rescale=100,
                       result_stats = ['mean', 'std', 'sem', 'median'],
                       all_results = False)
# The argument result_stats determines the types of summary results that are produced across the different trials.
# For example, the supplied list will produce the mean, standard deviation, standard error, and median values for each
# type of result. To return all individual results across all trials in addition to the summary stats, set the all_
# results argument to True.

# ----
# Examine the MonteCarloGE results
# ----
# The MonteCarloGE model populates with the same set of results attributes as the OneSectorGE model. For example, we can
# retrieve to main country level results:
country_results = monte_model.country_results
print(country_results.head(8))

# We can also check to see if any trials failed to solve.
print(monte_model.num_failed_trials)
# In this case, it is zero. However, it is possible that certain draws of coefficients may not yield a solution. In many
# cases, however, these issues can be resolved by adjusting the omr_rescale_factor/ For example, using omr_rescale=10
# will likely result in some trials failing to solve, depending on the seed.

# If the set of all results was selected via all_results=True, these results can be accessed from attributes labeled
# using the standard label with 'all_' as a prefix. For example, the 'country_results' will populate in the attribute
# monte_model.all_country_results

# ---
# Export the results
# ---
# Finally to export the results to a series of .csv files.
monte_model.export_results(directory="examples//", name = 'monte')

# Alternatively, you can return this outpur in python for further manipulation.
i,j,k = monte_model.export_results()

# Finally, all of the results can of course be exported individually using standard python or pandas procedures.




