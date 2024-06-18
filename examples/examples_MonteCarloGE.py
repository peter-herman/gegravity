__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "June 6, 2024"
__Description__ = '''A demonstration of the MonteCarloGE model/module'''

# ----
# Import some packages
# ----

# Note: To use the code from the packaged version of gegravity, use the import statement "import gegravity as ge".
#       Alternatively, to use the source code/github version of the code, use "import src.gegravity as ge"

import gegravity as ge
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)

# ---
# Prepare model data  inputs
# ---
# Load sample data and add a constant.
raw_data = pd.read_csv("examples\sample_data_set.dlm")
raw_data['constant'] = 1

# Define baseline data structure
baseline = ge.BaselineData(raw_data, trade_var_name='trade', imp_var_name='importer', exp_var_name='exporter',
                           year_var_name='year', output_var_name='Y', expend_var_name='E')


# ----
# Set up Cost information
# ----

# Costs estimated via stata:
'''
import delimited "D:\work\Peter_Herman\projects\gegravity\examples\sample_data_set.dlm"
encode importer, gen(imp_fe)
encode exporter, gen(exp_fe)
ppmlhdfe trade lndist  contiguity  common_language pta international, absorb(i.exp_fe i.imp_fe)
matrix list e(V)
'''

# Create DataFrame of coefficient estimates and standard errors
ests = [
    #            'var',      'beta',   'stderr'
    (         'lndist',  -0.3898623,   0.0729915),
    (     'contiguity',    0.891577,   0.1327354),
    ('common_language',   0.0326249,   0.0840702),
    (            'pta',   0.4711383,   0.1076578),
    (  'international',   -3.412584,   0.2151235),
    (       'constant',    16.32434,   0.4844137)]
ests = pd.DataFrame(ests, columns = ['var', 'beta', 'stderr'])

# Create dataframe of Variance/Covariance matrix
var_covar = [
    #            'var',     'lndist',  'contiguity',   'common_language',   'pta',    'international', 'constant'
    (         'lndist',     .00532776,     .00504029,     .00093778,     .00473823,    -.01440259,    -.03476407),
    (     'contiguity',     .00504029,     .01761868,    -.00175404,    -.00030881,    -.01492965,    -.03036031),
    ('common_language',     .00093778,    -.00175404,     .00706779,     .00249937,     .00136678,    -.01312498),
    (            'pta',     .00473823,    -.00030881,     .00249937,     .01159021,    -.01538926,    -.03255428),
    (  'international',    -.01440259,    -.01492965,     .00136678,    -.01538926,     .04627812,     .08911527),
    (       'constant',    -.03476407,    -.03036031,    -.01312498,    -.03255428,     .08911527,     .23465662),
]
var_covar = pd.DataFrame(var_covar, columns = ['var', 'lndist', 'contiguity', 'common_language', 'pta',
                                               'international', 'constant'])
# Set row variable labels as DataFrame index
var_covar.set_index('var', inplace = True)


# Define gegravity CostValues object to organize this info as inputs for the GE model
cost_params = ge.CostCoeffs(estimates = ests, identifier_col='var', coeff_col='beta', stderr_col='stderr',
                            covar_matrix=var_covar)




# ----
# Define the MonetCarloGE model
# ----

# Here we demonstrate a small Monte Carlo experiment using 10 trials. The model will randomly draw 10 sets of cost
# coefficients from their joint normal distribution and simulate 10 OneSectorGE gegravity corresponding to each draw.
# Most of the other MonteCarloGE arguments follow those from the OneSectorGE model. See that model for details.
mc_model = ge.MonteCarloGE(baseline,
                           year = '2006',
                           trials = 10,
                           reference_importer='DEU',
                           sigma=7,
                           cost_variables=['lndist', 'contiguity', 'common_language', 'pta', 'international', 'constant'],
                           cost_coeff_values=cost_params,
                           allow_singular_covar=True,
                           seed = 1)
# The seed argument sets the seed for the random draw. Setting a seed allows for repeatable/reproducible simulations.

# Examine the random parameter draws. Columns 0 to 9 are the 10 different trials
print(mc_model.coeff_sample)
'''
               var          0          1          2          3          4          5          6          7          8          9
0           lndist  -0.520534  -0.521725  -0.371895  -0.390785  -0.455785  -0.343360  -0.304334  -0.343764  -0.402571  -0.368141
1       contiguity   0.900885   0.830745   0.922490   0.853686   0.934542   1.043870   0.828643   0.797255   0.824927   0.771972
2  common_language   0.029730  -0.022368   0.011377   0.086987  -0.001718   0.057161  -0.009925   0.179760   0.043748   0.023636
3              pta   0.336452   0.331530   0.599859   0.356130   0.422941   0.510970   0.630710   0.602066   0.533864   0.512185
4    international  -3.118628  -3.174684  -3.573262  -3.310206  -3.267814  -3.510545  -3.723681  -3.457308  -3.348660  -3.479166
5         constant  17.128512  17.209114  16.222708  16.288851  16.786156  15.989807  15.813891  15.920395  16.368214  16.219798
'''

# Examine summary information about the random draws
print(mc_model.sample_stats)

# ----
# Define the counterfactual experiment and run the model trials
# ----
# Collect a copy of the baseline data to use for the counterfactual data. Be sure to include .copy() to that you are
# modifying a new dataframe instead of the existing baseline data.
exp_data = mc_model.baseline_data.copy()
exp_data.loc[(exp_data["importer"] == "CAN") & (exp_data["exporter"] == "JPN"), "pta"] = 1
exp_data.loc[(exp_data["importer"] == "JPN") & (exp_data["exporter"] == "CAN"), "pta"] = 1

# Run the trials by supplying the counterfactual experiment and (optionally) setting certain solver settings. For
# details on these settings, see OneSectorGE.build_baseline() and OneSectorGE.simulate().
mc_model.run_trials(experiment_data=exp_data,
                    omr_rescale=1,
                    result_stats = ['mean', 'std', 'sem', 'median'],
                    all_results = True)

# The argument result_stats determines the types of summary results that are produced across the different trials.
# For example, the supplied list will produce the mean, standard deviation, standard error, and median values for each
# type of result. To return all individual results across all trials in addition to the summary stats, set the all_
# results argument to True.



# ----
# Examine the MonteCarloGE results
# ----
# The MonteCarloGE model populates with the same set of results attributes as the OneSectorGE model. For example, we can
# retrieve to main country level results using the following. For the MC model, the dataframe contains summary stats
# across for the different outcomes across all trials
mc_country_results = mc_model.country_results
print(mc_country_results.head(8))
'''
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
'''

# If all_results = True, the model retains information about all of the trial runs. These can be accessed in a variety
# of ways. One option is to view all country-level results for each trial. Columns reflect trial numer and result type.
# (E.g. (0, 'welfare statistic') for the value of the Trial 0 welfare statistic).
all_mc_country_results = mc_model.all_country_results


# Alternatively, MonteCarloGE is based on running a series of OneSectorGE models using different cost parameters for
# each trial. These individual OneSectorGE models are stored in a list that can be accessed via the attribute
# trial_models. For example, we can examine the results from trial 2:
trial_2_model = mc_model.trial_models[2]
print(trial_2_model.country_results.head())




# ---
# Export the results
# ---
# Finally, a convenient function to export the main results to a series of .csv files.
mc_model.export_results(directory="examples//", name = 'monte')




# ----
# Dealing with Failed Trials
# ----
# Some GE gravity models are more difficult to solve than others. The repeated sampling and resolving of the model can
# result in cases where the model solves for some trials but not others. There are a few tools available to identify
# and mitigate these issues. For simplicity, the methods are demonstrated using the above example model, which does not
# exhibit any failed trials within the 10 specified trials, but can still illustrate the methods.

# We can check for failed trials, which are listed by their trial number (0 to N). In this case, there are none.
print(mc_model.failed_trials)

# One likely source of solver issues is the OMR rescale factor. There is a version of the check_omr_rescale method for
# the MonteCarloGE model that will check some or all of the trials for feasible OMR scaling factors. Here, we will check
# the values for trials 1, 4, and 5. Omitting that argument will check all trials.
mc_model_2 = ge.MonteCarloGE(baseline,
                           year = '2006',
                           trials = 10,
                           reference_importer='DEU',
                           sigma=7,
                           cost_variables=['lndist', 'contiguity', 'common_language', 'pta', 'international', 'constant'],
                           cost_coeff_values=cost_params,
                           allow_singular_covar=True,
                           seed = 1)

mc_omr_checks = mc_model_2.check_omr_rescale(omr_rescale_range=5, trials = [1, 4, 5])

# One potential option for dealing with a failed trial is to specify a different omr_rescale_factor to use for that
# Specific trial. FOr example, we could specify that trials 4 and 5 use a rescale factor of 100 instead of 1, which is
# used for all other trials.
mc_model_2.run_trials(experiment_data=exp_data,
                      omr_rescale=1,
                      trial_omr_rescale={4:100, 5:100})

# If the source of the failure cannot be overcome by adjusting the rescale factor, another option is to draw additional
# trials to replace those that failed. This has the advantage of insuring that a desired number of trials is run in
# total. However, it should also be noted that if certain parameter draws are failing for systemitic reasons, then
# redrawing failed trials could introduce an unwanted bias in the results. Therefore, this option should be used with
# some caution.
mc_model_2.run_trials(experiment_data=exp_data,
                      omr_rescale=1,
                      redraw_failed_trials=True)





