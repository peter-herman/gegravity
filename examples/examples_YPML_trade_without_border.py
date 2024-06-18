__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "February 16, 2021"
__Description__ = '''This example replicates the "trade without borders example from section 2.C of Yotov, Piermartini, 
Montiero, and Larch, 2016, An Advanced Guide to Trade Policy Analysis: The structural Gravity Model, Online Revised
Version, World Trade Organization and the United Nations Conference on Trade and Development. Required data files are 
available at https://vi.unctad.org/tpa/web/vol2/vol2home.html and https://yotoyotov.com/book.html'''


import pandas as pd
import gegravity as ge

# -----
# Setup the data
# -----

source_data_local = "Yotov et al (2016) files\\Chapter2\\Datasets\\1_TradeWithoutBorder.dta"
grav_vars = ['ln_DIST', 'CNTG', 'INTL_BRDR']



source_data = pd.read_stata(source_data_local)

# Drop FE columns, we don't need them
fe_columns = [col for col in source_data.columns if col.startswith('IMPORTER_FE') or col.startswith('EXPORTER_FE')]
source_data.drop(fe_columns, axis = 1, inplace = True)

# create non-float year variable
source_data['year']='2006'

baseline_data = ge.BaselineData(source_data, exp_var_name='exporter', imp_var_name='importer', year_var_name='year',
                                trade_var_name='trade', expend_var_name='E', output_var_name='Y')

# -----
# Define Cost Parameters Parameters
# -----

# Parameter values taken from page 104 of the Advanced Guide
params = [('ln_DIST', -0.791), ('CNTG', 0.674), ('INTL_BRDR',-2.474)]
params = pd.DataFrame(params, columns = ('variable', 'beta'))
cost_params = ge.CostCoeffs(params, identifier_col='variable', coeff_col='beta')



# ---
# Set up and run the GE Analysis
# ---

ge_model = ge.OneSectorGE(baseline=baseline_data,
                          cost_coeff_values=cost_params,
                          cost_variables=grav_vars,
                       year='2006', sigma=7, reference_importer='ZZZ')
# omr_check = ge_model.check_omr_rescale(mr_max_iter=5000)
ge_model.build_baseline(mr_max_iter=5000, omr_rescale=1000)

# Define the experiment
experiment_data = ge_model.baseline_data.copy()
experiment_data['INTL_BRDR'] = 0
ge_model.define_experiment(experiment_data)

# Simulate model
ge_model.simulate(ge_max_iter=5000)

# View results
cntry_results = ge_model.country_results

# Results are comparable the published version. The small quantitative differences can be attributed to numerical
# precision and differences in the numerical approaches used to solve each.