__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "February 16, 2021"
__Description__ = ''' '''


import pandas as pd
import gme as gme
from models.OneSectorGE import OneSectorGE


# -----
# Setup
# -----

source_data_local = "D:\work\Peter_Herman\projects\gegravity\Yotov et al (2016) files\Chapter2\Datasets\\1_TradeWithoutBorder.dta"
grav_vars = ['ln_DIST', 'CNTG', 'INTL_BRDR']



source_data = pd.read_stata(source_data_local)

# Drop FE columns, we don't need them
fe_columns = [col for col in source_data.columns if col.startswith('IMPORTER_FE') or col.startswith('EXPORTER_FE')]
source_data.drop(fe_columns, axis = 1, inplace = True)

# create non-float year variable
source_data['year']='2006'

# -----
# Estimate Parameters
# -----
gme_data = gme.EstimationData(source_data, exp_var_name='exporter', imp_var_name='importer',year_var_name='year',
                              trade_var_name='trade')

gme_model = gme.EstimationModel(gme_data, lhs_var='trade', rhs_var=grav_vars,
                                fixed_effects=[['exporter'],['importer']], omit_fixed_effect=[['importer']])
gme_model.estimate()
gme_model.results_dict['all'].summary()

##
# Check Estimates
##
if round(gme_model.results_dict['all'].params['ln_DIST'],3)!= -0.791:
    raise ValueError("'Distance estimate does not match expectation.")
if round(gme_model.results_dict['all'].params['CNTG'],3)!= 0.674:
    raise ValueError("Contiguity estimate does not match expectation.")
if round(gme_model.results_dict['all'].params['INTL_BRDR'],3)!= -2.474:
    raise ValueError("Contiguity estimate does not match expectation.")


# ---
# Set up GE Analysis
# ---

ge_model = OneSectorGE(estimation_model=gme_model,
                       expend_var_name='E',
                       output_var_name='Y',
                       year='2006', sigma=7, reference_importer='ZZZ')
# omr_check = ge_model.check_omr_rescale(mr_max_iter=5000)
ge_model.build_baseline(mr_max_iter=5000, omr_rescale=1000)

experiment_data = ge_model.baseline_data.copy()
experiment_data['INTL_BRDR'] = 0
ge_model.define_experiment(experiment_data)

ge_model.simulate(ge_max_iter=5000)

cntry_results = ge_model.country_results
