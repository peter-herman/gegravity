__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "May 05, 2020"
__Description__ = '''A method for Generating Monte Carlo GE models using the distributions of parameter estimates from 
the empirical model '''

import numpy as np
import pandas as pd
from gme.estimate.EstimationModel import EstimationModel
from models.OneSectorGE import OneSectorGE

class MonteCarloGE(object):
    def __init__(self,
                 estimation_model: EstimationModel,
                 trials: int,
                 cost_variables:list,
                 mc_variables:list = None,
                 results_key: str = 'all'):
        self._estimation_model = estimation_model
        self._cost_variables = cost_variables
        if mc_variables is None:
            self._mc_variables = self._cost_variables
        else:
            self._mc_variables = mc_variables
        self.results_key = results_key
        self.main_betas = self._estimation_model.results_dict[self.results_key].params
        self.main_stderrs = self._estimation_model.results_dict[self.results_key].bse
        self.trials = trials
        self.beta_sample = self.get_mc_params()

        # Create sumamry of sample distrbution
        sample_stats = self.beta_sample.T.describe().T
        new_col_names = ['sample_{}'.format(col) for col in sample_stats]
        sample_stats.columns = new_col_names
        main_cost_ests = pd.DataFrame({'beta_estimate':self.main_betas[self._mc_variables],
                                       'stderr_estimate':self.main_stderrs[self._mc_variables]})
        self.sample_stats = pd.concat([main_cost_ests, sample_stats], axis =1)


    def get_mc_params(self):
        var_samples = list()
        for var in self._mc_variables:
            beta = self.main_betas[var]
            stderr = self.main_stderrs[var]
            var_draw = np.random.normal(beta, stderr, self.trials)
            var_samples.append(pd.DataFrame({var:var_draw}))
        return pd.concat(var_samples, axis = 1).T



