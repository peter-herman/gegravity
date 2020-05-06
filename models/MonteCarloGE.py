__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "May 05, 2020"
__Description__ = '''A method for Generating Monte Carlo GE models using the distributions of parameter estimates from 
the empirical model '''

import numpy as np
import pandas as pd
from gme.estimate.EstimationModel import EstimationModel
from models.OneSectorGE import OneSectorGE
from typing import List


class MonteCarloGE(object):
    def __init__(self,
                 estimation_model: EstimationModel,
                 trials: int,
                 cost_variables: list,
                 mc_variables: list = None,
                 results_key: str = 'all',
                 seed:int = None):
        self._estimation_model = estimation_model
        self._cost_variables = cost_variables
        if mc_variables is None:
            self._mc_variables = self._cost_variables
        else:
            self._mc_variables = mc_variables
        if seed is None:
            self._seed = np.random.randint(0,10000)
        else:
            self._seed = seed
        self.results_key = results_key
        self.main_coeffs = self._estimation_model.results_dict[self.results_key].params
        self.main_stderrs = self._estimation_model.results_dict[self.results_key].bse
        self.trials = trials
        self.coeff_sample = self.get_mc_params()


        # Create sumamry of sample distrbution
        sample_stats = self.coeff_sample.T.describe().T
        new_col_names = ['sample_{}'.format(col) for col in sample_stats]
        sample_stats.columns = new_col_names
        main_cost_ests = pd.DataFrame({'beta_estimate': self.main_coeffs[self._mc_variables],
                                       'stderr_estimate': self.main_stderrs[self._mc_variables]})
        self.sample_stats = pd.concat([main_cost_ests, sample_stats], axis=1)

    def get_mc_params(self):
        var_samples = list()
        # Define seeds for each variable draw
        np.random.seed(self._seed)
        variable_seeds = np.random.randint(0, 10000, len(self._mc_variables))
        # Create sample for each mc variable
        for num, var in enumerate(self._mc_variables):
            beta = self.main_coeffs[var]
            stderr = self.main_stderrs[var]
            np.random.seed(variable_seeds[num])
            var_draw = np.random.normal(beta, stderr, self.trials)
            var_samples.append(pd.DataFrame({var: var_draw}))
        # Combine all variable samples
        mc_sample = pd.concat(var_samples, axis=1).T

        # Add rows without variation for cost variables not a part of mc
        costs_not_mc = [var for var in self._cost_variables if var not in self._mc_variables]
        for var in costs_not_mc:
            mc_sample.loc[var,:] = self.main_coeffs[var]
        return mc_sample

    def OneSectorGE(self,
                    estimation_model: EstimationModel,
                    year: str,
                    expend_var_name: str = 'expenditure',
                    output_var_name: str = 'output',
                    sigma: float = 5,
                    reference_importer: str = None,
                    omr_rescale: float = 1000,
                    imr_rescale: float = 1,
                    mr_method: str = 'hybr',
                    mr_max_iter: int = 1400,
                    mr_tolerance: float = 1e-8,
                    approach: str = None,
                    quiet: bool = True
                    ):
        models = list()
        for trial in range(self.trials):
            trial_coeffs = self.coeff_sample[trial]
            trial_model = OneSectorGE(self._estimation_model,
                                      cost_variables = None,
                                      cost_coeffs=trial_coeffs,
                                      year=year,
                                      expend_var_name=expend_var_name,
                                      output_var_name=output_var_name,
                                      sigma=sigma,
                                      reference_importer = reference_importer,
                                      omr_rescale = omr_rescale,
                                      imr_rescale = imr_rescale,
                                      mr_method = mr_method,
                                      mr_max_iter = mr_max_iter,
                                      mr_tolerance = mr_tolerance,
                                      approach = approach,
                                      quiet = quiet)
