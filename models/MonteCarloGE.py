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
                 year:str,
                 trials: int,
                 cost_variables: list,
                 mc_variables: list = None,
                 results_key: str = 'all',
                 seed:int = None):
        self._estimation_model = estimation_model
        self.meta_data = self._estimation_model.estimation_data._meta_data
        self._year = str(year)
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

        ##
        # Define Results attributes
        ##
        self.all_country_results = None
        self.country_results = None


        # prep baseline data
        _baseline_data = self._estimation_model.estimation_data.data_frame.copy()
        _baseline_data[self.meta_data.year_var_name] = _baseline_data[self.meta_data.year_var_name].astype(str)
        self.baseline_data = _baseline_data.loc[_baseline_data[self.meta_data.year_var_name] == self._year, :].copy()
        if self.baseline_data.shape[0] == 0:
            raise ValueError("There are no observations corresponding to the supplied 'year'")

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
                    experiment_data:pd.DataFrame,
                    expend_var_name: str = 'expenditure',
                    output_var_name: str = 'output',
                    sigma: float = 5,
                    reference_importer: str = None,
                    omr_rescale: float = 1000,
                    imr_rescale: float = 1,
                    mr_method: str = 'hybr',
                    mr_max_iter: int = 1400,
                    mr_tolerance: float = 1e-8,
                    ge_method:str = 'hybr',
                    ge_tolerance: float = 1e-8,
                    ge_max_iter: int = 1000,
                    approach: str = None,
                    quiet: bool = True
                    ):
        models = list()
        for trial in range(self.trials):
            print("\n Simulating trial {}".format(trial))
            trial_coeffs = self.coeff_sample[trial]
            try:
                trial_model = OneSectorGE(self._estimation_model,
                                          year=self._year,
                                          expend_var_name=expend_var_name,
                                          output_var_name=output_var_name,
                                          sigma=sigma,
                                          results_key=self.results_key,
                                          cost_variables=self._cost_variables,
                                          cost_coeffs=trial_coeffs,
                                          reference_importer = reference_importer,
                                          omr_rescale = omr_rescale,
                                          imr_rescale = imr_rescale,
                                          mr_method = mr_method,
                                          mr_max_iter = mr_max_iter,
                                          mr_tolerance = mr_tolerance,
                                          approach = approach,
                                          quiet = quiet)
                trial_model.define_experiment(experiment_data)
                trial_model.simulate(ge_method=ge_method,
                                     ge_tolerance=ge_tolerance,
                                     ge_max_iter=ge_max_iter)
                models.append(trial_model)
            except:
                print("Failed to solve model.\n")
        self.all_country_results, self.country_results = self.compile_country_results(models)

    def compile_country_results(self, models):
        # Combine all results
        combined_results_list = list()
        for num, model in enumerate(models):
            model_results = model.country_results.copy()
            # Label columns via multiindex with (trial, result)
            multi_columns = [(num,col) for col in model_results.columns]
            model_results.columns = pd.MultiIndex.from_tuples(multi_columns)
            combined_results_list.append(model_results)
        combined_results = pd.concat(combined_results_list, axis = 1)

        # Reshape trials to long format
        summary_results = combined_results.copy()
        summary_results = summary_results.stack(0).reset_index(level=1)
        summary_results.rename(columns={'level_1': 'trial'}, inplace=True)

        # Compute mean and std across trials
        agg_dict = dict()
        var_list = list(summary_results.columns)
        var_list.remove('trial')
        for var in var_list:
            agg_dict[var] = ['mean', 'std']
        summary_results = summary_results.groupby(level=0).agg(agg_dict)

        # Compute standard error for each result type
        for col in summary_results.columns:
            if col[1] == 'std':
                summary_results[(col[0], 'stderr')] = summary_results[col] / 3 ** 0.5
        summary_results = summary_results.stack(level=1).reset_index(level=1)
        summary_results.rename(columns = {'level_1':'statistic'},inplace = 0)
        return combined_results, summary_results

