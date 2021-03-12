__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "May 05, 2020"
__Description__ = '''A method for Generating Monte Carlo GE models using the distributions of parameter estimates from 
the empirical model '''
__all__ = ['MonteCarloGE']

import numpy as np
import pandas as pd
from gme.estimate.EstimationModel import EstimationModel
from models.OneSectorGE import OneSectorGE, CostCoeffs, _GEMetaData
from scipy.stats import multivariate_normal
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from typing import List

class MonteCarloGE(object):
    def __init__(self,
                 estimation_model: EstimationModel,
                 year:str,
                 trials: int,
                 expend_var_name: str,
                 output_var_name: str,
                 sigma: float,
                 reference_importer: str,
                 cost_variables: list,
                 mc_variables: list = None,
                 results_key: str = 'all',
                 seed: int = 0):
        '''

        Args:
            estimation_model(EstimationModel): A GME EstimationModel that must have been estimated with the option
                full_results = True (MonteCarlo simulation requires additional info from estimation compared to
                OneSectorGE)
            year:
            trials:
            expend_var_name:
            output_var_name:
            sigma:
            reference_importer:
            cost_variables:
            mc_variables:
            results_key:
            seed:
        '''

        # Store some inputs in model object
        self._estimation_model = estimation_model
        self.meta_data = _GEMetaData(estimation_model.estimation_data._meta_data, expend_var_name, output_var_name)
        self._year = str(year)
        self.sigma = sigma
        self._reference_importer = reference_importer
        self._cost_variables = cost_variables
        if mc_variables is None:
            self._mc_variables = self._cost_variables
        else:
            self._mc_variables = mc_variables
        if seed is None:
            self._seed = np.random.randint(0,10000)
        else:
            self._seed = seed
        self._results_key = results_key

        # Define Parameter values
        self.main_coeffs = self._estimation_model.results_dict[self._results_key].params
        self.main_stderrs = self._estimation_model.results_dict[self._results_key].bse
        self.trials = trials

        # Generate Sampling Distribution
        self.coeff_sample = self._draw_mc_trade_costs()

        ##
        # Define Results attributes
        ##
        self.num_failed_iterations = None
        self.all_country_results = None
        self.country_results = None
        self.all_country_mr_terms = None
        self.country_mr_terms = None
        self.all_outputs_expenditures = None
        self.outputs_expenditures = None
        self.all_factory_gate_prices = None
        self.factory_gate_prices = None
        self.all_aggregate_trade_results = None
        self.aggregate_trade_results = None
        self.all_bilateral_trade_results = None
        self.bilateral_trade_results = None
        self.all_bilateral_costs = None
        self.bilateral_costs = None
        self.solver_diagnostics = None


        # prep baseline data
        _baseline_data = self._estimation_model.estimation_data.data_frame.copy()
        _baseline_data[self.meta_data.year_var_name] = _baseline_data[self.meta_data.year_var_name].astype(str)
        self.baseline_data = _baseline_data.loc[_baseline_data[self.meta_data.year_var_name] == self._year, :].copy()
        if self.baseline_data.shape[0] == 0:
            raise ValueError("There are no observations corresponding to the supplied 'year'")

        # Create summary of sample distrbution
        sample_stats = self.coeff_sample.T.describe().T
        new_col_names = ['sample_{}'.format(col) for col in sample_stats]
        sample_stats.columns = new_col_names
        main_cost_ests = pd.DataFrame({'beta_estimate': self.main_coeffs[self._mc_variables],
                                       'stderr_estimate': self.main_stderrs[self._mc_variables]})
        self.sample_stats = pd.concat([main_cost_ests, sample_stats], axis=1)

    def _draw_mc_trade_costs(self):
        '''
        Draw coefficient values from multivariate normal distribution. For Poisson MLE,
        B-hat ~ Normal(B, (X'WX)^{-1}) where (X'WX)^{-1} is the covariance matrix. See An Introduction to Generalized
        Linear Models (2nd Ed) Annette J. Dobson, Chapman & Hall/CRC, Boca Raton Florida, Section 5.4.

        Returns: A dataframe of random draws of coefficients. Rows are cost variables from self._mc_variables, columns
            are different draws with the exception of a column with corresponding cost variable names ('index')
        '''
        # Get results and check that all needed info is available (i.e. covariance matrix in estimation model)
        est_results = self._estimation_model.results_dict[self._results_key]
        if not isinstance(est_results,GLMResultsWrapper):
            raise TypeError('MonteCarloGE requires that gme.EstimationModel be estimated with option full_results=True')
        betas = est_results.params.values

        cov = self._estimation_model.results_dict[self._results_key].cov_params()
        distribution_alt = multivariate_normal(betas, cov, seed=0)
        draws = list()
        for i in range(self.trials):
            draws.append(pd.Series(distribution_alt.rvs()))
        all_draws = pd.concat(draws, axis=1)
        all_draws.index = est_results.params.index
        all_draws = all_draws.loc[['LN_DIST', 'CNTG', 'BRDR'],:]
        return all_draws.reset_index()


    def run_trials(self,
                   experiment_data:pd.DataFrame,
                   omr_rescale: float = 1,
                   imr_rescale: float = 1,
                   mr_method: str = 'hybr',
                   mr_max_iter: int = 1400,
                   mr_tolerance: float = 1e-8,
                   ge_method:str = 'hybr',
                   ge_tolerance: float = 1e-8,
                   ge_max_iter: int = 1000,
                   quiet: bool = False,
                   result_stats:list = ['mean', 'std', 'sem']):
        '''
        Conduct Monte Carlo Simulation of OneSectorGE gravity model.
        Args:
            experiment_data (Pandas.DataFrame): A dataframe containing the counterfactual trade-cost data to use for the
                experiment. The best approach for creating this data is to copy the baseline data
                (MonteCarloGE.baseline_data.copy()) and modify columns/rows to reflect desired counterfactual experiment.
            omr_rescale (int): (optional) This value rescales the OMR values to assist in convergence. Often, OMR values
                are orders of magnitude different than IMR values, which can make convergence difficult. Scaling by a
                different order of magnitude can help. Values should be of the form 10^n. By default, this value is 1
                (10^0). However, users should be careful with this choice as results, even when convergent, may not be
                fully robust to any selection. The method OneSectorGE.check_omr_rescale() can help identify and compare
                feasible values for a given model.
            imr_rescale (int): (optional) This value rescales the IMR values to potentially aid in conversion. However,
                because the IMR for the reference importer is normalized to one, it is unlikely that there will be because
                because changing the default value, which is 1.
            mr_method (str): This parameter determines the type of non-linear solver used for solving the baseline and
                experiment MR terms. See the documentation for scipy.optimize.root for alternative methods. the default
                value is 'hybr'. (See also OneSectorGE.build_baseline())
            mr_max_iter (int): (optional) This parameter sets the maximum limit on the number of iterations conducted
                by the solver used to solve for MR terms. The default value is 1400.
                (See also OneSectorGE.build_baseline())
            mr_tolerance  (float): (optional) This parameter sets the convergence tolerance level for the solver used to
                solve for MR terms. The default value is 1e-8. (See also OneSectorGE.build_baseline())
            ge_method (str): (optional) The solver method to use for the full GE non-linear solver. See scipy.root()
                documentation for option. Default is 'hybr'.
            ge_tolerance (float): (optional) The tolerance for determining if the GE system of equations is solved.
                Default is 1e-8.
            ge_max_iter (int): (optional) The maximum number of iterations allowed for the full GE nonlinear solver.
                Default is 1000.
            quiet (bool): If True, suppress console printouts detailing the solver success/failures of each trial.
                Default is False.
            result_stats (list): A list of functions to compute in order to summarize the results across trials. The
                default is ['mean', 'std', 'sem'], which computes the mean, standard deviation, and standard mean error
                of the results, respectively. The model should accept any function that can be used with the
                pandas.DataFrame.agg() function.

        Returns:
            None
                No return but populates many results attributes of the MonteCarloGE model.

        '''
        models = list()
        num_failed_iterations = 0
        for trial in range(self.trials):
            print("\n* Simulating trial {} *".format(trial))
            param_values = CostCoeffs(self.coeff_sample, coeff_col=trial, identifier_col='index')
            try:
                trial_model = OneSectorGE(self._estimation_model,
                                          year=self._year,
                                          reference_importer=self._reference_importer,
                                          expend_var_name=self.meta_data.expend_var_name,
                                          output_var_name=self.meta_data.output_var_name,
                                          sigma=self.sigma,
                                          results_key=self._results_key,
                                          cost_variables=self._cost_variables,
                                          cost_coeff_values=param_values,
                                          # approach = approach,
                                          quiet = quiet)
                trial_model.build_baseline(omr_rescale=omr_rescale,
                                           imr_rescale=imr_rescale,
                                           mr_method=mr_method,
                                           mr_max_iter=mr_max_iter,
                                           mr_tolerance=mr_tolerance)
                trial_model.define_experiment(experiment_data)
                trial_model.simulate(ge_method=ge_method,
                                     ge_tolerance=ge_tolerance,
                                     ge_max_iter=ge_max_iter)
                models.append(trial_model)
            except:
                print("Failed to solve model.\n")
                num_failed_iterations+=1
        self.num_failed_iterations = num_failed_iterations
        self.all_country_results, self.country_results = self._compile_results(models, 'country_results', result_stats)
        self.all_country_mr_terms, self.country_mr_terms = self._compile_results(models, 'mr_terms', result_stats)
        self.all_outputs_expenditures, self.outputs_expenditures = self._compile_results(models, 'outputs_expenditures', result_stats)
        self.all_factory_gate_prices, self.factory_gate_prices = self._compile_results(models, 'factory_gate_prices', result_stats)
        self.all_aggregate_trade_results, self.aggregate_trade_results = self._compile_results(models, 'aggregate_trade_results', result_stats)
        self.all_bilateral_trade_results, self.bilateral_trade_results = self._compile_results(models, 'bilateral_trade', result_stats)
        self.all_bilateral_costs, self.bilateral_costs = self._compile_results(models, 'bilateral_costs', result_stats)
        self._compile_diagnostics(models)
        # ToDo: build some method for confidence intervals from Anderson Yotov (2016)


    def _compile_results(self, models, result_type, result_stats):
        '''
        Compile results across all trials.
        :param models: (List[OneSectorGE]) A list of solved OneSectorGE models.
        :param result_type: (str) Type of results to compile. Function works with:
            'country_results' - compiles results from OneSectorGE.country_mr_results
            'mr_terms' - compiles results from OneSectorGE.country_mr_terms
            'output_expenditures' - compiles results from OneSectorGE.output_expenditures
            'factory_gate_prices - compiles results from OneSectorGE.factory_gate_prices
            'aggregate_trade_results' - compiles results from OneSectorGE.aggregate_trade_results
            'bilateral_trade' - complies results from OneSectorGE.bilateral_trade_results
            'bilateral_costs' - compiles results from OneSectorGE.bilateral_costs
        :return:(pd.DataFrame, pd.DataFrame) Two dataframes. The first contains all results for each trial, with
            multiindex columns labeled (trial, result type). The second provides summary stats from all trials (mean,
            std, stderr)
        '''
        # Combine all results
        combined_results_list = list()
        for num, model in enumerate(models):
            if result_type == 'country_results':
                model_results = model.country_results.copy()
            if result_type == 'mr_terms':
                model_results = model.country_mr_terms
            if result_type == 'outputs_expenditures':
                model_results = model.outputs_expenditures
            if result_type == 'factory_gate_prices':
                model_results = model.factory_gate_prices
            if result_type == 'aggregate_trade_results':
                model_results = model.aggregate_trade_results
            if result_type == 'bilateral_trade':
                model_results = model.bilateral_trade_results
            if result_type == 'bilateral_costs':
                model_results = model.bilateral_costs

            # Label columns via multiindex with (trial #, result label)
            multi_columns = [(num,col) for col in model_results.columns]
            model_results.columns = pd.MultiIndex.from_tuples(multi_columns)
            combined_results_list.append(model_results)
        combined_results = pd.concat(combined_results_list, axis = 1)

        # Reshape trials to long format
        summary_results = combined_results.copy()
        if result_type in ['bilateral_trade','bilateral_costs']:
            # Bilateral trade has a two-part index (exporter and importer) and must be treated separately.
            summary_results = summary_results.stack(0).reset_index(level=2)
            summary_results.rename(columns={'level_2': 'trial'}, inplace=True)
        else:
            summary_results = summary_results.stack(0).reset_index(level=1)
            summary_results.rename(columns={'level_1': 'trial'}, inplace=True)

        # Compute mean and std across trials
        agg_dict = dict()
        var_list = list(summary_results.columns)
        var_list.remove('trial')
        for var in var_list:
            agg_dict[var] = result_stats
        if result_type in ['bilateral_trade','bilateral_costs']:
            summary_results = summary_results.groupby(level=[0, 1]).agg(agg_dict)
        else:
            summary_results = summary_results.groupby(level=0).agg(agg_dict)

        # Compute standard error for each result type
        # for col in summary_results.columns:
        #     if col[1] == 'std':
        #         summary_results[(col[0], 'stderr')] = summary_results[col] / (self.trials ** 0.5)
        if result_type in ['bilateral_trade','bilateral_costs']:
            summary_results = summary_results.stack(level=1).reset_index(level=2)
            summary_results.rename(columns = {'level_2':'statistic'},inplace = True)
        else:
            summary_results = summary_results.stack(level=1).reset_index(level=1)
            summary_results.rename(columns = {'level_1':'statistic'},inplace = True)
        return combined_results, summary_results

    def _compile_diagnostics(self, models):
        '''
        Compiles the diagnostics from each trial into a single dictionary, indexed by the trial number.
        Args:
            models: the list of OneSectorGE models associated with each trial

        Returns: None, Populates the attribute self.solver_daignostics

        '''
        combined_diagnostics = dict()
        for trial in range(self.trials):
            combined_diagnostics[0] = models[trial].solver_diagnostics
        self.solver_diagnostics = combined_diagnostics




