__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "May 05, 2020"
__Description__ = '''A method for Generating Monte Carlo GE gegravity using the distributions of parameter estimates from 
the empirical model '''
__all__ = ['MonteCarloGE']

import numpy as np
import pandas as pd
from pandas import DataFrame
from gme.estimate.EstimationModel import EstimationModel
from .OneSectorGE import OneSectorGE, CostCoeffs, _GEMetaData
from .BaselineData import BaselineData
from scipy.stats import multivariate_normal
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from typing import List

class MonteCarloGE(object):
    def __init__(self,
                 baseline: BaselineData,
                 year:str,
                 trials: int,
                 expend_var_name: str,
                 output_var_name: str,
                 sigma: float,
                 reference_importer: str,
                 cost_variables: list,
                 mc_variables: list = None,
                 cost_coeff_values=None,
                 seed: int = 0,
                 allow_singular_covar:bool = False):
        '''
        Define a Monte Carlo GE model.
        Args:
            baseline (BaselineData): Baseline data defined using the gegravity BaselineData class structure.
            year (str): The year to be used for the baseline model. Works best if estimation_model year column has been
                cast as string.
            trials (int): The number of trial simulations to conduct.
            expend_var_name (str): Column name of variable containing expenditure data in estimation_model.
            output_var_name (str): Column name of variable containing output data in estimation_model.
            sigma (float): Elasticity of substitution.
            reference_importer (str): Identifier for the country to be used as the reference importer (inward
                multilateral resistance normalized to 1 and other multilateral resistances solved relative to it).
            cost_variables (List[str]): (optional) A list of variables to use to compute bilateral trade costs. By
                default, all included non-fixed effect variables are used.
            mc_variables (List[str]): (optional) A subset of the cost_variables to randomly sample in the Monte Carlo
                experiment. Coefficients for the variables in this list are randomly drawn based on their estimated mean
                 and variance/covariance. Those excluded use their gravity estimated values only. By default, the model
                uses all cost variables (or those supplied to cost_variables argument) are
            cost_coeff_values (CostCoeffs): (optional) A set of parameter values or estimates to use for constructing
                trade costs. Should be of type gegravity.CostCoeffs, statsmodels.GLMResultsWrapper, or
                gme.SlimResults. If no values are provided, the estimates in the EstimationModel are used.
            seed (int): (optional) The seed to use for the random draws of cost coefficients in order to provide
                unchanging, consistent draws across runs. By default, the seed is randomly determined each time the
                model is constructed.
            allow_singular_covar (bool): If true, allow the covariance matrix to be singular when drawing coefficient
                values from multivariate normal distribution. Default is False.

        Attributes:
            baseline_data (pandas.DataFrame): Baseline data supplied to model in gme.EstimationModel.
            coeff_sample (Pandas.DataFrame): The randomly drawn sample of cost coefficients for each cost variable. Each
                column corresponds to a different trial.
            main_coeffs (pandas.Series): The main coefficient estimates for the cost varaibles supplied to the model.
            main_stderrs (pandas.Series): The standard errors for the main cost coefficients.
            trials (int): The number of trial simulations of the model.
            sample_stats (pandas.DataFrame): A dataframe depicting both the initially supplied estimate values for each
                cost variable ('beta_estimate' and 'stderr_estimate') as well as descriptive statistics for the randomly
                drawn values across all trials.
            sigma (int): The elasticity of substitution parameter value.
            ---
            **Attributes containing results populated after MonteCarloGE.run_trials()**:\n\n
            aggregate_trade_results (Pandas.DataFrame): Aggregate trade results summarized across all trials. See
                OneSectorGE ResultsLabels for description of results.
            bilateral_costs (Pandas.DataFrame): Baseline and counterfactual bilateral trade costs summarized across all
                trials. See OneSectorGE ResultsLabels for description of results.
            bilateral_trade_results (Pandas.DataFrame): Bilateral trade results summarized across all trials.
                See OneSectorGE ResultsLabels for description of results.
            country_mr_terms (Pandas.DataFrame): Baseline and experiment multilateral resistance terms summarized
                across all trials. See OneSectorGE ResultsLabels for description of results.
            country_results (Pandas.DataFrame): The main country level results summarized across
                all trials. See OneSectorGE ResultsLabels for description of results.
            factory_gate_prices (Pandas.DataFrame): Factory gate prices summarized across all trials. See OneSectorGE
                ResultsLabels for description of results.
            num_failed_trials (int): The number of trials for which the model failed to solve.
            outputs_expenditures (Pandas.DataFrame): Baseline and experiment output and expenditure values summarized
                across all trials. See OneSectorGE ResultsLabels for description of results.
            solver_diagnostics (dict): A dictionary containing diagnostic information for each individual trial. The
                solver diagnostics for each trial correspond to the three solution routines: baseline
                multilateral resistances, conditional multilateral resistances (partial equilibrium counterfactual
                effects) and the full GE model. See the diagnostic info from scipy.optimize.root for more details.
            ---
            **Additional results populated if MonteCarloGE.run_trials(all_results=True)**\n\n
            all_aggregate_trade_results (Pandas.DataFrame): All aggregate trade results for each individual trial.
                Columns are multi-indexed by the trial number and type of result. See OneSectorGE ResultsLabels
                for description of results.
            all_bilateral_costs (Pandas.DataFrame): Bilateral trade costs for each individual trial. Columns are
                multi-indexed by the trial number and type of result. See OneSectorGE ResultsLabels for description of
                results.
            all_bilateral_trade_results (Pandas.DataFrame): Bilateral trade results for all individual trials.
                Columns are multi-indexed by the trial number and type of result.
                See OneSectorGE ResultsLabels for description of results.
            all_country_mr_terms (Pandas.DataFrame): All baseline and experiment multilateral resistance terms for each
                individual trial. Columns are multi-indexed by the trial number and type of result. See OneSectorGE
                ResultsLabels for description of results.
            all_country_results (Pandas.DataFrame): Main results for all individual trials.
                Columns are multi-indexed by the trial number and type of result.
                See OneSectorGE ResultsLabels for description of results.
            all_factory_gate_prices (Pandas.DataFrame): Factory gate prices for each individual trial. Columns are
                multi-indexed by the trial number and type of result. See OneSectorGE ResultsLabels for description of
                results.
            all_outputs_expenditures (Pandas.DataFrame): Baseline and experiment output and expenditure values for each
                individual trial. Columns are multi-indexed by the trial number and type of result. See OneSectorGE
                ResultsLabels for description of results.
        '''


        # Store some inputs in model object
        self._estimation_model = baseline
        self.meta_data = _GEMetaData(baseline.meta_data, expend_var_name, output_var_name)
        self._year = str(year)
        self.sigma = sigma
        self._reference_importer = reference_importer
        self._cost_variables = cost_variables
        self._cost_coeffs = cost_coeff_values
        self._allow_singular = allow_singular_covar
        if mc_variables is None:
            self._mc_variables = self._cost_variables
        else:
            self._mc_variables = mc_variables
        if seed is None:
            self._seed = np.random.randint(0,10000)
        else:
            self._seed = seed

        # Define Parameter values
        self.main_coeffs = self._cost_coeffs.params
        self.main_stderrs = self._cost_coeffs.bse
        self.main_covar_mat = self._cost_coeffs.covar
        self.trials = trials

        # Generate Sampling Distribution
        self.coeff_sample = self._draw_mc_trade_costs()




        ##
        # Define Results attributes
        ##
        self.num_failed_trials = None
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
        _baseline_data = self._estimation_model.baseline_data.copy()
        _baseline_data[self.meta_data.year_var_name] = _baseline_data[self.meta_data.year_var_name].astype(str)
        self.baseline_data = _baseline_data.loc[_baseline_data[self.meta_data.year_var_name] == self._year, :].copy()
        if self.baseline_data.shape[0] == 0:
            raise ValueError("There are no observations corresponding to the supplied 'year'")

        # Create summary of sample distribution
        sample_stats = self.coeff_sample.copy()
        sample_stats.set_index(self._cost_coeffs._identifier_col, inplace= True)
        sample_stats = sample_stats.T.describe().T

        # sample_stats = self.coeff_sample.T.describe().T
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
        print('Deriving sample of cost parameters...')
        # Get results and check that all needed info is available (i.e. covariance matrix in estimation model)
        betas = self.main_coeffs
        cov = self.main_covar_mat
        # Define distribution of beta parameters
        distribution_alt = multivariate_normal(betas, cov, seed=self._seed, allow_singular=self._allow_singular)
        draws = list()
        for i in range(self.trials):
            draws.append(pd.Series(distribution_alt.rvs()))
        all_draws = pd.concat(draws, axis=1)
        all_draws.index = betas.index
        all_draws = all_draws.loc[self._mc_variables,:]
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
                   result_stats:list = ['mean', 'std', 'sem'],
                   all_results:bool = False):
        '''
        Conduct Monte Carlo Simulation of OneSectorGE gravity model.
        Args:
            experiment_data (pandas.DataFrame): A dataframe containing the counterfactual trade-cost data to use for the
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
            mr_max_iter (int): This parameter sets the maximum limit on the number of iterations conducted
                by the solver used to solve for MR terms. The default value is 1400.
                (See also OneSectorGE.build_baseline())
            mr_tolerance (float): This parameter sets the convergence tolerance level for the solver used to
                solve for MR terms. The default value is 1e-8. (See also OneSectorGE.build_baseline())
            ge_method (str): The solver method to use for the full GE non-linear solver. See scipy.root()
                documentation for option. Default is 'hybr'.
            ge_tolerance (float): The tolerance for determining if the GE system of equations is solved.
                Default is 1e-8.
            ge_max_iter (int): The maximum number of iterations allowed for the full GE nonlinear solver.
                Default is 1000.
            quiet (bool): If True, suppress console printouts detailing the solver success/failures of each trial.
                Default is False.
            result_stats (list): A list of functions to compute in order to summarize the results across trials. The
                default is ['mean', 'std', 'sem'], which computes the mean, standard deviation, and standard mean error
                of the results, respectively. The model should accept any function that can be used with the
                pandas.DataFrame.agg() function.
            all_results (bool): If true, MonteCarloGE attributes containing individual results for all trials are
                populated. Default is False to reduce memory use.

        Returns:
            None: No return but populates many results attributes of the MonteCarloGE model.

        '''
        models = list()
        failed_trials = list()
        num_failed_iterations = 0
        for trial in range(self.trials):
            if not quiet:
                print("\n* Simulating trial {} *".format(trial))
            # Define a new CostCoeff instance using one of the trial values
            param_values = CostCoeffs(self.coeff_sample, coeff_col=trial, identifier_col=self._cost_coeffs._identifier_col)
            try:
                trial_model = OneSectorGE(self._estimation_model,
                                          year=self._year,
                                          reference_importer=self._reference_importer,
                                          expend_var_name=self.meta_data.expend_var_name,
                                          output_var_name=self.meta_data.output_var_name,
                                          sigma=self.sigma,
                                          cost_variables=self._cost_variables,
                                          cost_coeff_values=param_values,
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
                if not quiet:
                    print("Failed to solve model.\n")
                num_failed_iterations+=1
                failed_trials.append(trial)

        # Get results labels from one of the OneSectorGE gegravity
        self.labels = models[0].labels
        self.num_failed_trials = num_failed_iterations
        self.failed_trials = failed_trials
        self.all_country_results, self.country_results = self._compile_results(models, 'country_results', result_stats, all_results)
        self.all_country_mr_terms, self.country_mr_terms = self._compile_results(models, 'mr_terms', result_stats, all_results)
        self.all_outputs_expenditures, self.outputs_expenditures = self._compile_results(models, 'outputs_expenditures', result_stats, all_results)
        self.all_factory_gate_prices, self.factory_gate_prices = self._compile_results(models, 'factory_gate_prices', result_stats, all_results)
        self.all_aggregate_trade_results, self.aggregate_trade_results = self._compile_results(models, 'aggregate_trade_results', result_stats, all_results)
        self.all_bilateral_trade_results, self.bilateral_trade_results = self._compile_results(models, 'bilateral_trade', result_stats, all_results)
        self.all_bilateral_costs, self.bilateral_costs = self._compile_results(models, 'bilateral_costs', result_stats, all_results)
        self._compile_diagnostics(models)
        # ToDo: build some method for confidence intervals from Anderson Yotov (2016)


    def _compile_results(self, models, result_type, result_stats, all_results):
        '''
        Compile results across all trials.
        :param models: (List[OneSectorGE]) A list of solved OneSectorGE gegravity.
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
        if all_results:
            return combined_results, summary_results
        else:
            return None, summary_results

    def _compile_diagnostics(self, models):
        '''
        Compiles the diagnostics from each trial into a single dictionary, indexed by the trial number.
        Args:
            models: the list of OneSectorGE gegravity associated with each trial

        Returns: None, Populates the attribute self.solver_daignostics

        '''
        combined_diagnostics = dict()
        for trial in range(len(models)):
            combined_diagnostics[trial] = models[trial].solver_diagnostics
        self.solver_diagnostics = combined_diagnostics

    def export_results(self, directory:str = None, name:str = '',
                       country_names:DataFrame = None, country:bool = True, bilateral:bool = True,
                       diagnostics:bool = True):
        '''
        Export results to csv files. Three files are stored containing (1) country-level results, (2) bilateral results,
        and (3) solver diagnostics.
        Args:
            directory (str): (optional) Directory in which to write results files. If no directory is supplied,
                three compiled dataframes are returned as a tuple in the order (Country-level results, bilateral
                results, solver diagnostics).
            name (str): (optional) Name of the simulation to prefix to the result file names.
            include_levels (bool): (optional) If True, includes additional columns reflecting the simulated changes in
                levels based on observed trade flows (rather than modeled trade flows). Values are those from the
                method calculate_levels.
            country_names (pandas.DataFrame): (optional) Adds alternative identifiers such as names to the returned
                results tables. The supplied DataFrame should include exactly two columns. The first column must be
                the country identifiers used in the model. The second column must be the alternative identifiers to
                add.
            country (boolean): (optional) If True, export country-level results to csv. If False, skip these results.
                Default is True.
            bilateral (boolean): (optional) If True, export bilateral results to csv. If False, skip these results.
                Default is True.
            diagnostics (boolean): (optional) If True, export diagnostic information to csv. If False, skip these
                results. Default is True.

        Returns:
            None or Tuple[DataFrame, DataFrame, DataFrame]: If a directory argument is supplied, the method returns
                nothing and writes three .csv files instead. If no directory is supplied, it returns a tuple of
                DataFrames.

        Examples:


        '''

        importer_col = self.meta_data.imp_var_name
        exporter_col = self.meta_data.exp_var_name

        country_result_set = [self.country_results, self.factory_gate_prices, self.aggregate_trade_results,
                              self.outputs_expenditures, self.country_mr_terms]
        country_results = pd.concat(country_result_set, axis = 1)
        # Order and select columns for inclusion, drop duplicates.
        country_results_cols = country_results.columns
        labs = self.labels
        # Country results to include
        results_cols = ['statistic'] + self.labels.country_level_labels
        included_columns = [col for col in results_cols if col in country_results_cols]
        country_results = country_results[included_columns]
        country_results = country_results.loc[:, ~country_results.columns.duplicated()]

        bilateral_results = self.bilateral_trade_results.reset_index()

        if country_names is not None:
            if country_names.shape[1]!=2:
                raise ValueError("country_names should have exactly 2 columns, not {}".format(country_names.shape[1]))
            code_col = country_names.columns[0]
            name_col = country_names.columns[1]
            country_names.set_index(code_col, inplace = True, drop = True)
            country_results = country_names.merge(country_results, how = 'right', left_index = True, right_index=True)

            # Add names to bilateral data
            for side in [exporter_col, importer_col]:
                side_names = country_names.copy()
                side_names.reset_index(inplace = True)
                side_names.rename(columns = {code_col:side, name_col:"{} {}".format(side,name_col)}, inplace = True)
                bilateral_results = bilateral_results.merge(side_names, how = 'left', on = side)

        # Create Dataframe with Diagnostic results
        column_list = list()
        diagnostics = self.solver_diagnostics
        for trial_num, trial in diagnostics.items():
            for results_type, results in trial.items():
                for key, value in results.items():
                    # Single Entry fields must be converted to list before creating DataFrame
                    if key in ['success', 'status', 'nfev', 'message']:
                        frame = pd.DataFrame({("trial_{}".format(trial_num), results_type, key): [value]})
                        column_list.append(frame)
                    # Vector-like fields Can be used as is. Several available fields are not included: 'fjac','r', and 'qtf'
                    elif key in ['x', 'fun']:
                        frame = pd.DataFrame({("trial_{}".format(trial_num), results_type, key): value})
                        column_list.append(frame)
        diag_frame = pd.concat(column_list, axis=1)
        diag_frame = diag_frame.fillna('')

        if directory is not None:
            if country:
                country_results.to_csv("{}/{}_country_results.csv".format(directory, name))
            if bilateral:
                bilateral_results.to_csv("{}/{}_bilateral_results.csv".format(directory, name), index=False)
            if diagnostics:
                diag_frame.to_csv("{}/{}_solver_diagnostics.csv".format(directory, name), index=False)
        else:
            return country_results, bilateral_results, diag_frame





