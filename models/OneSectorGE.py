__Author__ = "Peter Herman"
__Project__ = "Gravity Code"
__Created__ = "08/15/2018"
__Description__ = """A single sector or aggregate full GE model based on Larch and Yotov, 'General Equilibrium Trade
                  Policy Analysis with Structural Gravity," 2016. (WTO Working Paper ERSD-2016-08)"""

from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from gme.estimate.EstimationModel import EstimationModel
from scipy.optimize import root
from numpy import multiply
from warnings import warn
import math as math


'''
Convergence Tips:
    1. Modify the omr_rescale factor. Examining the estimates for different countries from non-convergent simulations 
        can help inform the correct rescale factor. Function values close to 1 seem to suggest MR initial values that 
        are too small. OMR rescaling will likely work better as IMR for the reference importer equals 1 by definition.
    2. Ensure data is square otherwise necessary fields end up empty (e.g. can't construct all necessary trade costs)
'''

class OneSectorGE(object):
    def __init__(self,
                 estimation_model: EstimationModel,
                 year: str,
                 expend_var_name: str = 'expenditure',
                 output_var_name: str = 'output',
                 sigma: float = 5,
                 results_key: str = 'all',
                 cost_variables: List[str] = None,
                 cost_coeffs: pd.Series = None,
                 reference_importer: str = None,
                 omr_rescale: float = 1000,
                 imr_rescale: float = 1,
                 mr_method: str = 'hybr',
                 mr_max_iter: int = 1400,
                 mr_tolerance: float = 1e-8,
                 approach: str = None,
                 quiet:bool = False):
        '''

        :param estimation_model:
        :param year:
        :param expend_var_name:
        :param output_var_name:
        :param sigma:
        :param results_key:
        :param cost_variables:
        :param cost_coeffs: (pd.Series) (optional) A pandas series containing coefficient values corresponding to each
            of the specified cost variables. Should be formatted the same way as GLMResultsWrapper.params or
            gme.SlimResults. If no values are provided, the estimates in the EstimationModel are used.
        :param reference_importer:
        :param omr_rescale:
        :param imr_rescale:
        :param mr_method:
        :param mr_max_iter:
        :param mr_tolerance:
        :param approach:
        :param quiet: (bool) If True, suppresses all console feedback from model during simulation. Default is False.
        '''
        if not isinstance(year, str):
            raise TypeError('year should be a string')

        # Check reference country (does not currently work)
        # try:
        #     omitted_fe = estimation_model.results_dict[results_key].params[('importer_fe_'+reference_importer)]
        #     raise ValueError('reference_importer should be the excluded importer fixed effect')
        # except:
        #     print('reference_importer OK')

        # ToDo: Modify meta_data load when GME update includes it differently
        self.meta_data = _GEMetaData(estimation_model.estimation_data._meta_data, expend_var_name, output_var_name)
        self._estimation_model = estimation_model
        self._estimation_results = self._estimation_model.results_dict[results_key]
        self._year = str(year)
        self.sigma = sigma
        self._reference_importer = reference_importer
        self._omr_rescale = omr_rescale
        self._imr_rescale = imr_rescale
        self._mr_max_iter = mr_max_iter
        self._mr_tolerance = mr_tolerance
        self._mr_method = mr_method
        self._ge_method = None
        self._ge_tolerance = None
        self._ge_max_iter = None
        self.country_set = None
        self._economy = None
        self.baseline_trade_costs = None
        self.experiment_trade_costs = None
        self.cost_shock = None
        self.experiment_data = None
        self.approach = approach
        self.quiet = quiet

        # Results fields
        self.bilateral_trade_results = None
        self.aggregate_trade_results = None
        self.solver_diagnostics = dict()
        self.factory_gate_prices = None
        self.outputs_expenditures = None
        self.country_results = None
        self.country_mr_terms = None


        # ---
        # Check inputs
        # ---
        if self.meta_data.trade_var_name is None:
            raise ValueError('\n Missing Input: Please insure trade_var_name is set in EstimationData object.')

        if cost_variables is None:
            self.cost_variables = self._estimation_model.specification.rhs_var
        else:
            self.cost_variables = cost_variables

        if cost_coeffs is not None:
            self.cost_coeffs = cost_coeffs
        else:
            self.cost_coeffs = self._estimation_results.params[self.cost_variables]



        # prep baseline data
        _baseline_data = estimation_model.estimation_data.data_frame.copy()
        _baseline_data[self.meta_data.year_var_name] = _baseline_data[self.meta_data.year_var_name].astype(str)
        self.baseline_data = _baseline_data.loc[_baseline_data[self.meta_data.year_var_name] == self._year, :].copy()
        if self.baseline_data.shape[0] == 0:
            raise ValueError("There are no observations corresponding to the supplied 'year'")


        # Build Baseline

        self.build_baseline()

    def build_baseline(self):
        """
        Builds the baseline values for the model
        """
        # Initialize a set of countries and the economy
        self.country_set = self._create_baseline_countries()
        self._economy = self._create_baseline_economy()
        # Calculate certain country values using info from the whole economy
        for country in self.country_set:
            self.country_set[country].calculate_baseline_output_expenditure_shares(self._economy)
        # Calculate baseline trade costs
        self.baseline_trade_costs = self._create_trade_costs(self.baseline_data)
        # Solve for the baseline multilateral resistance terms
        if self.approach == 'GEPPML':
            self._calculate_GEPPML_multilateral_resistance(version='baseline')
        else:
            self._calculate_multilateral_resistance(trade_costs=self.baseline_trade_costs, version='baseline')
        # Calculate baseline factory gate prices
        self._calculate_baseline_factory_gate_params()

        # ToDo: run some checks the ensure the baseline is solved (e.g. the betas solve the factory gat price equations)

    def _create_baseline_countries(self):
        """
        Initialize set of country objects
        """
        # Requires that the baseline data has output and expenditure data

        # Make sure the year data is in string form
        self.baseline_data[self.meta_data.year_var_name] = self.baseline_data.loc[:,
                                                           self.meta_data.year_var_name].astype(
            str)

        # Create Country-level observations
        year_data = self.baseline_data.loc[self.baseline_data[self.meta_data.year_var_name] == self._year, :]

        importer_info = year_data[[self.meta_data.imp_var_name, self.meta_data.expend_var_name]].copy()

        importer_info = importer_info.groupby([self.meta_data.imp_var_name])
        expenditures = importer_info.mean().reset_index()

        exporter_info = year_data[[self.meta_data.exp_var_name, self.meta_data.output_var_name]].copy()
        exporter_info = exporter_info.groupby([self.meta_data.exp_var_name])
        output = exporter_info.mean().reset_index()

        country_data = pd.merge(left=expenditures,
                                right=output,
                                how='outer',
                                left_on=[self.meta_data.imp_var_name],
                                right_on=[self.meta_data.exp_var_name])

        reference_expenditure = float(
            country_data.loc[country_data[self.meta_data.imp_var_name] == self._reference_importer,
                             self.meta_data.expend_var_name])

        # Convert DataFrame to a dictionary of country objects
        country_set = {}

        # Identify appropriate fixed effect naming convention and define function for creating them
        fe_specification = self._estimation_model.specification.fixed_effects
        # Importer FEs
        if [self.meta_data.imp_var_name] in fe_specification:
            def imp_fe_identifier(country_id):
                return "_".join([self.meta_data.imp_var_name,
                                 'fe', (country_id)])
        elif [self.meta_data.imp_var_name, self.meta_data.year_var_name] in fe_specification:
            def imp_fe_identifier(country_id):
                return "_".join([self.meta_data.imp_var_name, self.meta_data.year_var_name,
                                 'fe', (country_id + self._year)])
        else:
            raise ValueError("Fixed Effect Specification must feature {} or {}".format([self.meta_data.imp_var_name],
                                                                                       [self.meta_data.imp_var_name,
                                                                                        self.meta_data.year_var_name]))

        # Exporter FEs
        if [self.meta_data.exp_var_name] in fe_specification:
            def exp_fe_identifier(country_id):
                return "_".join([self.meta_data.exp_var_name,
                                 'fe', (country_id)])
        elif [self.meta_data.imp_var_name, self.meta_data.year_var_name] in fe_specification:
            def exp_fe_identifier(country_id):
                return "_".join([self.meta_data.exp_var_name, self.meta_data.year_var_name,
                                 'fe', (country_id + self._year)])
        else:
            raise ValueError(
                "Fixed Effect Specification must feature {} or {}".format([self.meta_data.exp_var_name],
                                                                          [self.meta_data.exp_var_name,
                                                                           self.meta_data.year_var_name]))

        for row in range(country_data.shape[0]):
            country_id = country_data.loc[row, self.meta_data.imp_var_name]

            # Get fixed effects if estimated
            try:
                bsln_imp_fe = self._estimation_results.params[imp_fe_identifier(country_id)]
            except:
                bsln_imp_fe = 'no estimate'
            try:
                bsln_exp_fe = self._estimation_results.params[exp_fe_identifier(country_id)]
            except:
                bsln_exp_fe = 'no estimate'

            # Build country
            try:
                country_ob = Country(identifier=country_id,
                                     year=self._year,
                                     baseline_output=country_data.loc[row, self.meta_data.output_var_name],
                                     baseline_expenditure=country_data.loc[row, self.meta_data.expend_var_name],
                                     baseline_importer_fe=bsln_imp_fe,
                                     baseline_exporter_fe=bsln_exp_fe,
                                     reference_expenditure=reference_expenditure)
            except:
                raise ValueError(
                    "Missing baseline information for {}. Check that there are output and expenditure data.".format(
                        country_id))

            country_set[country_ob.identifier] = country_ob

        return country_set

    def _create_baseline_economy(self):

        # Initialize Economy
        economy = Economy(sigma=self.sigma)
        economy.initialize_baseline_total_output_expend(self.country_set)

        # Calculate output/expenditure shares for each country
        # for country_id in self.country_set.keys():
        #     self.country_set[country_id].calculate_baseline_output_expenditure_shares(economy)

        return economy

    def _create_trade_costs(self,
                            data_set: object = None):
        # generate \hat{t}^{1-\sigma}_{ij}
        obs_id = [self.meta_data.imp_var_name,
                  self.meta_data.exp_var_name,
                  self.meta_data.year_var_name]
        weighted_costs = data_set[obs_id + self.cost_variables].copy()
        weighted_list = []
        for variable in self.cost_variables:
            weighted_costs[('cost_weighted_' + variable)] = self.cost_coeffs[variable] * \
                                                            weighted_costs[[variable]]
            weighted_list = weighted_list + [('cost_weighted_' + variable)]

        weighted_costs['trade_cost'] = np.exp(weighted_costs[weighted_list].sum(axis=1))

        return weighted_costs[obs_id + ['trade_cost']]

    def _create_cost_output_expend_params(self, trade_costs):
        # Prepare cost/expenditure and cost/output parameters
        # cost_output_share: t_{ij}^{1-\sigma} * Y_i / Y
        # cost_expend_share: t_{ij}^{1-\sigma} * E_j / Y
        cost_params = trade_costs.copy()
        cost_params['cost_output_share'] = -9999
        cost_params['cost_expend_share'] = -9999
        # Build actual values
        for row in cost_params.index:
            importer_key = cost_params.loc[row, self.meta_data.imp_var_name]
            exporter_key = cost_params.loc[row, self.meta_data.exp_var_name]
            cost_params.loc[row, 'cost_output_share'] = cost_params.loc[row, 'trade_cost'] \
                                                        * self.country_set[exporter_key].baseline_output_share
            cost_params.loc[row, 'cost_expend_share'] = cost_params.loc[row, 'trade_cost'] \
                                                        * self.country_set[importer_key].baseline_expenditure_share
        cost_params.sort_values([self.meta_data.exp_var_name, self.meta_data.imp_var_name], inplace=True)
        # Reshape to a Matrix with exporters as rows, importers as columns
        cost_exp_shr = cost_params.pivot(index=self.meta_data.exp_var_name,
                                         columns=self.meta_data.imp_var_name,
                                         values='cost_expend_share')
        cost_out_shr = cost_params.pivot(index=self.meta_data.exp_var_name,
                                         columns=self.meta_data.imp_var_name,
                                         values='cost_output_share')
        # Convert to numpy array to improve solver speed
        built_params = dict()
        built_params['cost_exp_shr'] = cost_exp_shr.values
        built_params['cost_out_shr'] = cost_out_shr.values
        return built_params

    def _calculate_multilateral_resistance(self,
                                           trade_costs: DataFrame,
                                           version: str,
                                           test=False,
                                           inputs_only=False):
        # Step 1: Build parameters for solver
        mr_params = dict()
        country_list = list(self.country_set.keys())
        mr_params['number_of_countries'] = len(country_list)
        mr_params['omr_rescale'] = self._omr_rescale
        mr_params['imr_rescale'] = self._imr_rescale
        # Calculate parameters reflecting trade costs, output shares, and expenditure shares
        cost_shr_params = self._create_cost_output_expend_params(trade_costs=trade_costs)
        mr_params['cost_exp_shr'] = cost_shr_params['cost_exp_shr']
        mr_params['cost_out_shr'] = cost_shr_params['cost_out_shr']

        # Step 2: Solve
        initial_values = [1] * (2 * mr_params['number_of_countries'] - 1)
        if test:
            # Option for testing and diagnosing the MR function
            test_diagnostics = dict()
            test_diagnostics['initial values'] = initial_values
            test_diagnostics['mr_params'] = mr_params
            if inputs_only:
                return test_diagnostics
            else:
                test_diagnostics['function_value'] = _multilateral_resistances(initial_values, mr_params)
                return test_diagnostics
        else:
            if not self.quiet:
                print('Solving for {} MRs...'.format(version))
            solved_mrs = root(_multilateral_resistances, initial_values, args=mr_params, method=self._mr_method,
                              tol=self._mr_tolerance,
                              options={'xtol': self._mr_tolerance, 'maxfev': self._mr_max_iter})
            if solved_mrs.message == 'The solution converged.':
                if not self.quiet:
                    print(solved_mrs.message)
            else:
                warn(solved_mrs.message)
            self.solver_diagnostics[version + "_MRs"] = solved_mrs

            # Step 3: Pack up results
            country_list.sort()
            imrs = solved_mrs.x[0:len(country_list) - 1] * mr_params['imr_rescale']
            imrs = np.append(imrs, 1)
            omrs = solved_mrs.x[len(country_list) - 1:] * mr_params['omr_rescale']
            mrs = pd.DataFrame(data={'imrs': imrs, 'omrs': omrs}, index=country_list)

            if version == 'baseline':
                for country in country_list:
                    self.country_set[country].baseline_imr = mrs.loc[country, 'imrs']  # 1 / P^{1-sigma}
                    self.country_set[country].baseline_omr = mrs.loc[country, 'omrs']  # 1 / Pi^{1-sigma}

            if version == 'conditional':
                for country in country_list:
                    self.country_set[country].conditional_imr = mrs.loc[country, 'imrs']  # 1 / P^{1-sigma}
                    self.country_set[country].conditional_omr = mrs.loc[country, 'omrs']  # 1 / Pi^{1-sigma}

    def _calculate_GEPPML_multilateral_resistance(self, version):
        '''
        Construct fixed effects according to Yotov, Piermartini, Monteiro, and Larch (2016),
        "An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model (Online Revised Version)
        Follows GEPPML approach and MRLs are based on equations (2-38) and (2-39)
        '''
        country_list = list(self.country_set.keys())

        # ToDo: Try recalculating the output expenditure measures

        def _GEPPML_OMR(Y_i, E_R, exp_fe_i):
            '''
            Calculate outward multilateral resistance based on equation (2-38): Pi_i^(1-sigma)
                Y_i: Output for exporter i
                E_r: Expenditure for the reference country
                exp_fe_i: Estimated exporter fixed effect for country i
            '''
            return (Y_i * E_R) / math.exp(exp_fe_i)

        def _GEPPML_IMR(E_j, E_R, imp_fe_j):
            '''
            Calculate inward multilateral resistance based on equation (2-39): P_j^(1-sigma)
                E_j: Expenditure for importer j
                E_R: Expenditure for the reference country
                imp_fe_j: Estimated importer fixed effect for country j
            '''
            return E_j / (math.exp(imp_fe_j) * E_R)

        if version == 'baseline':
            reference_expnd = self.country_set[self._reference_importer].baseline_expenditure
            for country in country_list:
                country_obj = self.country_set[country]

                # Set values for reference importer
                if country == self._reference_importer:

                    # Check that the estimation produced appropriate fixed effect estimates
                    if country_obj.baseline_importer_fe != 'no estimate':
                        warn("There exists an importer fixed effect estimate for the reference country."
                             " Check that the fixed effect specification correctly omits the reference country")
                    if country_obj.baseline_exporter_fe == 'no estimate':
                        raise ValueError("No exporter fixed effect estimate for {}".format(country))
                    # P_R = 1 by construction
                    imr = 1
                    # Pi_i^(1-sigma)
                    omr = _GEPPML_OMR(Y_i=country_obj.baseline_output, E_R=reference_expnd,
                                      exp_fe_i=country_obj.baseline_exporter_fe)
                    self.country_set[country].baseline_imr = 1 / imr  # 1 / P^{1-sigma}
                    self.country_set[country].baseline_omr = 1 / omr  # 1 / Pi^{1-sigma}

                # Set values for every other country
                else:
                    # Check that there exist fixed effect estimates
                    if country_obj.baseline_importer_fe == 'no estimate':
                        raise ValueError("No importer fixed effect estimate for {}".format(country))
                    if country_obj.baseline_exporter_fe == 'no estimate':
                        raise ValueError("No exporter fixed effect estimate for {}".format(country))
                    # Pi_i^(1-sigma)
                    omr = _GEPPML_OMR(Y_i=country_obj.baseline_output, E_R=reference_expnd,
                                      exp_fe_i=country_obj.baseline_exporter_fe)
                    # P_j^(1-sigma)
                    imr = _GEPPML_IMR(E_j=country_obj.baseline_expenditure, E_R=reference_expnd,
                                      imp_fe_j=country_obj.baseline_exporter_fe)

                    self.country_set[country].baseline_imr = 1 / imr  # 1 / P^{1-sigma}
                    self.country_set[country].baseline_omr = 1 / omr  # 1 / Pi^{1-sigma}

        if version == 'conditional':
            # Step 1: Re-estimate model
            baseline_specification = self._estimation_model.specification
            counter_factual_data = self.experiment_data.copy()
            counter_factual_data = counter_factual_data.merge(self.experiment_trade_costs, how='inner',
                                                              on=[self.meta_data.exp_var_name,
                                                                  self.meta_data.imp_var_name,
                                                                  self.meta_data.year_var_name])
            counter_factual_data['adjusted_trade'] = counter_factual_data[baseline_specification.lhs_var] / \
                                                     counter_factual_data['trade_cost']
            # ToDo: Perform estimation - May not work with GME.estimate() due to lack of rhs vars. If so, need to figure out how to deal with dropped FE in estimation stage.

            # ToDo: Step 2: Calculate shit.


    def _calculate_baseline_factory_gate_params(self):
        for country in self.country_set.keys():
            self.country_set[country].factory_gate_price_param = self.country_set[country].baseline_output_share \
                                                                 * self.country_set[country].baseline_omr

    def define_experiment(self, experiment_data: DataFrame = None):
        self.experiment_data = experiment_data
        self.experiment_trade_costs = self._create_trade_costs(self.experiment_data)
        cost_change = self.baseline_trade_costs.merge(right=self.experiment_trade_costs, how='outer',
                                                      on=[self.meta_data.imp_var_name,
                                                          self.meta_data.exp_var_name,
                                                          self.meta_data.year_var_name])
        cost_change.rename(columns={'trade_cost_x': 'baseline_trade_cost', 'trade_cost_y': 'experiment_trade_cost'},
                           inplace=True)
        self.cost_shock = cost_change.loc[cost_change['baseline_trade_cost'] != cost_change['experiment_trade_cost']]

    def simulate(self, ge_method: str = 'hybr', ge_tolerance: float = 1e-8, ge_max_iter: int = 1000):
        self._ge_method = ge_method
        self._ge_tolerance = ge_tolerance
        self._ge_max_iter = ge_max_iter
        # Step 1: Simulate conditional GE
        if self.approach == 'GEPPML':
            self._calculate_GEPPML_multilateral_resistance(version='conditional')
        else:
            self._calculate_multilateral_resistance(trade_costs=self.experiment_trade_costs, version='conditional')
        # Step 2: Simulate full GE
        self._calculate_full_ge()
        # Step 3: Generate post-simulation results
        [self.country_set[country].construct_terms_of_trade() for country in self.country_set.keys()]
        self._construct_experiment_output_expend()
        self._construct_experiment_trade()
        self._compile_results()

    def _calculate_full_ge(self):
        ge_params = dict()
        country_list = list(self.country_set.keys())
        country_list.sort()
        ge_params['number_of_countries'] = len(country_list)
        ge_params['omr_rescale'] = self._omr_rescale
        ge_params['imr_rescale'] = self._imr_rescale
        ge_params['sigma'] = self.sigma
        # Calculate parameters reflecting trade costs, output shares, and expenditure shares
        cost_shr_params = self._create_cost_output_expend_params(trade_costs=self.experiment_trade_costs)
        ge_params['cost_exp_shr'] = cost_shr_params['cost_exp_shr']
        ge_params['cost_out_shr'] = cost_shr_params['cost_out_shr']

        init_imr = list()
        init_omr = list()
        output_share = list()
        factory_gate_params = list()
        for country in country_list:
            init_imr.append(self.country_set[country].conditional_imr)
            init_omr.append(self.country_set[country].conditional_omr)
            output_share.append(self.country_set[country].baseline_output_share)
            factory_gate_params.append(self.country_set[country].factory_gate_price_param)

        init_imr = [mr / ge_params['imr_rescale'] for mr in init_imr]
        init_omr = [mr / ge_params['omr_rescale'] for mr in init_omr]

        ge_params['output_shr'] = output_share
        ge_params['factory_gate_param'] = factory_gate_params

        init_price = [1] * len(country_list)
        initial_values = init_imr[0:len(country_list) - 1] + init_omr + init_price
        initial_values = np.array(initial_values)
        if not self.quiet:
            print('Solving full GE model...')
        full_ge_results = root(_full_ge, initial_values, args=ge_params, method=self._ge_method, tol=self._ge_tolerance,
                               options={'xtol': self._ge_tolerance, 'maxfev': self._ge_max_iter})
        if full_ge_results.message == 'The solution converged.':
            if not self.quiet:
                print(full_ge_results.message)
        else:
            warn(full_ge_results.message)
        self.solver_diagnostics['full_GE'] = full_ge_results

        imrs = full_ge_results.x[0:len(country_list) - 1] * ge_params['imr_rescale']
        imrs = np.append(imrs, 1)
        omrs = full_ge_results.x[len(country_list) - 1:2 * len(country_list) - 1] * ge_params['omr_rescale']
        prices = full_ge_results.x[2 * len(country_list) - 1:]
        factory_gate_prices = pd.DataFrame({'exporter': country_list, 'experiment_factory_price': prices})
        self.factory_gate_prices = factory_gate_prices.set_index('exporter')
        for i, country in enumerate(country_list):
            self.country_set[country].experiment_imr = imrs[i]
            self.country_set[country].experiment_omr = omrs[i]
            self.country_set[country].experiment_factory_price = prices[i]
            self.country_set[country].factory_price_change = 100 * (prices[i] - 1)


    def _construct_experiment_output_expend(self):
        total_output = 0

        results_table = pd.DataFrame(columns=['country', 'baseline_output', 'experiment_output',
                                              'output_percent_change', 'baseline_expenditure',
                                              'experiment_expenditure', 'expenditure_percent_change'])
        # The first time looping through gets individual and total output and expenditure
        for country in self.country_set.keys():
            country_obj = self.country_set[country]
            country_obj.experiment_output = country_obj.experiment_factory_price * country_obj.baseline_output
            country_obj.experiment_expenditure = country_obj.experiment_factory_price * country_obj.baseline_expenditure
            total_output += country_obj.experiment_output

        # The second time looping through gets things that are dependent on total output/expenditure
        for country in self.country_set.keys():
            country_obj = self.country_set[country]
            country_obj.experiment_output_share = country_obj.experiment_output / total_output
            country_obj.output_change = 100 * (country_obj.experiment_output - country_obj.baseline_output) \
                                        / country_obj.baseline_output
            country_obj.expenditure_change = 100 * (country_obj.experiment_expenditure -
                                                    country_obj.baseline_expenditure) / country_obj.baseline_expenditure
            results_table = results_table.append({'country': country,
                                                  'baseline_output': country_obj.baseline_output,
                                                  'experiment_output': country_obj.experiment_output,
                                                  'output_percent_change': country_obj.output_change,
                                                  'baseline_expenditure': country_obj.baseline_expenditure,
                                                  'experiment_expenditure': country_obj.experiment_expenditure,
                                                  'expenditure_percent_change': country_obj.expenditure_change},
                                                 ignore_index=True)

        self._economy.experiment_total_output = total_output
        self._economy.output_change = 100 * (total_output - self._economy.baseline_total_output) \
                                      / self._economy.baseline_total_output
        self.outputs_expenditures = results_table.set_index('country')

    def _construct_experiment_trade(self):
        importer_col = self.meta_data.imp_var_name
        exporter_col = self.meta_data.exp_var_name
        year_col = self.meta_data.year_var_name
        trade_value_col = self.meta_data.trade_var_name

        countries = self.country_set.keys()
        trade_data = self.baseline_data[[exporter_col, importer_col, year_col, trade_value_col]].copy()
        trade_data = trade_data.loc[trade_data[year_col] == self._year, [exporter_col, importer_col, trade_value_col]]

        trade_data.rename(columns={trade_value_col: 'baseline_trade'}, inplace=True)

        trade_data['gravity'] = -9999

        for row in trade_data.index:
            importer = trade_data.loc[row, importer_col]
            exporter = trade_data.loc[row, exporter_col]
            imr = self.country_set[importer].experiment_imr
            omr = self.country_set[exporter].experiment_omr
            expend = self.country_set[importer].experiment_expenditure
            output_share = self.country_set[exporter].experiment_output_share

            gravity = (expend * output_share) / (imr * omr)

            trade_data.loc[row, 'gravity'] = gravity

            bsln_imr = self.country_set[importer].baseline_imr
            bsln_omr = self.country_set[exporter].baseline_omr
            bsln_expend = self.country_set[importer].baseline_expenditure
            bsln_output_share = self.country_set[exporter].baseline_output_share

            bsln_gravity = (bsln_expend * bsln_output_share) / (bsln_imr * bsln_omr)

            trade_data.loc[row, 'bsln_gravity'] = bsln_gravity

        trade_data = trade_data.merge(self.baseline_trade_costs, how='left', on=[importer_col, exporter_col])
        trade_data.rename(columns={'trade_cost': 'baseline_trade_cost'}, inplace=True)
        trade_data['baseline_modeled_trade'] = trade_data['baseline_trade_cost'] * trade_data['bsln_gravity']

        trade_data = trade_data.merge(self.experiment_trade_costs, how='left', on=[importer_col, exporter_col])

        trade_data['experiment_trade'] = trade_data['trade_cost'] * trade_data['gravity']

        trade_data['percent_change'] = 100 * (trade_data['experiment_trade'] - trade_data['baseline_modeled_trade']) \
                                       / trade_data['baseline_modeled_trade']

        self.bilateral_trade_results = trade_data[[exporter_col, importer_col, 'baseline_modeled_trade',
                                                   'experiment_trade', 'percent_change']]

        agg_imports = self.bilateral_trade_results.copy()
        agg_imports = agg_imports[[importer_col, 'baseline_modeled_trade', 'experiment_trade']]
        agg_imports = agg_imports.groupby([importer_col]).agg('sum')
        agg_imports.rename(columns={'baseline_modeled_trade': 'baseline_imports',
                                    'experiment_trade': 'experiment_imports'}, inplace=True)
        agg_imports['import_percent_change'] = 100 \
                                               * (agg_imports['experiment_imports'] - agg_imports['baseline_imports']) \
                                               / agg_imports['baseline_imports']

        agg_exports = self.bilateral_trade_results.copy()
        agg_exports = agg_exports[[exporter_col, 'baseline_modeled_trade', 'experiment_trade']]
        agg_exports = agg_exports.groupby([exporter_col]).agg('sum')
        agg_exports.rename(columns={'baseline_modeled_trade': 'baseline_exports',
                                    'experiment_trade': 'experiment_exports'}, inplace=True)
        agg_exports['export_percent_change'] = 100 \
                                               * (agg_exports['experiment_exports'] - agg_exports['baseline_exports']) \
                                               / agg_exports['baseline_exports']

        agg_trade = pd.concat([agg_exports, agg_imports], axis=1).reset_index()
        agg_trade.rename(columns={'index': 'country'}, inplace=True)
        for row in agg_trade.index:
            country = agg_trade.loc[row, 'country']
            country_obj = self.country_set[country]
            country_obj.baseline_imports = agg_trade.loc[row, 'baseline_imports']
            country_obj.baseline_exports = agg_trade.loc[row, 'baseline_exports']
            country_obj.experiment_imports = agg_trade.loc[row, 'experiment_imports']
            country_obj.experiment_exports = agg_trade.loc[row, 'experiment_exports']
            country_obj.imports_change = agg_trade.loc[row, 'import_percent_change']
            country_obj.exports_change = agg_trade.loc[row, 'export_percent_change']

        self.aggregate_trade_results = agg_trade.set_index('country')

    def _compile_results(self):
        results = list()
        mr_results = list()
        for country in self.country_set.keys():
            results.append(self.country_set[country].get_results())
            mr_results.append(self.country_set[country].get_mr_results())
        country_results = pd.concat(results, axis=0)
        self.country_results = country_results.set_index('country')
        country_mr_results = pd.concat(mr_results, axis=0)
        self.country_mr_terms = country_mr_results.set_index('country')


    def trade_share(self, importers: List[str], exporters: List[str]):
        bilat_trade = self.bilateral_trade_results
        columns = ['baseline_modeled_trade', 'experiment_trade']
        imports = bilat_trade.loc[bilat_trade['importer'].isin(importers), :].copy()
        exports = bilat_trade.loc[bilat_trade['exporter'].isin(exporters), :].copy()

        total_imports = imports[columns].agg('sum')
        total_exports = exports[columns].agg('sum')

        selected_imports = imports.loc[imports['exporter'].isin(exporters), columns].copy().agg('sum')
        selected_exports = exports.loc[exports['importer'].isin(importers), columns].copy().agg('sum')

        import_data = 100 * selected_imports / total_imports
        export_data = 100 * selected_exports / total_exports

        import_data['description'] = 'Percent of ' + ", ".join(importers) + ' imports from ' + ", ".join(exporters)
        export_data['description'] = 'Percent of ' + ", ".join(exporters) + ' exports to ' + ", ".join(importers)

        both = pd.concat([import_data, export_data], axis=1).T
        both = both[['description'] + columns]
        both['change (percentage point)'] = (both['experiment_trade'] - both['baseline_modeled_trade'])
        both['change (percent)'] = 100 * (both['experiment_trade'] - both['baseline_modeled_trade']) / \
                                   both['baseline_modeled_trade']

        return both

    # ---
    # Diagnostic Tools
    # ---
    def test_baseline_mr_function(self, inputs_only=False):
        test_diagnostics = self._calculate_multilateral_resistance(trade_costs=self.baseline_trade_costs,
                                                                   version='baseline', test=True,
                                                                   inputs_only=inputs_only)
        return test_diagnostics





def _multilateral_resistances(x, mr_params):
    # x should be length (n-1) + n (i.e. no IMR for the representative country)
    num_countries = mr_params['number_of_countries']
    cost_exp_shr = mr_params['cost_exp_shr']
    cost_out_shr = mr_params['cost_out_shr']
    imr_rescale = mr_params['imr_rescale']
    omr_rescale = mr_params['omr_rescale']

    # x_imr is IMR, N-1 elements
    x_imr = x[0:(num_countries - 1)]
    x_imr = [x * imr_rescale for x in x_imr]
    # x2 is OMR, N elements; multiplication by 1000 is done to correct the scaling problem
    x_omr = x[(num_countries - 1):]
    x_omr = [x * omr_rescale for x in x_omr]

    # Calculate IMR by looping over importers (j) excluding the reference country
    out = [1 - multiply(x_imr[j], sum(multiply(cost_out_shr[:, j], x_omr))) for j in range(num_countries - 1)]
    # Set last IMR for reference country equal to 1 for use in OMR calculation
    x_imr.append(1)
    # Calculate OMR by looping through exporters (i)
    out.extend([1 - multiply(x_omr[i], sum(multiply(cost_exp_shr[i, :], x_imr))) for i in range(num_countries)])
    return out


def _full_ge(x, ge_params):
    # Unpack Parameters
    num_countries = ge_params['number_of_countries']
    sigma_power = 1 - ge_params['sigma']
    out_share = ge_params['output_shr']
    cost_exp_shr = ge_params['cost_exp_shr']
    cost_out_shr = ge_params['cost_out_shr']
    beta = ge_params['factory_gate_param']
    omr_rescale = ge_params['omr_rescale']
    imr_rescale = ge_params['imr_rescale']

    # Break apart initial values vector
    # x_imr is IMR, N-1 elements
    x_imr = x[0:(num_countries - 1)] * imr_rescale
    # x2 is OMR, N elements; multiplication by 1000 is done to correct the scaling problem
    x_omr = x[(num_countries - 1):(2 * num_countries - 1)] * omr_rescale
    x_price = x[(2 * num_countries - 1):]

    # Calculate IMR by looping over importers (j) excluding the reference country
    out = [1 - multiply(x_imr[j], sum(multiply(cost_out_shr[:, j], x_omr))) for j in range(num_countries - 1)]
    # Set last IMR for reference country equal to 1 for use in OMR calculation
    x_imr = np.append(x_imr, 1)
    # Calculate OMR by looping through exporters (i)
    out.extend([1 - multiply(x_omr[i], sum(multiply(cost_exp_shr[i, :], x_imr))) for i in range(num_countries)])
    # Calculate factory gate prices by looping through countries (exporters)
    out.extend([1 - ((out_share[i] * x_omr[i]) / (beta[i] * x_price[i] ** sigma_power)) for i in range(num_countries)])
    return out


class Economy(object):
    def __init__(self,
                 sigma: float = 4):
        self.sigma = sigma
        self.experiment_total_output = None
        self.experiment_total_expenditure = None
        self.baseline_total_output = None
        self.baseline_total_expenditure = None
        self.output_change = None

    def initialize_baseline_total_output_expend(self, country_set):
        # Create baseline values for total output and expenditure
        total_output = 0
        total_expenditure = 0
        for country in country_set.keys():
            total_output += country_set[country].baseline_output
            total_expenditure += country_set[country].baseline_expenditure
        self.baseline_total_output = total_output
        self.baseline_total_expenditure = total_expenditure

    def __repr__(self):
        return "Economy \n" \
               "Sigma: {0} \n" \
               "Baseline Total Output: {1} \n" \
               "Baseline Total Expenditure: {2} \n" \
               "Experiment Total Output: {3} \n" \
               "Output Change (%): {4} \n" \
            .format(self.sigma,
                    self.baseline_total_output,
                    self.baseline_total_expenditure,
                    self.experiment_total_output,
                    self.output_change)


class Country(object):
    # This may need to be a country/year thing at some point
    def __init__(self,

                 identifier: str = None,
                 year: str = None,
                 baseline_output: float = None,
                 baseline_expenditure: float = None,
                 baseline_importer_fe: float = None,
                 baseline_exporter_fe: float = None,
                 reference_expenditure: float = None):
        self.identifier = identifier
        self.year = year
        self.baseline_output = baseline_output
        self.baseline_expenditure = baseline_expenditure
        self.baseline_importer_fe = baseline_importer_fe
        self.baseline_exporter_fe = baseline_exporter_fe
        self._reference_expenditure = reference_expenditure
        self.baseline_output_share = None
        self.baseline_expenditure_share = None
        self.baseline_export_costs = None
        self.baseline_import_costs = None
        self.baseline_imr = None  # \hat{P}^{1-sigma}_{j,t}
        self.baseline_omr = None  # \hat{\Pi}^{1-\sigma}_{i,t}
        self.factory_gate_price_param = None  # \beta_i
        self.baseline_factory_price = 1
        self.conditional_imr = None
        self.conditional_omr = None
        self.experiment_imr = None
        self.experiment_omr = None
        self.experiment_factory_price = None
        self.experiment_output = None
        self.experiment_expenditure = None
        self.baseline_terms_of_trade = None
        self.experiment_terms_of_trade = None
        self.output_change = None
        self.expenditure_change = None
        self.terms_of_trade_change = None
        self.factory_price_change = None
        self.baseline_imports = None
        self.baseline_exports = None
        self.experiment_imports = None
        self.experiment_exports = None
        self.imports_change = None
        self.exports_change = None


    def calculate_baseline_output_expenditure_shares(self, economy):
        self.baseline_expenditure_share = self.baseline_expenditure / economy.baseline_total_expenditure
        self.baseline_output_share = self.baseline_output / economy.baseline_total_output

    def construct_terms_of_trade(self):
        for value in [self.baseline_factory_price, self.baseline_imr,
                      self.experiment_factory_price, self.experiment_imr]:
            if value is None:
                warn("Not all necessary values for terms of trade have been calculated.")
        self.baseline_terms_of_trade = self.baseline_factory_price / self.baseline_imr
        self.experiment_terms_of_trade = self.experiment_factory_price / self.experiment_imr
        self.terms_of_trade_change = 100 * (self.experiment_terms_of_trade - self.baseline_terms_of_trade) \
                                     / self.baseline_terms_of_trade

    def get_results(self):
        row = pd.DataFrame(data={'country': [self.identifier],
                                 'factory_price_change': [self.factory_price_change],
                                 'output_change': [self.output_change],
                                 'expenditure_change': [self.expenditure_change],
                                 'export_change': [self.exports_change],
                                 'import_change': [self.imports_change],
                                 'terms_of_trade_change': [self.terms_of_trade_change]})
        return row

    def get_mr_results(self):
        row = pd.DataFrame(data={'country': [self.identifier],
                                 'baseline_imr': [self.baseline_imr],
                                 'conditional_expiriment_imr': [self.conditional_imr],
                                 'experiment_imr': [self.experiment_imr],
                                 'baseline_omr': [self.baseline_omr],
                                 'conditional_expiriment_omr': [self.conditional_omr],
                                 'experiment_omr': [self.experiment_omr]})
        return row

    def __repr__(self):
        return "Country: {0} \n" \
               "Year: {1} \n" \
               "Baseline Output: {2} \n" \
               "Baseline Expenditure: {3} \n" \
               "Baseline IMR: {4} \n" \
               "Baseline OMR: {5} \n" \
               "Experiment IMR: {6} \n" \
               "Experiment OMR: {7} \n" \
               "Experiment Factory Price: {8} \n" \
               "Output Change (%): {9} \n" \
               "Expenditure Change (%): {10} \n" \
               "Terms of Trade Change (%): {11} \n" \
            .format(self.identifier,
                    self.year,
                    self.baseline_output,
                    self.baseline_expenditure,
                    self.baseline_imr,
                    self.baseline_omr,
                    self.experiment_imr,
                    self.experiment_omr,
                    self.experiment_factory_price,
                    self.output_change,
                    self.expenditure_change,
                    self.terms_of_trade_change)


class _GEMetaData(object):
    '''
    Modified gme _MetaData object that includes output and expenditure column names
    '''
    def __init__(self, gme_meta_data, expend_var_name, output_var_name):
        self.imp_var_name = gme_meta_data.imp_var_name
        self.exp_var_name = gme_meta_data.exp_var_name
        self.year_var_name = gme_meta_data.year_var_name
        self.trade_var_name = gme_meta_data.trade_var_name
        self.sector_var_name = gme_meta_data.sector_var_name
        self.expend_var_name = expend_var_name
        self.output_var_name = output_var_name
