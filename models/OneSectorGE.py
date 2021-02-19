__Author__ = "Peter Herman"
__Project__ = "Gravity Code"
__Created__ = "08/15/2018"
__all__ = ['OneSectorGE', 'ParameterValues', 'Country', 'Economy']
__Description__ = """A single sector or aggregate full GE model based on Larch and Yotov, 'General Equilibrium Trade
                  Policy Analysis with Structural Gravity," 2016. (WTO Working Paper ERSD-2016-08)"""

# ToDo: Finish OneSectorGE attributes list, add attributes for Country and Economy classes.

from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from gme.estimate.EstimationModel import EstimationModel
from scipy.optimize import root
from numpy import multiply, median
from warnings import warn
import math as math
from models.ResultsLabels import ResultsLabels


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
                 reference_importer: str,
                 expend_var_name: str = 'expenditure',
                 output_var_name: str = 'output',
                 sigma: float = 5,
                 results_key: str = 'all',
                 cost_variables: List[str] = None,
                 parameter_values = None,
                 #approach: str = None,
                 quiet:bool = False):
        '''
        Define a general equilibrium (GE) gravity model.
        Args:
            estimation_model (gme.EstimationModel): A GME Estimation model
            year (str): The year to be used for the model. Works best if estimation_model year column has been cast as
                string too.
            reference_importer (str): Identifier for the country to be used as the reference importer (inward
                multilateral resistance normalized to 1 and other multilateral resistances solved relative to it).
            expend_var_name (str): Column name of variable containing expenditure data in estimation_model.
            output_var_name (str): Column name of variable containing output data in estimation_model.
            sigma (float): Elasticity of substitution.
            results_key (str): (optional) If using parameter estimates from estimation_model, this is the key (i.e.
                sector) corresponding to the estimates to be used. For single sector estimations (sector_by_sector =
                False in GME model), this key is 'all', which is the default.
            cost_variables (List[str]): (optional) A list of variables to use to compute bilateral trade costs. By
                default, all included non-fixed effect variables are used.
            parameter_values (ParameterValues): (optional) A set of parameter values or estimates to use for constructing
                trade costs. Should be of type gegravity.ParameterValues, statsmodels.GLMResultsWrapper, or
                gme.SlimResults. If no values are provided, the estimates in the EstimationModel are used.
            quiet (bool): (optional) If True, suppresses all console feedback from model during simulation. Default is False.

        Attributes:
            aggregate_trade_results (pandas.DataFrame): Country-level, aggregate results. See models.ResultsLabels for
                column details.
            baseline_trade_costs (pandas.DataFrame): The constructed baseline trade costs for each bilateral pair
                (t_{ij}^{1-sigma}). Calculated as exp{sum_k (B^k*x^k_ij)} for all cost variables x^k and estimate
                values B.
            bilateral_trade_results (pandas.DataFrame): Bilateral trade results. See models.ResultsLabels for
                column details.
            country_mr_terms (pandas.DataFrame): Baseline and counterfactual inward and outward multilateral resistance
                estimates. See models.ResultsLabels for column details.
            country_results (pandas.DataFrame): A collection of the main country-level simulation results. See
                models.ResultsLabels for column details.
            country_set (dict[Country]): A dictionary containing a Country object for each country in the model, keyed
                by their respective identifiers.
            cost_shock (pandas.DataFrame): The baseline and experiment trade costs combined.
            economy (Economy): The model's Economy object.
            experiment_data (pandas.DataFrame): The counterfactual experiment data.
            experiment_trade_costs (pandas.DataFrame): The constructed experiment trade costs for each bilateral pair
                (t_{ij}^{1-sigma}). Calculated as exp{sum_k (B^k*x^k_ij)} for all cost variables x^k and estimate
                values B
            factory_gate_prices (pandas.DataFrame): Counterfactual prices (baseline prices are all normalized to 1).
            outputs_expenditures (pandas.DataFrame): Baseline and counterfactual expenditure and output values. See
                models.ResultsLabels for column details.
            sigma (int): The elasticity of substitution parameter value
            solver_diagnostics (dict): A dictionary of solver diagnostics for the three solution routines: baseline
                multilateral resistances, conditional multilateral resistances (partial equilibrium counterfactual
                effects) and the full GE model. Each element contains a dictionary of various diagnostic info from
                scipy.optimize.root.

        Examples:

        '''
        # '''
        #
        # # ---
        # # Methods
        # # ---
        #
        # ##
        # # Model Construction and Simulation
        # ##
        #
        # build_baseline(omr_rescale: float = 1, imr_rescale: float = 1,  mr_method: str = 'hybr',
        #                mr_max_iter: int = 1400, mr_tolerance: float = 1e-8)
        #     Solve the baseline model. This primarily solves for the baseline Multilateral Resistance (MR) terms.
        #
        # define_experiment(experiment_data: DataFrame = None)
        #     Specify the counterfactual data to use for experiment.
        #
        # simulate(ge_method: str = 'hybr', ge_tolerance: float = 1e-8, ge_max_iter: int = 1000)
        #     Simulate the counterfactual scenario.
        #
        # ##
        # # Post-simulation Analysis
        # ##
        #
        # trade_share(importers: List[str], exporters: List[str])
        #     Calculate baseline and experiment import and export shares (in percentages) between user-supplied countries.
        #
        # export_results(directory:str = None, name:str = '')
        #     Export results to csv files. Three files are stored containing (1) country-level results, (2) bilateral
        #     results, and (3) solver diagnostics.
        #
        # calculate_levels(how: str = 'country'):
        #     Calculate changes in the level (value) of trade using baseline trade values and simulation outcomes. Results
        #     can be calculated at either the country level or bilateral level.
        #
        # ##
        # # Trouble-shooting
        # ##
        #
        # test_baseline_mr_function(inputs_only:bool=False)
        #     Test whether the multilateral resistance system of equations can be computed from baseline data. Helpful for
        #     debugging initial data problems. Note that the returned function values reflect those generated by the
        #     initial values and do not reflect a solution to the system.
        #
        # check_omr_rescale(omr_rescale_range:int = 10, mr_method: str = 'hybr', mr_max_iter: int = 1400,
        #                  mr_tolerance: float = 1e-8, countries:List[str] = [])
        #     Analyze different Outward Multilarteral Resistance (OMR) term rescale factors. This method can help identify
        #     feasible values to use for the omr_rescale argument in OneSectorGE.build_baseline().
        # '''
        if not isinstance(year, str):
            raise TypeError('year should be a string')

        # Check reference country (does not currently work)
        # try:
        #     omitted_fe = estimation_model.results_dict[results_key].params[('importer_fe_'+reference_importer)]
        #     raise ValueError('reference_importer should be the excluded importer fixed effect')
        # except:
        #     print('reference_importer OK')

        self.meta_data = _GEMetaData(estimation_model.estimation_data._meta_data, expend_var_name, output_var_name)
        self._estimation_model = estimation_model
        if parameter_values is None:
            self._estimation_results = self._estimation_model.results_dict[results_key]
        else:
            self._estimation_results = None
        self._year = str(year)
        self.sigma = sigma
        self._reference_importer = reference_importer
        self._omr_rescale = None
        self._imr_rescale = None
        self._mr_max_iter = None
        self._mr_tolerance = None
        self._mr_method = None
        self._ge_method = None
        self._ge_tolerance = None
        self._ge_max_iter = None
        self.country_set = None
        self.economy = None
        self.baseline_trade_costs = None # t_{ij}^{1-sigma}
        self.experiment_trade_costs = None # t_{ij}^{1-sigma}
        self.cost_shock = None
        self.experiment_data = None
        self.approach = None # Disabled until GEPPML is completed
        self.quiet = quiet

        # Results fields
        self.bilateral_trade_results = None
        self.aggregate_trade_results = None
        self.solver_diagnostics = dict()
        self.factory_gate_prices = None
        self.outputs_expenditures = None
        self.country_results = None
        self.country_mr_terms = None

        # Status checks
        self._baseline_built = False
        self._experiment_defined = False


        # ---
        # Check inputs
        # ---
        if self.meta_data.trade_var_name is None:
            raise ValueError('\n Missing Input: Please insure trade_var_name is set in EstimationData object.')

        if cost_variables is None:
            self.cost_variables = self._estimation_model.specification.rhs_var
        else:
            self.cost_variables = cost_variables

        if parameter_values is not None:
            self.cost_coeffs = parameter_values.params
        else:
            self.cost_coeffs = self._estimation_results.params[self.cost_variables]


        # prep baseline data
        _baseline_data = estimation_model.estimation_data.data_frame.copy()
        _baseline_data[self.meta_data.year_var_name] = _baseline_data[self.meta_data.year_var_name].astype(str)
        self.baseline_data = _baseline_data.loc[_baseline_data[self.meta_data.year_var_name] == self._year, :].copy()
        if self.baseline_data.shape[0] == 0:
            raise ValueError("There are no observations corresponding to the supplied 'year'")


        # Initialize a set of countries and the economy
        self.country_set = self._create_baseline_countries()
        self.economy = self._create_baseline_economy()
        # Calculate certain country values using info from the whole economy
        for country in self.country_set:
            self.country_set[country]._calculate_baseline_output_expenditure_shares(self.economy)
        # Calculate baseline trade costs
        self.baseline_trade_costs = self._create_trade_costs(self.baseline_data)




    def build_baseline(self,
                       omr_rescale: float = 1,
                       imr_rescale: float = 1,
                       mr_method: str = 'hybr',
                       mr_max_iter: int = 1400,
                       mr_tolerance: float = 1e-8):
        '''
        Solve the baseline model. This primarily solves for the baseline Multilateral Resistance (MR) terms.
        Args:
            omr_rescale (int): (optional) This value rescales the OMR values to assist in convergence. Often, OMR values
                are orders of magnitude different than IMR values, which can make convergence difficult. Scaling by a
                different order of magnitude can help. Values should be of the form 10^n. By default, this value is 1
                (10^0). However, users should be careful with this choice as results, even when convergent, may not be
                fully robust to any selection. The method OneSectorGE.check_omr_rescale() can help identify and compare
                feasible values.
            imr_rescale (int): (optional) This value rescales the IMR values to potentially aid in conversion. However,
                because the IMR for the reference importer is normalized to one, it is unlikely that there will be because
                because changing the default value, which is 1.
            mr_method (str): This parameter determines the type of non-linear solver used for solving the baseline and
                experiment MR terms. See the documentation for scipy.optimize.root for alternative methods. the default
                value is 'hybr'.
            mr_max_iter (int): (optional) This parameter sets the maximum limit on the number of iterations conducted
                by the solver used to solve for MR terms. The default value is 1400.
            mr_tolerance (float): (optional) This parameterset the convergence tolerance level for the solver used to
                solve for MR terms. The default value is 1e-8.

        Returns:
            None: Populates Attributes of model object.
        '''
        self._omr_rescale = omr_rescale
        self._imr_rescale = imr_rescale
        self._mr_max_iter = mr_max_iter
        self._mr_tolerance = mr_tolerance
        self._mr_method = mr_method

        # Solve for the baseline multilateral resistance terms
        if self.approach == 'GEPPML':
            if self._estimation_results is None:
                raise ValueError("GEPPML approach requires that the gme.EstimationModel be estimated and use importer and exporter fixed effects.")
            self._calculate_GEPPML_multilateral_resistance(version='baseline')
        else:
            self._calculate_multilateral_resistance(trade_costs=self.baseline_trade_costs, version='baseline')
        # Calculate baseline factory gate prices
        self._calculate_baseline_factory_gate_params()
        self._baseline_built = True

        # ToDo: run some checks the ensure the baseline is solved (e.g. the betas solve the factory gat price equations)

    def _create_baseline_countries(self):
        """
        Initialize set of country objects
        """
        # Requires that the baseline data has output and expenditure data

        # Make sure the year data is in string form
        self.baseline_data[self.meta_data.year_var_name] = self.baseline_data.loc[:,
                                                           self.meta_data.year_var_name].astype(str)

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
        economy._initialize_baseline_total_output_expend(self.country_set)
        return economy

    def _create_trade_costs(self,
                            data_set: object = None):
        '''
        Create bilateral trade costs. Returned values reflect \hat{t}^{1-\sigma}_{ij}, not \hat{t}. See equation (32)
        from Larch and Yotov (2016) "GENERAL EQUILIBRIUM TRADE POLICY ANALYSIS WITH STRUCTURAL GRAVITY"
        :param data_set: (DataFrame) The trade cost data to base trade costs on (either baseline or experimental)
        :return: (DataFrame) DataFrame of bilateral trade costs (t^{1-sigma})
        '''
        obs_id = [self.meta_data.imp_var_name,
                  self.meta_data.exp_var_name,
                  self.meta_data.year_var_name]
        weighted_costs = data_set[obs_id + self.cost_variables].copy()
        weighted_list = []
        # Cumulatively add each variable in the list of cost variables.
        for variable in self.cost_variables:
            weighted_costs[('cost_weighted_' + variable)] = self.cost_coeffs[variable] * \
                                                            weighted_costs[[variable]]
            weighted_list = weighted_list + [('cost_weighted_' + variable)]
        # Generate cost measure t^{1-sigma}
        weighted_costs['trade_cost'] = np.exp(weighted_costs[weighted_list].sum(axis=1))
        # Run some checks for completeness
        if weighted_costs.isna().any().any():
            warn("\n Calculated trade costs contain missing (nan) values. Check parameter values and trade cost variables in baseline or experiment data.")
        if weighted_costs.shape[0] != len(self.country_set.keys())**2:
            warn("\n Calculated trade costs are not square. Some bilateral costs are absent.")
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
        if np.isnan(cost_exp_shr.values).any():
            warn("\n 'cost_exp_share' values contain missing (nan) values. \n 1. Check that expenditure shares exist for all countries in country_set \n 2. Check that trade cost data is square and no bilateral pairs are missing.")
        if np.isnan(cost_out_shr.values).any():
            warn("\n 'cost_out_share' values contain missing (nan) values. \n 1. Check that output shares exist for all countries in country_set \n 2. Check that trade cost data is square no bilateral pairs are missing.")

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
        # cost_output_share: t_{ij}^{1-\sigma} * Y_i / Y
        # cost_expend_share: t_{ij}^{1-\sigma} * E_j / Y
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
                test_diagnostics['function_value'] = 'unsolved'
                test_diagnostics['function_value'] = _multilateral_resistances(initial_values, mr_params)
                return test_diagnostics
        # Actual Solver
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
                    self.country_set[country]._baseline_imr_ratio = mrs.loc[country, 'imrs']  # 1 / P^{1-sigma}
                    self.country_set[country]._baseline_omr_ratio = mrs.loc[country, 'omrs']  # 1 / π^{1-sigma}

            if version == 'conditional':
                for country in country_list:
                    self.country_set[country]._conditional_imr_ratio = mrs.loc[country, 'imrs']  # 1 / P^{1-sigma}
                    self.country_set[country]._conditional_omr_ratio = mrs.loc[country, 'omrs']  # 1 / π^{1-sigma}

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
            Calculate outward multilateral resistance based on equation (2-38): π_i^(1-sigma)
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
                    # π_i^(1-sigma)
                    omr = _GEPPML_OMR(Y_i=country_obj.baseline_output, E_R=reference_expnd,
                                      exp_fe_i=country_obj.baseline_exporter_fe)
                    self.country_set[country]._baseline_imr_ratio = 1 / imr  # 1 / P^{1-sigma}
                    self.country_set[country]._baseline_omr_ratio = 1 / omr  # 1 / π^{1-sigma}

                # Set values for every other country
                else:
                    # Check that there exist fixed effect estimates
                    if country_obj.baseline_importer_fe == 'no estimate':
                        raise ValueError("No importer fixed effect estimate for {}".format(country))
                    if country_obj.baseline_exporter_fe == 'no estimate':
                        raise ValueError("No exporter fixed effect estimate for {}".format(country))
                    # π_i^(1-sigma)
                    omr = _GEPPML_OMR(Y_i=country_obj.baseline_output, E_R=reference_expnd,
                                      exp_fe_i=country_obj.baseline_exporter_fe)
                    # P_j^(1-sigma)
                    imr = _GEPPML_IMR(E_j=country_obj.baseline_expenditure, E_R=reference_expnd,
                                      imp_fe_j=country_obj.baseline_exporter_fe)

                    self.country_set[country]._baseline_imr_ratio = 1 / imr  # 1 / P^{1-sigma}
                    self.country_set[country]._baseline_omr_ratio = 1 / omr  # 1 / π^{1-sigma}

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
                                                                 * self.country_set[country]._baseline_omr_ratio

    def define_experiment(self, experiment_data: DataFrame):
        '''
        Specify the counterfactual data to use for experiment.
        Args:
            experiment_data(Pandas.DataFrame): A dataframe contianing the counterfactual trade-cost data to use for the
                experiment. The best approach for creating this data is to copy the baseline data
                (OneSectorGE.baseline_data.copy()) and modify columns/rows to reflect desired counterfactual experiment.

        Returns:
            None:
                There is no return but the new information is added to model.
        '''
        if not self._baseline_built:
            raise ValueError("Baseline must be built first (i.e. ge_model.build_baseline() method")
        self.experiment_data = experiment_data
        self.experiment_trade_costs = self._create_trade_costs(self.experiment_data)
        cost_change = self.baseline_trade_costs.merge(right=self.experiment_trade_costs, how='outer',
                                                      on=[self.meta_data.imp_var_name,
                                                          self.meta_data.exp_var_name,
                                                          self.meta_data.year_var_name])
        cost_change.rename(columns={'trade_cost_x': 'baseline_trade_cost', 'trade_cost_y': 'experiment_trade_cost'},
                           inplace=True)
        self.cost_shock = cost_change.loc[cost_change['baseline_trade_cost'] != cost_change['experiment_trade_cost']]
        self._experiment_defined = True

    def simulate(self, ge_method: str = 'hybr', ge_tolerance: float = 1e-8, ge_max_iter: int = 1000):
        '''
        Simulate the counterfactual scenario
        Args:
            ge_method (str): (optional) The solver method to use for the full GE non-linear solver. See scipy.root()
                documentation for option. Default is 'hybr'.
            ge_tolerance (float): (optional) The tolerance for determining if the GE system of equations is solved.
                Default is 1e-8.
            ge_max_iter (int): (optional) The maximum number of iterations allowed for the full GE nonlinear solver.
                Default is 1000.

        Returns:
            None
                No return but populates new attributes of model.
        '''
        # '''
        # Simulate the counterfactual scenario.
        # :param ge_method: (str)
        # :param ge_tolerance: (float)
        #
        # :param ge_max_iter: (int)
        # :return: (None) No return but many fields in the model containing results are populated.
        # '''
        if not self._baseline_built:
            raise ValueError("Baseline must be built first (i.e. OneSectorGE.build_baseline() method")
        if not self._experiment_defined:
            raise ValueError("Expiriment must be defined first (i.e. OneSectorGE.define_expiriment() method")
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
        [self.country_set[country]._construct_country_measures(sigma=self.sigma) for country in self.country_set.keys()]
        self._construct_experiment_output_expend()
        self._construct_experiment_trade()
        self._compile_results()

    def _calculate_full_ge(self):
        # Solve Full GE model
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
            init_imr.append(self.country_set[country]._conditional_imr_ratio)
            init_omr.append(self.country_set[country]._conditional_omr_ratio)
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
        factory_gate_prices = pd.DataFrame({'exporter': country_list,
                                            country_results_labels['self.experiemnt_factory_price']: prices})
        self.factory_gate_prices = factory_gate_prices.set_index('exporter')
        for i, country in enumerate(country_list):
            self.country_set[country]._experiment_imr_ratio = imrs[i] # 1 / P^{1-sigma}
            self.country_set[country]._experiment_omr_ratio = omrs[i] # 1 / π^{1-sigma}
            self.country_set[country].experiment_factory_price = prices[i]
            self.country_set[country].factory_price_change = 100 * (prices[i] - 1)


    def _construct_experiment_output_expend(self):
        total_output = 0

        results_table = pd.DataFrame(columns=[country_results_labels['self.identifier'],
                                              country_results_labels['self.baseline_output'],
                                              country_results_labels['self.experiment_output'],
                                              country_results_labels['self.output_change'],
                                              country_results_labels['self.baseline_expenditure'],
                                              country_results_labels['self.experiment_expenditure'],
                                              country_results_labels['self.expenditure_change']])
        # The first time looping through gets calculates total output
        for country in self.country_set.keys():
            country_obj = self.country_set[country]
            total_output += country_obj.experiment_output

        # The second time looping through gets things that are dependent on total output/expenditure
        for country in self.country_set.keys():
            country_obj = self.country_set[country]
            country_obj.experiment_output_share = country_obj.experiment_output / total_output
            results_table = results_table.append({
                country_results_labels['self.identifier']: country,
                country_results_labels['self.baseline_output']: country_obj.baseline_output,
                country_results_labels['self.experiment_output']: country_obj.experiment_output,
                country_results_labels['self.output_change']: country_obj.output_change,
                country_results_labels['self.baseline_expenditure']: country_obj.baseline_expenditure,
                country_results_labels['self.experiment_expenditure']: country_obj.experiment_expenditure,
                country_results_labels['self.expenditure_change']: country_obj.expenditure_change},
                                                 ignore_index=True)
        # Store some economy-wide values to economy object
        self.economy.experiment_total_output = total_output
        self.economy.output_change = 100 * (total_output - self.economy.baseline_total_output) \
                                     / self.economy.baseline_total_output

        results_table = results_table.set_index(country_results_labels['self.identifier'])
        # Ensure all values are numeric
        for col in results_table.columns:
            results_table[col] = results_table[col].astype(float)
        # Save to model
        self.outputs_expenditures = results_table

    def _construct_experiment_trade(self):
        '''
        Construct simulated bilateral trade values.
        :return: None. It sets the values for self.bilateral_trade_results, self.aggregate_trade_results, and many of
        the trade attributes in the country objects.
        '''
        importer_col = self.meta_data.imp_var_name
        exporter_col = self.meta_data.exp_var_name
        year_col = self.meta_data.year_var_name
        trade_value_col = self.meta_data.trade_var_name

        countries = self.country_set.keys()
        trade_data = self.baseline_data[[exporter_col, importer_col, year_col, trade_value_col]].copy()
        trade_data = trade_data.loc[trade_data[year_col] == self._year, [exporter_col, importer_col, trade_value_col]]

        trade_data.rename(columns={trade_value_col: 'baseline_trade'}, inplace=True)

        # Set Placeholder value
        #trade_data['gravity'] = -9999

        # Construct Modeled trade for each country-pair
        for row in trade_data.index:
            # Collect importer and exporter IDs
            importer = trade_data.loc[row, importer_col]
            exporter = trade_data.loc[row, exporter_col]

            # Collect and generate Experiment Values
            exp_imr_ratio = self.country_set[importer]._experiment_imr_ratio
            exp_omr_ratio = self.country_set[exporter]._experiment_omr_ratio
            expend = self.country_set[importer].experiment_expenditure
            output_share = self.country_set[exporter].experiment_output_share
            # gravity = E_j  *   Y_i/Y      * 1/P_j^{1-sigma} * 1/π_i^{1-sigma}
            gravity = expend * output_share * exp_imr_ratio * exp_omr_ratio
            trade_data.loc[row, 'exper_gravity'] = gravity

            # Collect and generate baseline values
            bsln_imr_ratio = self.country_set[importer]._baseline_imr_ratio
            bsln_omr_ratio = self.country_set[exporter]._baseline_omr_ratio
            bsln_expend = self.country_set[importer].baseline_expenditure
            bsln_output_share = self.country_set[exporter].baseline_output_share
            # 'gravity' term = E_j     *   Y_i/Y           * 1/P_j^{1-sigma} * 1/π_i^{1-sigma}
            bsln_gravity = bsln_expend * bsln_output_share * bsln_imr_ratio * bsln_omr_ratio
            trade_data.loc[row, 'bsln_gravity'] = bsln_gravity

        trade_data = trade_data.merge(self.baseline_trade_costs, how='left', on=[importer_col, exporter_col])
        trade_data.rename(columns={'trade_cost': 'baseline_trade_cost'}, inplace=True)

        # Set column labels from label dictionary
        bsln_modeled_trade_label = trade_results_labels['baseline_modeled_trade']
        exper_trade_label = trade_results_labels['experiment_trade']
        trade_change_label = trade_results_labels['trade_change']

        trade_data[bsln_modeled_trade_label] = trade_data['baseline_trade_cost'] \
                                                                   * trade_data['bsln_gravity']

        trade_data = trade_data.merge(self.experiment_trade_costs, how='left', on=[importer_col, exporter_col])

        trade_data[exper_trade_label] = trade_data['trade_cost'] * trade_data['exper_gravity']

        trade_data[trade_change_label] = 100 * (trade_data[exper_trade_label] - trade_data[bsln_modeled_trade_label]) \
                                       / trade_data[bsln_modeled_trade_label]

        bilateral_trade_results = trade_data[[exporter_col, importer_col, bsln_modeled_trade_label,
                                                   exper_trade_label, trade_change_label]]
        self.bilateral_trade_results = bilateral_trade_results.set_index([exporter_col, importer_col])

        ##
        # Calculate total Imports (international and domestic)
        ##
        # set more labels from label dictionary
        bsln_agg_imports_label = country_results_labels['self.baseline_imports']
        exper_agg_imports_label = country_results_labels['self.experiment_imports']
        agg_import_change_label =  country_results_labels['self.imports_change']

        agg_imports = bilateral_trade_results.copy()
        agg_imports = agg_imports[[importer_col, bsln_modeled_trade_label, exper_trade_label]]
        agg_imports = agg_imports.groupby([importer_col]).agg('sum')
        agg_imports.rename(columns={bsln_modeled_trade_label: bsln_agg_imports_label,
                                    exper_trade_label: exper_agg_imports_label}, inplace=True)
        agg_imports[agg_import_change_label] = 100 \
                                               * (agg_imports[exper_agg_imports_label] - agg_imports[bsln_agg_imports_label]) \
                                               / agg_imports[bsln_agg_imports_label]
        ##
        # Calculate foreign imports
        ##
        # set more labels from label dictionary
        bsln_agg_frgn_imports_label = country_results_labels['self.baseline_foreign_imports']
        exper_agg_frgn_imports_label = country_results_labels['self.experiment_foreign_imports']
        agg_frgn_import_change_label = country_results_labels['self.foreign_imports_change']

        foreign_imports = bilateral_trade_results.copy()
        foreign_imports = foreign_imports.loc[foreign_imports[importer_col]!=foreign_imports[exporter_col],:]
        foreign_imports = foreign_imports[[importer_col, bsln_modeled_trade_label, exper_trade_label]]
        foreign_imports = foreign_imports.groupby([importer_col]).agg('sum')
        foreign_imports.rename(columns={bsln_modeled_trade_label: bsln_agg_frgn_imports_label,
                                    exper_trade_label: exper_agg_frgn_imports_label}, inplace=True)
        foreign_imports[agg_frgn_import_change_label] = 100 \
                                               * (foreign_imports[exper_agg_frgn_imports_label] - foreign_imports[bsln_agg_frgn_imports_label]) \
                                               / foreign_imports[bsln_agg_frgn_imports_label]

        ##
        # Calculate total exports (foreign + domestic)
        ##
        # Set labels from label dictionary
        bsln_agg_exports_label = country_results_labels['self.baseline_exports']
        exper_agg_exports_label = country_results_labels['self.experiment_exports']
        agg_exports_change_label = country_results_labels['self.exports_change']

        agg_exports = bilateral_trade_results.copy()
        agg_exports = agg_exports[[exporter_col, bsln_modeled_trade_label, exper_trade_label]]
        agg_exports = agg_exports.groupby([exporter_col]).agg('sum')
        agg_exports.rename(columns={bsln_modeled_trade_label: bsln_agg_exports_label,
                                    exper_trade_label: exper_agg_exports_label}, inplace=True)
        agg_exports[agg_exports_change_label] = 100 \
                                               * (agg_exports[exper_agg_exports_label] - agg_exports[bsln_agg_exports_label]) \
                                               / agg_exports[bsln_agg_exports_label]
        ##
        # Calculate foreign exports
        ##
        # Set labels from label dictionary
        bsln_agg_frgn_exports_label = country_results_labels['self.baseline_foreign_exports']
        exper_agg_frgn_exports_label = country_results_labels['self.experiment_foreign_exports']
        agg_frgn_exports_change_label = country_results_labels['self.foreign_exports_change']

        foreign_exports = bilateral_trade_results.copy()
        foreign_exports = foreign_exports.loc[foreign_exports[importer_col] != foreign_exports[exporter_col], :]
        foreign_exports = foreign_exports[[exporter_col, bsln_modeled_trade_label, exper_trade_label]]
        foreign_exports = foreign_exports.groupby([exporter_col]).agg('sum')
        foreign_exports.rename(columns={bsln_modeled_trade_label: bsln_agg_frgn_exports_label,
                                        exper_trade_label: exper_agg_frgn_exports_label}, inplace=True)
        foreign_exports[agg_frgn_exports_change_label] = 100 \
                                                           * (foreign_exports[exper_agg_frgn_exports_label] -
                                                              foreign_exports[bsln_agg_frgn_exports_label]) \
                                                           / foreign_exports[bsln_agg_frgn_exports_label]


        agg_trade = pd.concat([agg_exports, foreign_exports, agg_imports, foreign_imports], axis=1).reset_index()
        agg_trade.rename(columns={'index': country_results_labels['self.identifier']}, inplace=True)


        # ----
        # Get Intranational Trade
        # ----
        bsln_intra_label = country_results_labels['self.baseline_intranational_trade']
        exper_intra_label = country_results_labels['self.experiment_intranational_trade']
        intra_change_label = country_results_labels['self.intranational_trade_change']
        intranational = bilateral_trade_results.copy()
        intranational = intranational.loc[intranational[importer_col] == intranational[exporter_col], :]
        intranational.drop([importer_col], axis = 1, inplace = True)
        intranational.rename(columns= {exporter_col:country_results_labels['self.identifier'],
                                       trade_results_labels['baseline_modeled_trade']:bsln_intra_label,
                                       trade_results_labels['experiment_trade']:exper_intra_label,
                                       trade_results_labels['trade_change']:intra_change_label}, inplace = True)

        agg_trade = agg_trade.merge(intranational, on = country_results_labels['self.identifier'])

        # Store values in each country object
        for row in agg_trade.index:
            country = agg_trade.loc[row, country_results_labels['self.identifier']]
            country_obj = self.country_set[country]
            country_obj.baseline_imports = agg_trade.loc[row, bsln_agg_imports_label]
            country_obj.baseline_exports = agg_trade.loc[row, bsln_agg_exports_label]
            country_obj.baseline_foreign_imports = agg_trade.loc[row, bsln_agg_frgn_imports_label]
            country_obj.baseline_foreign_exports = agg_trade.loc[row, bsln_agg_frgn_exports_label]
            country_obj.experiment_imports = agg_trade.loc[row, exper_agg_imports_label]
            country_obj.experiment_exports = agg_trade.loc[row, exper_agg_exports_label]
            country_obj.imports_change = agg_trade.loc[row, agg_import_change_label]
            country_obj.exports_change = agg_trade.loc[row, agg_exports_change_label]
            country_obj.experiment_foreign_imports = agg_trade.loc[row, exper_agg_frgn_imports_label]
            country_obj.experiment_foreign_exports = agg_trade.loc[row, exper_agg_frgn_exports_label]
            country_obj.foreign_imports_change = agg_trade.loc[row, agg_frgn_import_change_label]
            country_obj.foreign_exports_change = agg_trade.loc[row, agg_frgn_exports_change_label]
            country_obj.baseline_intranational_trade = agg_trade.loc[row, bsln_intra_label]
            country_obj.experiment_intranational_trade = agg_trade.loc[row, exper_intra_label]
            country_obj.intranational_trade_change = agg_trade.loc[row, intra_change_label]

        self.aggregate_trade_results = agg_trade.set_index(country_results_labels['self.identifier'])

    def _compile_results(self):
        '''Generate and compile results after simulations'''
        results = list()
        mr_results = list()
        for country in self.country_set.keys():
            results.append(self.country_set[country].get_results())
            mr_results.append(self.country_set[country].get_mr_results())
        country_results = pd.concat(results, axis=0)
        self.country_results = country_results.set_index(country_results_labels['self.identifier'])
        country_mr_results = pd.concat(mr_results, axis=0)
        self.country_mr_terms = country_mr_results.set_index(country_results_labels['self.identifier'])


    def trade_share(self, importers: List[str], exporters: List[str]):
        '''
        Calculate baseline and experiment import and export shares (in percentages) between user-supplied countries.
        Args:
            importers (list[str]): A list of country codes to include as import partners.
            exporters (list[str]): A list of country codes to include as export partners.

        Returns:
            pandas.DataFrame: A dataframe expressing baseline, experiment, and changes in trade between each specified importer and exporter.
        '''
        # '''
        #
        # :param importers:  A list of country codes to include as import partners.
        # :param exporters: (list[str]) A list of country codes to include as export partners.
        # :return: (DataFrame)
        # '''
        importer_col = self.meta_data.imp_var_name
        exporter_col = self.meta_data.exp_var_name
        bsln_modeled_trade_label = trade_results_labels['baseline_modeled_trade']
        exper_trade_label = trade_results_labels['experiment_trade']

        bilat_trade = self.bilateral_trade_results.reset_index()
        columns = [bsln_modeled_trade_label, exper_trade_label]
        imports = bilat_trade.loc[bilat_trade[importer_col].isin(importers), :].copy()
        exports = bilat_trade.loc[bilat_trade['exporter'].isin(exporters), :].copy()

        total_imports = imports[columns].agg('sum')
        total_exports = exports[columns].agg('sum')

        selected_imports = imports.loc[imports['exporter'].isin(exporters), columns].copy().agg('sum')
        selected_exports = exports.loc[exports[importer_col].isin(importers), columns].copy().agg('sum')

        import_data = 100 * selected_imports / total_imports
        export_data = 100 * selected_exports / total_exports

        import_data['description'] = 'Percent of ' + ", ".join(importers) + ' imports from ' + ", ".join(exporters)
        export_data['description'] = 'Percent of ' + ", ".join(exporters) + ' exports to ' + ", ".join(importers)

        both = pd.concat([import_data, export_data], axis=1).T
        both = both[['description'] + columns]
        both['change (percentage point)'] = (both[exper_trade_label] - both[bsln_modeled_trade_label])
        both['change (%)'] = 100 * (both[exper_trade_label] - both[bsln_modeled_trade_label]) / \
                                   both[bsln_modeled_trade_label]

        return both

    def export_results(self, directory:str = None, name:str = '',
                       include_levels:bool = False, country_names:DataFrame = None):
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

        Returns:
            None or Tuple[DataFrame, DataFrame, DataFrame]: If a directory argument is supplied, the method returns
                nothing and writes three .csv files instead. If no directory is supplied, it returns a tuple of DataFrames.
        '''

        importer_col = self.meta_data.imp_var_name
        exporter_col = self.meta_data.exp_var_name

        country_result_set = [self.country_results, self.factory_gate_prices, self.aggregate_trade_results,
                              self.outputs_expenditures, self.country_mr_terms]
        country_results = pd.concat(country_result_set, axis = 1)
        # Order and select columns for inclusion, drop duplicates.
        country_results_cols = country_results.columns
        crl = country_results_labels
        results_cols = crl.keys()
        included_columns = [crl[col] for col in results_cols if crl[col] in country_results_cols]
        country_results = country_results[included_columns]
        country_results = country_results.loc[:, ~country_results.columns.duplicated()]

        bilateral_results = self.bilateral_trade_results.reset_index()

        if include_levels:
            country_levels = self.calculate_levels(how = 'country')
            duplicate_columns = [col for col in country_levels.columns if col in country_results.columns]
            country_levels.drop(duplicate_columns,axis = 1, inplace = True)
            country_results = country_results.merge(country_levels, how = 'left', left_index = True, right_index = True)

            bilateral_levels = self.calculate_levels(how='bilateral')
            duplicate_columns = [col for col in bilateral_levels.columns if (col in bilateral_results.columns)
                                 and col not in [exporter_col, importer_col]]
            bilateral_levels.drop(duplicate_columns, axis=1, inplace=True)
            bilateral_results = bilateral_results.merge(bilateral_levels, how='left', on = [exporter_col, importer_col])

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
        diagnostics = self.solver_diagnostics
        column_list = list()
        # Iterate through the three solver types: baseline_MRs, conditional_MRs, and Full_GE
        for results_type, results in diagnostics.items():
            for key, value in results.items():
                # Single Entry fields must be converted to list before creating DataFrame
                if key in ['success', 'status', 'nfev', 'message']:
                    frame = pd.DataFrame({(results_type, key): [value]})
                    column_list.append(frame)
                # Vector-like fields Can be used as is. Several available fields are not included: 'fjac','r', and 'qtf'
                elif key in ['x', 'fun']:
                    frame = pd.DataFrame({(results_type, key): value})
                    column_list.append(frame)
        diag_frame = pd.concat(column_list, axis=1)
        diag_frame = diag_frame.fillna('')


        if directory is not None:
            country_results.to_csv("{}/{}_country_results.csv".format(directory, name))
            bilateral_results.to_csv("{}/{}_bilateral_results.csv".format(directory, name), index = False)
            diag_frame.to_csv("{}/{}_solver_diagnostics.csv".format(directory, name), index = False)
        else:
            return country_results, bilateral_results, diagnostics

    def calculate_levels(self, how: str = 'country'):
        '''
        Calculate changes in the level (value) of trade using baseline trade values and simulation outcomes. Results
            can be calculated at either the country level or bilateral level.
        Args:
            how (str):  If 'country', returned values are calculated at the country level (total exports, imports,
                and intranational). If 'bilateral', returned results are at the bilateral level. Default is 'country'.

        Returns:
            pandas.DataFrame: A DataFrame containing baseline and experiment trade levels as well as the change expressed
                in levels and percentages. If calculated at the country level, these four measures are each returned for
                total imports, exports, and intranational trade. If calculated at the bilateral level, only one set of the
                measures is returned.
        '''
        if not self._baseline_built and self._experiment_defined:
            raise ValueError('Model must be fully solved before calculating levels.')
        exporter = self.meta_data.exp_var_name
        importer = self.meta_data.imp_var_name
        trade = self.meta_data.trade_var_name
        trade_flows = self.baseline_data.copy()
        trade_flows = trade_flows[[exporter, importer, trade]]
        bilateral_results = self.bilateral_trade_results[[trade_results_labels['trade_change']]].copy()
        bilateral_results.reset_index(inplace=True)
        crl = country_results_labels
        trl = trade_results_labels
        # Coumpute at the country level (importer, exporter, and intranational)
        if how == 'country':
            intra_national = trade_flows.loc[trade_flows[exporter] == trade_flows[importer], [exporter, trade]]
            intra_national.set_index(exporter, inplace=True)
            international = trade_flows.loc[trade_flows[exporter] != trade_flows[importer], :]
            # Aggregate trade values
            foreign_exports = international.groupby(exporter).agg({trade: 'sum'})
            foreign_imports = international.groupby(importer).agg({trade: 'sum'})
            # Combine
            country_trade = foreign_exports.merge(foreign_imports, how='outer', left_index=True, right_index=True)
            country_trade = country_trade.merge(intra_national, how='outer', left_index=True, right_index=True)
            country_trade.columns = [crl['self.baseline_observed_foreign_exports'],
                                     crl['self.baseline_observed_foreign_imports'],
                                     crl['self.baseline_observed_intranational_trade']]
            # Prep and add experiment change info
            experiment_results = self.country_results[[crl['self.foreign_exports_change'],
                                                       crl['self.foreign_imports_change']]].reset_index()
            intra_results = bilateral_results.loc[bilateral_results[exporter] == bilateral_results[importer],
                                                  [exporter, trade_results_labels['trade_change']]].copy()
            intra_results.rename(columns={exporter: 'country',
                                          trade_results_labels['trade_change']: crl['self.intranational_trade_change']},
                                          inplace=True)
            experiment_results = pd.merge(experiment_results, intra_results, on=crl['self.identifier'])
            experiment_results.set_index(crl['self.identifier'], inplace=True)
            country_trade = country_trade.merge(experiment_results, how='outer', left_index=True, right_index=True)
            # Compute new levels
            for level, change in [(crl['self.baseline_observed_foreign_exports'], crl['self.foreign_exports_change']),
                                  (crl['self.baseline_observed_foreign_imports'], crl['self.foreign_imports_change']),
                                  (crl['self.baseline_observed_intranational_trade'], crl['self.intranational_trade_change'])]:
                new_level_name = level.replace('baseline', 'experiment')
                level_change_name = change.replace('%','observed level')
                country_trade[new_level_name] = country_trade[level] * (1 + (experiment_results[change] / 100))
                country_trade[level_change_name] = country_trade[new_level_name] - country_trade[level]

            result_order = list()
            for result_type in ['exports','imports','intranational']:
                for sub_type in ['baseline','experiment','level','%']:
                    for col in country_trade.columns:
                        if (sub_type in col) and (result_type in col):
                            result_order.append(col)
            # Reorder columns
            # country_trade = country_trade[['baseline_foreign_exports', 'experiment_foreign_exports',
            #                                'foreign_export_change_level', 'foreign_export_change_percent',
            #                                'baseline_foreign_imports', 'experiment_foreign_imports',
            #                                'foreign_import_change_level', 'foreign_import_change_percent',
            #                                'baseline_intranational_trade', 'experiment_intranational_trade',
            #                                'intranational_change_level', 'intranational_change_percent']]
            return country_trade[result_order]
        # Compute at the bilateral level
        if how == 'bilateral':
            # Prep and add experiment change info
            experiment_results = bilateral_results
            bilat_trade = trade_flows.merge(experiment_results, on=[exporter, importer], how='outer')
            bilat_trade.rename(columns={trade: trl['baseline_observed_trade']},
                               inplace=True)
            # Create changes in levels

            bilat_trade[trl['experiment_observed_trade']] = bilat_trade[trl['baseline_observed_trade']] * (
                        1 + (bilat_trade[trl['trade_change']] / 100))
            bilat_trade[trl['trade_change_level']] = bilat_trade[trl['experiment_observed_trade']] - bilat_trade[trl['baseline_observed_trade']]
            bilat_trade = bilat_trade[[exporter, importer, trl['baseline_observed_trade'],
                                       trl['experiment_observed_trade'], trl['trade_change_level'], trl['trade_change']]]
            return bilat_trade

    def trade_weighted_shock(self, how:str = 'country', aggregations:list=['mean', 'sum', 'max']):
        '''
        Create measures of trade weighted policy shocks to better understand which countries are most affected. Results
        reflect the absolute value of the change in trade costs multiplied by the
        Args:
            how (str): Determines the level of the results. If 'country', weighted shocks are returned at the country
                level for both importer and exporter using specified methods of aggregation. If 'bilateral', it returns the
                weighted shocks for all bilateral pairs. Default is 'country'.
            aggregations (list[str]):  A list of methods by which to aggregate weighted shocks if how = 'country'.
                List entries must be selected from those that are functional with the pandas.DataFrame.agg() method. The
                default value is ['mean', 'sum', 'max'].
        Returns:
            pandas.DataFrame: A dataframe of trade-weighted trade cost shocks.
        '''
        # Collect needed results
        bilat_trade = self.bilateral_trade_results.copy()
        bilat_trade.reset_index(inplace=True)
        cost_shock = self.cost_shock.copy()

        # Define column names
        imp_col = self.meta_data.imp_var_name
        exp_col = self.meta_data.exp_var_name
        baseline_trade_col = trade_results_labels['baseline_modeled_trade']
        baseline_cost_col = 'baseline_trade_cost'
        exper_cost_col = 'experiment_trade_cost'
        cost_change = 'cost_change'
        weighted_col = 'weighted_shock'

        # Create change in costs and add to bilateral trade
        cost_shock[cost_change] = abs(cost_shock[exper_cost_col] - cost_shock[baseline_cost_col])
        trade_shock = bilat_trade.merge(cost_shock, how='left', on=[imp_col, exp_col])
        trade_shock = trade_shock[[exp_col, imp_col, baseline_trade_col, cost_change]]
        # Fill cases with no change in costs with zero
        trade_shock.fillna(0, inplace=True)
        # Calculate weighted costs and normalize my largest weighted shock
        trade_shock[weighted_col] = trade_shock[baseline_trade_col] * trade_shock[cost_change]
        max_shock = max(trade_shock[weighted_col])
        trade_shock[weighted_col] = trade_shock[weighted_col] / max_shock

        # Create aggregate measures at importer and exporter level
        exporter_shocks = trade_shock.groupby(exp_col).agg({weighted_col: aggregations})
        exporter_shocks.columns = pd.MultiIndex.from_product(exporter_shocks.columns.levels + [[exp_col]])
        importer_shocks = trade_shock.groupby(imp_col).agg({weighted_col: aggregations})
        importer_shocks.columns = pd.MultiIndex.from_product(importer_shocks.columns.levels + [[imp_col]])
        weighted_shocks = pd.concat([exporter_shocks, importer_shocks], axis=1)

        if how == 'country':
            return weighted_shocks
        if how == 'bilateral':
            return trade_shock


    # ---
    # Diagnostic Tools
    # ---
    def test_baseline_mr_function(self, inputs_only:bool=False):
        '''
        Test whether the multilateral resistance system of equations can be computed from baseline data. Helpful for
            debugging initial data problems. Note that the returned function values reflect those generated by the
            initial values and do not reflect a solution to the system.
        Args:
            inputs_only (bool): If False (default), the method tests the computability of the MR system of equations
                and returns both the inputs to the system and the output. If True, only the system inputs are return and
                the equations are not computed and can help diagnose input issues that raise errors.
        Returns:
            dict: A dictionary containing a collection of parameter and value inputs as well as the function
                values at the initial values.
        '''
        test_diagnostics = self._calculate_multilateral_resistance(trade_costs=self.baseline_trade_costs,
                                                                   version='baseline', test=True,
                                                                   inputs_only=inputs_only)
        return test_diagnostics

    def check_omr_rescale(self,
                         omr_rescale_range:int = 10,
                         mr_method: str = 'hybr',
                         mr_max_iter: int = 1400,
                         mr_tolerance: float = 1e-8,
                         countries:List[str] = []):
        '''
        Analyze different Outward Multilarteral Resistance (OMR) term rescale factors. This method can help identify
            feasible values to use for the omr_rescale argument in OneSectorGE.build_baseline().
        Args:
            omr_rescale_range (int): This parameter allows you to set the scope of the values tested. For example,
                if omr_rescale_range = 3, the model will check for convergence using omr_rescale values from the set
                [10^-3, 10^-2, 10^-1, 10^0, ..., 10^3]. The default value is 10.
            mr_method (str): This parameter determines the type of non-linear solver used for solving the baseline and
                experiment MR terms. See the documentation for scipy.optimize.root for alternative methods. the default
                value is 'hybr'.
            mr_max_iter (int): (optional) This parameter sets the maximum limit on the number of iterations conducted
                by the solver used to solve for MR terms. The default value is 1400.
            mr_tolerance (float): (optional) This parameterset the convergence tolerance level for the solver used to
                solve for MR terms. The default value is 1e-8.
            countries (List[str]):  A list of countries for which to return the estimated OMR values for user
                evaluation.
        Returns:
            pandas.DataFrame: A dataframe of diagnostic information for users to compare different omr_rescale factors.
                The returned dataframe contains the following columns:\n
                'omr_rescale': The rescale factor used\n
                'omr_rescale (alt format)': A string representation of the rescale factor as an exponential expression.\n
                'solved': If True, the MR model solved successfully. If False, it did not solve.\n
                'message': Description of the outcome of the solver.\n
                '..._func_value': Three columns reflelcting the maximum, mean, and median values from the solver
                    objective functions. Function values closer to zero imply a better solution to system of equations.
                'reference_importer_omr': The solution value for the reference importer's OMR value.\n
                '..._omr': The solution value(s) for the user supplied countries.
        '''

        self._mr_max_iter = mr_max_iter
        self._mr_tolerance = mr_tolerance
        self._mr_method = mr_method
        self._imr_rescale = 1

        # Set up procedure for identifying usable omr_rescale
        findings = list()
        value_index = 0
        # Create list of rescale factors to test
        scale_values = range(-1*omr_rescale_range,omr_rescale_range+1)

        for scale_value in scale_values:
            value_results = dict()
            rescale_factor = 10 ** scale_value

            if not self.quiet:
                print("\nTrying OMR rescale factor of {}".format(rescale_factor))
            self._omr_rescale = rescale_factor
            self._calculate_multilateral_resistance(trade_costs=self.baseline_trade_costs,
                                                    version='baseline')
            value_results['omr_rescale'] = rescale_factor
            value_results['omr_rescale (alt format)'] = '10^{}'.format(scale_value)
            value_results['solved'] = self.solver_diagnostics['baseline_MRs']['success']
            value_results['message'] = self.solver_diagnostics['baseline_MRs']['message']
            func_vals = self.solver_diagnostics['baseline_MRs']['fun']
            value_results['max_func_value'] = func_vals.max()
            value_results['mean_func_value'] = func_vals.mean()
            value_results['mean_func_value'] = median(func_vals)
            value_results['reference_importer_omr'] = self.country_set[self._reference_importer]._baseline_omr_ratio
            for country in countries:
                value_results['{}_omr'.format(country)] = self.country_set[country]._baseline_omr_ratio
            findings.append(value_results)
        findings_table = pd.DataFrame(findings)

        return findings_table


def _multilateral_resistances(x, mr_params):
    '''
    System of equartions for the multilateral resistances
    :param x: (list) Values for the endogenous variables OMR π_i^(1-sigma) and IMR P_j^(1-sigma).
    :param mr_params: (dict) Additional exogeneous parameters
    :return: (list) the function value evaluated at x given mr_params
    '''
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
    '''
    System of equations for the full-GE model
    :param x: (list) Values for the endogenous variables
    :param ge_params: (dict) Exogenous parameters for the equations including: number of countries, sigma, exogenous
        outpute, cost/expenditure, etc. shares, factory gate price parameter, and rescale factors.
    :return: (list) The value of the equations evaluated at x given ge_params.
    '''
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
    '''
    Object for storing economy-wide information.
    '''
    def __init__(self,
                 sigma: float = 4):
        self.sigma = sigma
        self.experiment_total_output = None
        self.experiment_total_expenditure = None
        self.baseline_total_output = None
        self.baseline_total_expenditure = None
        self.output_change = None

    def _initialize_baseline_total_output_expend(self, country_set):
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
        self.baseline_output = baseline_output # Y_i
        self.baseline_expenditure = baseline_expenditure # E_j
        self.baseline_importer_fe = baseline_importer_fe
        self.baseline_exporter_fe = baseline_exporter_fe
        self._reference_expenditure = reference_expenditure
        self.baseline_output_share = None # Y_i/Y
        self.baseline_expenditure_share = None # E_j/Y
        self.baseline_export_costs = None
        self.baseline_import_costs = None
        self._baseline_imr_ratio = None  # 1 / P^{1-sigma}
        self._baseline_omr_ratio = None  # 1 / π ^{1-\sigma}
        self.baseline_imr = None  # P
        self.baseline_omr = None  # π
        self.factory_gate_price_param = None  # Beta_i
        self.baseline_factory_price = 1
        self._conditional_imr_ratio = None  # 1 / P'^{1-sigma}
        self._conditional_omr_ratio = None  # 1 / π'^{1-\sigma}
        self.conditional_imr = None  # P'
        self.conditional_omr = None  # π'
        self._experiment_imr_ratio = None  # 1 / P*^{1-sigma}
        self._experiment_omr_ratio = None  # 1 / π*^{1-\sigma}
        self.experiment_imr = None  # P*
        self.experiment_omr = None  # π*
        self.imr_change = None # 100*(P*-P)/P
        self.omr_change = None # 100*(π*-π)/π
        self.experiment_factory_price = None # p*_i
        self.experiment_output = None # Y*_i
        self.experiment_expenditure = None # E*_j
        self.baseline_terms_of_trade = None  # ToT_i = p_i/P_i
        self.experiment_terms_of_trade = None  # ToT*_i = p*_i/P*_i
        self.terms_of_trade_change = None  # 100*(ToT* - ToT)/ToT
        self.output_change = None  # 100*(Y* - Y)/Y
        self.expenditure_change = None  # 100*(E* - E)/E
        self.factory_price_change = None # 100*(p*-p)/p or, in practice, 100*(P*-1)/1
        self.baseline_imports = None  # sum_i(X_ij) [modeled flows, not observed flows]
        self.baseline_exports = None  # sum_j(X_ij) [modeled flows, not observed flows]
        self.baseline_foreign_imports = None # sum_{i!=j}(X_ij) [modeled flows, not observed flows]
        self.baseline_foreign_exports = None  # sum_{j!=i}(X_ij) [modeled flows, not observed flows]
        self.experiment_imports = None # sum_i(X*_ij)
        self.experiment_exports = None # sum_j(X*_ij)
        self.experiment_foreign_imports = None # sum_{i!=j}(X*_ij)
        self.experiment_foreign_exports = None # sum_{j!=i}(X*_ij)
        self.foreign_imports_change = None # 100*(sum_{i!=j}(X*_ij) - sum_{i!=j}(X_ij))/sum_{i!=j}(X_ij)
        self.foreign_exports_change = None # 100*(sum_{j!=i}(X*_ij) - sum_{j!=i}(X_ij))/sum_{j!=i}(X_ij)
        self.baseline_intranational_trade = None # X_ii
        self.experiment_intranational_trade = None # X*_ii
        self.intranational_trade_change = None # 100*(X*_ii - X_ii)/X_ii
        self.baseline_gdp = None # GDP_j = Y_j/P_j
        self.experiment_gdp = None # GDP*_j = Y*_j/P*_j
        self.gdp_change = None # 100*(GDP* - GDP)/GDP
        self.phi = None  # φ_i = E_i / Y_i (Eqn. (30) from Larch and Yotov, 2016)
        self.welfare_stat = None  # (E_i/P_i)/(E*_i/P*_i)


    def _calculate_baseline_output_expenditure_shares(self, economy):
        self.baseline_expenditure_share = self.baseline_expenditure / economy.baseline_total_expenditure
        self.baseline_output_share = self.baseline_output / economy.baseline_total_output

    def _construct_country_measures(self, sigma):
        for value in [self.baseline_factory_price, self._baseline_imr_ratio,
                      self.experiment_factory_price, self._experiment_imr_ratio,
                      self._conditional_imr_ratio, self._conditional_omr_ratio]:
            if value is None:
                warn("Not all necessary values for terms of trade have been calculated.")

        # Create actual values for imr (P) and omr (π) from ratio representation
        sigma_inverse = 1 / (1 - sigma)
        self.baseline_imr = 1 / (self._baseline_imr_ratio**sigma_inverse)
        self.baseline_omr = 1 / (self._baseline_omr_ratio**sigma_inverse)
        self.conditional_imr = 1 / (self._conditional_imr_ratio ** sigma_inverse)
        self.conditional_omr = 1 / (self._conditional_omr_ratio ** sigma_inverse)
        self.experiment_imr = 1 / (self._experiment_imr_ratio ** sigma_inverse)
        self.experiment_omr = 1 / (self._experiment_omr_ratio ** sigma_inverse)

        # Calculate Output and Expenditure
        self.experiment_output = self.experiment_factory_price * self.baseline_output
        self.output_change = 100 * (self.experiment_output - self.baseline_output) \
                                    / self.baseline_output
        # Experiment Expenditure: E_i = φ_i Y_i (Eqn. (30) from Larch and Yotov, 2016) ->  E*_i = φ_i Y*_i
        self.phi = self.baseline_expenditure/self.baseline_output
        self.experiment_expenditure = self.phi * self.experiment_output
        self.expenditure_change = 100 * (self.experiment_expenditure -
                                                self.baseline_expenditure) / self.baseline_expenditure

        # Calculate Terms of Trade
        self.baseline_terms_of_trade = self.baseline_factory_price / self.baseline_imr
        self.experiment_terms_of_trade = self.experiment_factory_price / self.experiment_imr
        self.terms_of_trade_change = 100 * (self.experiment_terms_of_trade - self.baseline_terms_of_trade) \
                                     / self.baseline_terms_of_trade

        # Calculate GDP (from Stata code accompanying Yotov et al (2016): GDP_j = Y_j/P_j)
        self.baseline_gdp = self.baseline_output/self.baseline_imr
        self.experiment_gdp = self.experiment_output/self.experiment_imr
        self.gdp_change = 100 * (self.experiment_gdp - self.baseline_gdp)/ self.baseline_gdp

        # Calculate Arkolakis, Costinot and Rodríguez-Clare welfare statistic (Equation (25) from Larch and Yotov, 2016)
        # WF_i = W_i/W*_i = (E_i/P_i)/(E*_i/P*_i)
        self.welfare_stat = (self.baseline_expenditure/self.baseline_imr)/\
                            (self.experiment_expenditure/self.experiment_imr)

        # Calculate change in IMR (consumer price) and OMR (producer incidence) measure
        self.imr_change = 100*(self.experiment_imr - self.baseline_imr)/self.baseline_imr
        self.omr_change = 100*(self.experiment_omr - self.baseline_omr)/self.baseline_omr




    def get_results(self):
        '''
        Collect and return the country's main results.
        Returns:
            pandas.DataFrame: A one-row dataFrame containing columns of typical results.
        '''
        row = pd.DataFrame(data={country_results_labels['self.identifier']: [self.identifier],
                                 country_results_labels['self.factory_price_change']: [self.factory_price_change],
                                 country_results_labels['self.omr_change']: [self.omr_change],
                                 country_results_labels['self.imr_change']: [self.imr_change],
                                 country_results_labels['self.gdp_change']: [self.gdp_change],
                                 country_results_labels['self.welfare_stat']: [self.welfare_stat],
                                 country_results_labels['self.terms_of_trade_change']: [self.terms_of_trade_change],
                                 country_results_labels['self.output_change']: [self.output_change],
                                 country_results_labels['self.expenditure_change']: [self.expenditure_change],
                                 # country_results_labels['self.exports_change']: [self.exports_change],
                                 # country_results_labels['self.imports_change']: [self.imports_change],
                                 country_results_labels['self.foreign_exports_change']: [self.foreign_exports_change],
                                 country_results_labels['self.foreign_imports_change']: [self.foreign_imports_change],
                                 country_results_labels['self.intranational_trade_change']:self.intranational_trade_change,
                                 })
        return row

    def get_mr_results(self):
        '''
        Collect and return the country's MR terms (baseline, conditional, and experiment)
        Returns:
             pandas.DataFrame: A one-row dataFrame containing column of MR terms.
        '''
        row = pd.DataFrame(data={country_results_labels['self.identifier']: [self.identifier],
                                 country_results_labels['self.baseline_imr']: [self.baseline_imr],
                                 country_results_labels['self.conditional_imr']: [self.conditional_imr],
                                 country_results_labels['self.experiment_imr']: [self.experiment_imr],
                                 country_results_labels['self.baseline_omr']: [self.baseline_omr],
                                 country_results_labels['self.conditional_omr']: [self.conditional_omr],
                                 country_results_labels['self.experiment_omr']: [self.experiment_omr]})
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
    Modified gme._MetaData object that includes output and expenditure column names
    '''
    def __init__(self, gme_meta_data, expend_var_name, output_var_name):
        self.imp_var_name = gme_meta_data.imp_var_name
        self.exp_var_name = gme_meta_data.exp_var_name
        self.year_var_name = gme_meta_data.year_var_name
        self.trade_var_name = gme_meta_data.trade_var_name
        self.sector_var_name = gme_meta_data.sector_var_name
        self.expend_var_name = expend_var_name
        self.output_var_name = output_var_name


class ParameterValues(object):
    def __init__(self,
             estimates:DataFrame,
                 identifier_col: str,
                 coeff_col:str,
                 stderr_col:str = None):
                 #imp_fe_prefix: str = None, # Removed until completeion of GEPPML
                 #exp_fe_prefix: str = None # Removed until completeion of GEPPML

        '''
        Object for supplying non-gme.EstimationModel estimates such as those from Stata, R, the literature, or any other
        source of gravity estimates.
        Args:
            estimates (pandas.DataFrame): A dataframe containing gravity model estimates, which ought to include the
                following non-optional columns.
            identifier_col (str): The name of the column containing the identifiers for each estimate. These should
                correspond to the cost variables that you will use for trade costs in the simulation. This column is
                required.
            coeff_col (str): The name of the column containing the coefficient estimates for each variable. They
                should be numeric and are required.
            stderr_col (str):  The column name for the standard error estimates for each variable. This column is only
                required for the MonteCarloGE model and may be omitted for the OneSectorGE model.

        Returns:
            ParameterValues: An instance of a ParameterValues object.
        '''
        # '''
        # imp_fe_prefix (str):  The prefix used to identify the importer fixed effects. These fixed effects and
        #         prefix are only required for the GEPPML approach in OneSectorGE.
        #     exp_fe_prefix (str):  The prefix used to identify the exporter fixed effects. These fixed effects and
        #         prefix are only required for the GEPPML approach in OneSectorGE.
        # '''
        estimates = estimates.set_index(identifier_col)
        # Coefficient  Estimates
        self.params = estimates[coeff_col].copy()
        # Standard error estimates
        if stderr_col is not None:
            self.bse = estimates[stderr_col].copy()
        else:
            self.bse = None

        # self.imp_fe_prefix = imp_fe_prefix
        # self.exp_fe_prefix = exp_fe_prefix

#----
# Column Labels for the DataFrames of Results  (format is attribute:label)
#----
country_results_labels = {'self.identifier':'country',
                          'self.factory_price_change':'factory gate price change ((percent))',
                          'self.experiemnt_factory_price':'experiment factory gate price',
                          'self.terms_of_trade_change':'terms of trade change ((percent))',
                          'self.gdp_change':"GDP change (percent)",
                          'self.welfare_stat':'welfare statistic',
                          'self.baseline_output':'baseline output',
                          'self.experiment_output':'experiment output',
                          'self.output_change':'output change (percent)',
                          'self.baseline_expenditure':'baseline expenditure',
                          'self.experiment_expenditure':'experiment expenditure',
                          'self.expenditure_change':'expenditure change (percent)',
                          'self.baseline_exports':'baseline modeled shipments',
                          'self.experiment_exports':'experiment shipments',
                          'self.exports_change':'shipments change (percent)',
                          'self.baseline_imports':'baseline modeled consumption',
                          'self.experiment_imports':'experiment consumption',
                          'self.imports_change':'consumption change (percent)',
                          'self.baseline_foreign_exports':'baseline modeled foreign exports',
                          'self.experiment_foreign_exports':'experiment foreign exports',
                          'self.foreign_exports_change':'foreign exports change (percent)',
                          'self.baseline_observed_foreign_exports':'baseline observed foreign exports',
                          'self.baseline_foreign_imports':'baseline modeled foreign imports',
                          'self.experiment_foreign_imports':'experiment foreign imports',
                          'self.foreign_imports_change':'foreign imports change (percent)',
                          'self.baseline_observed_foreign_imports':'baseline observed foreign imports',
                          'self.baseline_intranational_trade':'baseline modeled intranational trade',
                          'self.experiment_intranational_trade':'experiment modeled intranational trade',
                          'self.intranational_trade_change':'intranational trade change (percent)',
                          'self.baseline_observed_intranational_trade': 'baseline observed intranational trade',
                          'self.baseline_imr':'baseline imr',
                          'self.conditional_imr':'conditional imr',
                          'self.experiment_imr': 'experiment imr',
                          'self.imr_change': 'imr change (percent)',
                          'self.baseline_omr':'baseline omr',
                          'self.conditional_omr':'conditional omr',
                          'self.experiment_omr':'experiment omr',
                          'self.omr_change': 'omr change (percent)'
}

trade_results_labels = {
    'baseline_modeled_trade':'baseline modeled trade',
    'experiment_trade':'experiment trade',
    'trade_change':'trade change (percent)',
    'trade_change_level':'trade change (observed level)',
    'baseline_observed_trade':'baseline observed trade',
    'experiment_observed_trade':'experiment observed trade'
}



