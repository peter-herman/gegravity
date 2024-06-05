__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "July 13, 2023"
__Description__ = '''Class to handle baseline data inputs without using the GME package.'''
__all__ = ['BaselineData']

from pandas import DataFrame
from typing import List
# from src.gegravity.OneSectorGE import _GEMetaData


class BaselineData(object):
    def __init__(self,
                 baseline_data:DataFrame,
                 imp_var_name: str = 'importer',
                 exp_var_name: str = 'exporter',
                 year_var_name: str = 'year',
                 trade_var_name: str = None,
                 expend_var_name: str = None,
                 output_var_name: str = None,
                 country_fixed_effects: DataFrame = None,
                 ):
        '''
        Create the baseline data input for the OneSectorGE model.
        Args:
            baseline_data: (pandas DataFrame) A pandas dataframe containing bilateral data used for the model. Data
                should include importer, exporter, and year identifiers; trade values; total output and expenditure
                values for each exporter and importer, respectively; and all trade cost variables, including any fixed
                effects to be used to compute trade costs. Fixed effects should be included as corresponding dummy
                variables.
            imp_var_name: (str) Name of the column containing the importer identifiers.
            exp_var_name: (str) Name of the column containing the exporter identifiers.
            year_var_name: (str) Name of the column containing the year identifiers.
            trade_var_name: (str) Name o the column containing bilateral trade values.
            expend_var_name: (str) Name of the column containing importer total expenditure values.
            output_var_name: (str) Name of the column containing exporter total output values.
            country_fixed_effects: (pandas DataFrame) Optional, estimated exporter and importer fixed effects. DataFrame
                must contain three columns: country identifier, exporter fixed effect, and importer fixed effect. The
                country identifier must appear in the first column. The fixed effects columns must be named with
                exp_var_name and imp_var_name, respectively. [Note: this argument currently serves no practical function
                but may in the future.]

        Atributes:
            baseline_data (pandas.DataFrame): The baseline_data dataframe
            baseline_columns (list): A list of the columns in the baseline_data dataframe
            meta_data (gegravity._MetaData): An object class that holds and organizes the column labels associated with
                various components of the data inputs.
            specification (gegravity.Specification) An object class holding information on the underlying econometric
                specification [Note: supported GME package integration but is largely unused as of v0.3]
            country_fixed_effects (pandas.DataFrame): The country_fixed_effects argument, if supplied.

        Examples:
            import gegravity as ge and pandas
            >>> import pandas as pd
            >>> import gegravity as ge

            Load the data.
            >>> grav_data = pd.read_csv(sample_data_set.dlm
            >>> grav_data.head()
              exporter importer  year  trade        Y       E  pta  contiguity  common_language  lndist  international
            0      GBR      AUS  2006   4310   925638  362227    0           0                1  9.7126              1
            1      FIN      AUS  2006    514   142759  362227    0           0                0  9.5997              1
            2      USA      AUS  2006  16619  5019964  362227    1           0                1  9.5963              1
            3      IND      AUS  2006    763   548517  362227    0           0                1  9.1455              1
            4      SGP      AUS  2006   8756   329817  362227    1           0                1  8.6732              1

            Define BaselineData object to hold and organize the baseline data inputs
            >>> baseline = ge.BaselineData(grav_data,
            ...                            imp_var_name='importer',
            ...                            exp_var_name='exporter',
            ...                            year_var_name='year', trade_var_name='trade',
            ...                            expend_var_name='E', output_var_name='Y')
        '''
        self.baseline_data = baseline_data
        self.baseline_columns = baseline_data.columns

        self.meta_data = _MetaData(imp_var_name=imp_var_name,
                                   exp_var_name=exp_var_name,
                                   year_var_name=year_var_name,
                                   trade_var_name=trade_var_name,
                                   sector_var_name=None,
                                   expend_var_name=expend_var_name,
                                   output_var_name=output_var_name)
        self.specification = Specification(lhs_var = trade_var_name, rhs_var=None)
        self.country_fixed_effects = country_fixed_effects

        # Convert year column to string to insure consistent dtype (OneSectorGE expects str)
        self.baseline_data[year_var_name] = self.baseline_data[year_var_name].astype(str)

    def __repr__(self):
        return("Baseline data columns: {}\n"
               "Baseline data dimensions: {}".format(", ".join(self.baseline_columns), self.baseline_data.shape))

# ToDo: Potentially Needed:
# ToDo: estimation_model.estimation_data._meta_data,
# ToDo: cost_coeff_values
# ToDo: estimation_model.estimation_data.data_frame
# ToDo: estimation_model.specification.fixed_effects
# ToDo: self._estimation_model.specification

class _MetaData(object):
    '''
    A class that contains certain information to simplify its passing around different procedures (from GME).
    '''

    def __init__(self,
                 imp_var_name: str = 'importer',
                 exp_var_name: str = 'exporter',
                 year_var_name: str = 'year',
                 trade_var_name: str = None,
                 sector_var_name: str = None,
                 expend_var_name: str = None,
                 output_var_name: str = None,
                 ):
        self.imp_var_name = imp_var_name
        self.exp_var_name = exp_var_name
        self.year_var_name = year_var_name
        self.trade_var_name = trade_var_name
        self.sector_var_name = sector_var_name
        self.expend_var_name = expend_var_name
        self.output_var_name = output_var_name

    def __repr__(self):
        return "imp_var_name: {0} \n" \
               "exp_var_name: {1} \n" \
               "year_var_name: {2} \n" \
               "trade_var_name: {3} \n" \
               "sector_var_name: {4} \n" \
               "expend_var_name: {5}\n" \
               "output_var_name: {6} \n" \
            .format(self.imp_var_name,
                    self.exp_var_name,
                    self.year_var_name,
                    self.trade_var_name,
                    self.sector_var_name,
                    self.expend_var_name,
                    self.output_var_name)


class Specification(object):
    def __init__(self,
                 # spec_name:str = 'default_name',
                 lhs_var: str = None,
                 rhs_var: List[str] = None,
                 # sector_by_sector: bool = False,
                 # drop_imp_exp: List[str] = [],
                 # drop_imp: List[str] = [],
                 # drop_exp: List[str] = [],
                 # keep_imp_exp: List[str] = [],
                 # keep_imp: List[str] = [],
                 # keep_exp: List[str] = [],
                 # drop_years: List[str] = [],
                 # keep_years: List[str] = [],
                 # drop_missing: bool = False,
                 # variables_to_drop_missing: List[str] = [],
                 fixed_effects:List[str] = [],
                 omit_fixed_effect:List[str] = [],
                 # std_errors:str = 'HC1',
                 # iteration_limit:int = 1000,
                 # drop_intratrade:bool = True,
                 # cluster: bool=False,
                 # cluster_on: str=None,
                 # verbose:bool = True
                 ):

        # self.spec_name = spec_name
        self.lhs_var = lhs_var
        self.rhs_var = rhs_var
        # self.sector_by_sector = sector_by_sector
        # self.drop_imp_exp = drop_imp_exp
        # self.drop_imp = drop_imp
        # self.drop_exp = drop_exp
        # self.keep_imp_exp = keep_imp_exp
        # self.keep_imp = keep_imp
        # self.keep_exp = keep_exp
        # self.drop_years = drop_years
        # self.keep_years = keep_years
        # self.drop_missing = drop_missing
        # self.variables_to_drop_missing = variables_to_drop_missing
        # self.fixed_effects = fixed_effects
        # self.omit_fixed_effect = omit_fixed_effect
        # self.std_errors = std_errors
        # self.iteration_limit = iteration_limit
        # self.drop_intratrade = drop_intratrade
        # self.cluster=cluster
        # self.cluster_on=cluster_on
        # self.verbose = verbose