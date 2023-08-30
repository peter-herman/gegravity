__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "July 13, 2023"
__Description__ = '''Class to handle baseline data inputs without using the GME package.'''

from pandas import DataFrame
from typing import List
# from src.gegravity.OneSectorGE import _GEMetaData


class BaselineModel(object):
    def __init__(self,
                 baseline_data:DataFrame,
                 imp_var_name: str = 'importer',
                 exp_var_name: str = 'exporter',
                 year_var_name: str = 'year',
                 trade_var_name: str = None,
                 sector_var_name: str = None,
                 expend_var_name: str = None,
                 output_var_name: str = None,):
                 # cost_coeff_values: CostCoeffs = None):
        self.baseline_data = baseline_data
        self.meta_data = _MetaData(imp_var_name=imp_var_name,
                                   exp_var_name=exp_var_name,
                                   year_var_name=year_var_name,
                                   trade_var_name=trade_var_name,
                                   sector_var_name=sector_var_name,
                                   expend_var_name=expend_var_name,
                                   output_var_name=output_var_name)
        self.specification =

# ToDo: Needed:
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
        if lhs_var is None:
            raise ValueError('lhs_var (left hand side variable) must be specified.')

        # self.spec_name = spec_name
        # self.lhs_var = lhs_var
        # self.rhs_var = rhs_var
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