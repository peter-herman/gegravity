__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "July 13, 2023"
__Description__ = '''Class to handle baseline data inputs without using the GME package.'''

from pandas import DataFrame
# from src.gegravity.OneSectorGE import _GEMetaData
from src.gegravity.OneSectorGE import CostCoeffs

class BaselineModel(object):
    def __init__(self,
                 baseline_data:DataFrame,
                 imp_var_name: str = 'importer',
                 exp_var_name: str = 'exporter',
                 year_var_name: str = 'year',
                 trade_var_name: str = None,
                 sector_var_name: str = None,
                 expend_var_name: str = None,
                 output_var_name: str = None,
                 cost_coeff_values: CostCoeffs = None):
        self.baseline_data = baseline_data
        self.meta_data = _MetaData(imp_var_name=imp_var_name,
                                   exp_var_name=exp_var_name,
                                   year_var_name=year_var_name,
                                   trade_var_name=trade_var_name,
                                   sector_var_name=sector_var_name,
                                   expend_var_name=expend_var_name,
                                   output_var_name=output_var_name)

# Needed:
# estimation_model.estimation_data._meta_data,
# cost_coeff_values
# estimation_model.estimation_data.data_frame
# estimation_model.specification.fixed_effects
# self._estimation_model.specification

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