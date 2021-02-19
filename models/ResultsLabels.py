__Author__ = "Peter Herman"
__Project__ = "gegravity"
__Created__ = "February 19, 2021"
__Description__ = '''Location to house the labels for different results '''
__all_ = ['ResultsLabels']



ResultsLabels = list()
'''
The following defines the labels used for various results throughout the package:

**'baseline modeled trade'** : Baseline modeled trade values (not the observed trade values from the source data).
'''


trade_results_labels = {
'baseline_modeled_trade':'baseline modeled trade',
'experiment_trade':'experiment trade',
'trade_change':'trade change (%)',
'trade_change_level':'trade change (observed level)',
'baseline_observed_trade':'baseline observed trade',
'experiment_observed_trade':'experiment observed trade'
}