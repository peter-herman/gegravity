# gegravity: General Equilibrium Gravity Modeling in Python
--------------------
gegravity is a Python package containing tools used to estimate general equilibrium (GE) structural gravity models and simulate counterfactual experiments. The package is based on the well established version of the gravity model described by Yotov, Piermartini, Monteiro, and Larch (2016) *An Advanced Guide to Trade  Policy  Analysis:  The  Structural  Gravity  Model*. It implements the structural GE gravity model in a general, robust, and easy to use manner in an effort to make GE gravity modeling more accessible for researchers and policy analysts.

The package provides several useful tools for structural gravity modeling.

1. It computes theory consistent estimates of the structural multilateral resistance terms of Anderson and van Wincoop (2003) "Gravity with Gravitas" from standard econometric gravity results.

2. It simulates GE effects from counterfactual experiments such as new trade agreements or changes to other trade costs. The model can be flexibly used to study aggregate or sector level trade as well as many different types of trade costs and policies.

3. It conducts Monte Carlo simulations that provide a means to compute standard errors and other measures of statistical precision for the GE model results.

For more information about the GE gravity model, its implementation, and the various components of the package, see the companion paper ["gegravity: General Equilibrium Gravity Modeling in Python"](https://usitc.gov/sites/default/files/publications/332/working_papers/herman_2021_gegravity_modeling_in_python.pdf) as well as the [technical documentation](https://peter-herman.github.io/gegravity/).

## Versions
* **Version 0.3** (Jun. 2024):
  * Package reworked to no longer rely on the GME package for data and parameter inputs. A new class, BaselineData, has been introduced as a simpler option for organizing and inputting baseline model data (trade flows, cost variables, expenditures, output, etc.). Similarly, gegravity class CostCoeffs is now the recommended way of inputting cost parameters (e.g. econometric coefficient values, standard errors, and variance/covariance matrices). GME inputs should generally still work for OneSectorGE models but now require the gme=True argument. MonteCarloGE models are no longer able to use GME inputs. 
  * New MonteCarloGE features added to help identify and resolve issues with trials that fail to solve, including check_omr_rescale fucntionality and option to replace failed trials.
  * Minor bug fixes and improvements to reliability.
* **Version 0.2** (Dec. 2022): Improved OneSectorGE speed and updated compatibility with dependencies.
* **Version 0.1** (Apr. 2021): Initial release.



## Citation and license
The package is publicly available and free to use under the MIT license. Users are asked to please cite the following document,

Herman, Peter (2021) "gegravity: General Equilibrium Gravity Modeling in Python." USITC Economics Working Paper 2021-04-B.
