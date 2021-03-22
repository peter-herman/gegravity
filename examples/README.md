# Readme for sample_data_set.dlm

## Description
The sample data set is meant to provide a minimal working sample to conduct example GE gravity analyses. It is a cross-sectional dataset that provides information about trade (foreign and domestic), output, expenditure for 30 countries in 2006. It also contains several common trade cost proxies reflecting preferential trade agreements (PTAs), common borders, common languages, geographic distance, and international borders. 

## Sources
The sample data was derived from two sources. 

1. The trade, output, and expenditure data was derrived from a sample dataset accompanying Yotov, Piermartini, Monteiro, and Larch's (2016) *An Advanced Guide toTrade  Policy  Analysis:  The  Structural  Gravity  Model* (Online  Revised  Version).  WorldTrade Organization and the United Nations Conference on Trade and Development. The source data ("1_TradeWithoutBorder.dta") is available at https://vi.unctad.org/tpa/web/vol2/vol2home.html.

2. The trade cost proxies were sourced from the Dynamic Gravity Dataset of Gurevis and Herman (2018) "he dynamic gravity dataset:  1948-2016," Office of Economics Working Paper 2018-02-A, U.S. International Trade Commission.

## Variable Definitions

* **exporter**: ISO 3-digit identifier for the exporting country.
* **importer**: ISO 3-digit identifier for the importing country.
* **year**: Year of the observation.
* **trade**: Trade value.
* **Y**: Value of the exporter's output
* **E**: Value of the importer's expenditure
* **pta**: Indicator taking the value of 1 if both countries belonged to a preferential trade agreement, 0 otherwise.
* **contiguity**: Indicator taking the value of 1 if both countries shared a common border, 0 otherwise.
* **common_language**: Indicator taking the value of 1 if both countries shared a common language, 0 otherwise.
* **lndist**: Population weighted, greater circle distance between countries, in logs.
* **international**: Indicator taking the value 1 if the trade flow is international (i.e. exporter is not the same as importer), 0 otherwise.

 