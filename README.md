## Running the Tests
Run each test module from the project root:

python -m test.test_cre_conf
python -m test.test_conf_detect
python -m test.test_conf_reso

## Running the Uncertainty-Quantification Module
The main script accepts different run modes (p, v, or pv) and an oi parameter.

Usage:
python -m uncertainty_quantification.main <p/v/pv> <oi>

Arguments:
- p — run in “p” mode
- v — run in “v” mode
- pv — run both p and v modes
- oi — object/identifier for the experiment

Example:
python -m uncertainty_quantification.main pv my_experiment