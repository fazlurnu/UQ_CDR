## Running the Tests
Run each test module from the project root:

`python -m test.test_cre_conf`
`python -m test.test_conf_detect`
`python -m test.test_conf_reso`

## Running the Uncertainty-Quantification Module
The main script accepts different run modes (p, v, or pv) and an oi parameter.

Usage:
`python -m uncertainty_quantification.main <p/v/pv> <oi>`

Arguments:
- p — position uncertainty
- v — velocity uncertainty
- pv — position & velocity uncertainty
- oi — ownship/intruder

Example:
python -m uncertainty_quantification.main pv oi