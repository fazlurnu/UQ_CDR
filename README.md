# README

## Running the Tests

Run each test module from the project root:

``` bash
python -m test.test_cre_conf
python -m test.test_conf_detect
python -m test.test_conf_reso
```

## Running the Uncertainty-Quantification Module

The main script supports three run modes and an ownship/intruder
selector.

**Usage**

``` bash
python -m uncertainty_quantification.main <p|v|pv> <oi>
```

**Arguments**

-   **p** --- position uncertainty\
-   **v** --- velocity uncertainty\
-   **pv** --- position & velocity uncertainty\
-   **oi** --- ownship/intruder

**Example**

``` bash
python -m uncertainty_quantification.main pv oi
```

## Running the Pairwise Simulation Module

Run the simulation by calling its main module with three arguments:

``` bash
python -m pairwise_sim.main <nav_uncertainty> <vehicle_uncertainty> <conf_reso_algo_select>
```

### Arguments

-   **nav_uncertainty**
    -   `p` --- position uncertainty\
    -   `v` --- velocity uncertainty\
    -   Use combinations such as `p`, `v`, or `pv`
-   **vehicle_uncertainty**
    -   `o` --- ownship\
    -   `i` --- intruder
-   **conf_reso_algo_select**
    -   `m` --- MVP\
    -   `v` --- VO

**Example**

``` bash
python -m pairwise_sim.main pv o m
```

**Defaults (when no arguments are provided)**

``` bash
pv oi m
```