#!/bin/bash

# Run conflict detection under different navigation uncertainty scenarios
echo "Running conflict detection simulations..."

for nav_uncertainty in p v pv
do
    echo "----> Running with nav_uncertainty=$nav_uncertainty and vehicle_uncertainty=oi"
    python -m uncertainty_quantification.run_conf_detect_UQ $nav_uncertainty oi
done

echo "✅ All simulations completed."