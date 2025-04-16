import sys

from uncertainty_quantification.monte_carlo_noise import ConflictResolutionSimulation

def main():
    """
    Orchestrates the entire conflict-resolution demonstration by:
      1. Selecting which uncertainties and sources are active
      2. Setting scenario noise parameters
      3. Running the main simulation loop (clustering + plotting + CSV logging).
    """
    # 1. Create a selector for user input

    if len(sys.argv) != 3:
        print("Usage: python main.py <nav_uncertainty> <vehicle_uncertainty>")
        print("  - nav_uncertainty: combination of s (speed), h (heading), p (position)")
        print("  - vehicle_uncertainty: combination 'o' (ownship) and 'i' (intruder)")
        
        print("Settings is set to default: shp oi")

        nav_uncertainty = 'shp'
        vehicle_uncertainty = 'oi'
    else:
        nav_uncertainty = sys.argv[1].lower().strip()  # e.g. "sh", "p", "s", "shp"
        vehicle_uncertainty = sys.argv[2].lower().strip()    # e.g. "o" or "i"

        print(f"Settings is set to: {nav_uncertainty} {vehicle_uncertainty}")

    # Validate nav_uncertainty letters
    allowed_uncertainty_letters = {'s', 'h', 'p'}
    if any(char not in allowed_uncertainty_letters for char in nav_uncertainty):
        raise ValueError("nav_uncertainty can only contain 's', 'h', 'p'")

    # Validate vehicle_uncertainty letters
    allowed_vehicle_letters = {'o', 'i'}
    if any(char not in allowed_vehicle_letters for char in vehicle_uncertainty):
        raise ValueError("vehicle_uncertainty can only be 'o' or 'i'")

    # 4. Initialize the conflict-resolution simulation
    print("I'm here")
    sim = ConflictResolutionSimulation(nav_uncertainty, vehicle_uncertainty)
    sim.run_simulation()

if __name__ == "__main__":
    main()