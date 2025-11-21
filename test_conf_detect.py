from shapely.geometry import Point
from autonomous_separation.conf_detect.state_based_pairwise import StateBasedWrapper
import numpy as np
from typing import Tuple

def meters_to_latlon(pos: Point, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Converts a local Cartesian Point(x,y) in meters to (Lat, Lon)
    based on a Reference Latitude and Longitude.
    """
    R_EARTH = 6371000.0 # Meters
    
    dlat = np.degrees(pos.x / R_EARTH)
    new_lat = ref_lat + dlat
    
    # dx = R * cos(lat) * dLon_rad
    radius_at_lat = R_EARTH * np.cos(np.radians(ref_lat))
    dlon = np.degrees(pos.y / radius_at_lat)
    new_lon = ref_lon + dlon
    
    return new_lat, new_lon


# ==========================================
# Inputs
# ==========================================

# 1. Create Shapely Points for ownship and intruder positions
own_pos = Point(0.0, 0.0) # here x is Northing, y is Easting
int_pos = Point(1000.0, 0.0) # here x is Northing, y is Easting

own_hdg = 0
int_hdg = 180
speed = 60.0      # m/s

own_latlon = meters_to_latlon(own_pos, ref_lat=0, ref_lon=0)
int_latlon = meters_to_latlon(int_pos, ref_lat=0, ref_lon=0)

# ==========================================
# Execution
# ==========================================

detector = StateBasedWrapper()

result = detector.conf_detect_hor(
    ownship_pos=own_latlon,       # Passing the Point object directly
    ownship_gs=speed,
    ownship_trk=own_hdg,
    intruder_pos=int_latlon,      # Passing the Point object directly
    intruder_gs=speed,
    intruder_trk=int_hdg,
    rpz=200,
    tlookahead=50
)

print("-" * 30)
print(f"Time CPA: {result[0]:.2f} s")
print(f"Time In:  {result[1]:.2f} s")
print(f"Time Out: {result[2]:.2f} s")
print(f"DCPA:     {result[3]:.2f} m")
print(f"Conflict: {result[4]}")