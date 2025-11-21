from autonomous_separation.traffic_generator import cre_conflict

X_OWN = 0 # in meter
Y_OWN = 0 # in meter
TRK_OWN = 0             # Heading of ownship in degrees
GS_OWN = 20             # Groundspeed of ownship

RPZ = 50                # Radius Protected Zone (example)
TLOSH = 15              # Time to Loss of Separation

SPD_INTRUDER = 15
dpsi_val = 10
dcpa_val = 0

x_int, y_int, trk_int, gs_int = cre_conflict(
    X_OWN, Y_OWN, TRK_OWN, GS_OWN, 
    dpsi_val, dcpa_val, TLOSH, SPD_INTRUDER, RPZ
)

print(x_int, y_int, trk_int, gs_int)