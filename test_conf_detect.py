from shapely.geometry import Point
from autonomous_separation.conf_detect.state_based import StateBasedDetection

# Dummy input for testing
ownship_position = Point(0, 0)
intruder_position = Point(1000, 0)

ownship_gs = 60  # m/s
intruder_gs = 60  # m/s

ownship_heading = 0    # degrees
intruder_heading = 180  # degrees

rpz = 200  # protected zone radius in meters
tlookahead = 50  # seconds

def test_state_based_detection():
    print("\n=== Testing StateBasedDetection ===")
    cd = StateBasedDetection()

    tin, tout, dcpa, is_conflict = cd.conf_detect_hor(
        ownship_position=ownship_position,
        ownship_gs=ownship_gs,
        ownship_heading=ownship_heading,
        intruder_position=intruder_position,
        intruder_gs=intruder_gs,
        intruder_heading=intruder_heading,
        rpz=rpz,
        tlookahead=tlookahead
    )

    print("Time In:", tin)
    print("Time Out:", tout)
    print("DCPA:", dcpa)
    print("Conflict:", is_conflict)

if __name__ == "__main__":
    test_state_based_detection()
