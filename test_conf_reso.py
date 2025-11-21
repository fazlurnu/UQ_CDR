import bluesky as bs
from shapely.geometry import Point
import numpy as np

# Import the resolution classes
# from autonomous_separation.conf_reso.algorithms.MVP_pairwise import MVPResolution
from autonomous_separation.conf_reso.algorithms.VO_pairwise import VOResolution
from autonomous_separation.conf_reso.algorithms.MVP_pairwise import MVPResolution

# Define dummy input for testing
ownship_position = Point(0, 0)
intruder_position = Point(1000, 20)

ownship_gs = 60  # m/s
intruder_gs = 60  # m/s

ownship_heading = 0   # degrees
intruder_heading = 180  # degrees

rpz = np.array([200])  # protected zone radius in meters

tlookahead = 15  # seconds
bs.init(mode='sim', detached=True)

def test_mvp():
    print("\n=== Testing MVPResolution ===")
    mvp = MVPResolution()
    vx, vy = mvp.resolve(
        ownship_position=ownship_position,
        ownship_gs=ownship_gs,
        ownship_trk=ownship_heading,
        intruder_position=intruder_position,
        intruder_gs=intruder_gs,
        intruder_trk=intruder_heading,
        rpz=rpz,
        tlookahead=tlookahead
    )
    print("Resolved Velocity (x, y, mag):", vx, vy, np.sqrt(vx**2 + vy**2))


def test_vo():
    print("\n=== Testing VOResolution ===")
    vo = VOResolution()
    vx, vy = vo.resolve(
        ownship_position=ownship_position,
        ownship_gs=ownship_gs,
        ownship_trk=ownship_heading,
        intruder_position=intruder_position,
        intruder_gs=intruder_gs,
        intruder_trk=intruder_heading,
        rpz=rpz,
        tlookahead=tlookahead,
        method=0  # try 1, 2, or 4 for different strategies
    )
    print("Resolved Velocity (x, y, mag):", vx, vy, np.sqrt(vx**2 + vy**2))


if __name__ == "__main__":
    test_mvp()
    test_vo()
