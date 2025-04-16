import numpy as np
from shapely.geometry import Point
from typing import Tuple

from autonomous_separation.conf_detect.conf_detect import ConflictDetection

class StateBasedDetection(ConflictDetection):
    def __init__(self):
        super().__init__()
        self.detector_name = "State-Based"

    def conf_detect_hor(self,
        ownship_position: Point,
        ownship_gs: float,
        ownship_trk: float,
        intruder_position: Point,
        intruder_gs: float,
        intruder_trk: float,
        rpz: float,
        tlookahead: float
    ) -> Tuple[float, float, float, bool]:

        
        own_heading_rad = np.radians(ownship_trk)
        int_heading_rad = np.radians(intruder_trk)

        own_velocity = Point(ownship_gs * np.cos(own_heading_rad),
                             ownship_gs * np.sin(own_heading_rad))
        int_velocity = Point(intruder_gs * np.cos(int_heading_rad),
                             intruder_gs * np.sin(int_heading_rad))

        dx = ownship_position.x - intruder_position.x
        dy = ownship_position.y - intruder_position.y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        dvx = own_velocity.x - int_velocity.x
        dvy = own_velocity.y - int_velocity.y
        vrel = np.sqrt(dvx ** 2 + dvy ** 2)

        tcpa = -(dx * dvx + dy * dvy) / (dvx ** 2 + dvy ** 2)

        dcpa2 = (dist*dist - tcpa * tcpa * vrel * vrel)
        epsilon = 1e-9

        if np.isclose(dcpa2, 0.0, atol=epsilon):
            dcpa = 0.0
        else:
            dcpa = np.sqrt(dcpa2)

        LOS = dcpa < rpz

        if LOS:
            tcrosshi = tcpa + np.sqrt(rpz ** 2 - dcpa ** 2) / vrel
            tcrosslo = tcpa - np.sqrt(rpz ** 2 - dcpa ** 2) / vrel

            tin = max(0.0, min(tcrosslo, tcrosshi))
            tout = max(tcrosslo, tcrosshi)

            is_in_conflict = (dcpa < rpz) and (tin < tlookahead)

            return (tin, tout, dcpa, is_in_conflict)
        else:
            return (0.0, 1e4, dcpa, False)
