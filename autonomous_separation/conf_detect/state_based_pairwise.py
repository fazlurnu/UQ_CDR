import numpy as np
from shapely.geometry import Point
from typing import Tuple
from types import ModuleType
import sys
from .conf_detect import ConflictDetection
from .state_based import StateBased 

# ==============================================================================
# 1. BASE CLASS & WRAPPER
# ==============================================================================

class ConflictDetection:
    def __init__(self): pass

class TrafficAdapter:
    def __init__(self, lat, lon, gs, trk, acid):
        self.id = acid
        self.lat = np.array([lat])
        self.lon = np.array([lon])
        self.gs = np.array([gs])
        self.trk = np.array([trk])
        self.alt = np.array([0.0] * len(acid))
        self.vs = np.array([0.0] * len(acid))

        self.ntraf = len(acid)

class StateBasedWrapper(ConflictDetection):
    def __init__(self):
        super().__init__()
        self._backend = StateBased()

    def conf_detect_hor(self, ownship_pos, ownship_gs, ownship_trk,
                        intruder_pos, intruder_gs, intruder_trk,
                        rpz, tlookahead) -> Tuple[float, float, float, float, bool]:
        
        # 1. Batch inputs into lists [Ownship, Intruder]
        lats = [ownship_pos.x, intruder_pos.x]
        lons = [ownship_pos.y, intruder_pos.y]
        gss  = [ownship_gs, intruder_gs]
        trks = [ownship_trk, intruder_trk]
        ids  = ["OWN", "INT"]

        # 2. Create ONE traffic object with 2 aircraft
        traf = TrafficAdapter(lats, lons, gss, trks, ids)

        # 3. Pass the SAME object as both arguments (All-vs-All detection)
        self._backend.detect(traf, traf, rpz=rpz, hpz=100, dtlookahead=tlookahead)
        
        # 4. Extract Results
        if hasattr(self._backend.tcpa, '__len__') and len(self._backend.tcpa) > 0:
            # A conflict was found in the matrix
            # Because it's a symmetric matrix (0 hits 1, and 1 hits 0), we just take the first valid hit.
            return (float(self._backend.tcpa[0]), 
                    float(self._backend.tLOS[0]), 
                    float(2 * self._backend.tcpa[0] - self._backend.tLOS[0]), 
                    float(self._backend.dcpa[0]), True)
        else:
            # No conflict found
            return (1e8, 1e8, -1e8, 1e9, False)