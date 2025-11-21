import numpy as np
from shapely.geometry import Point
from .cr_mvp import MVP
from autonomous_separation.conf_reso.conf_reso import ConflictResolution

class MVPResolution(ConflictResolution):
    def __init__(self):
        self.reso_name = "MVP"
        self.algo = MVP() # Instance of the BlueSky logic

    def _prepare_data(self, own_p, own_gs, own_trk, int_p, int_gs, int_trk, rpz, tlookahead):
        """
        Prepares Mock Traffic objects and calculates geometric parameters (dist, qdr, tcpa)
        required by the BlueSky MVP function.
        """
        # 1. Create Mock Traffic Arrays (Index 0=Ownship, Index 1=Intruder)
        class MockTraffic:
            def __init__(self):
                self.lat = np.zeros(2) # Not used in core MVP logic
                self.lon = np.zeros(2) # Not used in core MVP logic
                self.alt = np.zeros(2) # Assuming flat 2D for this wrapper
                self.vs = np.zeros(2)
                self.gseast = np.zeros(2)
                self.gsnorth = np.zeros(2)

        traf = MockTraffic()
        
        # Convert Track/Speed to East/North components
        # Note: Aviation Track 0 deg = North (Y), 90 deg = East (X)
        def get_components(gs, trk):
            rad = np.radians(trk)
            return gs * np.cos(rad), gs * np.sin(rad) # (East, North)

        ve_own, vn_own = get_components(own_gs, own_trk)
        ve_int, vn_int = get_components(int_gs, int_trk)

        traf.gseast[0], traf.gsnorth[0] = ve_own, vn_own
        traf.gseast[1], traf.gsnorth[1] = ve_int, vn_int

        # 2. Create Mock Config
        class MockConf:
            def __init__(self):
                self.rpz = np.array([rpz, rpz])
                self.hpz = np.array([100, 100])
                self.dtlookahead = np.array([tlookahead, tlookahead])
        
        conf = MockConf()

        # 3. Calculate Geometry (Dist, QDR, TCPA)
        # MVP expects these as inputs.
        dx = int_p.x - own_p.x
        dy = int_p.y - own_p.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # QDR: Bearing from Ownship to Intruder (Degrees, Clockwise from North)
        # atan2(dx, dy) gives radians from North (where dx is East, dy is North)
        qdr_rad = np.arctan2(dx, dy)
        qdr = np.degrees(qdr_rad) % 360

        # TCPA Calculation
        dvx = ve_own - ve_int
        dvy = vn_own - vn_int
        vrel_sq = dvx**2 + dvy**2
        
        if vrel_sq < 1e-6:
            tcpa = 0
        else:
            # Note: Formula is -(dr . dv) / |dv|^2
            # But we need relative position and relative velocity consistent
            # drel = int - own (dx, dy)
            # vrel = int - own (ve_int-ve_own, vn_int-vn_own) = -(dvx, dvy)
            # It matches standard TCPA formulation
            tcpa = -(dx * (ve_int - ve_own) + dy * (vn_int - vn_own)) / \
                   ((ve_int - ve_own)**2 + (vn_int - vn_own)**2)

        return traf, conf, qdr, dist, tcpa

    def resolve(self, ownship_position, ownship_gs, ownship_trk,
                      intruder_position, intruder_gs, intruder_trk,
                      rpz, tlookahead=300, resofach=1.05):
        
        # Update safety factor if passed dynamically
        self.algo.resofach = resofach

        # Prepare data
        traf, conf, qdr, dist, tcpa = self._prepare_data(
            ownship_position, ownship_gs, ownship_trk,
            intruder_position, intruder_gs, intruder_trk,
            rpz, tlookahead
        )

        # Call the original MVP function
        # ownship=traf, intruder=traf (BlueSky passes the whole traffic array object)
        # idx1=0 (Ownship), idx2=1 (Intruder)
        dv_mvp, _ = self.algo.MVP(traf, traf, conf, qdr, dist, tcpa, 0, 0, 1)

        # Apply resolution
        # The BlueSky logic: dv_mvp is the vector to ADD to intruder or SUBTRACT from ownship
        # to solve conflict. Since we are moving ownship:
        # New_V = Old_V - dv_mvp (See applyprio/resolve logic in original file)
        
        current_ve = traf.gseast[0]
        current_vn = traf.gsnorth[0]
        
        # dv_mvp[0] is East change, dv_mvp[1] is North change
        # Note: standard MVP logic often halves the resolution for cooperative.
        # Assuming non-cooperative (only I move), we subtract the full vector.
        new_ve = current_ve - dv_mvp[0]
        new_vn = current_vn - dv_mvp[1]

        return new_ve, new_vn

    def resolve_with_dcpa(self, ownship_position, ownship_gs, ownship_trk,
                      intruder_position, intruder_gs, intruder_trk,
                      rpz, tlookahead, resofach=1.05):
        
        # Standard resolution
        vx, vy = self.resolve(
            ownship_position, ownship_gs, ownship_trk,
            intruder_position, intruder_gs, intruder_trk,
            rpz, tlookahead, resofach
        )

        # Re-calculate DCPA for return
        # (We calculate this inside resolve, but to keep signatures clean we re-do or extract it)
        # Ideally, we calculate it manually here to ensure we return the vector based on original paths
        dx = intruder_position.x - ownship_position.x
        dy = intruder_position.y - ownship_position.y
        
        ve_own = ownship_gs * np.sin(np.radians(ownship_trk))
        vn_own = ownship_gs * np.cos(np.radians(ownship_trk))
        ve_int = intruder_gs * np.sin(np.radians(intruder_trk))
        vn_int = intruder_gs * np.cos(np.radians(intruder_trk))
        
        dvx = ve_int - ve_own
        dvy = vn_int - vn_own
        
        if (dvx**2 + dvy**2) < 1e-6:
            tcpa = 0
        else:
            tcpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)

        dcpa = np.array([dx + dvx * tcpa, dy + dvy * tcpa])

        return vx, vy, dcpa