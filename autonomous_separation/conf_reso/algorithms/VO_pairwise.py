import numpy as np
from shapely.geometry import Point, LineString
from shapely.affinity import translate
from shapely.ops import nearest_points
from math import degrees, atan2, sqrt, sin, cos, radians

# Assuming cr_vo.py contains the VO class
from .cr_vo import VO 
from autonomous_separation.conf_reso.conf_reso import ConflictResolution

class VOResolution(ConflictResolution):
    def __init__(self):
        self.reso_name = "VO"
        self.algo = VO() # Instance of the core VO logic

    def resolve(self, ownship_pos, ownship_gs, ownship_trk,
                      intruder_pos, intruder_gs, intruder_trk,
                      rpz, tlookahead, method=0, scale=1.0, resofach=1.05):
        
        # Apply safety factor to Protected Zone
        rpz_m = rpz * resofach 

        # 1. Set up Positions and Velocities
        
        # Use OWN_P = Origin for VO geometry simplification
        own_p = Point(0, 0) 
        
        # Relative Position (Intruder - Ownship)
        dx = intruder_pos.x - ownship_pos.x
        dy = intruder_pos.y - ownship_pos.y
        int_p_rel = Point(dx, dy) 
        
        # Aviation Math: Vx (East) = Sin(trk), Vy (North) = Cos(trk)
        rad_own = np.radians(ownship_trk)
        rad_int = np.radians(intruder_trk)
        
        ve_own = ownship_gs * np.cos(rad_own)
        vn_own = ownship_gs * np.sin(rad_own)
        ve_int = intruder_gs * np.cos(rad_int)
        vn_int = intruder_gs * np.sin(rad_int)
        
        # Ownship Velocity Vector (Point at V_x, V_y relative to origin)
        own_v = Point(ve_own, vn_own)
        # Intruder Velocity Vector
        int_v = Point(ve_int, vn_int)

        # 2. Get Tangency Points
        # We call the external helper to get the tangency points
        tp_1, tp_2 = self.algo.get_cc_tp(own_p, int_p_rel, rpz_m)
        
        if((tp_1 is not None) and (tp_2 is not None)):
            
            # 3. Calculate Velocity Obstacle Geometry (VO)
            
            # VO origin is V_intruder
            vo_0 = Point(int_v.x, int_v.y)
            
            # Translate TP1 and TP2 by V_intruder
            vo_1 = translate(tp_1, xoff = int_v.x, yoff = int_v.y)
            vo_2 = translate(tp_2, xoff = int_v.x, yoff = int_v.y)

            vo_line_1 = LineString([vo_0, vo_1])
            vo_line_2 = LineString([vo_0, vo_2])
            
            # 4. Determine New Velocity (cp) based on Method
            
            # Initialize cp for fallback
            cp = own_v 
            
            # Method 0: Optimal (Closest point to current velocity)
            if method == 0:
                cp_1 = nearest_points(vo_line_1, own_v)[0]
                cp_2 = nearest_points(vo_line_2, own_v)[0]

                cp = cp_1 if cp_1.distance(own_v) <= cp_2.distance(own_v) else cp_2
            
            # Method 4: SVO/Priority Check (Requires additional logic)
            elif method == 4:
                cp_1 = nearest_points(vo_line_1, own_v)[0]
                cp_2 = nearest_points(vo_line_2, own_v)[0]

                # Current/Target headings need to be in degrees from North
                curr_hdg_rad = atan2(own_v.x, own_v.y) # atan2(East, North) -> angle from North
                curr_hdg = degrees(curr_hdg_rad) % 360

                hdg_1 = degrees(atan2(cp_1.x, cp_1.y)) % 360
                hdg_2 = degrees(atan2(cp_2.x, cp_2.y)) % 360

                cp = self.choose_right_vector(curr_hdg, hdg_1, hdg_2, cp_1, cp_2)

                # SVO-specific maneuver check (Do I maneuver or does the intruder?)
                do_maneuver = self.do_maneuver_check(ownship_gs, ownship_trk, intruder_gs, intruder_trk)

                if not do_maneuver:
                    cp = own_v # If I shouldn't maneuver, new velocity is old velocity
            
            # Methods 1/2 are complex and often modified versions of method 0/4;
            # they are not fully implemented here for brevity as they require modifying own_v 
            # *before* checking nearest points, which is unusual for a resolution function.
            # Fallback to current velocity if method is not 0 or 4.
            else:
                pass # cp remains own_v or is set by one of the methods above
            
            # New velocity components are the coordinates of cp (Vx, Vy)
            new_ve = cp.x
            new_vn = cp.y

            return new_ve, new_vn
        else:
            # Not in conflict, maintain current velocity
            return ve_own, vn_own

    # --- Helper Methods ---

    def do_maneuver_check(self, gs_own, hdg_own, gs_int, hdg_int):      
        """ SVO maneuver responsibility check (simplified version) """
        
        # Convert intruder velocity to ownship's local frame
        rad_own = np.radians(-hdg_own) # Rotation angle
        
        # Aviation Math: Sin=East, Cos=North
        vn_int = gs_int * np.cos(np.radians(hdg_int))
        ve_int = gs_int * np.sin(np.radians(hdg_int))

        # 2D Rotation: 
        vn_int_local = vn_int * np.cos(rad_own) - ve_int * np.sin(rad_own)
        ve_int_local = vn_int * np.sin(rad_own) + ve_int * np.cos(rad_own)

        # Bearing angle of intruder in ownship's frame (0 = forward)
        # atan2(East, North) gives angle from North (which is the forward axis in the local frame)
        v_brg_A = np.degrees(np.arctan2(ve_int_local, vn_int_local))

        # SVO Rules (simplified priority)
        if -70 <= v_brg_A <= 70:
            # Head-on/Overtaking from behind: Higher speed has priority (Maneuver if gs_own < gs_int)
            return gs_own <= gs_int # Maneuver if Ownship is slower
        elif 70 < v_brg_A <= 160:
            # Intruder coming from Ownship's right side (standard right-of-way)
            return True # Ownship maneuvers (intruder has priority)
        else: # Intruder coming from Ownship's left side
            return False # Ownship has priority (Intruder maneuvers)

    def normalize_angle(self, angle):
        while angle <= -180:
            angle += 360
        while angle > 180:
            angle -= 360
        return angle

    def angle_difference(self, current, target):
        diff = self.normalize_angle(target - current)
        if diff < 0:
            diff += 360
        return diff

    def choose_right_vector(self, current, hdg_1, hdg_2, vector_1, vector_2):
        """ Chooses the velocity vector (cp) that requires the smallest heading change. """
        diff1 = self.angle_difference(current, hdg_1)
        diff2 = self.angle_difference(current, hdg_2)
        return vector_1 if diff1 < diff2 else vector_2
    
    