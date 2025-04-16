import numpy as np
from shapely.geometry import Point, LineString
from shapely.affinity import translate
from shapely.ops import nearest_points

from autonomous_separation.conf_reso.conf_reso import ConflictResolution

class VOResolution(ConflictResolution):
    def __init__(self):
        self.reso_name = "VO"

    def resolve(self, ownship_position, ownship_gs, ownship_trk,
                      intruder_position, intruder_gs, intruder_trk,
                      rpz, tlookahead, method=0, scale=1.0, resofach=1.05):

        rpz = np.max(rpz * resofach)
        ownship_trk_rad = np.radians(ownship_trk)
        intruder_trk_rad = np.radians(intruder_trk)
        
        tp_1, tp_2 = self.get_cc_tp(ownship_position, intruder_position, rpz)
        
        ownship_velocity = Point(ownship_position.x + ownship_gs * np.cos(ownship_trk_rad), ownship_position.y + ownship_gs * np.sin(ownship_trk_rad))
        intruder_velocity = Point(intruder_gs * np.cos(intruder_trk_rad), intruder_gs * np.sin(intruder_trk_rad))
        
        if((tp_1 != None) & (tp_2 != None)):
            vo_0 = translate(ownship_position, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
            vo_1 = translate(tp_1, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
            vo_2 = translate(tp_2, xoff = intruder_velocity.x, yoff = intruder_velocity.y)

            vo_line_1 = LineString([vo_0, vo_1])
            vo_line_2 = LineString([vo_0, vo_2])

            # method = 0: opt, 1: spd change, 2: hdg change
            if(method == 0):
                cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
                cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

                cp_1_dist = cp_1.distance(ownship_velocity)
                cp_2_dist = cp_2.distance(ownship_velocity)
                
                if(cp_1_dist <= cp_2_dist):
                    cp = cp_1
                else:
                    cp = cp_2
                    
            if(method == 1):
                if(ownship_gs > intruder_gs):
                    ownship_velocity = Point(ownship_position.x + ownship_gs * scale * np.cos(ownship_trk_rad), ownship_position.y + ownship_gs * scale * np.sin(ownship_trk_rad))
                else:
                    ownship_velocity = Point(ownship_position.x + ownship_gs / scale * np.cos(ownship_trk_rad), ownship_position.y + ownship_gs / scale * np.sin(ownship_trk_rad))
                
                cp_1 = nearest_points(vo_line_1, Point(ownship_velocity.x, ownship_velocity.y))[0]
                cp_2 = nearest_points(vo_line_2, Point(ownship_velocity.x, ownship_velocity.y))[0]

                cp_1_dist = cp_1.distance(ownship_velocity)
                cp_2_dist = cp_2.distance(ownship_velocity)

                if(cp_1_dist <= cp_2_dist):
                    cp = cp_1
                else:
                    cp = cp_2
                    
            if(method == 2):
                # scale = 0.95 why this scale here?
                ownship_velocity = Point(ownship_position.x + ownship_gs * scale * np.cos(ownship_trk_rad), ownship_position.y + ownship_gs * scale * np.sin(ownship_trk_rad))
                cp_1 = nearest_points(vo_line_1, Point(ownship_velocity.x, ownship_velocity.y))[0]
                cp_2 = nearest_points(vo_line_2, Point(ownship_velocity.x, ownship_velocity.y))[0]

                cp_1_dist = cp_1.distance(ownship_velocity)
                cp_2_dist = cp_2.distance(ownship_velocity)

                if(cp_1_dist <= cp_2_dist):
                    cp = cp_1
                else:
                    cp = cp_2
                    
            if(method == 4):
                cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
                cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

                curr_hdg = np.degrees(np.arctan2(ownship_velocity.y, ownship_velocity.x))
                hdg_1 = np.degrees(np.arctan2(cp_1.y, cp_1.x))
                hdg_2 = np.degrees(np.arctan2(cp_2.y, cp_2.x))

                cp = self.choose_right_vector(curr_hdg, hdg_1, hdg_2, cp_1, cp_2)

                gs_own = np.sqrt(ownship_velocity.x **2 + ownship_velocity.y **2)
                hdg_own = np.arctan2(ownship_velocity.y, ownship_velocity.x)

                gs_int = np.sqrt(intruder_velocity.x **2 + intruder_velocity.y **2)
                hdg_int = np.arctan2(intruder_velocity.y, intruder_velocity.x)

                do_maneuver = self.do_maneuver_check(gs_own, hdg_own, gs_int, hdg_int)

                if(do_maneuver):
                    cp = cp
                else:
                    cp = ownship_velocity

            return cp.x - ownship_position.x, cp.y - ownship_position.y
        else:
            return ownship_velocity.x - ownship_position.x, ownship_velocity.y - ownship_position.y

    # --- Helper Methods ---
    # Here check whether ownship should do a manuever or not
    # This is based on the SVO algorithm
    # The bearing angle can be redefined later
    def do_maneuver_check(self, gs_own, hdg_own, gs_int, hdg_int):      
        vn_int = gs_int * np.cos(np.radians(hdg_int))
        ve_int = gs_int * np.sin(np.radians(hdg_int))

        vn_int_local = vn_int * np.cos(np.radians(-hdg_own)) - ve_int * np.sin(np.radians(-hdg_own))
        ve_int_local = vn_int * np.sin(np.radians(-hdg_own)) + ve_int * np.cos(np.radians(-hdg_own))

        v_brg_A = np.degrees(np.arctan2(ve_int_local, vn_int_local))

        if -70 <= v_brg_A <= 70:
            return gs_own >= gs_int
        elif 70 < v_brg_A <= 160:
            return False
        else:
            return True

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
        diff1 = self.angle_difference(current, hdg_1)
        diff2 = self.angle_difference(current, hdg_2)
        return vector_1 if diff1 < diff2 else vector_2

    def get_cc_tp(self, ownship_position, intruder_position, rpz):
        dx = intruder_position.x - ownship_position.x
        dy = intruder_position.y - ownship_position.y

        d = np.sqrt(dx**2 + dy**2)

        if(d > rpz):
            theta = np.arctan2(dy, dx)
            beta = np.arcsin(rpz/d)
            side = np.sqrt(d**2 - rpz**2)

            tp_1_x = ownship_position.x + side * np.cos(theta - beta)
            tp_1_y = ownship_position.y + side * np.sin(theta - beta)
            tp_2_x = ownship_position.x + side * np.cos(theta + beta)
            tp_2_y = ownship_position.y + side * np.sin(theta + beta)

            return Point(tp_1_x, tp_1_y), Point(tp_2_x, tp_2_y)
        
        else:
            return None, None
