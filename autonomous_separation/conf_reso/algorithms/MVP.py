import numpy as np
from shapely.geometry import Point

from autonomous_separation.conf_reso.conf_reso import ConflictResolution

class MVPResolution(ConflictResolution):
    def __init__(self):
        self.reso_name = "MVP"

    def resolve(self, ownship_position, ownship_gs, ownship_trk,
                      intruder_position, intruder_gs, intruder_trk,
                      rpz, tlookahead=15, resofach=1.05):

        rpz_m = np.max(rpz * resofach)
        
        dx = ownship_position.x - intruder_position.x
        dy = ownship_position.y - intruder_position.y
        dist = np.sqrt(dx**2 + dy**2)
        qdr = np.arctan2(dy, dx)
        drel = np.array([np.sin(qdr) * dist, np.cos(qdr) * dist])

        ownship_trk_rad = np.radians(ownship_trk)
        intruder_trk_rad = np.radians(intruder_trk)
        
        ownship_velocity = Point(ownship_gs * np.cos(ownship_trk_rad),
                                 ownship_gs * np.sin(ownship_trk_rad))
        intruder_velocity = Point(intruder_gs * np.cos(intruder_trk_rad),
                                  intruder_gs * np.sin(intruder_trk_rad))

        dvx = ownship_velocity.x - intruder_velocity.x
        dvy = ownship_velocity.y - intruder_velocity.y
        tcpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)

        v1 = np.array([ownship_velocity.y, ownship_velocity.x])
        v2 = np.array([intruder_velocity.y, intruder_velocity.x])
        vrel = v1 - v2

        dcpa = drel + vrel * tcpa
        dabsH = np.sqrt(dcpa[0]**2 + dcpa[1]**2)
        iH = rpz_m - dabsH

        threshold = 0.001
        if dabsH <= threshold:
            dabsH = threshold
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH

        if rpz_m < dist and dabsH < dist:
            erratum = np.cos(np.arcsin(rpz_m / dist) - np.arcsin(dabsH / dist))
            dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

        dv = np.array([dv1, dv2, 0])
        return ownship_velocity.x + dv[1], ownship_velocity.y + dv[0]
    
    def resolve_with_dcpa(self, ownship_position, ownship_gs, ownship_trk,
                      intruder_position, intruder_gs, intruder_trk,
                      rpz, tlookahead, resofach=1.05):

        rpz_m = np.max(rpz * resofach)
        
        dx = ownship_position.x - intruder_position.x
        dy = ownship_position.y - intruder_position.y
        dist = np.sqrt(dx**2 + dy**2)
        qdr = np.arctan2(dy, dx)
        drel = np.array([np.sin(qdr) * dist, np.cos(qdr) * dist])

        ownship_trk_rad = np.radians(ownship_trk)
        intruder_trk_rad = np.radians(intruder_trk)
        
        ownship_velocity = Point(ownship_gs * np.cos(ownship_trk_rad),
                                 ownship_gs * np.sin(ownship_trk_rad))
        intruder_velocity = Point(intruder_gs * np.cos(intruder_trk_rad),
                                  intruder_gs * np.sin(intruder_trk_rad))

        dvx = ownship_velocity.x - intruder_velocity.x
        dvy = ownship_velocity.y - intruder_velocity.y
        tcpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)

        v1 = np.array([ownship_velocity.y, ownship_velocity.x])
        v2 = np.array([intruder_velocity.y, intruder_velocity.x])
        vrel = v1 - v2

        dcpa = drel + vrel * tcpa
        dabsH = np.sqrt(dcpa[0]**2 + dcpa[1]**2)
        iH = rpz_m - dabsH

        threshold = 0.001
        if dabsH <= threshold:
            dabsH = threshold
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH

        if rpz_m < dist and dabsH < dist:
            erratum = np.cos(np.arcsin(rpz_m / dist) - np.arcsin(dabsH / dist))
            dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

        dv = np.array([dv1, dv2, 0])
        return ownship_velocity.x + dv[1], ownship_velocity.y + dv[0], dcpa
