import numpy as np

def cre_conflict(xref, yref, trkref, gsref,
                 dpsi, dcpa, tlosh, spd, rpz = 50):
    
    trkref_rad = np.radians(trkref)
    
    trk = trkref + dpsi
    trk_rad = np.radians(trk)
    
    gsx = spd * np.cos(trk_rad)
    gsy = spd * np.sin(trk_rad)
    
    vrelx = gsref * np.cos(trkref_rad) - gsx
    vrely = gsref * np.sin(trkref_rad) - gsy
    
    vrel = np.sqrt(vrelx*vrelx + vrely*vrely)
    
    if(dcpa == 0):
        drelcpa = (tlosh*vrel + np.sqrt(rpz*rpz - dcpa*dcpa))
    elif(dcpa > rpz):
        drelcpa = 0
    else:
        drelcpa = tlosh*vrel + np.sqrt(rpz*rpz - dcpa*dcpa)

    dist = np.sqrt(drelcpa*drelcpa + dcpa*dcpa)
    
    # Rotation matrix diagonal and cross elements for distance vector
    rd      = drelcpa / dist
    rx      = dcpa / dist
    # Rotate relative velocity vector to obtain intruder bearing
    brn     = np.degrees(np.arctan2(-rx * vrelx + rd * vrely,
                             rd * vrelx + rx * vrely))
    
    xint, yint = dist * np.cos(np.radians(brn)), dist * np.sin(np.radians(brn))

    return xint, yint, trk, spd