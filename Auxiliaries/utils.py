import numpy as np
import math

def quaternion_from_euler(ai, aj, ak): # RPY(roll, pitch, yaw)
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q # [x, y, z, w]


def quat2euler(q): # q, as np.array() [qw, qx, qy, qz]
    
    q0, q1, q2, q3 = q.squeeze().tolist()
    
    m=np.eye(3,3)
    m[0,0] = 1.0 - 2.0*(q2*q2 + q3*q3)
    m[0,1] = 2.0*(q1*q2 - q0*q3)
    m[0,2] = 2.0*(q1*q3 + q0*q2)
    m[1,0] = 2.0*(q1*q2 + q0*q3)
    m[1,1] = 1.0 - 2.0*(q1*q1 + q3*q3)
    m[1,2] = 2.0*(q2*q3 - q0*q1)
    m[2,0] = 2.0*(q1*q3 - q0*q2)
    m[2,1] = 2.0*(q2*q3 + q0*q1)
    m[2,2] = 1.0 - 2.0*(q1*q1 + q2*q2)
    phi = math.atan2(m[2,1], m[2,2])
    theta = -math.asin(m[2,0])
    psi = math.atan2(m[1,0], m[0,0])
    return phi, theta, psi # RPY(roll, pitch, yaw)

def lonlat2xyz(lat, lon, lat0 = 40.07217788696289, lon0 = -88.20980072021484): 
    # WGS84 ellipsoid constants:
    a = 6378137
    b = 6356752.3142
    e = math.sqrt(1-b**2/a**2)
    
    x = a*math.cos(math.radians(lat0))*math.radians(lon-lon0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,0.5)
    y = a*(1 - e**2)*math.radians(lat-lat0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,1.5)
    
    return x, y # x and y coordinates in a reference frame with the origin in lat0, lon0

def euler_to_quaternion(r):
    (roll, pitch, yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]