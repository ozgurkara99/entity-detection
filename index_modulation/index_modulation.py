import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
# util functions 
def parametrize(t0, t1, p):
    par = np.transpose(np.array(t0)) + np.transpose(np.array(t1) - np.array(t0)) * p
    return par

def euclidian_dist(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return np.sqrt(abs(np.sum(((arr1 - arr2) ** 2), axis=-1)))

def cart_to_spherical(x, y, z):
    hxy = np.hypot(x, y)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el

def mirror_point_over_plane(a, b, c, d, x1, y1, z1):  
    k =(-a * x1 - b * y1 - c * z1 - d)/float((a * a + b * b + c * c)) 
    x2 = a * k + x1 
    y2 = b * k + y1 
    z2 = c * k + z1 
    x3 = 2 * x2-x1 
    y3 = 2 * y2-y1 
    z3 = 2 * z2-z1 
    return [x3, y3, z3]

class Simulation():
    def __init__(self, num_of_tx=8, num_of_rx=1, r_rx=5, r_tx=4.5, D=79.4,step=0.001, time=0.75, d_yz=10, d_x=10, center_of_rx = [0,0,0]):
        print("Simulation is starting...")
        self.num_of_tx = num_of_tx
        self.num_of_rx = num_of_rx
        self.r_rx = r_rx
        self.r_tx = r_tx
        self.D = D
        self.step = step
        self.time = time
        self.d_yz = d_yz
        self.d_x = d_x
        self.center_of_rx = center_of_rx
        self.center_of_UCA = [center_of_rx[0] + d_x, center_of_rx[1], center_of_rx[2]]
    
    def start_simulation(self):
        pass
   
    def find_azimuth_elevation(self, coords):
        coords = np.array(coords)
        azimuth, elevation = cart_to_spherical(coords[0] - self.center_of_rx[0], coords[1] - self.center_of_rx[1], coords[2] - self.center_of_rx[2])
        return azimuth, elevation

    def find_nearest_coord(self, t0, t1, center_of_sphere, r):
        h = np.array([parametrize(t0,t1,x) for x in np.arange(0,1,0.0001)])
        a = np.full((h.shape[0], h.shape[1]), center_of_sphere)
        x = euclidian_dist(a,h)
        return np.array([h[np.argmin(abs(abs(x) - r)),0], h[np.argmin(abs(abs(x) - r)),1], h[np.argmin(abs(abs(x) - r)),2]])
    
    def tx_positions(self):
        d = np.arange(0,1,1/(self.num_of_tx))
        theta = d * 2 * PI
        y = self.center_of_UCA[1] + 4 * np.cos(theta)
        z = self.center_of_UCA[2] + 4 * np.sin(theta)
        x = np.full((d.shape[0]), self.center_of_UCA[0])
        return np.stack((x,y,z), axis=-1)
    
    def tx_reflection(self, t0, t1, center_of_tx):
        coords = self.find_nearest_coord(t0,t1,center_of_tx, self.r_tx )
        a = coords[0] - center_of_tx[0]
        b = coords[1] - center_of_tx[1]
        c = coords[2] - center_of_tx[2]
        d = -1 * (a * coords[0] + b * coords[1] + c * coords[2])
        reflected = mirror_point_over_plane(a, b, c, d, t1[0], t1[1], t1[2])
        return np.array(reflected), np.array(coords)
        

m = Simulation()
coords = m.find_nearest_coord([0,7,0] , [0,3,0], m.center_of_rx, m.r_rx)
az, el = m.find_azimuth_elevation(coords)
pos_tx = m.tx_positions()
reflected, coord = m.tx_reflection([0,6,7] , [0,0,4], [0,0,0])


plt.gcf().gca().add_artist(plt.Circle((0, 0), 4.5, color='r', fill=False))
h = np.array([[6,7], [0,4], [0,0], [reflected[1], reflected[2]], [coord[1],coord[2]]])
plt.scatter(h[:,0], h[:,1])
dist1 = np.sum(abs((coord - np.array([0,1,2]))**2), axis=-1)
dist2 = np.sum(abs((coord - reflected)**2), axis=-1)
plt.plot([6,0],[7,4],'ro-')
plt.plot([0,2*coord[1]],[0,2*coord[2]],'g-')
plt.plot([reflected[1], coord[1]], [reflected[2], coord[2]], '-bo')
plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(pos_tx[:,1],pos_tx[:,2])
plt.show()
