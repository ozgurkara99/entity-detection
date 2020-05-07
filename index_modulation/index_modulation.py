import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
# util functions 
def parametrize(t0, t1, p):
    par = np.transpose(np.array(t0)) + np.transpose(np.array(t1) - np.array(t0)) * p
    return par

def euclidian_dist(arr1, arr2):
    return np.sqrt(abs(np.sum((arr1 ** 2),axis=1) - np.sum((arr2 ** 2), axis=1)))

def cart_to_spherical(x, y, z):
    hxy = np.hypot(x, y)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el



class Simulation():
    def __init__(self, num_of_tx=8, num_of_rx=1, r_rx=5, r_tx=1, D=79.4,step=0.001, time=0.75, d_yz=10, d_x=10, center_of_rx = [0,0,0]):
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

    def find_nearest_coord(self, t0, t1):
        h = np.array([parametrize(t0,t1,x) for x in np.arange(0,1,0.001)])
        a = np.full((h.shape[0], h.shape[1]),self.center_of_rx)
        x = euclidian_dist(a,h)
        return np.array([h[np.argmin(abs(abs(x) - self.r_rx)),0], h[np.argmin(abs(abs(x) - self.r_rx)),1], h[np.argmin(abs(abs(x) - self.r_rx)),2]])
    
    def rx_positions(self):
        d = np.arange(0,1,1/(self.num_of_tx))
        theta = d * 2 * PI
        y = self.center_of_UCA[1] + 4 * np.cos(theta)
        z = self.center_of_UCA[2] + 4 * np.sin(theta)
        x = np.full((d.shape[0]), self.center_of_UCA[0])
        return np.stack((x,y,z), axis=-1)
    

m = Simulation()
coords = m.find_nearest_coord([0,7,0] , [0,3,0])
az, el = m.find_azimuth_elevation(coords)
pos_rx = m.rx_positions()

plt.scatter(pos_rx[:,1],pos_rx[:,2])
plt.show()
