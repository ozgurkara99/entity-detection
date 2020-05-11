import numpy as np
import matplotlib.pyplot as plt 
import time

PI = np.pi
# util functions 

def parametrize(t0, t1, p):
    par = np.transpose(t0) + np.transpose(t1 - t0) * p
    return par

def euclidian_dist(arr1, arr2):
    #arr1 = np.array(arr1)
    #arr2 = np.array(arr2)
    return np.sqrt(abs(np.sum(((arr1 - arr2) ** 2), axis=-1)))

def cart_to_spherical(x, y, z):
    hxy = np.hypot(x, y)
    el = np.arctan2(hxy, z) #(0,0,1) has 0 elevation
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
    return np.stack((x3,y3,z3), axis=-1)

def delete_duplicated(pos, pos2):
    differ = np.array(list(set(pos2).difference(set(pos))))
    return differ

class Simulation():
    """
    A class used to represent a Simulation

    Attributes
    ----------
    num_of_tx : int
        the number of transmitter in simulation topology
    num_of_rx : int
        the number of receiver in simulation topology
    r_rx : float
        the radius of each receiver which is in spherical form
    r_tx : float
        the radius of each transmitter which is in spherical form
    D : float
        the diffusion coefficient
    step : float
        the time step that molecules move in each step
    time : float
        total time that simulation works
    d_yz : float
        the length between the center of UCA and the closest point of a transmitter
    d_x : float
        the length between the center of UCA + r_tx and the closest point of receiver
    center_of_rx : array (3x1)
        the coordinates of the center of receiver
    mol_number : int
        the number of moles

    Methods
    -------
    start_simulation()
        Starts the simulation
    detect_indices(pos, radius, coord)
        detect the indices of "pos" array's values which is in the sphere whose radius and center coordinates are input
    find_azimuth_elevation(coords)
        finds the azimuth and elevation values of given coordinates according to receiver
    find_with_quadratic(t0, t1, center_of_sphere, r):
        finds the point in [t0,t1] line segment that intersects with the sphere whose radius and center coordinates are input
    tx_positions()
        creates the transmitter position circular with given number 
    tx_reflection(t0, t1, center_of_tx, r)
        reflects the molecule with given t0 position and t1 position and the sphere
    plot_3d_scatter(pos)
        it plots the molecules and receiver
    """
    def __init__(self, num_of_tx=8, num_of_rx=1, r_rx=5, r_tx=1, D=79.4,step=0.0001, time=0.75, d_yz=10, d_x=10, center_of_rx = np.array([0,0,0]), mol_number=100):
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
        self.center_of_UCA = np.array([center_of_rx[0] + d_x + r_rx +  r_tx, center_of_rx[1], center_of_rx[2]])
        self.sigma = np.sqrt(2 * D * step)
        self.mol_number = mol_number
        self.mu = 0
    
    def start_simulation(self):
        print("Simulation is starting...")
        pos = np.full((self.mol_number,self.tx_positions()[0].shape[0]), self.tx_positions()[0])
        self.output = []
        self.output_coordinates = []
        for i in range(int(self.time / self.step)):
            delta = self.sigma * np.random.randn(self.mol_number, pos.shape[1]) + self.mu
            self.plot_3d_scatter(pos)
            pos2 = pos + delta
            rx_indices = self.detect_indices(pos2, self.r_rx, self.center_of_rx)
            if(len(rx_indices) != 0):    
                coords = []
                for h in rx_indices:
                    coords.append(self.find_with_quadratic(pos[h], pos2[h], self.center_of_rx, self.r_rx))
                coords = np.array(coords)
                self.azimuth_data, self.elevation_data = self.find_azimuth_elevation(coords)
                pos2[rx_indices,0] = -1000
                self.output_coordinates.append(coords)
                self.output.append([self.azimuth_data, self.elevation_data])
            tx_block_indices = np.where(pos2[:,0] >= (self.center_of_UCA[0] + self.r_tx))[0]
            pos2[tx_block_indices] = mirror_point_over_plane(1, 0, 0, -1 * (self.center_of_UCA[0] + self.r_tx), pos2[tx_block_indices,0], pos2[tx_block_indices,1], pos2[tx_block_indices,2])
            
            for each_tx in range(self.num_of_tx):
                center_of_tx = self.tx_positions()[each_tx]
                tx_indices_1 = self.detect_indices(pos[np.where(pos[:,0] >= self.d_x)], self.r_tx, center_of_tx)
                tx_indices_2 = self.detect_indices(pos2[np.where(pos[:,0] >= self.d_x)], self.r_tx, center_of_tx)
                tx_indices = delete_duplicated(tx_indices_1, tx_indices_2)
                if(len(tx_indices)!=0):
                    for h in tx_indices:
                        pos2[h] = self.tx_reflection(pos[h], pos2[h], center_of_tx, self.r_tx)   
            pos = pos2
        return self.output, self.output_coordinates
    
    def detect_indices(self, pos, radius, coord):
        distance = euclidian_dist(pos, coord)
        indices = np.where(distance<=radius)[0]
        return indices
    
    def find_azimuth_elevation(self, coords):
        azimuth, elevation = cart_to_spherical(coords[:,0] - self.center_of_rx[0], coords[:,1] - self.center_of_rx[1], coords[:,2] - self.center_of_rx[2])
        return azimuth, elevation

    def find_with_quadratic(self,t0, t1, center_of_sphere, r):
        t0 = t0 - center_of_sphere
        t1 = t1 - center_of_sphere
        coefs = np.zeros((4))
        coefs[0] = t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2
        coefs[1] = 2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[1] - t0[1]) + t0[2] * (t1[2] - t0[2]))
        coefs[2] = (t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2 + (t1[2] - t0[2]) ** 2 
        coefs[3] = r**2
        xn = (-1*coefs[1] + np.sqrt(coefs[1]**2 - 4 * coefs[2] * (coefs[0] - coefs[3]))) / (2*coefs[2])
        xn2 = (-1*coefs[1] - np.sqrt(coefs[1]**2 - 4 * coefs[2] * (coefs[0] - coefs[3]))) / (2*coefs[2])
        root1 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn, t0[1] + (t1[1] - t0[1]) * xn , t0[2] + (t1[2] - t0[2]) * xn]))
        root2 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn2, t0[1] + (t1[1] - t0[1]) * xn2 , t0[2] + (t1[2] - t0[2]) * xn2]))
        if(np.sum((root1-t0)**2,axis=-1) >= np.sum((root2-t0)**2,axis=-1)):
            return root2
        else:
            return root1
        
    def tx_positions(self):
        d = np.arange(0,1,1/(self.num_of_tx))
        theta = d * 2 * PI
        y = self.center_of_UCA[1] + 4 * np.cos(theta)
        z = self.center_of_UCA[2] + 4 * np.sin(theta)
        x = np.full((d.shape[0]), self.center_of_UCA[0])
        return np.stack((x,y,z), axis=-1)
    
    def tx_reflection(self, t0, t1, center_of_tx, r):
        coords = self.find_with_quadratic(t0, t1, center_of_tx, r)
        #ax + by + cz + d = 0
        a = coords[0] - center_of_tx[0]
        b = coords[1] - center_of_tx[1]
        c = coords[2] - center_of_tx[2]
        d = -1 * (a * coords[0] + b * coords[1] + c * coords[2])
        #reflect t1 point over the plane ax + by + cz + d = 0
        reflected = mirror_point_over_plane(a, b, c, d, t1[0], t1[1], t1[2])
        return reflected
        
    def plot_3d_scatter(self, pos):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='r', marker='o')
        for i in range(self.num_of_tx):
            center_of_tx = self.tx_positions()[i]
            ax.scatter(center_of_tx[0], center_of_tx[1], center_of_tx[2], c='g', marker='o')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = self.r_rx * np.outer(np.cos(u), np.sin(v))
        y = self.r_rx * np.outer(np.sin(u), np.sin(v))
        z = self.r_rx * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, rstride=10, cstride=10, color='b')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=0, azim=90)
        plt.show()
        plt.close()
        

m = Simulation()

start = time.time()
output, output_coordinates = m.start_simulation()
print("total time = " + str(time.time() - start))
"""
x = 10
t0 = np.full((x,3),[-9,0,0])
t1 = np.full((x,3),[2,1,0])
center_of_sphere = [1,0,0]
r = 3
reflected, coord = m.tx_reflection(t0[0], t1[0], center_of_sphere, r)

fig = plt.figure()

plt.gcf().gca().add_artist(plt.Circle((center_of_sphere[0], center_of_sphere[1]), m.r_tx, color='r', fill=False))
h = np.array([[center_of_sphere[0], center_of_sphere[1]],[t0[0,0], t0[0,1]], [t1[0,0], t1[0,1]],[reflected[0], reflected[1]], [coord[0],coord[1]]])
plt.scatter(h[:,0], h[:,1])

plt.plot([t0[0,0],t1[0,0]],[t0[0,1],t1[0,1]],'ro-')
plt.plot([reflected[0], coord[0]], [reflected[1], coord[1]], '-bo')
plt.plot([center_of_sphere[0], coord[0]], [center_of_sphere[1], coord[1]], '-go')


plt.plot([0,2*coord[1]],[0,2*coord[2]],'g-')

plt.plot([reflected[1], coord[1]], [reflected[2], coord[2]], '-bo')

plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(pos_tx[:,1],pos_tx[:,2])
plt.show()
"""


"""
    def find(self,t0, t1, center_of_sphere, r):
        a = self.find_nearest_coord_newton(t0, t1, center_of_sphere,  r)
        if(type(a)==int):
            return self.find_nearest_coord(t0, t1, center_of_sphere, r)
        else:
            return self.find_nearest_coord_newton(t0, t1, center_of_sphere, r)
        
    def find_nearest_coord_newton(self, t0, t1, center_of_sphere,  r, x0 = 1, epsilon=0.001,max_iter=100): 
        coefs = np.zeros((4))
        t0 = t0 - center_of_sphere
        t1 = t1 - center_of_sphere
        coefs[0] = t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2
        coefs[1] = 2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[1] - t0[1]) + t0[2] * (t1[2] - t0[2]))
        coefs[2] = (t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2 + (t1[2] - t0[2]) ** 2 
        coefs[3] = r**2
        f = lambda x: (coefs[0]) + (coefs[1] * x) + (coefs[2] * x**2) - coefs[3]
        Df = lambda x: coefs[1] + 2 * x * coefs[2]
        xn = x0
        for n in range(0,max_iter):
            fxn = f(xn)
            print(fxn)
            if abs(fxn) < epsilon:
                print('Found solution after',n,'iterations.')
                return  np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn, t0[1] + (t1[1] - t0[1]) * xn , t0[2] + (t1[2] - t0[2]) * xn])) + center_of_sphere
            Dfxn = Df(xn)
            if Dfxn == 0:
                #print('Zero derivative. No solution found.')
                return -1
            xn = xn - fxn/Dfxn
        print('max_iter')
        return -1
            
    def find_nearest_coord(self, t0, t1, center_of_sphere, r):
        h = np.array([parametrize(t0,t1,x) for x in np.arange(0,1,0.01)])
        a = np.full((h.shape[0], h.shape[1]), center_of_sphere)
        x = euclidian_dist(a,h)
        return [h[np.argmin(abs(abs(x) - r)),0], h[np.argmin(abs(abs(x) - r)),1], h[np.argmin(abs(abs(x) - r)),2]]
"""