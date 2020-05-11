import numpy as np
import time

def find_nearest_coord_newton(t0, t1, center_of_sphere,  r, x0 = 1, epsilon=0.0000001,max_iter=10):
    t0 = t0 - center_of_sphere
    t1 = t1 - center_of_sphere
    coefs = np.zeros((4))
    coefs[0] = t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2
    coefs[1] = 2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[1] - t0[1]) + t0[2] * (t1[2] - t0[2]))
    coefs[2] = (t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2 + (t1[2] - t0[2]) ** 2 
    coefs[3] = r**2
    f = lambda x: (coefs[0]) + (coefs[1] * x) + (coefs[2] * x**2) - coefs[3]
    Df = lambda x: coefs[1] + 2 * x * coefs[2]
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn, t0[1] + (t1[1] - t0[1]) * xn , t0[2] + (t1[2] - t0[2]) * xn]))
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return -1
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return -1

def find_with_quadratic(t0, t1, center_of_sphere, r):
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

def parametrize(t0, t1, p):
    par = np.transpose(t0) + np.transpose(t1 - t0) * p
    return par

def euclidian_dist(arr1, arr2):
    #arr1 = np.array(arr1)
    #arr2 = np.array(arr2)
    return np.sqrt(abs(np.sum(((arr1 - arr2) ** 2), axis=-1)))

def find_nearest_coord(t0, t1, center_of_sphere, r):
    h = np.array([parametrize(t0,t1,x) for x in np.arange(0,1,0.0001)])
    a = np.full((h.shape[0], h.shape[1]), center_of_sphere)
    x = euclidian_dist(a,h)
    return [h[np.argmin(abs(abs(x) - r)),0], h[np.argmin(abs(abs(x) - r)),1], h[np.argmin(abs(abs(x) - r)),2]]


def find(t0, t1, center_of_sphere, r):
    a = find_nearest_coord_newton(t0, t1, center_of_sphere,  r)
    if(type(a)==int):
        return find_nearest_coord( t0, t1, center_of_sphere, r)
    else:
        return find_nearest_coord_newton( t0, t1, center_of_sphere, r)
    
x = 1000000
t0 = np.full((x,3),[5,0,0])
t1 = np.full((x,3),[2,0,0])
center_of_sphere = [0,0,0]
r = 2
st = time.time()
t = []
for i in range(x):
    t.append(find_with_quadratic(t0[i], t1[i], center_of_sphere,  r))
t = np.array(t)
print("t2: " + str(st - time.time()))

st = time.time()
t = []
for i in range(x):
    t.append(find_with_quadratic(t0[i], t1[i], center_of_sphere,  r))
t = np.array(t)
print("t2: " + str(st - time.time()))


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