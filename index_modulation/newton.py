import numpy as np

def find_nearest_coord_newton(t0, t1, center_of_sphere,  r, x0 = 1, epsilon=0.0000001,max_iter=10):
    t0 = t0 - center_of_sphere
    t1 = t1 - center_of_sphere
    coefs = np.zeros((4))
    coefs[0] = t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2
    coefs[1] = 2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[2] - t0[1]) + t0[2] * (t1[2] - t0[2]))
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
    
x = 10
t0 = np.full((x,3),[1, 2.43065363, 4.63531517])
t1 = np.full((x,3),[0.72598089, 2.2827281, 3.74937278])
center_of_sphere = [1,0,3]
r = 5
 
t = []
for i in range(x):
    t.append(find_nearest_coord(t0[i], t1[i], center_of_sphere,  r))
t = np.array(t)