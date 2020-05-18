#include <iostream>
#include "Simulation.h"
#include "util.h"
#include <vector>
#include <cmath>
#include <random>

const double PI = 3.1415926535;

std::vector<double> Simulation::center_of_rx_def (3,0);
 
std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y) 
{
    std::vector<double> z;
    for(int i=0; i<x.size(); i++)
    {
    	z.push_back(x[i] + y[i]);
	}
	return z;
}

std::vector<double> operator-(const std::vector<double> &x, const std::vector<double> &y) 
{
    std::vector<double> z;
    for(int i=0; i<x.size(); i++)
    {
    	z.push_back(x[i] - y[i]);
	}
	return z;
}

std::vector<double> operator* (const std::vector<double> &x, const std::vector<double> &y) 
{
    std::vector<double> z;
    for(int i=0; i<x.size(); i++)
    {
    	z.push_back(x[i] * y[i]);
	}
	return z;
}

std::vector<std::vector<double> > operator+ (const std::vector<std::vector<double> > &x, const std::vector<std::vector<double> > &y) 
{
    std::vector<std::vector<double> > z;
    for(int i=0; i<x.size(); i++)
    {
    	std::vector<double> one_row(3);
    	for(int j=0;j<x[0].size(); j++)
    	{
    		one_row[j] = (x[i][j] + y[i][j]);
		}
		z.push_back(one_row);	
	}
	return z;
}

Simulation::Simulation(std::vector<double> center_of_rx, int num_of_tx, int num_of_rx, int mol_number, double r_rx, double r_tx, double D, double step, double time, double d_yz, double d_x) 
{
	std::cout << "Constructor started " << std::endl;
	this->num_of_tx = num_of_tx;
	this->num_of_rx = num_of_rx;
	this->mol_number = mol_number;
	this->r_rx = r_rx;
	this->r_tx = r_tx;
	this->D = D;
	this->step = step;
	this->time = time;
	this->d_yz = d_yz;
	this->d_x = d_x;
	this->center_of_rx = center_of_rx;
	this->sigma = sqrt(2 * this->D * this->step);
	this->mu = 0;
	set_center_of_UCA(this->center_of_rx, this->d_x, this->r_rx, this->r_tx);
}

void Simulation::set_center_of_UCA(std::vector<double> a, double d_x, double r_rx, double r_tx)
{
	center_of_UCA.push_back(a[0] + d_x + r_rx + r_tx);
	center_of_UCA.push_back(a[1]);
	center_of_UCA.push_back(a[2]);
}

std::vector<double> Simulation::detect_indices(std::vector<std::vector<double> > pos, double radius, std::vector<double> coord)
{
/*    
	Detect the indices of values in pos which are inside of the sphere that is defined
    by given radius and center.

    Parameters
    ----------
    pos : array
        the position matrix of molecules
    radius : float
        the radius of sphere
    coord : array
        the center point of sphere [x,y,z]
    Returns
    -------
    array
        the indices of the "pos" array that are detected inside the sphere
*/
	std::vector<double> indices;
	
	double distance = 0;
	for(int i=0; i<pos.size();i++)
	{
		distance = helper_func::euclidian_dist(pos[i], coord);
		if(distance<=radius)
		{
			indices.push_back(i);
		}
	}
	helper_func::print_array(indices);
    return indices;
}

std::vector<double> Simulation::find_azimuth_elevation(std::vector<double> coords)
{
/*    
	Find the azimuth and elevation values of "coords" array with 
    respect to self.center_of_rx 

    Parameters
    ----------
    coords : array
        the coordinates array of molecules that will be converted to spherical coordinates
    Returns
    -------
    azimuth : float
        the azimuth value
    elevation : float
        the elevation value
*/   
	double az, el= 0;
	helper_func::cart_to_spherical(coords[0] - center_of_rx[0], coords[1] - center_of_rx[1], coords[2] - center_of_rx[2], &az, &el);
	std::vector<double> arr;
	arr.push_back(az);
	arr.push_back(el);
    return arr;
}

std::vector<double> Simulation::find_with_quadratic(std::vector<double> t0, std::vector<double> t1, std::vector<double> center_of_sphere, double r)
{    
/*    
    Find the intersection point of the line segment that is crossing through 
    t0 and t1, and the sphere which is defined by center_of_sphere and r (radius) value

    Parameters
    ----------
    t0, t1 : array
        the points through which the line segment is crossing 
    center_of_sphere : array
        the center point of sphere [x,y,z]
    r : float
        the radius value of array
    Returns
    -------
    array
        the point of intersection [x,y,z]
*/      
    
    t0 = t0 - center_of_sphere;
    t1 = t1 - center_of_sphere;
    helper_func::print_array(t0*t0);
    helper_func::print_array(t1);
    std::vector<double> coefs(4);
    coefs[0] = helper_func::vector_sum(t0 * t0);
    coefs[1] = 2 * helper_func::vector_sum(t0 * (t1 - t0)); //b
    coefs[2] = helper_func::vector_sum((t1 - t0) * (t1 - t0)); //a
    coefs[3] = r * r; //coefs[0] -  coefs[3] = c
    helper_func::print_array(coefs);
    double delta = sqrt(coefs[1] * coefs[1] - 4 * coefs[2] * (coefs[0] - coefs[3]));
    double xn = (-1*coefs[1] + delta)/ (2*coefs[2]);
    double xn2 = (-1*coefs[1] - delta)/ (2*coefs[2]);
    std::vector<double> root1;
    std::vector<double> root2;
    for(int i=0;i<3;i++)
    {
    	root1.push_back(t0[i] + (t1[i] - t0[i]) * xn);  
		root2.push_back(t0[i] + (t1[i] - t0[i]) * xn2);      	
	}
    root1 = center_of_sphere + root1;
    root2 = center_of_sphere + root2;
    if(helper_func::vector_sum((root1 - t0) * (root1 - t0)) >= helper_func::vector_sum((root2 - t0) * (root2 - t0)))
    {
    	return root2;
	}
	else
	{
		return root1;
	}
/*      
    coefs = np.zeros((4))

    # f = lambda x: (coefs[0]) + (coefs[1] * x) + (coefs[2] * x**2) - coefs[3]
    coefs[0] = np.sum(t0**2, axis=-1) #t0[0] ** 2 + t0[1] ** 2 + t0[2] ** 2 
    coefs[1] = 2 * np.sum((t0 * (t1-t0)), axis=-1) #2 * (t0[0] * (t1[0] - t0[0]) + t0[1] * (t1[1] - t0[1]) + t0[2] * (t1[2] - t0[2]))
    coefs[2] = np.sum((t1-t0)**2, axis=-1)#(t1[0] - t0[0]) ** 2 + (t1[1] - t0[1]) ** 2 + (t1[2] - t0[2]) ** 2 
    coefs[3] = r**2
    delta = np.sqrt(coefs[1]**2 - 4 * coefs[2] * (coefs[0] - coefs[3])) 
    xn = (-1*coefs[1] + delta)/ (2*coefs[2])
    xn2 = (-1*coefs[1] - delta)/ (2*coefs[2])
    root1 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn, t0[1] + (t1[1] - t0[1]) * xn , t0[2] + (t1[2] - t0[2]) * xn]))
    root2 = center_of_sphere + np.transpose(np.array([t0[0] + (t1[0] - t0[0]) * xn2, t0[1] + (t1[1] - t0[1]) * xn2 , t0[2] + (t1[2] - t0[2]) * xn2]))
    if(np.sum((root1-t0)**2,axis=-1) >= np.sum((root2-t0)**2,axis=-1)):
        return root2
    else:
        return root1
*/     
}

void Simulation::tx_positions()
{
    
/*
    Create the transmitter's coordinates in a circular region and distributed 
    uniformly in angle depending on the self.num_of_tx

    Returns
    -------
    array
        the coordinates of created transmitters
  */        
    
    std::vector<double> d = helper_func::arange(0, 1, num_of_tx + 1);
    std::cout<<"d: ";
    helper_func::print_array(d);
    std::vector<double> h(d.size(), double(2) * PI);
    std::cout<<"h: ";
    helper_func::print_array(h);
    std::vector<double> theta = d * h;
    std::cout<<"theta: ";
    helper_func::print_array(theta);
    std::vector<double> temp(3);
    for(int i=0;i<num_of_tx;i++)
    {
    	temp[0] = center_of_UCA[0];
    	temp[1] = center_of_UCA[1] + (d_yz + r_tx) * cos(theta[i]);
    	temp[2] = center_of_UCA[2] + (d_yz + r_tx) * sin(theta[i]);
    	tx_pos.push_back(temp);
	}
}

std::vector<double> Simulation::tx_reflection(std::vector<double> t0, std::vector<double> t1, std::vector<double> center_of_tx, double r)
{    
/*
	Find the reflection point of the line segment which is crossing through t0 and t1 and 
    intersects with the sphere that is defined by parameters. Then returns this 
    reflected point
    Parameters
    ----------
    t0, t1 : array
        the points through which the line segment is crossing 
    center_of_tx : array
        the center point of sphere [x,y,z]
    r : float
        the radius value of array
    Returns
    -------
    array
        the coordinates of reflected point [x,y,z]
*/         
    
    std::vector<double> coords = Simulation::find_with_quadratic(t0, t1, center_of_tx, r);
    //ax + by + cz + d = 0
    double a = coords[0] - center_of_tx[0];
    double b = coords[1] - center_of_tx[1];
    double c = coords[2] - center_of_tx[2];
    double d = -1 * (a * coords[0] + b * coords[1] + c * coords[2]);
    //reflect t1 point over the plane ax + by + cz + d = 0
    return helper_func::mirror_point_over_plane(a, b, c, d, t1[0], t1[1], t1[2]);
}

bool Simulation::is_inside(std::vector<double> pos, double r, std::vector<double> center)
{
	double dist = helper_func::euclidian_dist(pos, center);
	if(dist <= r)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void Simulation::start_simulation()
{
/*    
    Starts the simulation, molecules are reflected over transmitter sphere and transmitter
    block and it returns the molecules' coordinates that hits the receiver
    
    Returns
    -------
    output : array
        the list of spherical coordinates of each point
    output_coordinates : array
        the list of cartesian coordinates of each point
*/       
    
    std::cout << "Simulation is starting..." << std::endl;
    Simulation::tx_positions();
    std::vector<std::vector<double> > pos = helper_func::fill_with_vector(mol_number, tx_pos[0]);
    //self.output = []
    //self.output_coordinates = []
    
    for(int i=0; i<int(time/step); i++)
    {
    	std::vector<std::vector<double> > delta = helper_func::create_random_2d(sigma, mu, mol_number, 3);
    	std::vector<std::vector<double> > pos2 = pos + delta;
    	
    	//reflect over tx_block
    	for(int j=0;j<mol_number;j++)
    	{
    		if(pos2[j][0] >= center_of_UCA[0] + r_tx)
    		{
    			pos2[j] = helper_func::mirror_point_over_plane(1, 0, 0, -1 * (center_of_UCA[0] + r_tx), pos2[j][0], pos2[j][1], pos2[j][2]);
			}
		}
		
		//tx_reflection
		
		for (int j=0;j<num_of_tx;j++)
		{
			std::vector<double> center_of_tx = tx_pos[j];
			for(int k=0;k<mol_number;k++)
			{
				//pos2 is inside but pos not inside the sphere
				if((Simulation::is_inside(pos2[k], r_tx, center_of_tx)) && (!(Simulation::is_inside(pos[k], r_tx, center_of_tx))))
				{
					pos2[k] = Simulation::tx_reflection(pos[k], pos2[k], center_of_tx, r_tx);
				}
			}
		}
		
		
		
		pos = pos2;
	}
    
}


void Simulation::print()
{
		std::cout << "Printing parameters..."<< std::endl;
    	std::cout << "number of tx: "<< num_of_tx << std::endl;
    	std::cout << "number of rx: "<< num_of_rx << std::endl;
    	std::cout << "mol number: "<< mol_number << std::endl;
    	std::cout << "radius of rx: "<< r_rx << std::endl;
    	std::cout << "radius of tx: "<< r_tx << std::endl;
    	std::cout << "diffusion coefficient: "<< D << std::endl;
    	std::cout << "step: "<< step << std::endl;
    	std::cout << "time: "<< time << std::endl;
    	std::cout << "d_yz: "<< d_yz << std::endl;
    	std::cout << "d_x: "<< d_x << std::endl;
    	std::cout << "center of rx: ";
		helper_func::print_array(center_of_rx);
		std::cout << "center of tx: ";
		helper_func::print_array(center_of_UCA);
 
}



void Simulation::deneme(std::vector<std::vector<double> > x, std::vector<std::vector<double> > y)
{
	for(int i=0;i<x.size();i++)
		helper_func::print_array((x + y)[i]);
}
