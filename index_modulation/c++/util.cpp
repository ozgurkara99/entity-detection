#include "util.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

double helper_func::euclidian_dist(std::vector<double> arr1, std::vector<double> arr2)
{

/*	    
    Find the euclidian distance between 2 points

    Parameters
    ----------
    arr1 : vector array
        first point [x,y,z]
    arr2 : vector array
        second point [x,y,z]
    Returns
    -------
    double
        the euclidian distance between points
*/    
	double sum1 = 0;
    for(int i=0;i<arr1.size();i++)
    {
    	sum1 = sum1 + (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
	}
    return sqrt(sum1);
    
}

void helper_func::cart_to_spherical(double x, double y, double z, double * az, double * el)
{
/*    
    Convert the given cartesian coordinate system to spherical coordinate system
    ex: (0,0,1) -> elevation is 0
    ex: (1,0,0) -> azimuth is 0

    Parameters
    ----------
    x : double
        x coordinate of point
    y : double
        y coordinate of point
    z : double
        z coordinate of point
    az : double pointer
    	returns the azimuth value
    el : double pointer
    	returns the elevation value
    """
*/    
    double hxy = sqrt(x*x + y*y);
    *el = atan2(hxy, z);
    *az = atan2(y, x);
}

std::vector<double> helper_func::mirror_point_over_plane(double a, double b, double c, double d, double x1, double y1, double z1)
{
/*    
    Mirror given point (x1,y1,z1) over given plane that has equation ax + by + cz + d = 0

    Parameters
    ----------
    a, b, c, d : double
        plane equation coefficients
    x1, y1, z1 : double
        the point coordinates that will be mirrored
    Returns
    -------
    vector array
        the coordinates of reflected point [x1,y1,z1]
*/   

    double k =(-a * x1 - b * y1 - c * z1 - d)/double(a * a + b * b + c * c); 
    double x2 = a * k + x1;
    double y2 = b * k + y1;
    double z2 = c * k + z1; 
    std::vector<double> vec (3);
    vec[0] = 2 * x2-x1; 
    vec[1] = 2 * y2-y1; 
    vec[2] = 2 * z2-z1; 
    return vec;
}

std::vector<double> helper_func::delete_duplicated(std::vector<double> pos, std::vector<double> pos2)
 {
 	
/*
    Find the set difference of pos2 from pos

    Parameters
    ----------
    pos : vector array
        first array
    pos2 : vector array
        second array
    Returns
    -------
    vector array
        the values of set(pos2)/set(pos)
*/           

	std::vector<double> vec(pos.size());
	std::sort (pos.begin(), pos.end());
	std::sort (pos2.begin(), pos2.end());
	std::vector<double>::iterator p;
  	p = std::set_difference (pos.begin(), pos.end(), pos2.begin(), pos2.end(), vec.begin());
	
    return vec;
}

void helper_func::print_array(std::vector<double> arr)
{
	for(int i=0;i<arr.size();i++)
	{
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

double helper_func::vector_sum(std::vector<double> arr)
{
	double sum = 0;
	for(int i=0;i<arr.size();i++)
	{
		sum = sum + arr[i];
	}
	return sum;
}

std::vector<double> helper_func::arange(double start, double stop, double number)
{
	std::vector<double> arr;
	double interval = (stop - start) / (number - 1);
	for(int i=0;i<number;i++)
	{
		arr.push_back(start + double(i) * interval);
	}
	return arr;
}

std::vector<std::vector<double> > helper_func::fill_with_vector(int number, std::vector<double> fill)
{
	std::vector<std::vector<double> > filled;
	for(int i=0; i<number; i++)
	{
		filled.push_back(fill);
	}
	return filled;
}

std::vector<std::vector<double> > helper_func::transpose(std::vector<std::vector<double> > b)
{

    std::vector<std::vector<double> > trans_vec(b[0].size(), std::vector<double>());

    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < b[i].size(); j++)
        {
            trans_vec[j].push_back(b[i][j]);
        }
    }

    return trans_vec;    // <--- reassign here
}

std::vector<std::vector<double> > helper_func::create_random_2d(double sigma, double mu, int row_num, int col_num)
{
	// random device class instance, source of 'true' randomness for initializing random seed
    std::random_device rd; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd()); 
    std::normal_distribution<double> d(mu, sigma);
    int i;
    double sample;

    std::vector<std::vector<double> > delta;
    
    for(int j=0; j<col_num; j++)
    {
    	std::vector<double> one_row(row_num);
	    for(i = 0; i < row_num; ++i)
	    {    
	    	
	        // get random number with normal distribution using gen as random source
	        sample = d(gen); 
			one_row[i] = sample;
	        // profit
	        std::cout << "sample: " << sample << std::endl;
	        
	    }
	    delta.push_back(one_row);
	}	
	return helper_func::transpose(delta);
}
