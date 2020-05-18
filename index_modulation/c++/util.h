#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED
#include <vector>
namespace helper_func{
	
	int add(int a, int b); 
	double euclidian_dist(std::vector<double> arr1, std::vector<double> arr2);
	void cart_to_spherical(double x, double y, double z, double * az, double * el);
	std::vector<double> mirror_point_over_plane(double a, double b, double c, double d, double x1, double y1, double z1);
	std::vector<double> delete_duplicated(std::vector<double> pos, std::vector<double> pos2);
	void print_array(std::vector<double> arr);
	double vector_sum(std::vector<double> arr);
	std::vector<double> arange(double start, double stop, double number);
	std::vector<std::vector<double> > fill_with_vector(int number, std::vector<double> fill);
	std::vector<std::vector<double> > transpose(std::vector<std::vector<double> > b);
	std::vector<std::vector<double> > create_random_2d(double sigma, double mu, int row_num, int col_num);
	
}
#endif
