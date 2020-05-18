#include <iostream>
#include "Simulation.h" 
#include "util.h"
#include <vector>
#include <algorithm>


using namespace std;

int main()
{
	std::vector<double> t0 {10,10,20};
	std::vector<double> t1 {0,2,1};
	std::vector<double> center_of_sphere (3,1);
	Simulation obje1;
	double r = 4;
	
	helper_func::print_array(obje1.find_with_quadratic(t0, t1,center_of_sphere , r));
	helper_func::print_array(helper_func::arange(1,10,10));
	
	obje1.tx_positions();
	cout<<"geliyor: " << std::endl;
	for(int i=0;i<obje1.tx_pos.size();i++)
	{
		helper_func::print_array(obje1.tx_pos[i]);
	}
	
	std::vector<std::vector<double> > den = helper_func::fill_with_vector(5,t0);
	for(int i=0;i<den.size();i++)
		helper_func::print_array(den[i]);
	den = helper_func::transpose(den);
	for(int i=0;i<den.size();i++)
		helper_func::print_array(den[i]);
		
	den = helper_func::create_random_2d(obje1.sigma, obje1.mu, 10, 3);
	cout << "den: "<< endl;
	for(int i=0;i<den.size();i++)
		helper_func::print_array(den[i]);	
	cout << "toplam: " << endl;
	obje1.deneme(den,den);
	/*
	vector<vector<double> > a;
	
	a.push_back(deneme);
	a.push_back(coord2);
	a.push_back(coord2);
	a.push_back(coord2);
	
	for(int i=0;i<a.size();i++)
	{
		helper_func::print_array(obje1.find_azimuth_elevation(a[i]));
	}
	cout << a.size()<<endl;
	obje1.detect_indices(a,5, obje1.center_of_rx);
	
	//obje1.print();
	//Simulation obje2 (coord2);
	//obje2.print();
	
	
	
	
	
	double a[5] = { 10, 20, 30, 40, 50 };
	std::vector<double> vect(a,a+5);
	double b[5] = { 20, 30, 60, 10, 50 };
	std::vector<double> vect2(b,b+5);
	std::vector<double> vec = helper_func::delete_duplicated(vect,vect2);
     for (int i=0;i<vec.size();i++)
	 {
	 	std::cout << vec[i] << " ";
	  } 
         */
    return 0;
}
