#ifndef SIMULATION_H
#define SIMULATION_H
#include <iostream>
#include <vector>

class Simulation
{
  private:
 	static std::vector<double> center_of_rx_def;
  public:
  	friend std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y);
  	friend std::vector<std::vector<double> > operator+ (const std::vector<std::vector<double> > &x, const std::vector<std::vector<double> > &y);
  	friend std::vector<double> operator-(const std::vector<double> &x, const std::vector<double> &y);
  	friend std::vector<double> operator*(const std::vector<double> &x, const std::vector<double> &y);
  	std::vector<double> find_with_quadratic(std::vector<double> t0, std::vector<double> t1, std::vector<double> center_of_sphere, double r);
  	std::vector<double> tx_reflection(std::vector<double> t0, std::vector<double> t1, std::vector<double> center_of_tx, double r);
  	bool is_inside(std::vector<double> pos, double r, std::vector<double> center);
  	void start_simulation();
  	void tx_positions();
 	int num_of_tx;
 	int num_of_rx;
 	int mol_number;
 	std::vector<std::vector<double> > tx_pos;
 	double r_rx;
 	double r_tx;
 	double D;
 	double step;
 	double time;
 	double d_yz;
 	double d_x;
 	std::vector<double> center_of_rx;
 	std::vector<double> center_of_UCA;
 	double sigma;
 	//void Simulation::tx_positions();
 	double mu;
 	void set_center_of_UCA(std::vector<double> a, double d_x, double r_rx, double r_tx);
	std::vector<double> detect_indices(std::vector<std::vector<double> > pos, double radius, std::vector<double> coord); 
	std::vector<double> find_azimuth_elevation(std::vector<double> coords);
	Simulation(std::vector<double> vec=center_of_rx_def, int num_of_tx=8, int num_of_rx=1, int mol_number=10000, double r_rx=5, double r_tx=0.5, double D=79.4, double step=0.0001, double time=0.75, double d_yz=10, double d_x=10);
 	void print();
	void deneme(std::vector<std::vector<double> > x, std::vector<std::vector<double> > y);
};

std::vector<double> operator+(const std::vector<double> &x, const std::vector<double> &y);
#endif


