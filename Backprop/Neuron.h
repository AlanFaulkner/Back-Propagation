#include <random>
#include <cmath>
#include <Eigen/Eigen>

// This class details a individual neuron. Its constructor initializes a random weights for each
// connection into the neuron and a neuron bias. Further information used in calculating the back
// propagation algorithm are also stored.

class Neuron
{
public:

	// Neuron Class Constructor
	Neuron(int Number_Of_Connections, int Seed);

	// Public accessible values related to Neuron.

	//// Public accessible Matrices
	Eigen::MatrixXd Output;
	Eigen::MatrixXd Error;
	Eigen::MatrixXd Delta;
	Eigen::MatrixXd Weights;
	Eigen::MatrixXd Input;
	Eigen::MatrixXd RMS_Error;

	// Backprop on-line
	double Sigma_error;
	Eigen::MatrixXd Weight_Update;
	Eigen::MatrixXd Weight_Update_Old;
	Eigen::MatrixXd Weight_Update_Oldest;
};