#include "Neuron.h"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

struct Dynamic_Sort {
	Dynamic_Sort(int paramA) { this->paramA = paramA; }
	bool operator () (std::vector<double> i, std::vector<double> j) { return i[paramA] < j[paramA]; }

	int paramA;
};

class ANN
{
public:

	enum Activation_Function { Idenity, Sigmoid, TanH, ArcTan, Softsign, Sinusoid, Gaussian };
	std::vector<int>Network_Description;                      // Full network description - number of Inputs, number of neurons in each hidden layer, number of outputs
	std::vector<std::vector<Neuron>> Network;				  // As above but does not contain input layer
	Activation_Function Choosen_Activation_Function = TanH;   // Chooses which activation function is used in calculations

	// Network Functions - Working
	void Create_Network(std::vector<int> Network_Description, int Seed = time(NULL)); // Auto defines seed to current time unless a seed value is given by user.
	void Load_Network(std::string filename);
	void Save_Network(std::string filename);
	void Train_Network(std::vector<std::vector<double>>Training_Data, double Training_Rate = 0.1, double Target_Error = 1e-10, double Max_Iterations = 100000);

	// Diagnostic functions
	void Print_Neuron_Info(int x, int y);
	void Print_Diagnostics();
	void Print_Network_Weights();
	void Print_Network_Outputs();
	void Print_Network_Error();
	void Print_Network_Delta();
	void Print_Network_Inputs();

	// Network Functions - Beta testing
	std::vector<double> Get_Output_Single_Data_Set(std::vector<double> Input_Data);
	std::vector<std::vector<double>> Get_Output_Multi_Data_Set(std::vector<std::vector<double>> Input_Data);
	std::vector<std::vector<double>> Load_Training_Data(std::string Filename);

private:

	int Epoch = 0;			// Number of iterations spent training
	double Network_Error;
	Eigen::MatrixXd Layer;  // Used as temporary store for layer output results when feeding data forward through the network

	// Private Functions
	void Calculate_Network_Output(std::vector<std::vector<double>> Input_Data);
	double Activation_Functions(double X, bool Differential);
};