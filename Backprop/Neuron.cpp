#include "Neuron.h"

Neuron::Neuron(int Number_Of_Connections, int Seed)
{
	// Random Number Generation. - Seed defined in Neuron Class (Private Variable)
	std::mt19937_64 Generate(Seed);
	std::uniform_real_distribution<double> Weight(-1, 1);
	std::uniform_real_distribution<double> Bias(0, 1);

	// Resize the weight matrix for the number of connection, plus one additional weight that acts as the bias
	Neuron::Weights.resize((Number_Of_Connections + 1), 1);

	// Randomize input weights.
	for (int a = 0; a < Number_Of_Connections; a++) {
		Neuron::Weights(a) = Weight(Generate);
	}

	Neuron::Weights((Number_Of_Connections)) = Bias(Generate); // Randomize neuron bias.
	Neuron::Weight_Update_Old = Eigen::MatrixXd::Constant(Neuron::Weights.cols(), Neuron::Weights.rows(), 0);
}