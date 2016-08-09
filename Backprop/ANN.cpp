#include "ANN.h"

// Working Functions
void ANN::Create_Network(std::vector<int> Network_Description, int Seed)
{
	ANN::Network_Description = Network_Description;
	if (Network_Description.size() < 2) {
		std::cout << "The description of neural net you have entered is not valid!\n\nA valid description must contain at least two values:\n   The number of inputs into the network\n   The number of output neurons\n\n In both cases the minimum value allowed is 1!\n\n";
		return;
	}

	for (std::vector<int>::size_type i = 0; i < Network_Description.size(); i++) {
		if (Network_Description[i] == 0) {
			std::cout << "The description of neural net you have entered is not valid!\n\n The minimum allowed number of neurons or inputs is 1\n\n";
			return;
		}
	}

	srand(Seed); // Use seed value to initial random number generation - Used to set random weights and basis for each neuron

	for (std::vector<int>::size_type i = 1; i < Network_Description.size(); i++) { // Loop through Network description staring with first hidden layer - Input layer has no neurons!
		std::vector<Neuron>Layer; // Create a layer of neurons
		for (int j = 0; j < Network_Description[i]; j++) {
			if (i == 1) {
				Neuron Neuron(Network_Description[0], rand());
				Layer.push_back(Neuron);
			}
			else {
				Neuron Neuron(Network_Description[i - 1], rand());
				Layer.push_back(Neuron);
			}
		}
		ANN::Network.push_back(Layer);
	}

	std::cout << "Network has been successfully created.\n\n";
	return;
}
void ANN::Load_Network(std::string filename)
{
	std::vector<int> Description;
	std::ifstream Input(filename);
	std::string line;
	std::string Data = "Data";
	if (!Input.is_open()) { std::cout << "Cannot open file!" << std::endl; }
	while (getline(Input, line) && line != Data) {
		std::stringstream ss(line);
		int c = 0;
		ss >> c;
		if (c != 0) { Description.push_back(c); }
	}

	if (Description.size() < 2) {
		std::cout << "The description of neural net you have entered is not valid!\n\nA valid description must contain at least two values:\n   The number of inputs into the network\n   The number of output neurons\n\n In both cases the minimum value allowed is 1!\n\n";
		return;
	}

	for (std::vector<int>::size_type i = 0; i < Network_Description.size(); i++) {
		if (Description[i] == 0) {
			std::cout << "The description of neural net you have entered is not valid!\n\n The minimum allowed number of neurons or inputs is 1\n\n";
			return;
		}
	}

	ANN::Create_Network(Description);

	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			for (int k = 0; k < Network[i][j].Weights.rows(); k++) {
				getline(Input, line);
				std::stringstream ss(line);
				double c = 0.0;
				ss >> c;
				Network[i][j].Weights(k, 0) = c;
			}
		}
	}

	Input.close();
	std::cout << "\nNetwork data loaded successfully.\n\n";
}
void ANN::Save_Network(std::string filename)
{
	std::ofstream Output(filename);

	for (std::vector<int>::size_type i = 0; i < ANN::Network_Description.size(); i++) {
		Output << ANN::Network_Description[i] << std::endl;
	}

	Output << "Data\n";

	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			for (int k = 0; k < Network[i][j].Weights.rows(); k++) {
				Output << Network[i][j].Weights(k, 0) << std::endl;
			}
		}
	}

	Output.close();

	std::cout << "\nNetwork data has been saved to file.\n\n";
}
void ANN::Train_Network(std::vector<std::vector<double>> Training_Data, double Training_Rate, double Target_Error, double Max_Iterations)
{
	std::vector<std::vector<double>> Training_Inputs;																// Extract training inputs from input data.
	Eigen::MatrixXd Training_Results(Training_Data.size(), ANN::Network[ANN::Network.size() - 1].size());			// store outputs as in matrix for manipulation later

																													// One training data set per row
	for (std::vector<double>::size_type i = 0; i < Training_Data.size(); i++) {
		std::vector<double>Row;
		for (int j = 0; j < ANN::Network_Description[0]; j++) {
			Row.push_back(Training_Data[i][j]);
		}

		Training_Inputs.push_back(Row);
		Row.clear();

		for (std::vector<double>::size_type j = ANN::Network_Description[0]; j < Training_Data[0].size(); j++) {
			Training_Results(i, j - ANN::Network_Description[0]) = Training_Data[i][j];
		}
	}

	std::cout << " Training.......\n\n";
	std::cout << " Training Results\n------------------\n\n";
	std::cout << " Epoch      Error\n\n";

	do {
		ANN::Network_Error = 0;

		for (std::vector<std::vector<double>>::size_type Z = 0; Z < Training_Inputs.size(); Z++) {
			std::vector<std::vector<double>> Element;
			Element.push_back(Training_Inputs[Z]);
			Eigen::MatrixXd Element_Target = Training_Results.row(Z);

			ANN::Calculate_Network_Output(Element); // Calculates output of network using current weights. - if this code work fix this section
			int A = ANN::Network.size() - 1; // Used to access neuron in reverse

			for (std::vector<std::vector<Neuron>>::reverse_iterator i = ANN::Network.rbegin(); i != ANN::Network.rend(); ++i, --A) {
				for (std::vector<std::vector<Neuron>>::size_type j = 0; j < ANN::Network[A].size(); j++) {
					if (A == (ANN::Network.size() - 1)) {
						ANN::Network[A][j].Sigma_error = (Element_Target(0, j) - ANN::Network[A][j].Output(0, 0))*ANN::Activation_Functions(ANN::Network[A][j].Output(0, 0), true);
						ANN::Network_Error += ANN::Network[A][j].Sigma_error * ANN::Network[A][j].Sigma_error*0.5;
						ANN::Network[A][j].Weight_Update = ANN::Network[A][j].Input*ANN::Network[A][j].Sigma_error*Training_Rate;
						ANN::Network[A][j].Weights = ANN::Network[A][j].Weights + ANN::Network[A][j].Weight_Update.transpose() + ANN::Network[A][j].Weight_Update_Old.transpose();
						ANN::Network[A][j].Weight_Update_Oldest = ANN::Network[A][j].Weight_Update_Old;
						ANN::Network[A][j].Weight_Update_Old = ANN::Network[A][j].Weight_Update * 0.7; // Momentum
					}

					else {
						for (std::vector<Neuron>::size_type X = 0; X < ANN::Network[A + 1].size(); X++) {
							ANN::Network[A][j].Sigma_error += ANN::Network[A + 1][X].Sigma_error * ANN::Network[A + 1][X].Weights(X, 0);
						}

						ANN::Network[A][j].Sigma_error = ANN::Network[A][j].Sigma_error*ANN::Activation_Functions(ANN::Network[A][j].Output(0, 0), true);
						ANN::Network[A][j].Weight_Update = ANN::Network[A][j].Input*ANN::Network[A][j].Sigma_error*Training_Rate;
						ANN::Network[A][j].Weights = ANN::Network[A][j].Weights + ANN::Network[A][j].Weight_Update.transpose() + ANN::Network[A][j].Weight_Update_Old.transpose();
						ANN::Network[A][j].Weight_Update_Oldest = ANN::Network[A][j].Weight_Update_Old;
						ANN::Network[A][j].Weight_Update_Old = ANN::Network[A][j].Weight_Update * 0.7;
					}
				}
			}
		}
		ANN::Network_Error = ANN::Network_Error / Training_Inputs.size();
		ANN::Epoch++;

		if (Epoch % 10000 == 0) { std::cout << std::setw(7) << Epoch << " : " << std::setw(10) << Network_Error << std::endl; }
	} while (Network_Error >= Target_Error && Epoch <= Max_Iterations);

	std::cout << "\n Training complete! \n \n";
}

// Diagnostic Functions

// Beta-Testing Functions
void ANN::Print_Network_Weights()
{
	std::cout << " Network Weights and biases\n-----------------------------\n\n";
	std::cout << "Number of Inputs: " << ANN::Network_Description[0] << std::endl << std::endl;
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << "Hidden Layer: " << i + 1 << std::endl << std::endl; }
		else { std::cout << "Output Layer." << std::endl << std::endl; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			std::cout << Network[i][j].Weights << std::endl << std::endl;
		}
		std::cout << std::endl;
	}
}

void ANN::Print_Network_Outputs()
{
	std::cout << " Network Outputs\n-----------------\n\n";
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << "Hidden Layer: " << i + 1 << std::endl << std::endl; }
		else { std::cout << "Output Layer." << std::endl << std::endl; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			std::cout << Network[i][j].Output << std::endl << std::endl;
		}
		std::cout << std::endl;
	}
}

void ANN::Print_Network_Error()
{
	std::cout << " Network Error\n-----------------------\n\n";
	std::cout << "Number of Inputs: " << ANN::Network_Description[0] << std::endl << std::endl;
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << "Hidden Layer: " << i + 1 << std::endl << std::endl; }
		else { std::cout << "Output Layer." << std::endl << std::endl; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			std::cout << Network[i][j].Error << std::endl << std::endl;
		}
		std::cout << std::endl;
	}
}

void ANN::Print_Network_Delta()
{
	std::cout << " Network Delta\n-----------------------------\n\n";
	std::cout << "Number of Inputs: " << ANN::Network_Description[0] << std::endl << std::endl;
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << "Hidden Layer: " << i + 1 << std::endl << std::endl; }
		else { std::cout << "Output Layer." << std::endl << std::endl; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			std::cout << Network[i][j].Delta << std::endl << std::endl;
		}
		std::cout << std::endl;
	}
}

void ANN::Print_Network_Inputs()
{
	std::cout << " Network Inputs\n-----------------------------\n\n";
	std::cout << "Number of Inputs: " << ANN::Network_Description[0] << std::endl << std::endl;
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << "Hidden Layer: " << i + 1 << std::endl << std::endl; }
		else { std::cout << "Output Layer." << std::endl << std::endl; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			std::cout << Network[i][j].Input << std::endl << std::endl;
		}
		std::cout << std::endl;
	}
}

// Diagnostic functions
void ANN::Print_Neuron_Info(int x, int y)
{
	std::cout << " Neuron ID - Network[" << x << "][" << y << "]" << "\n---------------------------\n";
	std::cout << "\n Inputs                                 :  " << ANN::Network[x][y].Input;
	std::cout << "\n Weight + Bias                          :  " << ANN::Network[x][y].Weights.transpose();
	std::cout << "\n Output                                 :  " << ANN::Network[x][y].Output;
	std::cout << "\n Sigmoid error [(Target-out)(1-out)out] : " << ANN::Network[x][y].Sigma_error;
	std::cout << "\n Weight Update [Input*Sigmoid error]    : " << ANN::Network[x][y].Weight_Update;
	std::cout << "\n Weight Update Old [From previous step] : " << ANN::Network[x][y].Weight_Update_Oldest;
	std::cout << std::endl << std::endl;
}
void ANN::Print_Diagnostics()
{
	std::cout << "########################\n# Network Diagnostics #\n########################\n\n";
	for (std::vector<Neuron>::size_type i = 0; i < ANN::Network.size(); i++) {
		if (i != Network.size() - 1) { std::cout << " Hidden Layer: " << i + 1 << "\n----------------\n"; }
		else { std::cout << " Output Layer\n--------------\n\n"; }
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			ANN::Print_Neuron_Info(i, j);
		}
		std::cout << std::endl;
	}
	std::cout << "##############################\n# Network Diagnostics - END #\n##############################\n\n";
}

// Output Functions
std::vector<double> ANN::Get_Output_Single_Data_Set(std::vector<double> Input_Data)
{
	std::vector<double> Output;
	std::vector<std::vector<double>> Input;
	Input.push_back(Input_Data);
	ANN::Calculate_Network_Output(Input);
	for (std::vector<std::vector<Neuron>>::size_type i = 0; i < ANN::Network[ANN::Network.size() - 1].size(); i++) {
		Output.push_back(ANN::Network[ANN::Network.size() - 1][i].Output(0, 0));
	}

	return Output;
}

std::vector<std::vector<double>> ANN::Get_Output_Multi_Data_Set(std::vector<std::vector<double>> Input_Data)
{
	ANN::Calculate_Network_Output(Input_Data);
	std::vector<std::vector<double>> Output;

	for (std::vector<std::vector<double>>::size_type i = 0; i < Input_Data.size(); i++) {
		std::vector<double> Output_row;
		for (std::vector<std::vector<Neuron>>::size_type j = 0; j < ANN::Network[ANN::Network.size() - 1].size(); j++) {
			Output_row.push_back(ANN::Network[ANN::Network.size() - 1][j].Output(i, 0));
		}
		Output.push_back(Output_row);
	}

	return Output;
}

std::vector<std::vector<double>> ANN::Load_Training_Data(std::string Filename)
{
	std::vector<std::vector<double>> Training_Data;
	std::ifstream Input(Filename);
	if (Input.fail()) { std::cout << "Error - Can't open file!" << std::endl; return Training_Data; } // Error checking

	int Data_Input_Size = ANN::Network_Description[0] + ANN::Network_Description[ANN::Network.size()];

	double Data;

	while (Input >> Data) {
		std::vector<double>Data_Row;
		for (int i = 0; i < Data_Input_Size - 1; i++) {
			Data_Row.push_back(Data);
			Input >> Data;
		}
		Data_Row.push_back(Data);
		Training_Data.push_back(Data_Row);
	}

	Input.close();

	return Training_Data;
}

std::vector<std::vector<double>> ANN::Calculate_Batch_Output(std::vector<std::vector<double>> Input_Data)
{
	// Standardized format of data entry: One set of Inputs is equal to one row
	// Covert vector into eigen matrix - currently this is for convergence might resort to just using std::vectors in future

	// Build input eigen matrix
	Eigen::MatrixXd Input(Input_Data.size(), Input_Data[0].size() + 1);										// This allows for input data to hold more than one set of inputs
	for (std::vector<double>::size_type i = 0; i < Input_Data.size(); i++) {
		for (std::vector<double>::size_type j = 0; j < Input_Data[0].size(); j++) {
			Input(i, j) = Input_Data[i][j];
		}
		Input(i, Input_Data[0].size()) = 1;																	// This is the input for the neuron bias. It is always set to 1. Value allows for easy matrix multiplication
	}

	for (std::vector<std::vector<Neuron>>::size_type i = 0; i < Network.size(); i++) {
		ANN::Layer.resize(Input_Data.size(), ANN::Network[i].size());										// Stores output of each neuron in given layer in a row. Each row represents one set of inputs.
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			// Calculate the output of neuron.
			Network[i][j].Input = Input;																	// Stores a copy of inputs in neuron - useful for error checking
			Network[i][j].Output = Network[i][j].Input*Network[i][j].Weights;								// Calculate output of neuron
			for (int y = 0; y < Network[i][j].Output.rows(); y++) {
				for (int x = 0; x < Network[i][j].Output.cols(); x++) {
					Network[i][j].Output(y, x) = Activation_Functions(Network[i][j].Output(y, x), false);	// Apply activation function
				}
			}

			// Store output of neuron in Layer matrix to be used as input for next layer
			for (int y = 0; y < ANN::Layer.rows(); y++) {
				ANN::Layer(y, j) = Network[i][j].Output(y, 0);
			}
		}

		// Update input data for next layer
		Input.resize(Layer.rows(), Layer.cols());
		Input = ANN::Layer;

		// Add the value of 1 to each row to take into account neuron bias needs an input value of one.
		Input.conservativeResize(Input.rows(), Input.cols() + 1);											// Resize matrix without data loss
		Eigen::VectorXd Neuron_Bias_Input = Eigen::VectorXd::Constant(Input.rows(), 1);						// Create a temporary vector of size = training set size and initialized to 1
		Input.col(Input.cols() - 1) = Neuron_Bias_Input;													// Add vector to input matrix
	}
	return std::vector<std::vector<double>>();
}

// Private functions
void ANN::Calculate_Network_Output(std::vector<std::vector<double>> Input_Data)
{
	// Standardized format of data entry: One set of Inputs is equal to one row
	// Covert vector into Eigen matrix - currently this is for convergence might resort to just using std::vectors in future

	// Build input Eigen matrix
	Eigen::MatrixXd Input(Input_Data.size(), Input_Data[0].size() + 1);										// This allows for input data to hold more than one set of inputs
	for (std::vector<double>::size_type i = 0; i < Input_Data.size(); i++) {
		for (std::vector<double>::size_type j = 0; j < Input_Data[0].size(); j++) {
			Input(i, j) = Input_Data[i][j];
		}
		Input(i, Input_Data[0].size()) = 1;																	// This is the input for the neuron bias. It is always set to 1. Value allows for easy matrix multiplication
	}

	for (std::vector<std::vector<Neuron>>::size_type i = 0; i < Network.size(); i++) {
		ANN::Layer.resize(Input_Data.size(), ANN::Network[i].size());										// Stores output of each neuron in given layer in a row. Each row represents one set of inputs.
		for (std::vector<Neuron>::size_type j = 0; j < Network[i].size(); j++) {
			// Calculate the output of neuron.
			Network[i][j].Input = Input;																	// Stores a copy of inputs in neuron - useful for error checking
			Network[i][j].Output = Network[i][j].Input*Network[i][j].Weights;								// Calculate output of neuron
			for (int y = 0; y < Network[i][j].Output.rows(); y++) {
				for (int x = 0; x < Network[i][j].Output.cols(); x++) {
					Network[i][j].Output(y, x) = Activation_Functions(Network[i][j].Output(y, x), false);	// Apply activation function
				}
			}

			// Store output of neuron in Layer matrix to be used as input for next layer
			for (int y = 0; y < ANN::Layer.rows(); y++) {
				ANN::Layer(y, j) = Network[i][j].Output(y, 0);
			}
		}

		// Update input data for next layer
		Input.resize(Layer.rows(), Layer.cols());
		Input = ANN::Layer;

		// Add the value of 1 to each row to take into account neuron bias needs an input value of one.
		Input.conservativeResize(Input.rows(), Input.cols() + 1);											// Resize matrix without data loss
		Eigen::VectorXd Neuron_Bias_Input = Eigen::VectorXd::Constant(Input.rows(), 1);						// Create a temporary vector of size = training set size and initialized to 1
		Input.col(Input.cols() - 1) = Neuron_Bias_Input;													// Add vector to input matrix
	}
}
double ANN::Activation_Functions(double X, bool Differential)
{
	switch (Choosen_Activation_Function) {
	case(Idenity):
		// Limits -inf -> inf
		if (Differential == true) { return 1; }
		else return X;
		break;
	case(Sigmoid):
		// Limits 0 -> 1
		if (Differential == true) { return X*(1 - X); }
		else return 1 / (1 + exp(-X));
		break;
	case(TanH):
		// Limits -1 -> 1
		if (Differential == true) { return (4 * cosh(X)*cosh(X)) / ((cosh(2 * X) + 1)*(cosh(2 * X) + 1)); }
		else return tanh(X);
		break;
	case(ArcTan):
		// Limits -pi/2 -> pi/2
		if (Differential == true) { return 1 / (X*X + 1); }
		else return atan(X);
		break;
	case(Softsign):
		// Limits -1 -> 1
		if (Differential == true) { return 1 / ((1 - abs(X))*(1 - abs(X))); }
		else return X / (1 - abs(X));
		break;
	case(Sinusoid):
		// Limits -1 -> 1
		if (Differential == true) { return cos(X); }
		else return sin(X);
		break;
	case(Gaussian):
		// Limits 0 -> 1
		if (Differential == true) { return -2 * X*exp(-1 * X*X); }
		else return exp(-1 * X*X);
		break;

	default:
		break;
	}

	return 0;
}