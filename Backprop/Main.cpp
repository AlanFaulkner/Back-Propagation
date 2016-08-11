#include "ANN.h"
#include <fstream>

int main() {
	// Build credit data matrix

	std::vector<std::vector<double>> Credit_Data;

	std::ifstream Input("Credit_Data.txt");
	int id = 0;
	while (Input.good()) {
		std::vector<double> Row;
		Row.push_back(id);
		std::string line_str;
		std::getline(Input, line_str);
		std::istringstream line(line_str);
		while (line.good()) {
			double x;
			line >> x;
			Row.push_back(x);
		}
		id++;
		Credit_Data.push_back(Row);
	}
	Credit_Data.pop_back(); // crude way to deal with blank line at end of input file.

	// Randomly shuffle credit data before selecting training sample
	std::random_shuffle(Credit_Data.begin(), Credit_Data.end());

	// build training set. comprises of 100 elements only
	std::ofstream Tdata("Training_Data.dat");
	std::vector<std::vector<double>> Training_Data;
	for (int i = 0; i < 500; i++) {
		std::vector<double> Row;
		for (std::vector<std::vector<double>>::size_type j = 1; j < Credit_Data[i].size(); j++) {
			Row.push_back(Credit_Data[i][j]);
			Tdata << std::setprecision(5) << std::setw(9) << Credit_Data[i][j] << " ";
		}
		Training_Data.push_back(Row);
		Tdata << std::endl;
	}

	// Save training data
	Tdata.close();

	// Train Network

	ANN Network;
	//Network.Load_Network("Trained.net");
	Network.Create_Network({ 15,10,1 });
	Network.Choosen_Activation_Function = Network.Activation_Function::Sigmoid;
	Network.Train_Network(Training_Data);
	Network.Save_Network("Trained.net");

	// Validation of credit data
	// Just use all credit data
	std::cout << "Validation\n\n";
	std::ofstream Out("Results.dat");
	std::ofstream Valid("Validation.text");
	for (std::vector<std::vector<double>>::size_type i = 0; i < Credit_Data.size(); i++) {
		std::vector<double> Element = Credit_Data[i];
		Element.pop_back();
		Element.erase(Element.begin()); // Remove data ID column.

		for (int q = 0; q < Element.size(); q++) { Valid << std::setprecision(5) << std::setw(9) << Element[q] << " "; }
		Valid << std::endl;

		double Target = Credit_Data[i][Credit_Data[i].size() - 1];
		std::vector<double> Result = Network.Get_Output_Single_Data_Set(Element);
		
		for (std::vector<double>::size_type a = 0; a < Result.size(); a++) { Out << Result[a] << " "; }
		Out << Target << std::endl;
	}
	Out.close();
	Valid.close();

	//std::cout << "Data Validation: " << Score << "/" << Credit_Data.size() - 1 << std::endl << std::endl;

	return 0;
}