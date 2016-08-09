#include "ANN.h"

int main() {
	ANN Network;
	Network.Create_Network({ 2,1 });
	Network.Choosen_Activation_Function = Network.Activation_Function::Sigmoid;
	
	std::vector<std::vector<double>> Data = { {0,0,0},{0,1,1},{1,0,1},{1,1,1} }; // OR
	Network.Train_Network(Data);
	Network.Print_Diagnostics();

	// Testing
	std::vector<std::vector<double>> Test{ {0,0},{0,1},{1,0},{1,1} };
	std::vector<std::vector<double>> Out = Network.Get_Output_Multi_Data_Set(Test);
	for (auto i = 0; i < Out.size(); i++) {
		for (auto j = 0; j < Out[i].size(); j++) {
			std::cout << Out[i][j] << "  ";
		}
		std::cout << std::endl;
	}
	return 0;
}