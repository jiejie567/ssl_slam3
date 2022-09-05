#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]){
    if(argc != 2){
        std::cerr << "usage: main <path-to-exported-script-module>\n";
        return -1;
    }

    torch::Device device(torch::kCUDA);
    // Deserialize the ScriptModule from a file using torch::jit::load()
    torch::jit::script::Module module = torch::jit::load(argv[1]);
    module.to(device);

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(device));

    // Exectute the model
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << output.slice(/*dims=*/1, /*start=*/0, /*end=*/5) << '\n';

    std::cout << "ok\n";
}
