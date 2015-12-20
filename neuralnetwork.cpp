//This program is to use neural network algorithm to predict USDCAD rate.
//Input files part: I: train.xlsx, input variables are high correlated with USDCAD rate.
//It includes original data sets: 1 WTI oil price, 2 Brent oil price, 3 SP500 index, 4 SPTSX index (SP Toronto)
// 5 US10Y yield, 6 CAD10Y yield, 7 Gold Spot price, 8 dollar index
// All data sets are normalized by feature scaling method.
//About 90% data sets are used for training, 10% for test.
//The neuralnetwork.cpp is the main program, input train01.txt and test01.txt.
//The output results (i=5000, iteration loop for the deep learning)are showed in train.xlsx.
//This program is free to use, you can redistribute it
//and modify it under the software license.
//Program Author: Zhen Qian (Martin), Rutgers University
//Email:qianzhen77@hotmail.com, or zhen.qian@rutgers.edu.com
//This program is distributed in the hope that it will be useful,
//but without any warranty for a particular purpose.
//This program is passed by Code::Blocks.
// Copyright (c) Dec 2015

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

using namespace std;

struct neuron;

//define the weight file, deltaWeight is used for activation function derivative term
struct connection{
    double weight;
    double deltaWeight;
};

typedef vector<neuron> Layer;
typedef vector<connection> connections;

//=========class neuron===============
class neuron{
public:
    neuron(int numOutput, int index_);
    void setOutputValue(double value){outputValue = value;}
    double getOutputValue() const { return outputValue; }
    void feedForward(const Layer& prevLayer);
    void calcOutputGradients(double targetValue);
    void calcHiddenGradients(const Layer& nextLayer);
    void updateWeights(Layer& prevLayer);

private:
    static double eta;
    static double alpha;

    int index;
    connections outputWeights;
    double gradient;
    double outputValue;
//use the tanh[-1...1] for optimization
//can try other functions (Sigmoid function) for optimization.
//Derivative function is used to calculate the delta weight change
    double activationFunction(double x) { return tanh(x); }
    double activationFunctionDer(double x) {   return 1.0 - x * x; }
//random number [0...1]
    double randomWeight() { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer& nextLayer) const;
};

//eta:learning rate, [0...1]is the gradient decent contribution
//alpha: momentum term, [0...1] that keeps a moving average
//of gradient descent weight change contribution, thus smooth the overall weight changes
double neuron::eta   =  0.10;
double neuron::alpha =  0.30;

neuron::neuron(int numOutput, int index_){
    for(int i = 0; i < numOutput; ++i){
        outputWeights.push_back(connection());
//Initialization the weights by random numbers.
        outputWeights.back().weight=randomWeight();
    }
    index = index_;
}

//calculation the neuron's output by f(x)=weight[i]*input[i]
void neuron::feedForward(const Layer& prevLayer){
    double sum = 0.0;
    for(int n = 0; n < prevLayer.size(); ++n){
        sum+= prevLayer[n].getOutputValue()*prevLayer[n].outputWeights[index].weight;
    }
    setOutputValue(activationFunction(sum));
}

//Output layer gradient descent
void neuron::calcOutputGradients(double targetValue){
    double delta=targetValue-outputValue;
    gradient=delta*activationFunctionDer(outputValue);
}

//hidden layer gradient descent
void neuron::calcHiddenGradients(const Layer& nextLayer){
    double dow=sumDOW(nextLayer);
    gradient=dow*activationFunctionDer(outputValue);
}

//sum the neuron's error
double neuron::sumDOW(const Layer& nextLayer) const{
    double sum=0.0;
    for(int i = 0; i < nextLayer.size() - 1; ++i){
        sum+=outputWeights[i].weight*nextLayer[i].gradient;
    }
    return sum;
}

//update the weights, old weight+ deltaWeight
void neuron::updateWeights(Layer& prevLayer){
    for(int i = 0; i < prevLayer.size(); ++i){
        neuron& neuron = prevLayer[i];
        double oldDeltaWeight=neuron.outputWeights[index].deltaWeight;
        double newDeltaWeight=eta*neuron.outputValue*gradient+alpha*oldDeltaWeight;
        neuron.outputWeights[index].deltaWeight=newDeltaWeight;
        neuron.outputWeights[index].weight+=newDeltaWeight;
    }
}

//==========================define the network class================
class network{
public:
    network(const vector <int> & topology);
    void feedForward(const vector <double> & inputValue);
    void backProp(const vector <double> & targetValue);
    void getOutput(vector <double> & outputValue) const;

private:
    typedef vector<Layer> Layers;
    Layers layers;
    double rmse;
    double recentAvgError;
    double recentAvgSmoothingFactor;
};

//set up layers and fill up with neurons
network::network(const vector <int> & topology){
    int layerNum_=topology.size();
    for(int layerNum=0; layerNum<layerNum_;++layerNum){
        layers.push_back(Layer());
        int numOutput=layerNum==layerNum_-1 ? 0:topology[layerNum+1];
        Layer& currentLayer = layers.back();
        for(int neuronNum=0; neuronNum<=topology[layerNum]; ++neuronNum){
            currentLayer.push_back(neuron(numOutput,neuronNum));
        }
//set up the bias term value 1.0 for input and hidden layer
        currentLayer.back().setOutputValue(1.0);
    }
}

void network::feedForward(const vector <double> & inputValue) {
 //   assert(inputValue.size() == layers[0].size() - 1);
    for(int i = 0; i < inputValue.size(); ++i){
        layers[0][i].setOutputValue(inputValue[i]);
    }
    for(int layerNum = 1; layerNum < layers.size(); ++layerNum){
        Layer& layer = layers[layerNum];
        Layer& prevLayer = layers[layerNum-1];
        for(int n = 0; n < layer.size()-1; ++n){
            layer[n].feedForward(prevLayer);
        }
    }
}

void network::backProp(const vector <double> & targetValue) {
// Calculate rmse
    Layer& outputLayer = layers.back();
    rmse=0.0;
    for(int i = 0; i < targetValue.size(); ++i){
        double delta = targetValue[i] - outputLayer[i].getOutputValue();
        rmse+=delta*delta;
    }
    rmse=sqrt(rmse/targetValue.size());
// Calculate a recent average rmse
    recentAvgError=(recentAvgError*recentAvgSmoothingFactor+rmse)/(recentAvgSmoothingFactor+1.0);
// Calculate output gradient
    for(int i = 0; i < outputLayer.size() - 1; ++i){
        outputLayer[i].calcOutputGradients(targetValue[i]);
    }
// Calculate hidden gradients
    for(int i=layers.size()-2; i > 0; --i){
        Layer& hiddenLayer=layers[i];
        Layer& nextLayer=layers[i+1];
        for(int j = 0; j < hiddenLayer.size(); ++j){
            hiddenLayer[j].calcHiddenGradients(nextLayer);
        }
    }
// Update weights
    for(int i=layers.size() - 1; i > 0; i--){
        Layer& layer= layers[i];
        Layer& prevLayer= layers[i-1];
        for(int j = 0; j < layer.size()-1; ++j){
            layer[j].updateWeights(prevLayer);
        }
    }
};

void network::getOutput(vector <double> & outputValue) const{
    outputValue.clear();
    const Layer& outputLayer = layers.back();
    for(int i = 0; i < outputLayer.size() - 1; ++i){
        outputValue.push_back(outputLayer[i].getOutputValue());
    }
}

//=========================training part===================
void train(network& net, vector <double> && inputValue, vector <double> && targetValue){
    net.feedForward(inputValue);
    net.backProp(targetValue);

}

//=============================test part================
double test(network& net, vector <double> && inputValue){
    net.feedForward(inputValue);
    vector <double>  result;
    net.getOutput(result);
//   for(double value:inputValue){
//       printf("%.8f ", value);
//   }
//    printf("%.8f\n", result[0]);
    return result[0];
}

int main(){

//========build the 8 input neurons, 9 hidden neurons and 1 output neuron.
    vector <int>  topology = {8, 10, 1};
//    vector <int>  topology = {2, 4, 1};
    network net(topology);

//==================reading the training data sets from train01.txt
    for (int i=0; i<5000; ++i){
    double v1,v2,v3,v4,v5,v6,v7,v8,v9;
    ifstream infile;
    infile.open("train01.txt");
    assert (!infile.fail( ));
    while(infile>>v1>>v2>>v3>>v4>>v5>>v6>>v7>>v8>>v9){
 //       cout<<"The train data are:"<<endl;
 //       cout<<v1<<"||"<<v2<<v3<<v4<<v5<<v6<<v7<<v8<<"||"<<v9<<endl;
        train(net, {v1, v2, v3, v4, v5, v6, v7, v8},{v9});
    }
    infile.close();

    }

//==================test prediction program by test01.txt file and get the results
    double t1,t2,t3,t4,t5,t6,t7,t8;
    ifstream in_file;
    ofstream outfile;
    in_file.open("test01.txt");
    outfile.open("out.txt");
    assert (!in_file.fail());
    assert (!outfile.fail());
    while(in_file>>t1>>t2>>t3>>t4>>t5>>t6>>t7>>t8){
 //       cout<<"The test data are :"<<endl;
 //       cout<<t1<<"||"<<t2<<t3<<t4<<t5<<t6<<t7<<t8<<"||"<<endl;
        outfile<<test(net, {t1, t2, t3, t4, t5, t6, t7, t8})<<endl;
    }
    in_file.close();
    outfile.close();

//    for(int i = 0; i < 50000; ++i){
//        train(net, {0, 0}, {0});
//        train(net, {0, 1}, {1});
//        train(net, {1, 0}, {1});
//        train(net, {1, 1}, {0});
//    }
//    test(net, {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8});
//    test(net, {0.2, 0.3,0.4,0.1,0.6,0.6,0.7,0.7});
 //   test(net, {0.8, 1});
 //   test(net, {1, 1});
}

