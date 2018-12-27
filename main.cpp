#include <iostream>
#include "InOut.h"
#include "Sample.h"
//#include "FeatureFactory.h"
//#include "Node.h"
#include "Forest.h"
//#include "utils.h"
#include <chrono>

int main(int argc, char **argv)
{
	bool functionDebug = false;
	if (functionDebug)
	{
		InOut ioObj;
		Eigen::MatrixXf neigh;
		ioObj.readPoints("./TestEnv/neighborhood1.txt", neigh);
		Features feat;
		feat._featType = 24;
		feat._numVoxels = 1;
		feat._pointId.push_back(5);
		FeatureFactory nodeFeat(neigh, feat);
		float result = 0.0f;
		bool flag = nodeFeat.project(result);
	}
	else
	{
		// change here to enter debug mode
		bool debug = false;
		// to control whether training from scratch or use a pretrained model
		bool directTrain = false;

		// give respective dataset/label path
		std::string trainSetPath = "./datasets/bildstein_station1_xyz_intensity_rgb_downsample_1000.txt";
		std::string trainLabelPath = "./datasets/bildstein_station1_xyz_intensity_rgb_downsample_1000.labels";
		std::string valSetPath = "./datasets/testsets/bildstein3_part_2.txt";
		std::string valLabelPath = "./Results/exp4predicted.labels";
		std::string cloudPath = "./datasets/other datasets/bildstein_station3_xyz_intensity_rgb_dropped.txt";
		//std::string truthPath = "./datasets/bildstein_station1_xyz_intensity_rgb_dropped.labels";
		// if training the real dataset, give a path to save the model
		std::string modelPath = "./models/exp3.model";
		// statistics file of the model
		std::string statsPath = "./models/exp3stats.txt";

		// parameters to modify
		int numTrees = 50;
		int maxDepth = 20;
		int minSamplesPerLeaf = 20;
		int featsPerNode = 30;

		if (debug)
		{
			modelPath = "./TestEnv/test.model";
			statsPath = "./TestEnv/test_stats.txt";
			trainSetPath = "./TestEnv/downsampled.txt";
			trainLabelPath = "./TestEnv/downsampled.labels";
			valSetPath = "./TestEnv/downsampled.txt";
			valLabelPath = "./TestEnv/predict.labels";
			cloudPath = "./datasets/bildstein_station1_xyz_intensity_rgb_downsample.txt";
			numTrees = 2;
			maxDepth = 5;
			minSamplesPerLeaf = 5;
			featsPerNode = 10;
		}

		// in this work it's 8
		int numClasses = 8;

		if (directTrain)
		{
			// prepare dataset
			InOut trainObj;
			Eigen::MatrixXf cloud;
			//Eigen::VectorXi truths;
			Eigen::MatrixXf trainset;
			Eigen::VectorXi trainlabels;

			// reading dataset and labels from given file path
			trainObj.readPoints(cloudPath.c_str(), cloud);
			//trainObj.readLabels(truthPath.c_str(), truths);
			trainObj.readPoints(trainSetPath.c_str(), trainset);
			trainObj.readLabels(trainLabelPath.c_str(), trainlabels);

			Eigen::MatrixXi trainIndex;
			Eigen::MatrixXf trainDists;
			Eigen::VectorXi predictedLabels;

			trainObj.searchNN(cloud, trainset, trainIndex, trainDists);
			//trainObj.writeToDisk("./toy_dataset/trainsetIndex.txt", trainIndex);
			//trainObj.writeToDisk("./toy_dataset/trainsetDist.txt", trainDists);

			RandomForest randObj(numTrees, maxDepth, minSamplesPerLeaf);
			auto start = std::chrono::system_clock::now();
			randObj.train(&trainset, &trainlabels, &trainIndex, &trainDists,
						  numClasses, featsPerNode, &cloud);
			auto end = std::chrono::system_clock::now();
			double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
			std::cout << "Training takes: " << elapsed << "s" << std::endl;
			randObj.saveModel(modelPath.c_str(), statsPath.c_str());

			start = std::chrono::system_clock::now();
			//randObj.predict(valSetPath.c_str(), predictedLabels);
			randObj.predict(cloudPath.c_str(), valSetPath.c_str(), predictedLabels);
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
			std::cout << "Predicting takes: " << elapsed << "s" << std::endl;

			trainObj.writeToDisk(valLabelPath.c_str(), predictedLabels);
		}
		else
		{
			RandomForest randObj(modelPath.c_str());
			Eigen::VectorXi predictedLabels;
			auto start = std::chrono::system_clock::now();
			//randObj.predict(valSetPath.c_str(), predictedLabels);
			randObj.predict(cloudPath.c_str(), valSetPath.c_str(), predictedLabels);
			auto end = std::chrono::system_clock::now();
			double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
			std::cout << "Predicting takes: " << elapsed << "s" << std::endl;
			InOut testObj;
			testObj.writeToDisk(valLabelPath.c_str(), predictedLabels);
		}
	}
	system("pause");
	return 0;
}
