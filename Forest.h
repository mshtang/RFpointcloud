#pragma once

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"Tree.h"
#include"Sample.h"

class RandomForest
{
public:
	/*************************************************************
	numTrees: number of trees in this forest
	maxDepth: the max possible depth of each tree
	minSamplesPerLeaf: set the terminating condition for growing a tree
	**************************************************************/
	RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf);
	
	RandomForest(const char* modelPath);
	~RandomForest();

	/***************************************************************
	trainset:: the training set with each row repsresenting a datapoint (dims: N*7)
	labels: the corresponding labels for each datapoint (dims: N*1)
	inidces: the indices of nearest neighbors for each datapoint (dims: N*k with k being
	the number of nearest neighbors in a neighborhood)
	dists: the corresponding dists from the query point to its neighbors (dims: N*k)
	numClasses: number of classes, for this work it's 8
	numFeatsPerNode: number of features used at each node, the potential number of features
	at each node is huge, so only a small fraction of that is used to limit
	the computational effort
	**************************************************************/
	void train(Eigen::MatrixXf *trainset, Eigen::VectorXi *labels, 
			   Eigen::MatrixXi *indices, Eigen::MatrixXf *dists, 
			   int numClasses, int numFeatsPerNode,
			   Eigen::MatrixXf *cloud, Eigen::VectorXi *truths);
	
	/***************************************************************************
	 * pass in the file path to the test dataset. within this method, an InOut
	 * object will be instantiated will then will handle the search for nearest
	 * neighbors. predictedLabels stores the predicted label for each data point.
	 ***************************************************************************/
	void predict(const char* testDataPath, Eigen::VectorXi &predictedLabels);
	int predict(Eigen::MatrixXf &testNeigh);

	void saveModel(const char* path, const char* statFilePath=nullptr);
	void readModel(const char* path);

	void setTrainSample(Sample* trainsample) { _trainSample = trainsample; }

private:
	//the feature number used in a node while training (stop criterion)
	int _numFeatsPerNode;  
	int _numTrees;
	//the max depth which a tree can reach (stop criterion)
	int _maxDepth; 
	//the number of classes
	int _numClasses;  
	//terminate condition the min samples at a node (stop criterion)
	int _minSamplesPerLeaf;  

	std::vector<Tree*> _forest;
	//hold the whole dataset and some other infomation
	Sample *_trainSample;  
};
