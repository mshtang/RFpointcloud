#include "Node.h"
#include <vector>
#include "FeatureFactory.h"
#include <iostream>
#include "utils.h"

Node::Node():
	_isLeaf(false),
	_samples(nullptr),
	_bestFeat(Features()),
	_class(-1),
	_prob(0.0)
{}


float Node::computeGini(std::vector<int> & samplesId, std::vector<float> &probs)
{
	int numSamples = samplesId.size();
	int numClasses = _samples->getNumClasses();
	probs.resize(numClasses, 0);
	
	for (int i = 0; i < numSamples; ++i)
	{
		// count the number of instances of each class
		probs[(*(_samples->_labels))[samplesId[i]]]++;
	}
	
	float gini = 0;
	for (int i = 0; i < numClasses; ++i)
	{
		probs[i] = probs[i] / (float)numSamples;
		gini += (probs[i] * probs[i]);
	}
	
	return 1 - gini;
}
// calculate the gini of the samples in this node
void Node::computeNodeGini()
{
	Eigen::VectorXi samplesId = _samples->getSelectedSamplesId();
	// convert Eigen::Vector to std::vector
	std::vector<int> samplesIdVec = toStdVec(samplesId);
	_gini = computeGini(samplesIdVec, _probs);
}

// split the node into two children and compute the gini at each child node
void Node::computeInfoGain(std::vector<Node*> &nodes, int nodeId)
{
	Eigen::MatrixXf data = *(_samples->_dataset);
	Eigen::VectorXi labels =*( _samples->_labels);
	//Eigen::MatrixXf cloud = *(_samples->_cloud);
	//Eigen::VectorXi truths = *(_samples->_truths);
	// randomly samples some points from the cloud
	// and store the selected points id in sampleId
	Eigen::VectorXi sampleId = _samples->getSelectedSamplesId();
	int numSamples = sampleId.size();
	// for this work, numClasses is 8
	int numClasses = _samples->getNumClasses();
	_samples->randomSampleFeatures();
	std::vector<Features> featCandidates = _samples->getSelectedFeatures();
	int numFeats = _samples->getNumFeatures();

	// variables for storing the final parameters
	float bestInfoGain = 0;
	float bestLeftGini = 0;
	float bestRightGini = 0;
	Features bestFeat = featCandidates[0];
	std::vector<int> bestLeftChild;
	std::vector<int> bestRightChild;
	std::vector<float> bestLeftProbs;
	std::vector<float> bestRightProbs;

	for (int i = 0; i < numFeats; ++i)
	{
		// apply each candidate feature to the samples at this node
		float infoGain = 0;
		float leftGini = 0;
		float rightGini = 0;
		Features feat = featCandidates[i];
		std::vector<int> leftChildSamples;
		std::vector<int> rightChildSamples;
		std::vector<float> leftProbs;
		std::vector<float> rightProbs;
		
		for (int j = 0; j < numSamples; ++j)
		{
			Eigen::MatrixXf neigh = _samples->buildNeighborhood(sampleId[j]);
			//InOut tmp;
			//tmp.writeToDisk("./toy_dataset/firstNeigh.txt", neigh);
			FeatureFactory nodeFeat(neigh, feat);
			if (nodeFeat.computeFeature() == false)
				leftChildSamples.push_back(sampleId[j]);
			else
				rightChildSamples.push_back(sampleId[j]);
		}
		
		if (leftChildSamples.size() == 0 or rightChildSamples.size() == 0)
		{
			continue;
		}

		leftGini = computeGini(leftChildSamples, leftProbs);
		rightGini = computeGini(rightChildSamples, rightProbs);
		float leftRatio = leftChildSamples.size() / (float)numSamples;
		float rightRatio = rightChildSamples.size() / (float)numSamples;
		infoGain = _gini - leftGini * leftRatio - rightGini * rightRatio;

		if (infoGain > bestInfoGain)
		{
			bestInfoGain = infoGain;
			bestLeftGini = leftGini;
			bestRightGini = rightGini;
			bestFeat = feat;
			bestLeftChild = leftChildSamples;
			bestRightChild = rightChildSamples;
			bestLeftProbs = leftProbs;
			bestRightProbs = rightProbs;
		}
	}
	
	if (bestLeftChild.size() == 0 or bestRightChild.size()==0 or bestInfoGain<0)
	{
		createLeaf(nodes[0]->_probs);
	}
	else
	{
		_bestFeat = bestFeat;
		nodes[nodeId * 2 + 1] = new Node();
		nodes[nodeId * 2 + 2] = new Node();
		(nodes[nodeId * 2 + 1])->_gini = bestLeftGini;
		(nodes[nodeId * 2 + 2])->_gini = bestRightGini;
		(nodes[nodeId * 2 + 1])->_probs = bestLeftProbs;
		(nodes[nodeId * 2 + 2])->_probs = bestRightProbs;

		Eigen::VectorXi bestLeftChildVec = toEigenVec(bestLeftChild);
		Eigen::VectorXi bestRightChildVec = toEigenVec(bestRightChild);

		Sample *leftSamples = new Sample(_samples, bestLeftChildVec);
		Sample *rightSamples = new Sample(_samples, bestRightChildVec);
		(nodes[nodeId * 2 + 1])->_samples = leftSamples;
		(nodes[nodeId * 2 + 2])->_samples = rightSamples;
	}
}

// make the most frequent class as the class of this leaf node
void Node::createLeaf(std::vector<float> priorDistr)
{
	_class = 0;
	int numClasses = _samples->getNumClasses();
	int nullClasses = 0; // classes have no instances
	float normalizeFactor = 0;
	for (int i = 0; i < numClasses; ++i)
	{
		if (priorDistr[i] == 0)
		{
			_probs[i] = 0;
			nullClasses++;
		}
		else
		{
			_probs[i] = _probs[i] / priorDistr[i];
			normalizeFactor += _probs[i];
		}
	}
	for (int i = 0; i < numClasses; ++i)
	{
		_probs[i] /= normalizeFactor;
	}
	
	_prob = _probs[0];
	for (int i = 1; i < numClasses; ++i)
	{
		if (_probs[i] > _prob)
		{
			_class = i;
			_prob = _probs[i];
		}
	}
	_isLeaf = true;
}


bool Node::isHomogenous(){
	int numSamples = _samples->getSelectedSamplesId().size();
	int pointId1 = _samples->getSelectedSamplesId()[0];
	for (int i = 1; i < numSamples - 1; ++i)
	{
		if (_samples->getSelectedSamplesId()[i] != pointId1)
			return false;
	}
	return true;
}
