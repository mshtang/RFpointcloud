#include "Tree.h"

Tree::Tree(int maxDepth, int numFeatPerNode, int minNumSamplesPerLeaf, float giniThresh):
	_maxDepth(maxDepth),
	_numFeatPerNode(numFeatPerNode),
	_minNumSamplesPerLeaf(minNumSamplesPerLeaf),
	_giniThresh(giniThresh),
	_numNodes(static_cast<int>(std::pow(2, _maxDepth)-1)),
	_treeNodes(_numNodes, nullptr)
{
}

void Tree::train(Sample *sample)
{
	// for all the possible features, only numFeats features at each node
	// is calculated
	//int numFeats = sample->getNumFeatures();
	Eigen::VectorXi selectedSamplesId = sample->getSelectedSamplesId();
	Sample *nodeSample = new Sample(sample, selectedSamplesId);
	_treeNodes[0] = new Node();
	_treeNodes[0]->_samples = nodeSample;
	// calculate the probability and gini
	_treeNodes[0]->computeNodeGini();
	for (int i = 0; i < _numNodes; ++i)
	{
		int parentId = (i - 1) / 2;
		// if current node's parent is null, continue
		if (_treeNodes[parentId] == nullptr)
			continue;
		// if current node's parent is a leaf, continue
		if (i > 0 and _treeNodes[parentId]->isLeaf())
			continue;
		// if maxDepth is reached, set current node as a leaf node
		if (i * 2 + 1 >= _numNodes)
		{
			_treeNodes[i]->createLeaf();
			continue;
		}
		// if current samples in this node is less than the threshold
		// set current node as a leaf node and continue
		if ((_treeNodes[i]->_samples->getNumSelectedSamples() <= _minNumSamplesPerLeaf) or
			_treeNodes[i]->isHomogenous())
		{
			_treeNodes[i]->createLeaf();
			continue;
		}
		//_treeNodes[i]->_samples->randomSampleFeatures();
		_treeNodes[i]->computeInfoGain(_treeNodes, i, _giniThresh);
	}
}

void Tree::createLeaf(int nodeId, int classLabel, float prob)
{
	_treeNodes[nodeId] = new Node();
	_treeNodes[nodeId]->setLeaf(true);
	_treeNodes[nodeId]->setClass(classLabel);
	_treeNodes[nodeId]->setProb(prob);
}

std::vector<float> Tree::predict(Eigen::MatrixXf &testNeigh)
{
	int nodeId = 0;
	nodeId = searchNode(testNeigh, 0);
	return _treeNodes[nodeId]->_probs;
}

int Tree::searchNode(Eigen::MatrixXf &testNeigh, int nodeId)
{
	if (_treeNodes[nodeId]->isLeaf())
		return nodeId;
	else
	{
		FeatureFactory testNodeFeat(testNeigh, _treeNodes[nodeId]->getBestFeature());
		if (testNodeFeat.computeFeature() == false)
			return searchNode(testNeigh, nodeId * 2 + 1);
		else
			return searchNode(testNeigh, nodeId * 2 + 2);
	}
}