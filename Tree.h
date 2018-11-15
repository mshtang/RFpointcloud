#pragma once

#include <iostream>
#include <cmath>
#include "Sample.h"
#include "Node.h"
#include "FeatureFactory.h"


/*****************************************************************************
 * This class holds the basic element of a forest. A tree is grown by Nodes. It 
 * stops growing if either the maximal depths is reached, or the number of 
 * samples at a node is less than minNumSamplesPerLeaf (in which case the 
 * subtree is stopped from growing but not necessarily other subtrees)
 * **************************************************************************/
class Tree
{
public:

	Tree(int maxDepth, int numFeatPerNode, int minNumSamplesPerLeaf);
	
	void train(Sample *sample);
	
	std::vector<float> predict(Eigen::MatrixXf &testNeigh);
	
	inline std::vector<Node*> getTreeNodes() { return _treeNodes; }

	void createLeaf(int nodeId, int classLabel, float prob);
	
	int searchNode(Eigen::MatrixXf & testNeigh, int nodeId);

private:
	int _maxDepth;
	int _numNodes; // number of nodes = 2^_maxDepth - 1;
	int _minNumSamplesPerLeaf;
	int _numFeatPerNode;
	std::vector<Node*> _treeNodes;
};