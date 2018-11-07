/***************************************************
This class is used for projecting a high dimensional
data to a real value (1d) so that simple comparison 
at a given node is possible
***************************************************/

#pragma once
#include "Sample.h"

class FeatureFactory
{
public:

	FeatureFactory(Eigen::MatrixXf& neighborhood, Features feat);
	bool computeFeature();

	// find the k-nn of each point in this local neighborhood (here k is half the 
	// number of points in the neighborhood) so that voxels of different sizes
	// can be constructed 
	void localNeighbors();
	void buildVoxels();
	Eigen::MatrixXi getLocalNeighbors() { return _localIndices; }
	Eigen::MatrixXf getLocalDists() { return _localDists; }
	static const int numOfPossibleProjections = 10;

private:
	bool redColorDiff();
	bool greenColorDiff();
	bool blueColorDiff();
	bool xDiff();
	bool yDiff();
	bool zDiff();
	Eigen::MatrixXf _neighborhood;
	Features _feat;
	Eigen::MatrixXi _localIndices;
	Eigen::MatrixXf _localDists;
	std::vector<Eigen::MatrixXf> _voxels;
};



