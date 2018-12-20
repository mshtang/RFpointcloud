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
	float castProjection();

	// find the k-nn of each point in this local neighborhood (here k is half the 
	// number of points in the neighborhood) so that voxels of different sizes
	// can be constructed 
	Eigen::MatrixXi getLocalNeighbors() { return _localIndices; }
	Eigen::MatrixXf getLocalDists() { return _localDists; }

	//TODO: modify according to the final number of projections
	static const int numOfPossibleProjections = 17;
	std::vector<Eigen::Matrix3f> computeCovarianceMatrix();

	Eigen::Vector3f computeEigenValues(Eigen::Matrix3f covMat);

	std::vector<Eigen::Vector3f> computeEigenValues(std::vector<Eigen::Matrix3f> covTensor);

private:
	void localNeighbors();
	void buildVoxels();
	std::vector<Eigen::VectorXf> averageVoxels();
	float compareChannels(std::vector<Eigen::VectorXf> avg_voxels, int channelNo, bool convertToHSV=false);

	Eigen::Matrix3f computeCovarianceMatrix(Eigen::MatrixXf neigh);

	
	Eigen::MatrixXf _neighborhood;
	Features _feat;
	Eigen::MatrixXi _localIndices;
	Eigen::MatrixXf _localDists;
	std::vector<Eigen::MatrixXf> _voxels;

	Eigen::Matrix<float, 4, 1> pt;
	
};



