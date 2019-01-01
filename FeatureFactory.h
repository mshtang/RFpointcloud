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
	// if (one of the subvoxel) is empty or the projection cannot be
	// cast (e.g. a subvoxel has less than 3 points but projection type
	// is eigen value based.), false is returned, otherwise true and the
	// result is stored in res.
	bool project(float &res);
	bool castProjection(const Eigen::MatrixXf &voxel, int featType, float& res);

	// find the k-nn of each point in this local neighborhood (here k is half the 
	// number of points in the neighborhood) so that voxels of different sizes
	// can be constructed 
	Eigen::MatrixXi getLocalNeighbors() { return _localIndices; }
	Eigen::MatrixXf getLocalDists() { return _localDists; }

	//TODO: modify according to the final number of projections
	static const int numOfPossibleProjections = 27;
	
	void computeEigens(const Eigen::Matrix3f &covMat, float & majorVal, float & middleVal, float & minorVal, 
					   Eigen::Vector3f & majorAxis, Eigen::Vector3f & middleAxis, Eigen::Vector3f & minorAxis);

	void computeOBB(Eigen::MatrixXf & neigh, Eigen::MatrixXf &neighR, Eigen::Vector3f & obbMinP, Eigen::Vector3f & obbMaxP);
	std::vector<std::vector<Eigen::VectorXf>> partitionSpace(Eigen::MatrixXf & neigh);


private:
	void buildVoxels(std::vector<std::vector<Eigen::VectorXf>> &partitions);
	std::vector<Eigen::VectorXf> averageVoxels();
	float selectChannel(Eigen::VectorXf avg_voxel, int channelNo, bool convertToHSV=false);

	Eigen::Matrix3f computeCovarianceMatrix(Eigen::MatrixXf neigh);

	
	Eigen::MatrixXf _neighborhood;
	Features _feat;
	Eigen::MatrixXi _localIndices;
	Eigen::MatrixXf _localDists;
	std::vector<Eigen::MatrixXf> _voxels;
	
};



