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

	FeatureFactory(std::vector<Eigen::MatrixXf>& voxels, Features feat, 
				   Eigen::VectorXf & ptEValues, Eigen::VectorXf & ptEVectors, 
				   Eigen::MatrixXf & voxelEValues, Eigen::MatrixXf & voxelEVectors);
	
	// if (one of the subvoxel) is empty or the projection cannot be
	// cast (e.g. a subvoxel has less than 3 points but projection type
	// is eigen value based.), false is returned, otherwise true and the
	// result is stored in res.
	bool project(float &res);
	

	//TODO: modify according to the final number of projections
	static const int numOfPossibleProjections = 27;
	
	Eigen::MatrixXf recoverNeighborhood(std::vector<Eigen::MatrixXf> voxels);



	//private:
	bool castProjection(const Eigen::MatrixXf & voxel, int featType, Eigen::VectorXf & selectedEigenValues, Eigen::VectorXf & selectedEigenVectors, float & testResult);
	float selectChannel(Eigen::VectorXf avg_voxel, int channelNo, bool convertToHSV=false);

	std::vector<Eigen::MatrixXf> _voxels;
	Features _feat;
	Eigen::VectorXf _ptEValues;
	Eigen::VectorXf _ptEVectors;
	Eigen::MatrixXf _voxelEValues;
	Eigen::MatrixXf _voxelEVectors;
	
};



