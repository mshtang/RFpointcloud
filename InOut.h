/*************************************************************
This class is used for preprocessing the dataset, including
building the index matrix of nearest neighbors that can be 
used in further step in a simple indexing way (constant time).
**************************************************************/

#pragma once

#include <vector>
#include "c:/Eigen/Eigen/Dense"

class InOut {
public:
	
	void readPoints(const char* filename, Eigen::MatrixXf &cloud);

	void readLabels(const char* filename, Eigen::VectorXi &vec);

	// search the k nearest neighbors for each point the cloud
	// `indices` and `dists` are stored so that the query of a 
	// point's neighborhood can be done in constant time
	void searchNN(const Eigen::MatrixXf &cloud, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists);
	// for each point in the dataset search for its k nearest neighbors
	// in the cloud
	void searchNN(const Eigen::MatrixXf &cloud, const Eigen::MatrixXf &dataset, Eigen::MatrixXi & indices, Eigen::MatrixXf & dists);
	
	// partition the neighborhood of each point into 27 subvoxels
	// input: cloud, the original point cloud
	// input: points, during training, they are a subset of the cloud, during test,
	//		  they are equivalent to cloud
	// output: indices, (numOfPoints, k), the indices of nn for each point
	// output: dists, (numOfPoint, k), the dists of nn for each point
	// output: ptEigenValues, (numOfPoints, 3), eigenvalues associated with the neighborhood
	//		   of each point in descending order
	// output: ptEigenVectors, (numOfPoints, 9), eigenvectors associated with the neighborhood
	//		   of each point. Each row respresents three eigen vectors, with first three the first
	//		   eigen vector (corresponding to the largest eigen value) and so on.
	// output: partsEigenValues, (numOfPoints, 27, 3), eigenvalues associated with the subvoxels
	//		   in the neighborhood of each point. The neighborhood of each point is partitioned into
	//		   27 subvoxels. If a subvoxel contains less than 3 points, then the eigenvalues can't be
	//		   computed and the eigenvalues are set to -1, -1, -1;
	// output: partsEigenVectors, (numOfPoints, 27, 9), similar to ptEigenVectors
	// output: vecVecVecIndices, (numOfPoints, 27, x), the indices mapped to the orignal cloud of the 
	//		   points in each subvoxel. Since the number of points in each subvoxel is not fixed,
	//		   so, the innermost vec, vector<int>, stores the actual indices; 27 such vecInt is combined
	//		   into a vecVecIndices, vector<vector<int>>, for each point. Finally all the vecVecIndices
	//		   are gathered in vecVecVecIndices.
	void partitionNeighborhood(Eigen::MatrixXf & cloud, Eigen::MatrixXf & points, 
							   Eigen::MatrixXi & indices, Eigen::MatrixXf & dists, 
							   Eigen::MatrixXf & ptEigenValues, Eigen::MatrixXf & ptEigenVectors, 
							   std::vector<Eigen::MatrixXf>& partsEigenValues, std::vector<Eigen::MatrixXf>& partEigenVectors, 
							   std::vector<std::vector<std::vector<int>>>& vecVecVecIndices);
	
	// for debugging purposes
	void writeToDisk(const char *filename, Eigen::MatrixXf &data);
	void writeToDisk(const char * filename, Eigen::MatrixXi & data);
	void writeToDisk(const char * filename, Eigen::VectorXi & data);


	void computeOBB(Eigen::MatrixXf & neigh, Eigen::MatrixXf & neighR, Eigen::Vector3f & obbMinP, Eigen::Vector3f & obbMaxP, float & majorVal, float & midVal, float & minorVal, Eigen::Vector3f & majorAxis, Eigen::Vector3f & midAxis, Eigen::Vector3f & minorAxis);

	Eigen::Matrix3f computeCovarianceMatrix(Eigen::MatrixXf & neigh);

	void computeEigens(const Eigen::Matrix3f & covMat, float & majorVal, float & middleVal, float & minorVal, Eigen::Vector3f & majorAxis, Eigen::Vector3f & middleAxis, Eigen::Vector3f & minorAxis);

	static const size_t numOfNN = 100;
};
