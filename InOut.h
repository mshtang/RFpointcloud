/*************************************************************
This class is used for preprocessing the dataset, including
building the index matrix of nearest neighbors that can be 
used in further step in a simple indexing way (constant time).
**************************************************************/

#pragma once

#include "c:/Eigen/Eigen/Dense"


class InOut {
public:
	void readPoints(const char* filename, Eigen::MatrixXf &cloud);

	void readLabels(const char* filename, Eigen::VectorXi &vec);

	// search the k nearest neighbors for each point the cloud
	// `indices` and `dists` are stored so that the query of a 
	// point's neighborhood can be done in constant time
	void searchNN(const Eigen::MatrixXf &cloud, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists);
	// for debug use
	void searchNN(const Eigen::MatrixXf & cloud, const Eigen::MatrixXf &points, Eigen::MatrixXi & indices, Eigen::MatrixXf & dists);
	
	// for debugging purposes
	void writeToDisk(const char *filename, Eigen::MatrixXf &data);
	void writeToDisk(const char * filename, Eigen::MatrixXi & data);

	void writeToDisk(const char * filename, Eigen::VectorXi & data);

	static const size_t numOfNN = 50;
};
