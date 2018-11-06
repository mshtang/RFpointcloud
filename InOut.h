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
	void searchNN(const Eigen::MatrixXf &cloud, const size_t k, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists);
	
	// for debugging purposes
	void writeToDisk(const char *filename, Eigen::MatrixXf &data);
	void writeToDisk(const char * filename, Eigen::MatrixXi & data);

	/*template<typename Der>
	void writeToDisk(const char *filename, Eigen::MatrixBase<Der> &data);*/
};

//template<typename Der>
//inline void InOut::writeToDisk(const char * filename, Eigen::MatrixBase<Der> &data)
//{	
//
//	std::ofstream output(filename);
//	for (int r = 0; r < data.rows(); ++r)
//	{
//		for (int c = 0; c < data.cols(); ++c)
//		{
//			output << data(r, c) << " ";
//		}
//		output << std::endl;
//	}
//}
