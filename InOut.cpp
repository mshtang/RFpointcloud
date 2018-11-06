#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "InOut.h"
#include "nanoflann.hpp"

void InOut::readPoints(const char* filename, Eigen::MatrixXf &cloud)
{
	// open file
	std::fstream ifs(filename);
	if (!ifs.is_open())
	{
		std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
		std::cerr << "Press enter to continue." << std::endl;
		std::cin.get();
		exit(-1);
	}

	// read points to list
	std::string lineBuffer;
	std::vector<std::vector<float>> points;
	int numPoints = 0;
	while (std::getline(ifs, lineBuffer))
	{
		numPoints++;
		// convert std::string to std::stringstream
		std::stringstream currLine(lineBuffer);
		std::string buffer;
		std::vector<float> point;
		while (std::getline(currLine, buffer, ' '))
		{
			point.push_back(std::atof(buffer.c_str()));
		}
		points.push_back(point);
	}
	
	cloud.resize(numPoints, 7);
	int i = 0;
	for (std::vector<std::vector<float>>::iterator p = points.begin(); p != points.end(); ++p, ++i) {
		cloud(i, 0) = (*p).at(0);
		cloud(i, 1) = (*p).at(1);
		cloud(i, 2) = (*p).at(2);
		cloud(i, 3) = (*p).at(3);
		cloud(i, 4) = (*p).at(4);
		cloud(i, 5) = (*p).at(5);
		cloud(i, 6) = (*p).at(6);
	}

	std::cout << numPoints << " points read." << std::endl;
}

void InOut::readLabels(const char* filename, Eigen::VectorXi &labels) {
	std::fstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
		std::cerr << "Press enter to continue." << std::endl;
		std::cin.get();
		exit(-1);
	}
	// read labels to list
	std::string buffer;
	std::vector<int> vlabels;
	int numLabels = 0;
	while (std::getline(ifs, buffer)) {
		numLabels++;
		vlabels.push_back(std::atoi(buffer.c_str()));
	}
	
	labels.resize(numLabels);
	int i = 0;
	for (std::vector<int>::iterator p = vlabels.begin(); p != vlabels.end(); ++p, ++i) {
		labels(i, 0) = *p;
	}

	std::cout << "Reading labels finished." << std::endl;
}

void InOut::searchNN(const Eigen::MatrixXf & cloud, const size_t k, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists)
{
	// Eigen::MatrixXf uses colMajor as default
	// copy the coords to a RowMajor matrix and search in this matrix
	// the nearest points for each datapoint
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> Matrix3fRow;
	Matrix3fRow coords = cloud.leftCols(3);
	
	// different max_leaf values only affect the search speed 
	// and any value between 10 - 50 is reasonable
	const int max_leaf = 10;
	nanoflann::KDTreeEigenMatrixAdaptor<Matrix3fRow> mat_index(coords, max_leaf);
	mat_index.index->buildIndex();
	/*Eigen::MatrixXi ret_indices_mat(cloud.rows(), k);
	Eigen::MatrixXf ret_dists_mat(cloud.rows(), k);*/
	indices.resize(cloud.rows(), k);
	dists.resize(cloud.rows(), k);
	// do a knn search
	for (int i = 0; i < coords.rows(); ++i) 
	{
		// coords is RowMajor so coords.data()[i*3+0 / +1  / +2] represents the ith row of coords
		std::vector<float> query_pt{ coords.data()[i*3+0], coords.data()[i*3+1], coords.data()[i*3+2] };
		
		std::vector<size_t> ret_indices(k);
		std::vector<float> out_dists_sqr(k);
		nanoflann::KNNResultSet<float> resultSet(k);
		resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		for (size_t j = 0; j < k; ++j) 
		{
			indices(i, j) = ret_indices[j];
			dists(i, j) = std::sqrt(out_dists_sqr[j]);
		}
	}
	std::cout << "Searching for knn finished. " << std::endl;
}



void InOut::writeToDisk(const char *filename, Eigen::MatrixXf &data)
{
	std::ofstream output(filename);
	for (int r = 0; r < data.rows(); ++r) 
	{
		for (int c = 0; c < data.cols(); ++c) 
		{
			output << data(r, c) << " ";
		}
		output << std::endl;
	}
}

void InOut::writeToDisk(const char *filename, Eigen::MatrixXi &data)
{
	std::ofstream output(filename);
	for (int r = 0; r < data.rows(); ++r) 
	{
		for (int c = 0; c < data.cols(); ++c) 
		{
			output << data(r, c) << " ";
		}
		output << std::endl;
	}
}