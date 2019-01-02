#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "InOut.h"
#include "nanoflann.hpp"
#include <chrono>

void InOut::readPoints(const char* filename, Eigen::MatrixXf &cloud)
{
	std::cout << "Reading points ... ";
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

	std::cout << "Done! " << numPoints << " points read." << std::endl;
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

void InOut::searchNN(const Eigen::MatrixXf &cloud, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists)
{
	std::cout << "Searching kNN starts ... ";
	int k1 = numOfNN + 1;
	// Eigen::MatrixXf uses colMajor as default
	// copy the coords to a RowMajor matrix and search in this matrix
	// the nearest points for each datapoint
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> Matrix3fRow;
	Matrix3fRow coords = cloud.leftCols(3);

	auto start = std::chrono::system_clock::now();
	// different max_leaf values only affect the search speed 
	// and any value between 10 - 50 is reasonable
	const int max_leaf = 10;
	nanoflann::KDTreeEigenMatrixAdaptor<Matrix3fRow> mat_index(coords, max_leaf);
	mat_index.index->buildIndex();
	/*Eigen::MatrixXi ret_indices_mat(cloud.rows(), k);
	Eigen::MatrixXf ret_dists_mat(cloud.rows(), k);*/
	indices.resize(coords.rows(), k1);
	dists.resize(coords.rows(), k1);
	// do a knn search
	for (int i = 0; i < coords.rows(); ++i) // for each point
	{
		// coords is RowMajor so coords.data()[i*3+0 / +1  / +2] represents the ith row of coords
		std::vector<float> query_pt{ coords.data()[i * 3 + 0], coords.data()[i * 3 + 1], coords.data()[i * 3 + 2] };

		std::vector<size_t> ret_indices(k1);
		std::vector<float> out_dists_sqr(k1);
		nanoflann::KNNResultSet<float> resultSet(k1);
		resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		for (size_t j = 0; j < k1; ++j)
		{
			indices(i, j) = ret_indices[j];
			dists(i, j) = std::sqrt(out_dists_sqr[j]);
		}
	}
	auto end = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
	std::cout << " Finished! in " << elapsed << "s." << std::endl;	
}

void InOut::searchNN(const Eigen::MatrixXf &cloud, const Eigen::MatrixXf &dataset, Eigen::MatrixXi &indices, Eigen::MatrixXf &dists)
{
	std::cout << "Searching kNN starts ... ";
	int k1 = numOfNN+1;
	// Eigen::MatrixXf uses colMajor as default
	// copy the coords to a RowMajor matrix and search in this matrix
	// the nearest points for each datapoint
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> Matrix3fRow;
	Matrix3fRow coords = cloud.leftCols(3);
	Matrix3fRow pointCoords = dataset.leftCols(3);

	auto start = std::chrono::system_clock::now();
	// different max_leaf values only affect the search speed 
	// and any value between 10 - 50 is reasonable
	const int max_leaf = 10;
	nanoflann::KDTreeEigenMatrixAdaptor<Matrix3fRow> mat_index(coords, max_leaf);
	mat_index.index->buildIndex();
	/*Eigen::MatrixXi ret_indices_mat(cloud.rows(), k);
	Eigen::MatrixXf ret_dists_mat(cloud.rows(), k);*/
	indices.resize(pointCoords.rows(), k1);
	dists.resize(pointCoords.rows(), k1);
	// do a knn search
	for (int i = 0; i < pointCoords.rows(); ++i) // for each point
	{
		// coords is RowMajor so coords.data()[i*3+0 / +1  / +2] represents the ith row of coords
		std::vector<float> query_pt{ pointCoords.data()[i * 3 + 0], pointCoords.data()[i * 3 + 1], pointCoords.data()[i * 3 + 2] };
		
		std::vector<size_t> ret_indices(k1);
		std::vector<float> out_dists_sqr(k1);
		nanoflann::KNNResultSet<float> resultSet(k1);
		resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		for (size_t j = 0; j < k1; ++j) 
		{
			indices(i, j) = ret_indices[j];
			dists(i, j) = std::sqrt(out_dists_sqr[j]);
		}
	}
	auto end = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
	std::cout << " Finished! in " << elapsed << "s." << std::endl;
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

void InOut::writeToDisk(const char *filename, Eigen::VectorXi &data)
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

void InOut::partitionNeighborhood(Eigen::MatrixXf &cloud, Eigen::MatrixXf &points,
								  Eigen::MatrixXi &indices, Eigen::MatrixXf &dists,
								  Eigen::MatrixXf &ptEigenValues, Eigen::MatrixXf &ptEigenVectors,
								  std::vector<Eigen::MatrixXf> &partsEigenValues, std::vector<Eigen::MatrixXf> &partsEigenVectors,
								  std::vector<std::vector<std::vector<int>>> &vecVecVecIndices)
{
	searchNN(cloud, points, indices, dists);
	int numPoints = points.rows();
	// for each point, there is a neighborhood associated with it
	// hence a corresponding covmat and further eigenvals and eigenvecs
	ptEigenValues.resize(numPoints, 3);
	// to simplify the data structure, eigenvectors of each covmat are 
	// flattened out and stored in a row, so the first three elements of
	// a row is the first eigenvector, the next three the second, the last
	// theree the third, in descending order
	ptEigenVectors.resize(numPoints, 9);
	// for each neighborhood of a point, we partition it into 27 subvoxels
	// so for each subvoxel, we have corresponding covmat and furhter
	// eigenvalues and eigenvectors
	// std::vector<std::vector<Eigen::MatrixXf>> partitions;
	// each element in partsEigenValues has shape (27, 3)
	// std::vector<Eigen::MatrixXf> partsEigenValues;
	// each element in partsEigenVectors has shape(27, 9)
	// std::vector<Eigen::MatrixXf> partsEigenVectors;
	// partition the neighborhood of each point
	std::cout << "Precomputing begins ... \n";
	auto startpoing = std::chrono::system_clock::now();
	for (int i = 0; i < numPoints; ++i)
	{
		float ratio = (i+1) / static_cast<float>(numPoints) * 100;
		if(static_cast<int>(ratio) % 10==0)
			std::cout << static_cast<int>(ratio) << "% of the points processed.\n";
		// all neighboring points indices
		Eigen::VectorXi candidates = indices.row(i);
		// recover neighborhood
		int n = candidates.rows();
		int d1 = cloud.cols()+1;
		Eigen::MatrixXf neigh(n, d1);
		for (int j = 0; j < n; ++j)
		{
			neigh.row(j).leftCols(d1-1) = cloud.row(candidates(j));
			neigh(j, d1 - 1) = candidates(j);
		}
		Eigen::MatrixXf neighWithoutIndex = neigh.leftCols(d1 - 1);
		Eigen::MatrixXf neighR;
		Eigen::Vector3f obbMinP;
		Eigen::Vector3f obbMaxP;
		float majorValue = 0, midValue = 0, minorValue = 0;
		Eigen::Vector3f majorAxis(0, 0, 0);
		Eigen::Vector3f midAxis(0, 0, 0);
		Eigen::Vector3f minorAxis(0, 0, 0);
		computeOBB(neighWithoutIndex, neighR, obbMinP, obbMaxP, 
				   majorValue, midValue, minorValue,
				   majorAxis, midAxis, minorAxis);
		ptEigenValues(i, 0) = majorValue;
		ptEigenValues(i, 1) = midValue;
		ptEigenValues(i, 2) = minorValue;
		ptEigenVectors(i, 0) = majorAxis.x();
		ptEigenVectors(i, 1) = majorAxis.y();
		ptEigenVectors(i, 2) = majorAxis.z();
		ptEigenVectors(i, 3) = midAxis.x();
		ptEigenVectors(i, 4) = midAxis.y();
		ptEigenVectors(i, 5) = midAxis.z();
		ptEigenVectors(i, 6) = minorAxis.x();
		ptEigenVectors(i, 7) = minorAxis.y();
		ptEigenVectors(i, 8) = minorAxis.z();
		
		// partition neighborhood into 3*3*3 small cubes
		// divide the length/width/height of the bounding box into three parts
		// that is the dimensions of the small cubes
		// to avoid divisions in successive steps, the inverse of each is used
		float inverse_dlength = 3.0f / (obbMaxP.x() - obbMinP.x());
		float inverse_dwidth = 3.0f / (obbMaxP.y() - obbMinP.y());
		float inverse_dheight = 3.0f / (obbMaxP.z() - obbMinP.z());
		std::vector<std::vector<Eigen::VectorXf>> voxels(27);
		// for each point in the neighborhood, put it into one of the 27 subvoxels
		for (int j = 0; j < n; ++j)
		{
			int ijk0 = static_cast<int>(floor(neighR(j, 0)*inverse_dlength));
			if (ijk0 >= 3) ijk0 = 2;
			int ijk1 = static_cast<int>(floor(neighR(j, 1)*inverse_dwidth));
			if (ijk1 >= 3) ijk1 = 2;
			int ijk2 = static_cast<int>(floor(neighR(j, 2)*inverse_dheight));
			if (ijk2 >= 3) ijk2 = 2;
			int idx = ijk0 + ijk1 * 3 + ijk2 * 9;
			voxels[idx].push_back(neigh.row(j));
		}
		// transform a vector<vector<VectorXf>> into a vector<MatrixXf>
		std::vector<Eigen::MatrixXf> vecMatVoxels;
		// instead of storing the entire subvoxel points,
		// only the indices of points in each subvoxel are stored in vecVecIndices
		// std::vector<std::vector<std::vector<int>>> vecVecVecIndices;
		std::vector<std::vector<int>> vecVecIndices;
		for (int k = 0; k < voxels.size(); ++k)
		{
			std::vector<int> vecIndices;
			// empty voxel
			if (voxels[k].size() == 0)
			{
				Eigen::MatrixXf tmp(1, 1);
				tmp << 0;
				vecMatVoxels.push_back(tmp);
				vecIndices.push_back(-1);
			}
			else
			{
				int dd = voxels[k].size();
				Eigen::MatrixXf tmp(dd, voxels[k][0].size()-1);
				for (int m = 0; m < dd; ++m)
				{
					tmp.row(m) = voxels[k][m].transpose().leftCols(d1-1);
					vecIndices.push_back(static_cast<int>(voxels[k][m](d1 - 1)));
				}
				vecMatVoxels.push_back(tmp);
			}
			vecVecIndices.push_back(vecIndices);
		}
		vecVecVecIndices.push_back(vecVecIndices);
		//partitions.push_back(vecMatVoxels);

		Eigen::MatrixXf tmpEigenValues(27, 3);
		Eigen::MatrixXf tmpEigenVectors(27, 9);
		for (int k = 0; k < vecMatVoxels.size(); ++k)
		{
			float majorValue = 0, midValue = 0, minorValue = 0;
			Eigen::Vector3f majorAxis(0, 0, 0);
			Eigen::Vector3f midAxis(0, 0, 0);
			Eigen::Vector3f minorAxis(0, 0, 0);
			// if the subvoxel has less than 3 points
			// no eigenvalues can be computed
			// set the eigenvalues and eigenvectors to negative 1
			if (vecMatVoxels[k].size() == 1 or vecMatVoxels[k].rows()==1 or vecMatVoxels[k].rows()==2)
			{
				tmpEigenValues(k, 0) = -1;
				tmpEigenValues(k, 1) = -1;
				tmpEigenValues(k, 2) = -1;
				tmpEigenVectors(k, 0) = -1;
				tmpEigenVectors(k, 1) = -1;
				tmpEigenVectors(k, 2) = -1;
				tmpEigenVectors(k, 3) = -1;
				tmpEigenVectors(k, 4) = -1;
				tmpEigenVectors(k, 5) = -1;
				tmpEigenVectors(k, 6) = -1;
				tmpEigenVectors(k, 7) = -1;
				tmpEigenVectors(k, 8) = -1;
			}
			else
			{
				Eigen::Matrix3f covmat = computeCovarianceMatrix(vecMatVoxels[k]);
				computeEigens(covmat, majorValue, midValue, minorValue, majorAxis, midAxis, minorAxis);
				tmpEigenValues(k, 0) = majorValue;
				tmpEigenValues(k, 1) = midValue;
				tmpEigenValues(k, 2) = minorValue;
				tmpEigenVectors(k, 0) = majorAxis.x();
				tmpEigenVectors(k, 1) = majorAxis.y();
				tmpEigenVectors(k, 2) = majorAxis.z();
				tmpEigenVectors(k, 3) = midAxis.x();
				tmpEigenVectors(k, 4) = midAxis.y();
				tmpEigenVectors(k, 5) = midAxis.z();
				tmpEigenVectors(k, 6) = minorAxis.x();
				tmpEigenVectors(k, 7) = minorAxis.y();
				tmpEigenVectors(k, 8) = minorAxis.z();
			}
		}
		partsEigenValues.push_back(tmpEigenValues);
		partsEigenVectors.push_back(tmpEigenVectors);
	}
	auto endpoint = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(endpoint - startpoing).count();
	std::cout << "Precomputing done! in " << elapsed << "s.\n";
}

// compute the axis algined orientated bounding box
// neighR is the neigh being rotated by the local frame
void InOut::computeOBB(Eigen::MatrixXf &neigh, Eigen::MatrixXf &neighR, Eigen::Vector3f &obbMinP, Eigen::Vector3f &obbMaxP,
					   float &majorVal, float &midVal, float &minorVal,
					   Eigen::Vector3f &majorAxis, Eigen::Vector3f &midAxis, Eigen::Vector3f &minorAxis)
{

	Eigen::Matrix3f covMat = computeCovarianceMatrix(neigh);
	Eigen::VectorXf meanVal = neigh.colwise().mean();

	// get the eigenvalues and eigenvector of the covmat 
	majorVal = 0; midVal = 0; minorVal = 0;
	majorAxis << 0, 0, 0;
	midAxis << 0, 0, 0;
	minorAxis << 0, 0, 0;
	computeEigens(covMat, majorVal, midVal, minorVal, majorAxis, midAxis, minorAxis);

	obbMinP << 0, 0, 0;
	obbMaxP << 0, 0, 0;
	obbMinP.x() = std::numeric_limits<float>::max();
	obbMinP.y() = std::numeric_limits<float>::max();
	obbMinP.z() = std::numeric_limits<float>::max();

	obbMaxP.x() = std::numeric_limits<float>::min();
	obbMaxP.y() = std::numeric_limits<float>::min();
	obbMaxP.z() = std::numeric_limits<float>::min();

	neighR = neigh;

	// express the points in the local frame formed by the eigenvectors
	// P_A = P_AB + R_AB * P_B
	// point expressed in frame {A} is equivalent to this point expressed
	// in frame {B} rotated by R_AB (rotation of frame {B} relative to frame {A})
	// plus the translation of origin of {B} to {A}
	// suppose {A} is the universal frame and {B} is the local frame
	// then P_B = R_AB^-1 * (P_A - P_AB) = R_AB^T * (P_A - P_AB).
	for (int i = 0; i < neigh.rows(); ++i)
	{
		float x = (neigh(i, 0) - meanVal(0))*majorAxis(0)
			+ (neigh(i, 1) - meanVal(1))*midAxis(0)
			+ (neigh(i, 2) - meanVal(2))*minorAxis(0);
		float y = (neigh(i, 0) - meanVal(0))*majorAxis(1)
			+ (neigh(i, 1) - meanVal(1))*midAxis(1)
			+ (neigh(i, 2) - meanVal(2))*minorAxis(1);
		float z = (neigh(i, 0) - meanVal(0))*majorAxis(2)
			+ (neigh(i, 1) - meanVal(1))*midAxis(2)
			+ (neigh(i, 2) - meanVal(2))*minorAxis(2);

		neighR(i, 0) = x;
		neighR(i, 1) = y;
		neighR(i, 2) = z;

		if (x <= obbMinP.x())
			obbMinP.x() = x;
		if (y <= obbMinP.y())
			obbMinP.y() = y;
		if (z <= obbMinP.z())
			obbMinP.z() = z;

		if (x >= obbMaxP.x())
			obbMaxP.x() = x;
		if (y >= obbMaxP.y())
			obbMaxP.y() = y;
		if (z >= obbMaxP.z())
			obbMaxP.z() = z;
	}

	// rotation matrix
	/*obbR.setZero();
	obbR << majorAxis(0), midAxis(0), minorAxis(0),
	majorAxis(1), midAxis(1), minorAxis(1),
	majorAxis(2), midAxis(2), minorAxis(2);*/

	// translation vector
	Eigen::Vector3f trans((obbMaxP.x() - obbMinP.x()) / 2.0,
		(obbMaxP.y() - obbMinP.y()) / 2.0,
						  (obbMaxP.z() - obbMinP.z()) / 2.0);
	// translate the origin of the local frame to the minimal point
	neighR.leftCols(3).rowwise() -= obbMinP.transpose();
	obbMaxP -= obbMinP;
	obbMinP -= obbMinP;
}

Eigen::Matrix3f InOut::computeCovarianceMatrix(Eigen::MatrixXf &neigh)
{
	Eigen::Matrix3f covMat;
	covMat.setZero();
	Eigen::VectorXf	 meanNeigh = neigh.colwise().mean();
	for (int j = 0; j < neigh.rows(); ++j) // for every point in each neighborhood
	{
		float px = neigh.row(j).x() - meanNeigh[0];
		float py = neigh.row(j).y() - meanNeigh[1];
		float pz = neigh.row(j).z() - meanNeigh[2];
		covMat(1, 1) += py * py;
		covMat(1, 2) += py * pz;
		covMat(2, 2) += pz * pz;
		float pxx = px * px;
		float pxy = px * py;
		float pxz = px * pz;
		covMat(0, 0) += pxx;
		covMat(0, 1) += pxy;
		covMat(0, 2) += pxz;
	}
	covMat(1, 0) = covMat(0, 1);
	covMat(2, 0) = covMat(0, 2);
	covMat(2, 1) = covMat(1, 2);
	covMat /= neigh.rows();
	return covMat;
}

void InOut::computeEigens(const Eigen::Matrix3f &covMat, float &majorVal, float &middleVal, float &minorVal,
								   Eigen::Vector3f &majorAxis, Eigen::Vector3f &middleAxis, Eigen::Vector3f &minorAxis)
{
	Eigen::EigenSolver<Eigen::Matrix3f> es;
	// eigenvalues are not in a particular order
	es.compute(covMat);
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvectorsType eigenVecs;
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvalueType eigenVals;
	eigenVecs = es.eigenvectors();
	eigenVals = es.eigenvalues();

	// sort the eigenvalues in descending order
	unsigned int temp = 0;
	unsigned int majorIdx = 0;
	unsigned int middleIdx = 1;
	unsigned int minorIdx = 2;
	if (eigenVals.real() (majorIdx) < eigenVals.real() (middleIdx))
	{
		temp = majorIdx;
		majorIdx = middleIdx;
		middleIdx = temp;
	}
	if (eigenVals.real() (majorIdx) < eigenVals.real() (minorIdx))
	{
		temp = majorIdx;
		majorIdx = minorIdx;
		minorIdx = temp;
	}
	if (eigenVals.real() (middleIdx) < eigenVals.real() (minorIdx))
	{
		temp = minorIdx;
		minorIdx = middleIdx;
		middleIdx = temp;
	}

	majorVal = eigenVals.real() (majorIdx);
	middleVal = eigenVals.real() (middleIdx);
	minorVal = eigenVals.real() (minorIdx);

	majorAxis = eigenVecs.col(majorIdx).real();
	middleAxis = eigenVecs.col(middleIdx).real();
	minorAxis = eigenVecs.col(minorIdx).real();

	// normalize
	majorAxis.normalize();
	middleAxis.normalize();
	minorAxis.normalize();

	float det = majorAxis.dot(middleAxis.cross(minorAxis));
	if (det <= 0.0f)
	{
		majorAxis(0) = -majorAxis(0);
		majorAxis(1) = -majorAxis(1);
		majorAxis(2) = -majorAxis(2);
	}
}