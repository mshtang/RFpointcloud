#include "FeatureFactory.h"
#include <ctime>


FeatureFactory::FeatureFactory(Eigen::MatrixXf& neighborhood, Features feat) :
	_neighborhood(neighborhood),
	_feat(feat)
{}

bool FeatureFactory::computeFeature()
{	
	bool result = false;
	switch (_feat._featType)
	{
		case 0: 
			result = FeatureFactory::redColorDiff(); 
			break;
		case 1: 
			result = FeatureFactory::greenColorDiff(); 
			break;
		case 2: 
			result = FeatureFactory::blueColorDiff(); 
			break;
		case 3: 
			result = FeatureFactory::xDiff(); 
			break;
		case 4: 
			result = FeatureFactory::yDiff(); 
			break;
		case 5: 
			result = FeatureFactory::zDiff(); 
			break;
		/*case 7: absRedColorDiff(); break;
		case 8: absGreenColorDiff(); break;
		case 9: absBlueColorDiff(); break;
		case 10: localDensity(); breakneighborhood
		case 11: localPlanarity(); break;
		case 12: localSufaceDev(); break;*/
	}
	return result;
}

inline bool FeatureFactory::redColorDiff()
{
	return _neighborhood(_feat._point1, 4) < _neighborhood(_feat._point2, 4) ? true : false;
}

inline bool FeatureFactory::greenColorDiff()
{
	return _neighborhood(_feat._point1, 5) < _neighborhood(_feat._point2, 5) ? true : false;
}

inline bool FeatureFactory::blueColorDiff()
{
	return _neighborhood(_feat._point1, 6) < _neighborhood(_feat._point2, 6) ? true : false;
}

inline bool FeatureFactory::xDiff()
{
	return _neighborhood(_feat._point1, 0) < _neighborhood(_feat._point2, 0) ? true : false;
}

inline bool FeatureFactory::yDiff()
{
	return _neighborhood(_feat._point1, 1) < _neighborhood(_feat._point2, 1) ? true : false;
}

inline bool FeatureFactory::zDiff()
{
	return _neighborhood(_feat._point1, 2) < _neighborhood(_feat._point2, 2) ? true : false;
}
