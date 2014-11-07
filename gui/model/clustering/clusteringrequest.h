#ifndef CLUSTERINGREQUEST_H
#define CLUSTERINGREQUEST_H

#include <model/clustering/clusteringmethod.h>
#include <model/representation.h>

/** Struct to store a clustering request from the GUI to the model. */
struct ClusteringRequest {
	explicit ClusteringRequest(
			ClusteringMethod::t m,
			representation::t r,
			bool lsh )
		: method(m), repr(r), lsh(lsh)
	{}

	// default copy and assignment are OK

	ClusteringMethod::t method;
	representation::t repr;
	bool lsh;
};

#endif // CLUSTERINGREQUEST_H
