#ifndef CLUSTERINGMETHOD_H
#define CLUSTERINGMETHOD_H

#include <QVariant>

struct ClusteringMethod {
	enum t { FAMS, PSPMS, FSPMS};
};

Q_DECLARE_METATYPE(ClusteringMethod::t)

#endif // CLUSTERINGMETHOD_H
