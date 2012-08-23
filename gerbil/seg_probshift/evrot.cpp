/**********
 *
 *  mex interface to compute the gradient of the eigenvectors
 *  alignment quality
 *
 *  To mexify:
 *               mex  evrot.cpp;
 *
 *  [clusters,Quality,Vrot] = evrot(V,method);
 *
 *  Input:
 *    V = eigenvecors, each column is a vector
 *    method = 1   gradient descent
 *             2   approximate gradient descent
 *
 *  Output:
 *    clusts - Resulting cluster assignment
 *    Quality = The final quality
 *    Vr = The rotated eigenvectors
 *
 *
 *  Lihi Zelnik (Caltech) March.2005
 *
 *
 ************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>



#define DEBUG 0    /* set to 1 and mex to see print outs */

using namespace cv;
using std::vector;

//class EvRot {


/******* compute V ***********/
/** Gradient of a single Givens rotation **/
Mat_<double> *gradU(vector<double> &theta, int k,
			   int* ik, int* jk, int dim)
{

	Mat_<double> *V = new Mat_<double>(dim, dim);
	V->setTo(0);

    // variant A
    (*V)(ik[k], ik[k]) = -sin(theta[k]);
    (*V)(ik[k], jk[k]) = cos(theta[k]);
    (*V)(jk[k], ik[k]) = -cos(theta[k]);
    (*V)(jk[k], jk[k]) = -sin(theta[k]);
    
    // variant B
    // (*V)(ik[k], ik[k]) = -sin(theta[k]);
    // (*V)(jk[k], ik[k]) = cos(theta[k]);
    // (*V)(ik[k], jk[k]) = -cos(theta[k]);
    // (*V)(jk[k], jk[k]) = -sin(theta[k]);

	return V;
}



/******* compute Uab ***********/
/** Givens rotation for angles a to b **/
Mat_<double> *build_Uab(vector<double> &theta, int a, int b,
			   int* ik, int* jk, int dim)
{
	Mat_<double> *Uab = new Mat_<double>(dim, dim);
	Uab->setTo(0);
	int k,i,j;
	/* set Uab to be an identity matrix */
	for( j=0; j<dim; j++ ){
		(*Uab)(j, j) = 1.0;
	}

	if( b < a ) {
		return Uab;
	}

	double tt,u_ik;
	for( k=a; k<=b; k++ ){
		tt = theta[k];
		for( i=0; i<dim; i++ ){
            // u_ik = (*Uab)(ik[k], i) * cos(tt) - (*Uab)(jk[k], i) * sin(tt);
            // (*Uab)(jk[k], i) = (*Uab)(ik[k], i) * sin(tt) + (*Uab)(jk[k], i) * cos(tt);
            // (*Uab)(ik[k], i) = u_ik;

            // i interpreted as row
            u_ik = (*Uab)(i, ik[k]) * cos(tt) - (*Uab)(i, jk[k]) * sin(tt);
            (*Uab)(i, jk[k]) = (*Uab)(i, ik[k]) * sin(tt) + (*Uab)(i, jk[k]) * cos(tt);
            (*Uab)(i, ik[k]) = u_ik;
		}
	}
	return Uab;
}

/** Rotate vectors in X with Givens rotation according to angles in theta **/
Mat_<double> rotate_givens(const Mat_<double> &X, vector<double> &theta,
					   int* ik, int* jk, int angle_num, int dim)
{
	Mat_<double> *G = build_Uab(theta, 0, angle_num-1,ik,jk,dim);
    Mat_<double> Y = X * (*G);
	delete G;
	return Y;
}



/****** quality gradient *******************/

double evqualitygrad(const Mat_<double> &X, vector<double> &theta,
					 int *ik, int *jk,
					 int angle_num,int angle_index,
					 int dim,int ndata)
{
	/* build V,U,A */
	Mat_<double> *V = gradU(theta,angle_index,ik,jk,dim);
	if( DEBUG )
		printf("Computed gradU\n");

	Mat_<double> *U1 = build_Uab(theta,0,angle_index-1,ik,jk,dim);
	Mat_<double> *U2 = build_Uab(theta,angle_index+1,angle_num-1,ik,jk,dim);
	if( DEBUG )
		printf("Computed Uab\n");

	//Mat_<double> A = X * (*U1) * (*V) * (*U2);
	Mat_<double> A = X * ((*U1) * ((*V) * (*U2)));
	if( DEBUG )
		printf("Built A\n");

	/* get rid of no longer needed arrays */
	delete V;
	delete U1;
	delete U2;

	/* rotate vecs according to current angles */
	Mat_<double> Y = rotate_givens(X,theta,ik,jk,angle_num,dim);
	if( DEBUG )
		printf("Rotated according to Givens successfully\n");

	/* find max of each row */
    vector<double> max_values(ndata, 0);
    vector<int> max_index(ndata, 0);
    cv::Point maxLoc;
	for (int i=0; i<ndata; i++ ) { /* loop over all rows */
        cv::minMaxLoc(Y.row(i), NULL, &max_values[i], NULL, &maxLoc);
        max_index[i] = maxLoc.x;
    }
	if( DEBUG )
		printf("Found max of each row\n");

	/* compute gradient */
    /* FYI: A.rows=ndata, A.cols=dim */
    int i, j;
	double dJ=0, tmp1, tmp2;
	for( j=0; j<dim; j++ ){  /* loop over all columns */
        for( i=0; i<ndata; i++ ){ /* loop over all rows */
            // ind = j*ndata + i

            tmp1 = A(i, j) * Y(i, j) / (max_values[i]*max_values[i]);
            tmp2 = A(i, max_index[i]) * (Y(i, j) * Y(i, j)) / (max_values[i]*max_values[i]*max_values[i]);

    //        printf("A(%d/%d, %d/%d)\n", i, A.rows, max_index[i], A.cols);
			dJ += tmp1-tmp2;
		}
	}
	dJ = 2*dJ/ndata/dim;
	if( DEBUG )
		printf("Computed gradient = %g\n",dJ);

	return dJ;
}


/******** alignment quality ***********/
double evqual(const Mat_<double> &X,
			  int *ik, int *jk,
			  int dim,int ndata)
{
	/* take the square of all entries and find max of each row */
    vector<double> max_values(ndata, 0);
    vector<int> max_index(ndata, 0);
	int i,j;
	for( j=0; j<dim; j++ ){  /* loop over all columns */
		for( i=0; i<ndata; i++ ){ /* loop over all rows */
            double sq = X(i, j) * X(i, j);
			if( max_values[i] <= sq){
				max_values[i] = sq;
				max_index[i] = j;
			}
		}
	}
	if( DEBUG )
		printf("Found max of each row\n");

	/* compute cost */
	double J=0;
	for( j=0; j<dim; j++ ){  /* loop over all columns */
		for( i=0; i<ndata; i++ ){ /* loop over all rows */
			J += X(i, j) * X(i, j) / max_values[i];
		}
	}
	J = 1.0 - (J/ndata -1.0)/dim;
	if( DEBUG )
		printf("Computed quality = %g\n",J);

	return J;
}


/********** cluster assignments ************/
vector< vector<int> > cluster_assign(const Mat_<double> &X,
						int *ik, int *jk,
						int dim,int ndata)
{
	/* take the square of all entries and find max of each row */
    vector<double> max_values(ndata, 0);
    vector<int> max_index(ndata, -1);
    vector<int> cluster_count(dim, 0);
	int i,j;
	for( j=0; j<dim; j++ ){  /* loop over all columns */
		for( i=0; i<ndata; i++ ){ /* loop over all rows */
			if( max_values[i] <= X(i, j) * X(i, j)){
				if( max_index[i] >= 0 )
					cluster_count[max_index[i]]--;
				cluster_count[j]++;
				max_values[i] = X(i, j)*X(i, j);
				max_index[i] = j;
			}
		}
	}
	if( DEBUG )
		printf("Found max of each row\n");

	/* allocate memory for cluster assignments */
	vector< vector<int> > clusters(dim);
	for( j=0; j<dim; j++ ){  /* loop over all columns */
		clusters[j].resize(cluster_count[j]);
	}

	/* prepare cluster assignments */
	int cind;
	for( j=0; j<dim; j++ ){  /* loop over all columns */
		vector<int> &cluster = clusters[j];
		cind = 0;
		for( i=0; i<ndata; i++ ){ /* loop over all rows */
			if( max_index[i] == j ){
				cluster[cind] = i;
				cind++;
			}
		}
	}

	return clusters;
}









/***************************   //////////////// main   */

  void
  evrot (const Mat_<double> &X, vector< vector<int> > &ret_clusters, double &ret_quality, Mat_<double> &ret_Xrot)
{

	/* get the number and length of eigenvectors dimensions */
	const int ndata = X.rows;
	const int dim = X.cols;
	if( DEBUG )
		printf("Got %d vectors of length %d\n",dim,ndata);

	/* get the number of angles */
	int angle_num;
	angle_num = (int)(dim*(dim-1)/2);
	vector<double> theta(angle_num, 0);
	if( DEBUG )
		printf("Angle number is %d\n",angle_num);

    int method = 2;

	/* build index mapping */
	int i,j,k;
	int* ik = (int*)calloc(angle_num,sizeof(int));
	int* jk = (int*)calloc(angle_num,sizeof(int));
	assert(ik && jk);
	k=0;
	for( i=0; i<dim-1; i++ ){
		for( j=i+1; j<=dim-1; j++ ){
			ik[k] = i;
			jk[k] = j;
			k++;
		}
	}
	if( DEBUG )
		printf("Built index mapping for %d angles\n",k);

	/* definitions */
	int max_iter = 200;
    double Q_up, Q_down;
	double dQ,Q,Q_new,Q_old1,Q_old2;
	double alpha;
	int iter,d;
	Mat_<double> Xrot;

	vector<double> theta_new = theta;

	Q = evqual(X,ik,jk,dim,ndata); /* initial quality */
	if( DEBUG )
		printf("Q = %g\n",Q);
	Q_old1 = Q;
	Q_old2 = Q;
	iter = 0;
	while( iter < max_iter ){ /* iterate to refine quality */
		iter++;
		for( d = 0; d < angle_num; d++ ){
            if( method == 2 ){ /* descend through numerical drivative */
                alpha = 0.1;
                /* move up */                
                theta_new[d] = theta[d] + alpha;
                Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim);
                Q_up = evqual(Xrot,ik,jk,dim,ndata);
                /* move down */
                theta_new[d] = theta[d] - alpha;
                Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim);
                Q_down = evqual(Xrot,ik,jk,dim,ndata);

                /* update only if at least one of them is better */
                if( Q_up > Q || Q_down > Q){
                    if( Q_up > Q_down ){ 
                        theta[d] = theta[d] + alpha;
                        theta_new[d] = theta[d];
                        Q = Q_up;
                    } else {
                        theta[d] = theta[d] - alpha;
                        theta_new[d] = theta[d];
                        Q = Q_down;
                    }
                }
            } else { /* descend through true derivative */
                alpha = 1.0;
                dQ = evqualitygrad(X,theta,ik,jk,angle_num,d,dim,ndata);
                theta_new[d] = theta[d] - alpha * dQ;
                Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim);
                Q_new = evqual(Xrot,ik,jk,dim,ndata);
                if( Q_new > Q){
                    theta[d] = theta_new[d];
                    Q = Q_new;
                }
                else{
                    theta_new[d] = theta[d];
                }
            }
		}
		/* stopping criteria */
		if( iter > 2 ){
			if( Q - Q_old2 < 1e-3 ){
				break;
            }
		}
		Q_old2 = Q_old1;
		Q_old1 = Q;
	}

	if( DEBUG )
		printf("Done after %d iterations, Quality is %g\n",iter,Q);

	Xrot = rotate_givens(X,theta_new,ik,jk,angle_num,dim);
	ret_clusters = cluster_assign(Xrot,ik,jk,dim,ndata);

	/** prepare output **/
	if (DEBUG)
		printf("preparing output\n");
//	cout << "Xrot=" << endl << Xrot << endl;
	ret_quality = Q;
//	ret_Xrot = Mat_<double>(Xrot, true); // XXX: is copyData=true necessary, here?
	ret_Xrot = Xrot;
//	Xrot.copyTo(ret_Xrot);

	/* free allocated memory */
	free(ik);
	free(jk);

	if( DEBUG )
		printf("Done evrot\n");

	return;
}

//};
