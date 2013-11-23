#include "supportRoutines.h"

#include <cstdio>
#include <cmath>
#include <iostream>
#include "rbc_include.h"
#include <sys/time.h>

char *dataFileX, *dataFileQ, *dataFileXtxt, *dataFileQtxt, *outFile, *outFiletxt;
char runBrute=0, runEval=0;
/** number of input points (database) */
unint n = 0;
/** number queries */
unint m = 0;
/** dimensionality */
unint d = 0;
/** number of representatives */
unint numReps = 0;
/** device number */
unint deviceNum = 0;

int old_main(int argc, char**argv){

    try{
        OclContextHolder::oclInit();

        matrix x, q;
        intMatrix nnsRBC;
        matrix distsRBC;
        struct timeval tvB,tvE;
        // cudaError_t cE;
        ocl_rbcStruct rbcS;

        printf("*****************\n");
        printf("RANDOM BALL COVER\n");
        printf("*****************\n");

        parseInput(argc,argv);

        //  gettimeofday( &tvB, NULL );
        //  printf("Using GPU #%d\n",deviceNum);
        //  if(cudaSetDevice(deviceNum) != cudaSuccess){
        //    printf("Unable to select device %d.. exiting. \n",deviceNum);
        //    exit(1);
        //  }

        //  size_t memFree, memTot;
        //  cudaMemGetInfo(&memFree, &memTot);
        //  printf("GPU memory free = %lu/%lu (MB) \n",(unsigned long)memFree/(1024*1024),(unsigned long)memTot/(1024*1024));
        //  gettimeofday( &tvE, NULL );
        //  printf(" init time: %6.2f \n", timeDiff( tvB, tvE ) );

        //Setup matrices
        initMat( &x, n, d );
        initMat( &q, m, d );
        x.mat = (real*)calloc( sizeOfMat(x), sizeof(*(x.mat)));
        q.mat = (real*)calloc( sizeOfMat(q), sizeof(*(q.mat)));

        //Load data
        if( dataFileXtxt )
            readDataText( dataFileXtxt, x );
        else
            readData( dataFileX, x );

        if( dataFileQtxt )
            readDataText( dataFileQtxt, q );
        else
            readData( dataFileQ, q );

        //Allocate space for NNs and dists
        initIntMat( &nnsRBC, m, KMAX );  //KMAX is defined in defs.h
        initMat( &distsRBC, m, KMAX );
        nnsRBC.mat = (unint*)calloc( sizeOfIntMat(nnsRBC), sizeof(*nnsRBC.mat));
        distsRBC.mat = (real*)calloc( sizeOfMat(distsRBC), sizeof(*distsRBC.mat));

        //Build the RBC
        printf("building the rbc..\n");
        gettimeofday( &tvB, NULL );
        buildRBC( x, &rbcS, numReps, numReps );
        gettimeofday( &tvE, NULL );
        printf( "\t.. build time = %6.4f \n", timeDiff(tvB,tvE) );

        //This finds the 32-NNs; if you are only interested in the 1-NN, use queryRBC(..) instead
        gettimeofday( &tvB, NULL );

        kqueryRBC( q, rbcS, nnsRBC, distsRBC );

        //simpleKqueryRBC(q, rbcS, nnsRBC, distsRBC);

        gettimeofday( &tvE, NULL );
        printf( "\t.. query time for krbc = %6.4f \n", timeDiff(tvB,tvE) );

        //Shows how to run brute force search
        if(runBrute)
        {
            intMatrix nnsBrute;
            matrix distsBrute;
            initIntMat( &nnsBrute, m, KMAX );
            nnsBrute.mat = (unint*)calloc(sizeOfIntMat(nnsBrute),
                                          sizeof(*nnsBrute.mat));
            initMat( &distsBrute, m, KMAX );
            distsBrute.mat = (real*)calloc( sizeOfMat(distsBrute),
                                            sizeof(*distsBrute.mat) );

            printf("running k-brute force..\n");
            gettimeofday( &tvB, NULL );
            bruteK( x, q, nnsBrute, distsBrute );
            gettimeofday( &tvE, NULL );
            printf( "\t.. time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

            writeNeighbs(NULL, "output_brute.txt", nnsBrute, distsBrute);

            free( nnsBrute.mat );
            free( distsBrute.mat );
        }

        //  cE = cudaGetLastError();
        //  if( cE != cudaSuccess ){
        //    printf("Execution failed; error type: %s \n", cudaGetErrorString(cE) );
        //  }

        if(runEval)
            evalKNNerror(x,q,nnsRBC);

        if(outFile || outFiletxt)
            writeNeighbs( outFile, outFiletxt, nnsRBC, distsRBC );

        destroyRBC( &rbcS );
        // cudaThreadExit();
        free(nnsRBC.mat);
        free(distsRBC.mat);
        free(x.mat);
        free(q.mat);
    }
    catch(cl::Error error)
    {
        std::cerr << "ERROR: " << error.what()
                  << "(" << error.err() << ")" << std::endl;
    }
}


void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -x datafileX -q datafileQ  -n numPts (DB) -m numQueries -d dim -r numReps [-o outFile] [-g GPU num] [-b] [-e]\n\n");
    printf("\tdatafileX    = binary file containing the database\n");
    printf("\tdatafileQ    = binary file containing the queries\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionality\n");
    printf("\tnumReps      = number of representatives (must be at least 32)\n");
    printf("\toutFile      = binary output file (optional)\n");
    printf("\tGPU num      = ID # of the GPU to use (optional) for multi-GPU machines\n");
    printf("\n\tuse -b to run brute force in addition the RBC\n");
    printf("\tuse -e to run the evaluation routine (implicitly runs brute force)\n");
    printf("\n\n\tTo input/output data in text format (instead of bin), use the \n\t-X and -Q and -O switches in place of -x and -q and -o (respectively).\n");
    printf("\n\n");
    exit(0);
  }

  while(i<argc){
    if(!strcmp(argv[i], "-x"))
      dataFileX = argv[++i];
    else if(!strcmp(argv[i], "-q"))
      dataFileQ = argv[++i];
    else if(!strcmp(argv[i], "-X"))
      dataFileXtxt = argv[++i];
    else if(!strcmp(argv[i], "-Q"))
      dataFileQtxt = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-o"))
      outFile = argv[++i];
    else if(!strcmp(argv[i], "-O"))
      outFiletxt = argv[++i];
    else if(!strcmp(argv[i], "-g"))
      deviceNum = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-b"))
      runBrute=1;
    else if(!strcmp(argv[i], "-e"))
      runEval=1;
    else{
      fprintf(stderr,"%s : unrecognized option.. exiting\n",argv[i]);
      exit(1);
    }
    i++;
  }

  if( !n || !m || !d || !numReps  ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (!dataFileX && !dataFileXtxt) || (!dataFileQ && !dataFileQtxt) ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (dataFileX && dataFileXtxt) || (dataFileQ && dataFileQtxt) ){
    fprintf(stderr,"you can only give one database file and one query file.. exiting\n");
    exit(1);
  }
  if( numReps>n ){
    fprintf(stderr,"can't have more representatives than points.. exiting\n");
    exit(1);
  }
  if( numReps<32 ){
    fprintf(stderr, "number of representatives must be at least 32\n");
    exit(1);
  }
}


void readData(char *dataFile, matrix x){
  unint i;
  FILE *fp;
  unint numRead;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }

  for( i=0; i<x.r; i++ ){ //can't load everything in one fread
                           //because matrix is padded.
    numRead = fread( &x.mat[IDX( i, 0, x.ld )], sizeof(real), x.c, fp );
    if(numRead != x.c){
      fprintf(stderr,"error reading file.. exiting \n");
      exit(1);
    }
  }
  fclose(fp);
}


void readDataText(char *dataFile, matrix x){
  FILE *fp;
  double t;
  unint i,j;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }

  for(i=0; i<x.r; i++){
    for(j=0; j<x.c; j++){
      if(fscanf(fp,"%lf ", &t)==EOF){
    fprintf(stderr,"error reading file.. exiting \n");
    exit(1);
      }
      x.mat[IDX( i, j, x.ld )]=(real)t;
    }
  }
  fclose(fp);
}

void writeNeighbs(const char *file, const char *filetxt,
                  intMatrix NNs, matrix dNNs)
{
  unint i,j;

  if( filetxt ) { //write text

    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    for( i=0; i<m; i++ ){
      for( j=0; j<KMAX; j++ )
    fprintf( fp, "%u ", NNs.mat[IDX( i, j, NNs.ld )] );
      fprintf(fp, "\n");
    }

    for( i=0; i<m; i++ ){
      for( j=0; j<KMAX; j++ )
    fprintf( fp, "%f ", dNNs.mat[IDX( i, j, dNNs.ld )]);
      fprintf(fp, "\n");
    }
    fclose(fp);

  }

  if( file ){ //write binary

    FILE *fp = fopen(file,"wb");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    for( i=0; i<m; i++ )
      fwrite( &NNs.mat[IDX( i, 0, NNs.ld )], sizeof(*NNs.mat), KMAX, fp );
    for( i=0; i<m; i++ )
      fwrite( &dNNs.mat[IDX( i, 0, dNNs.ld )], sizeof(*dNNs.mat), KMAX, fp );

    fclose(fp);
  }
}


//evals the error rate of k-nns
void evalKNNerror(matrix x, matrix q, intMatrix NNs)
{
    struct timeval tvB, tvE;
    unint i,j,k;

    unint m = q.r;
    printf("\nComputing error rates (this might take a while)\n");

    unint *ol = (unint*)calloc( q.r, sizeof(*ol) );

    intMatrix NNsB;
    matrix distsBrute;

    initIntMat( &NNsB, q.r, KMAX );
    initMat( &distsBrute, q.r, KMAX );
    NNsB.mat = (unint*)calloc( sizeOfIntMat(NNsB), sizeof(*NNsB.mat) );
    distsBrute.mat = (real*)calloc( sizeOfMat(distsBrute), sizeof(*distsBrute.mat) );

    gettimeofday(&tvB,NULL);
    bruteK(x,q,NNsB,distsBrute);

    gettimeofday(&tvE,NULL);

    //calc overlap
    for(i=0; i<m; i++)
    {
        for(j=0; j<KMAX; j++)
        {
            for(k=0; k<KMAX; k++)
            {
                ol[i] += (NNs.mat[IDX(i, j, NNs.ld)] == NNsB.mat[IDX(i, k, NNsB.ld)]);
            }
        }
    }

    long int nc=0;
    for(i=0;i<m;i++)
    {
        nc += ol[i];
    }

    double mean = ((double)nc)/((double)m);
    double var = 0.0;

    for(i=0;i<m;i++)
    {
        var += (((double)ol[i])-mean)*(((double)ol[i])-mean)/((double)m);
    }
    printf("\tavg overlap = %6.4f/%d; std dev = %6.4f \n", mean, KMAX, sqrt(var));

    real *ranges = (real*)calloc(q.pr,sizeof(*ranges));
    for(i=0;i<q.r;i++)
    {
        ranges[i] = distVec(q,x,i,NNs.mat[IDX(i, KMAX-1, NNs.ld)]);
    }

    unint *cnts = (unint*)calloc(q.pr,sizeof(*cnts));
    bruteRangeCount(x,q,ranges,cnts);

    nc=0;
    for(i=0;i<m;i++)
    {
        nc += cnts[i];
    }
    mean = ((double)nc)/((double)m);
    var = 0.0;
    for(i=0;i<m;i++)
    {
        var += (((double)cnts[i])-mean)*(((double)cnts[i])-mean)/((double)m);
    }
    printf("\tavg actual rank of 32nd NN returned by the RBC = %6.4f; std dev = %6.4f \n\n", mean, sqrt(var));
    printf("(brute k-nn took %6.4f) \n", timeDiff(tvB, tvE));

    free(cnts);
    free(ol);
    free(NNsB.mat);
    free(distsBrute.mat);
}

// NEVER USED
/*void evalNNerror(matrix x, matrix q, unint *NNs){
  struct timeval tvB, tvE;
  unint i;

  printf("\nComputing error rates (this might take a while)\n");
  real *ranges = (real*)calloc(q.pr,sizeof(*ranges));
  for(i=0;i<q.r;i++){
    if(NNs[i]>n) printf("error");
    ranges[i] = distVec(q,x,i,NNs[i]) - 10e-6;
  }

  unint *cnts = (unint*)calloc(q.pr,sizeof(*cnts));
  gettimeofday(&tvB,NULL);
  bruteRangeCount(x,q,ranges,cnts);
  gettimeofday(&tvE,NULL);

  long int nc=0;
  for(i=0;i<m;i++){
    nc += cnts[i];
  }
  double mean = ((double)nc)/((double)m);
  double var = 0.0;
  for(i=0;i<m;i++) {
    var += (((double)cnts[i])-mean)*(((double)cnts[i])-mean)/((double)m);
  }
  printf("\tavg rank = %6.4f; std dev = %6.4f \n\n", mean, sqrt(var));
  printf("(range count took %6.4f) \n", timeDiff(tvB, tvE));

  free(ranges);
  free(cnts);
}*/
