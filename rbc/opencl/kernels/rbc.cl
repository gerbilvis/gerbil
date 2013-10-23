typedef float real;
typedef uint unint;

#define BLOCK_SIZE 16

//Row major indexing
#define IDX(i,j,ld) (((size_t)(i)*(ld))+(j))

#define DIST(i,j) ( ( (i)-(j) )*( (i)-(j) ) )  // L_2

#define MAX(a,b) max(a,b)
#define MIN(a,b) min(a,b)


//Computes all pairs of distances between Q and X.

//typedef struct {
//  real *mat;
//  unint r; //rows
//  unint c; //cols
//  unint pr; //padded rows
//  unint pc; //padded cols
//  unint ld; //the leading dimension (in this code, this is the same as pc)
//} matrix;


__kernel void dist1Kernel(__global const real* Q_mat,
                         unint Q_r,
                         unint Q_c,
                         unint Q_pr,
                         unint Q_pc,
                         unint Q_ld,
                         unint qStart,
                         __global const real* X_mat,
                         unint X_r,
                         unint X_c,
                         unint X_pr,
                         unint X_pc,
                         unint X_ld,
                         unint xStart,
                         __global real* D_mat,
                         unint D_r,
                         unint D_c,
                         unint D_pr,
                         unint D_pc,
                         unint D_ld)
{
  unint c, i, j;

  size_t threadIdx_x = get_local_id(0);
  size_t threadIdx_y = get_local_id(1);

  size_t blockIdx_x = get_group_id(0);
  size_t blockIdx_y = get_group_id(1);

  unint qB = blockIdx_y*BLOCK_SIZE + qStart;
  unint q  = threadIdx_y;
  unint xB = blockIdx_x*BLOCK_SIZE + xStart;
  unint x = threadIdx_x;

  real ans=0;

  //This thread is responsible for computing the dist between Q[qB+q] and X[xB+x]

  __local real Qs[BLOCK_SIZE][BLOCK_SIZE];
  __local real Xs[BLOCK_SIZE][BLOCK_SIZE];


  for(i=0 ; i<Q_pc/BLOCK_SIZE; i++){
    c=i*BLOCK_SIZE; //current col block

    Qs[x][q] = Q_mat[ IDX(qB+q, c+x, Q_ld) ];
    Xs[x][q] = X_mat[ IDX(xB+q, c+x, X_ld) ];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(j=0 ; j<BLOCK_SIZE ; j++)
      ans += DIST( Qs[j][q], Xs[j][x] );

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  D_mat[ IDX( qB+q, xB+x, D_ld ) ] = ans;
}


//This function is used by the rbc building routine.  It find an appropriate range
//such that roughly cntWant points fall within this range.  D is a matrix of distances.
__kernel void findRangeKernel(__global const real* D_mat,
                              unint D_r,
                              unint D_c,
                              unint D_pr,
                              unint D_pc,
                              unint D_ld,
                              unint numDone,
                              __global real* ranges,
                              unint cntWant)
{

  size_t blockIdx_y = get_group_id(1);

    size_t threadIdx_x = get_local_id(0);
    size_t threadIdx_y = get_local_id(1);

  unint row = blockIdx_y*(BLOCK_SIZE/4)+threadIdx_y + numDone;
  unint ro = threadIdx_y;
  unint co = threadIdx_x;
  unint i, c;
  real t;

  const unint LB = (90*cntWant)/100 ;
  const unint UB = cntWant;

  __local real smin[BLOCK_SIZE/4][4*BLOCK_SIZE];
  __local real smax[BLOCK_SIZE/4][4*BLOCK_SIZE];

//  real min= MAX_REAL;
    real min_val = FLT_MAX;


  real max_val=0;
  for(c=0 ; c<D_pc ; c+=(4*BLOCK_SIZE)){
    if( c+co < D_c ){
      t = D_mat[ IDX( row, c+co, D_ld ) ];
      min_val = MIN(t,min_val);
      max_val = MAX(t,max_val);
    }
  }

  smin[ro][co] = min_val;
  smax[ro][co] = max_val;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=2*BLOCK_SIZE ; i>0 ; i/=2){
    if( co < i ){
      smin[ro][co] = MIN( smin[ro][co], smin[ro][co+i] );
      smax[ro][co] = MAX( smax[ro][co], smax[ro][co+i] );
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  //Now start range counting.

  unint itcount=0;
  unint cnt;
  real rg;

  __local unint scnt[BLOCK_SIZE/4][4*BLOCK_SIZE];
  __local char cont[BLOCK_SIZE/4];

  if(co==0)
    cont[ro]=1;

  do{
    itcount++;

    barrier(CLK_LOCAL_MEM_FENCE);

    if( cont[ro] )  //if we didn't actually need to cont, leave rg as it was.
      rg = ( smax[ro][0] + smin[ro][0] ) / ((real)2.0) ;

    cnt=0;
    for(c=0 ; c<D_pc ; c+=(4*BLOCK_SIZE)){
      cnt += (c+co < D_c && row < D_r && D_mat[ IDX( row, c+co, D_ld ) ] <= rg);
    }

    scnt[ro][co] = cnt;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(i=2*BLOCK_SIZE ; i>0 ; i/=2){
      if( co < i ){
        scnt[ro][co] += scnt[ro][co+i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(co==0){
      if( scnt[ro][0] < cntWant )
        smin[ro][0]=rg;
      else
        smax[ro][0]=rg;
    }

    // cont[ro] == this row needs to continue
    if(co==0)
      cont[ro] = row<D_r && ( scnt[ro][0] < LB || scnt[ro][0] > UB );

    barrier(CLK_LOCAL_MEM_FENCE);

    // Determine if *any* of the rows need to continue
    for(i=BLOCK_SIZE/8 ; i>0 ; i/=2){
      if( ro < i && co==0)
        cont[ro] |= cont[ro+i];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  } while(cont[0]);

  if(co==0 && row<D_r )
    ranges[row]=rg;

}
