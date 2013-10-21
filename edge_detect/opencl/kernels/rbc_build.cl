#define BLOCK_SIZE 16

//Row major indexing
#define IDX(i,j,ld) (((size_t)(i)*(ld))+(j))

__kernel void find_ranges(__global const float* distances,
                          int rows,
                          int cols,
                          int padded_rows,
                          int padded_cols,
                          int num_done,
                          __global float* ranges,
                          int cnt_want)
{
    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int row = group_id_y * (BLOCK_SIZE / 4) + local_id_y + num_done;

    int ro = local_id_x;
    int co = local_id_y;

    int i, c;
    float t;

    const int LB = (90 * cnt_want) / 100 ;
    const int UB = cnt_want;

//    __shared__ real smin[BLOCK_SIZE/4][4*BLOCK_SIZE];
//    __shared__ real smax[BLOCK_SIZE/4][4*BLOCK_SIZE];

    __local float smin[BLOCK_SIZE/4][4*BLOCK_SIZE];
    __local float smax[BLOCK_SIZE/4][4*BLOCK_SIZE];

    float min = MAX_REAL;
    float max = 0;
    for(c = 0; c < padded_cols; c += (4*BLOCK_SIZE))
    {
        if(c + co < cols)
        {
            t = D.mat[IDX(row, c+co, padded_cols)];
            min = min(t, min);
            max = max(t, max);
        }
    }

    smin[ro][co] = min;
    smax[ro][co] = max;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(i = 2 * BLOCK_SIZE; i > 0; i /= 2)
    {
        if(co < i)
        {
            smin[ro][co] = MIN( smin[ro][co], smin[ro][co+i]);
            smax[ro][co] = MAX( smax[ro][co], smax[ro][co+i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Now start range counting.

    int itcount=0;
    int cnt;
    float rg;
    __local int scnt[BLOCK_SIZE/4][4*BLOCK_SIZE];
    __local char cont[BLOCK_SIZE/4];

    if(co == 0)
        cont[ro] = 1;

    do
    {
        itcount++;

        barrier(CLK_LOCAL_MEM_FENCE);

        if(cont[ro])  //if we didn't actually need to cont, leave rg as it was.
          rg = (smax[ro][0] + smin[ro][0]) / 2.f;

        cnt = 0;
        for(c=0; c < padded_cols; c += (4 * BLOCK_SIZE))
        {
            cnt += (c + co < cols && row < rows
                    && distances[IDX(row, c+co, padded_cols)] <= rg);
        }

        scnt[ro][co] = cnt;
        barrier(CLK_LOCAL_MEM_FENCE);

        for(i = 2*BLOCK_SIZE; i > 0; i /= 2)
        {
            if( co < i ){
                scnt[ro][co] += scnt[ro][co+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(co == 0)
        {
            if(scnt[ro][0] < cntWant)
                smin[ro][0] = rg;
            else
                smax[ro][0] = rg;
        }

        // cont[ro] == this row needs to continue
        if(co == 0)
            cont[ro] = row<D.r && ( scnt[ro][0] < LB || scnt[ro][0] > UB );

        barrier(CLK_LOCAL_MEM_FENCE);

        // Determine if *any* of the rows need to continue
        for(i = BLOCK_SIZE / 8; i > 0; i /= 2)
        {
            if( ro < i && co==0)
                cont[ro] |= cont[ro+i];

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    } while(cont[0]);

    if(co == 0 && row < rows)
    ranges[row]=rg;
}
