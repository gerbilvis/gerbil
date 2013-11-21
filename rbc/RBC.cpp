#include "RBC.h"
#include "utilsGPU.h"

#include "rbc_include.h"
#include <sys/time.h>

#include "supportRoutines.h"

RBC::RBC()
    : vole::Command("RBC", config,
                    "Michal Cieslak",
                    "michal.cieslak@hotmail.com")
{}

RBC::RBC(const vole::RBCConfig &cfg)
    : vole::Command("RBC", config,
                    "Michal Cieslak",
                    "michal.cieslak@hotmail.com"),
      config(cfg)
{}

void generateRandomQueries(const matrix x, int numOfReps, matrix& q)
{
    q.c = x.c;
    q.pc = x.pc;
    q.r = numOfReps;
    q.pr = PAD(numOfReps);
    q.ld = q.pc;

    q.mat = (real*)calloc(q.pc * q.pr, sizeof(real));

    for(int i = 0; i < numOfReps; ++i)
    {
        int idx = rand() % x.r;

        real* src_ptr = x.mat + IDX(idx, 0, x.ld);
        real* dst_ptr = q.mat + IDX(i, 0, q.ld);

        std::copy(src_ptr, src_ptr + x.pc, dst_ptr);
    }
}

int RBC::execute()
{
    if(config.old_impl)
    {

        const char* argv[] = {"testRBC",
                              "-X", "/home/mcieslak/SOCIS/RandomBallCover-GPU-master/sample_input/sample_db.txt",
                              "-Q", "/home/mcieslak/SOCIS/RandomBallCover-GPU-master/sample_input/sample_queries.txt",
                              //"-X", "/home/michalc/SOCIS/RandomBallCover-GPU-master/sample_input/sample_db.txt",
                              //"-Q", "/home/michalc/SOCIS/RandomBallCover-GPU-master/sample_input/sample_queries.txt",
                              "-n", "1024",
                              "-m", "128",
                              "-d", "16",
                              "-r", "128",
                              "-O", "output.txt"};

        return old_main(15, (char**)argv);
    }

    try
    {
        OclContextHolder::oclInit();

        std::cout << "rbc execution" << std::endl;

        std::cout << "input file: " << config.input_file << std::endl;

        multi_img img;
        img.minval = 0.;
        img.maxval = 1.;
        img.read_image(config.input_file);
        if (img.empty())
        {
            std::cout << "loading file failed!" << std::endl;
            return -1;
        }
        img.rebuildPixels(false);


        //float* buffer = new float[img.width * img.height * img.size()];
        //imgToPlainBuffer(img, buffer);

        matrix database;
        matrix queries;
        ocl_rbcStruct rbcS;

        std::cout << "image size: " << img.width
                  << "x" << img.height
                  << "x" << img.size() << std::endl;

        int numReps = sqrt(img.width * img.height) * 5;
        int numQueries = img.width * img.height;
        int pointsPerRepresentative = 1024*16;

        std::cout << "number of representatives: " << numReps << std::endl;
        std::cout << "number of queries: " << numQueries << std::endl;
        std::cout << "number of points per representative: "
                  << pointsPerRepresentative << std::endl;

        imgToMatrix(img, database);
        generateRandomQueries(database, numQueries, queries);

        std::cout << "calcualting rbc" << std::endl;


        cl::Context& context = OclContextHolder::context;
        cl::CommandQueue& queue = OclContextHolder::queue;

        cl_int err;
        cl::Buffer repsPilots(context, CL_MEM_READ_WRITE,
                              PAD(numReps) * sizeof(real), 0, &err);
        checkErr(err);

        cl::Buffer newRepsPilots(context, CL_MEM_READ_WRITE,
                                 PAD(numReps) * sizeof(real), 0, &err);

        checkErr(err);

        buildRBC(database, &rbcS, numReps, pointsPerRepresentative,
                 repsPilots, 512);

        real* repsPilotsHost = new real[PAD(numReps)];

        err = queue.enqueueReadBuffer(repsPilots, CL_TRUE, 0,
                                      PAD(numReps) * sizeof(real),
                                      repsPilotsHost);
        checkErr(err);

        for(int i = 0; i < 20; ++i)
        {
            std::cout << repsPilotsHost[(rand() % PAD(numReps))] << std::endl;
        }


        delete repsPilotsHost;

        std::cout << "querying" << std::endl;


        cl::Buffer allPilots(context, CL_MEM_READ_WRITE,
                              PAD(database.pr) * sizeof(real), 0, &err);
        checkErr(err);

        computePilots(database, repsPilots, rbcS, allPilots);

        /*
        intMatrix nnsRBC;
        matrix distsRBC;

        //Allocate space for NNs and dists
        initIntMat(&nnsRBC, numQueries, KMAX);  //KMAX is defined in defs.h
        initMat(&distsRBC, numQueries, KMAX);

        std::cout << "query indexes size: "
                  << ((float)(numQueries * KMAX * sizeof(unint))) / (1024 * 1024)
                  << "MB" << std::endl;

        std::cout << "query distances size: "
                  << ((float)(numQueries * KMAX * sizeof(real))) / (1024 * 1024)
                  << "MB" << std::endl;

        nnsRBC.mat = (unint*)calloc(sizeOfIntMat(nnsRBC), sizeof(*nnsRBC.mat));
        distsRBC.mat = (real*)calloc(sizeOfMat(distsRBC), sizeof(*distsRBC.mat));

        kqueryRBC(queries, rbcS, nnsRBC, distsRBC);*/

        ocl_matrix output_means;

        copyAndMove(&output_means, &database);

        meanshiftKQueryRBC(database, rbcS, output_means,
                           repsPilots, newRepsPilots, 1024);


        free(database.mat);
        free(queries.mat);

        //delete[] buffer;
    }
    catch(cl::Error error)
    {
        std::cerr << "ERROR: " << error.what()
                  << "(" << error.err() << ")" << std::endl;
    }
}

void RBC::imgToPlainBuffer(multi_img &img, float *buffer) const
{
    int width = img.width;
    int height = img.height;
    int depth = img.size();

    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            multi_img::Pixel pixel = img(i, j);

            float* ptr = buffer + (i * width + j) * depth;

            std::copy(pixel.begin(), pixel.end(), ptr);
        }
    }
}


void RBC::imgToMatrix(multi_img &img, matrix &matrix)
{
    int width = img.width;
    int height = img.height;
    int depth = img.size();

    int size = width * height;

    matrix.r = size;
    matrix.c = depth;
    matrix.pr = PAD(size);
    matrix.pc = PAD(depth);
    matrix.ld = PAD(depth);

    matrix.mat = (real*)calloc(matrix.pr * matrix.pc, sizeof(real));

    int point_counter = 0;

    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            multi_img::Pixel pixel = img(i, j);

            float* ptr = matrix.mat + point_counter * matrix.pc;
            std::copy(pixel.begin(), pixel.end(), ptr);
            point_counter++;
        }
    }
}

void RBC::printShortHelp() const
{
    std::cout << "rbc short help" << std::endl;
}

void RBC::printHelp() const
{
    std::cout << "rbc help" << std::endl;
}
