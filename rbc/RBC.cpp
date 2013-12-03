#include "RBC.h"
#include "utilsGPU.h"

#include "rbc_include.h"
#include <sys/time.h>

#include "supportRoutines.h"

#include "meanshift.h"

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
                              "-Q", "/home/mcieslak/SOCIS/RandomBallCover-GPU-master/sample_input/sample_db.txt",
                              //"-X", "/home/michalc/SOCIS/RandomBallCover-GPU-master/sample_input/sample_db.txt",
                              //"-Q", "/home/michalc/SOCIS/RandomBallCover-GPU-master/sample_input/sample_queries.txt",
                              "-n", "1024",
                              "-m", "1024",
                              "-d", "16",
                              "-r", "256",
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

        std::cout << "image size: " << img.width
                  << "x" << img.height
                  << "x" << img.size() << std::endl;

        matrix database;
        imgToMatrix(img, database);

       // for(int i = 0; i < 20; ++i)
       meanshift_rbc(database, img.width, img.height);

        free(database.mat);
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
