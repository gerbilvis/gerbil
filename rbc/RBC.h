#ifndef RBC_H
#define RBC_H

#include "RBC_config.h"
#include "rbc_ops.h"
#include "multi_img.h"

#include "meanshift_config.h"

#include <command.h>

class RBC : public vole::Command {

    public:
        RBC();
        // use only with pre-filled config!
        RBC(const vole::RBCConfig &config);
        int execute();

        void printShortHelp() const;
        void printHelp() const;

        void imgToPlainBuffer(multi_img& img, float* buffer) const;

        /**
         * @brief imgToMatrix
         * @param img image
         * @param matrix uninitialized matrix
         */
        void imgToMatrix(multi_img& img, matrix& matrix);

    protected:
        int executeSimple();

        vole::RBCConfig config;
        vole::MeanShiftConfig config_ms;
};


#endif // RBC_H
