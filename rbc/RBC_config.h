#ifndef RBC_CONFIG_H
#define RBC_CONFIG_H

#include <vole_config.h>

namespace vole {

class RBCConfig : public Config {

public:
    RBCConfig(const std::string& prefix = std::string());

    virtual ~RBCConfig() {}

    // input image filename
    std::string input_file;
    // working directory
    std::string output_dir;

    bool old_impl;

    virtual std::string getString() const;

    protected:

    #ifdef WITH_BOOST
        virtual void initBoostOptions();
    #endif // WITH_BOOST
};



}


#endif // RBC_CONFIG_H
