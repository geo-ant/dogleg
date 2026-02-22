#include <glog/logging.h>

extern "C" void init_glog_for_ceres(int verbosity) {
    google::InitGoogleLogging("ceres");
    FLAGS_logtostderr = 1;
    FLAGS_v = verbosity;
}
