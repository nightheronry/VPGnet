# Declare $PATH_TO_DATASET_DIR and $PATH_TO_DATASET_LIST

../../build/tools/convert_driving_data ../../../caltech-lanes-dataset/ ../../../caltech-lanes-dataset/c1w1w2.txt LMDB_train
../../build/tools/compute_driving_mean LMDB_train ./driving_mean_train.binaryproto lmdb

../../build/tools/convert_driving_data ../../../caltech-lanes-dataset/ ../../../caltech-lanes-dataset/cordova2.txt LMDB_test


