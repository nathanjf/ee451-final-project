#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include <dirent.h>

#include <pthread.h>

#include <iostream>
#include <fstream>

// Operating modes
#define NO_MODE         0
#define TRAINING_MODE   1
#define TESTING_MODE    2
#define ANIMATION_MODE  3

// Config values
#define TOTAL_CONFIG_LINES 2
#define UNSET              "NULL"

// Program values
#define MAX_THREAD_COUNT 32  

typedef std::pair<std::string, std::pair<int, int>> image_path_mean_type;


// Global variables
std::vector<std::string> image_paths;
std::vector<image_path_mean_type> image_paths_means;

// Thread id array
pthread_t tid[MAX_THREAD_COUNT];

// kmean clustering specific variables , 1920*1080*3
#define KMEANS_ITERATIONS 100
int kmeans[4] = {256*144*3, 426*240*3, 740*480*3, 1280*720*3};
struct kmeansArgs {
    std::vector<image_path_mean_type> image_paths_means;
};
kmeansArgs kmeansArgs_tid[MAX_THREAD_COUNT];
pthread_mutex_t kmeansMutex;

void * kmeansClustering(void * arg) {
    kmeansArgs * kmeansArgs_tid = (kmeansArgs *)arg;

    /*
        For every iteration check the value of the vector's points 
    */

    for(int i = 0; i < KMEANS_ITERATIONS; i++) {
        for(int j = 0; j < kmeansArgs_tid->image_paths_means.size(); j++) {
            int pixel_count = image_paths_means[j].second.first;
            int kmean = image_paths_means[j].second.second;
            int shortest_distance = INT32_MAX;
            for(int k = 0; k < sizeof(kmeans)/sizeof(int); k++) {
                if (shortest_distance > abs(kmeans[k] - pixel_count)) {
                    shortest_distance = abs(kmeans[k] - pixel_count);
                    kmean = k;
                }
            }        
            image_paths_means[j].second.second = kmean;
            pthread_mutex_lock(&kmeansMutex);
            kmeans[kmean] = (kmeans[kmean] + pixel_count) / 2;
            pthread_mutex_unlock(&kmeansMutex);
        }
    }
    return NULL;
}

int main(int argc, char * argv[]) {
    
    int MODE = NO_MODE;

    int VALID_CONFIG_LINES = 0;
    std::fstream CONFIG;
    std::string TRAINING_IMAGE_DIRECTORY    = UNSET;
    std::string TRAINING_DATA_PATH          = UNSET;
    std::string TEST_IMAGE_DIRECTORY        = UNSET;
    std::string AMINATION_IMAGE_DIRECTORY   = UNSET;
    
    if(argc == 3) {
        /*
            Parse the mode and exit if it is not valid
        */
        std::string MODE_ARG = argv[1];
        MODE = std::stoi(MODE_ARG);
        switch(MODE){
            case TRAINING_MODE:
                break;

            case TESTING_MODE:
                break;

            case ANIMATION_MODE:
                break;

            default:
                // **TODO**
                // Add a print for all the valid modes and the mode that was input
                std::cout << "INVALID MODE: The argument was not a valid mode" << std::endl;
                return 0;
        }

        /*
            Parse the config lines and exit if they are not there
        */

        std::string PATH_ARG = argv[2];
        CONFIG.open(PATH_ARG, std::fstream::in);
        if(CONFIG.is_open()){
            std::string line;
            while(std::getline(CONFIG, line)) {
                std::string setting_prefix = line.substr(0, line.find("="));
                if(setting_prefix == "TRAINING_IMAGE_DIRECTORY") {
                    TRAINING_IMAGE_DIRECTORY = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "TRAINING_DATA_PATH") {
                    TRAINING_DATA_PATH = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                
            }
            if(VALID_CONFIG_LINES != TOTAL_CONFIG_LINES) {
                if(TRAINING_IMAGE_DIRECTORY == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TRAINING_IMAGE_DIRECTORY\" is not set" << std::endl;
                }
                if(TRAINING_DATA_PATH == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TRAINING_DATA_PATH\" is not set" << std::endl;
                }
            
                //**TODO**
                // Add a check for the remaining settings
                return 0;
            }
            CONFIG.close();

            std::cout << "\"TRAINING_IMAGE_DIRECTORY\" is set to: " << TRAINING_IMAGE_DIRECTORY << std::endl;
            std::cout << "\"TRAINING_DATA_PATH\" is set to: " << TRAINING_DATA_PATH << std::endl;
            
            //**TODO**
            // Print the remaining config lines aswell
        
        }
        else {
            //**TODO**
            // Print that there is no valid config
            return 0;
        }

    }
    else {
        //**TODO**
        // Print that that the arguments are invalid and print what valid arguments look like
        return 0;    
    }

    switch(MODE) {
        case TRAINING_MODE :
            
            DIR *               training_dir;
            struct dirent *     training_dirent;

            /*
                Load all the images into a vector of image paths and error check
            */
            
            training_dir = opendir(TRAINING_IMAGE_DIRECTORY.c_str());
            if(training_dir) {
                while((training_dirent = readdir(training_dir)) != NULL) {
                    std::string filename(training_dirent->d_name);
                    if(filename.find(".JPEG") != std::string::npos || filename.find(".jpeg") != std::string::npos) {
                        image_paths.push_back(TRAINING_IMAGE_DIRECTORY + filename);
                    }
                }
                closedir(training_dir);
            }
            else {
                //**TODO**
                //Not a valid directory
            }

            /*
                If no images were found print an error statement
            */

            if(image_paths.size() == 0) {
                //**TODO**
                //No JPEGs in directory
                std::cout << "ERROR: No \".jpeg\" found in target directory" << std::endl;
                return 0;
            }
            else {
                std::cout << "Sucessfully loaded " << image_paths.size() << " images" << std::endl;
            }
                    
            /*
                Initialize the data by creating the training data vector and 
                tagging each image with their pixel counts and their closest
                initial centroid
            */

            for (std::string image_path : image_paths) {
                
                cv::Mat image = cv::imread(image_path);
                int pixel_count = image.rows * image.cols * 3;
                int init_mean = INT32_MAX;
                int shortest_distance = INT32_MAX;
                for(int i = 0; i < sizeof(kmeans)/sizeof(int); i++) {
                    if (shortest_distance > abs(kmeans[i] - pixel_count)) {
                        shortest_distance = abs(kmeans[i] - pixel_count);
                        init_mean = i;
                    }
                }
                std::pair<int,int> image_data (pixel_count, init_mean);
                image_path_mean_type image_path_mean (image_path, image_data);
                image_paths_means.push_back(image_path_mean);
            }

            for(int i = 0; i < sizeof(kmeans)/sizeof(int); i++) {
                std::cout << kmeans[i] << " ";
            }
            std::cout << std::endl;

            /*
                Partition the vector to the threads and spawn all the threads
                If there is a remainder in the partition allocate those to the last thread
            */


            for(int i = 0; i < MAX_THREAD_COUNT; i++) {
                if(i < MAX_THREAD_COUNT - 1) {
                    int offset = image_paths_means.size()/MAX_THREAD_COUNT;
                    for(int j = i * offset; j < (i * offset) + offset; j++) {
                        kmeansArgs_tid[i].image_paths_means.push_back(image_paths_means[j]);
                    }
                }
                else {
                    if(image_paths_means.size()%MAX_THREAD_COUNT == 0) {    
                        int offset = image_paths_means.size()/MAX_THREAD_COUNT;
                        for(int j = i * offset; j < (i * offset) + offset; j++) {
                            kmeansArgs_tid[i].image_paths_means.push_back(image_paths_means[j]);
                        }
                    }
                    else {
                        int remainder = image_paths_means.size()%MAX_THREAD_COUNT;
                        int offset = image_paths_means.size()/MAX_THREAD_COUNT;
                        for(int j = i * offset; j < (i * offset) + offset + remainder; j++) {
                            kmeansArgs_tid[i].image_paths_means.push_back(image_paths_means[j]);
                        }
                    }
                }
                pthread_create(&tid[i], NULL, kmeansClustering, (void*)&kmeansArgs_tid[i]);
                
            }

            /*
                Clear the current values in the vector and write the new values
                the threads have calculated
            */

            image_paths_means.clear();
            for(int i = 0; i < MAX_THREAD_COUNT; i++) {
                pthread_join(tid[i], NULL);
                for(std::pair<std::string, std::pair<int, int>> image_path_mean: kmeansArgs_tid[i].image_paths_means)
                {
                    image_paths_means.push_back(image_path_mean);
                }
            }

            for(int i = 0; i < sizeof(kmeans)/sizeof(int); i++) {
                std::cout << kmeans[i] << " ";
            }
            std::cout << std::endl;
            std::cout << image_paths_means.size() << std::endl;

            /*
                At this point the kmeans are calculated based off the seed centroids
                and every image in the vector will have the proper tagging
            */

            //**TODO**
            // EDGE

            //**TODO**
            // CORNER

            //**TODO**
            // HOUGH TRANSFORM

            break;

        case TESTING_MODE :
            break;

        case ANIMATION_MODE :
            break;
    }
        
            // processing(for each thread count)
                // Edge detection
                // Moravec Corner Detection
                // Hough Transform

                // Update image's optimal thread count's and execution times

    std::cout << "No errors so far" << std::endl;

    return 0;
}