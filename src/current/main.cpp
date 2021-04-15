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

// Edge detection specific variables
int edge_optimal_threads[4] = {0, 0, 0 ,0};
struct edge_args {
    cv::Mat * input_image;
    cv::Mat * output_image;
};
edge_args edge_args_tid[MAX_THREAD_COUNT];


// Moravec Corner detection specific variables
int moravec_optimal_threads[4] = {0, 0, 0, 0};
struct moravec_args {
    cv::Mat * input_image;
    cv::Mat * output_image;
};
moravec_args moravec_args_tid[MAX_THREAD_COUNT];

// Hough Transform specific variables
int hough_optimal_threads[4] = {0, 0, 0, 0};

int color2gray(int blue, int green, int red) {
    int gray = 0.299 * (double)red + 0.587 * (double)green + 0.114 * (double)blue;
    //int gray = (red + green + blue) / 3;
    return gray;
}

void * kmeansClustering(void * arg) {
    kmeansArgs * kmeansArgs_tid = (kmeansArgs *)arg;

    /*
        For every iteration check the distance of each point to each centroid
        Assign the point to its closest centroid and then update the value of that centroid
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

void * edgeDetection(void * arg) {
    edge_args * edge_args_tid = (edge_args *)arg;
    cv::Mat * temp = new cv::Mat(edge_args_tid->output_image->rows, edge_args_tid->output_image->cols, CV_8UC1, cv::Scalar(0));

    // Convert to gray
    for(int x = 0; x < edge_args_tid->input_image->cols; x++) {
        for(int y = 0; y < edge_args_tid->input_image->rows; y++) {
            cv::Vec3b pixel = edge_args_tid->input_image->at<cv::Vec3b>(cv::Point(x, y));
            temp->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);            
        }
    }

    for(int x = 1; x < edge_args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < edge_args_tid->output_image->rows - 1; y++) {
            int Gb = 
                    1  * temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    1  * temp->at<uchar>(cv::Point(x    , y - 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    1  * temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x    , y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x - 1, y    )) +
                    1  * temp->at<uchar>(cv::Point(x    , y    )) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y    ));
            Gb = Gb / 9;
            temp->at<uchar>(cv::Point(x, y)) = Gb;
        }
    }

    for(int x = 1; x < edge_args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < edge_args_tid->output_image->rows - 1; y++) {
            int Gy = 
                    -1 * temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    -2 * temp->at<uchar>(cv::Point(x    , y - 1)) +
                    -1 * temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    1  * temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    2  * temp->at<uchar>(cv::Point(x    , y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y + 1));

            int Gx =
                    -1 * temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    -2 * temp->at<uchar>(cv::Point(x - 1, y    )) +
                    -1 * temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    2  * temp->at<uchar>(cv::Point(x + 1, y    )) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y + 1));

            int G = (int)sqrt(Gx * Gx + Gy * Gy);
            
            if(G > 175) {
                G = 255;
            }
            else {
                G = 0;
            }
            edge_args_tid->output_image->at<uchar>(cv::Point(x, y)) = G;
        }
    }

    delete temp;
    
    return NULL;
}

void * moravecCorner(void * arg) {
    moravec_args * moravec_args_tid = (moravec_args *)arg;
    cv::Mat * temp = new cv::Mat(moravec_args_tid->output_image->rows, moravec_args_tid->output_image->cols, CV_8UC1, cv::Scalar(0));

    for(int x = 0; x < edge_args_tid->input_image->cols; x++) {
        for(int y = 0; y < edge_args_tid->input_image->rows; y++) {
            cv::Vec3b pixel = edge_args_tid->input_image->at<cv::Vec3b>(cv::Point(x, y));
            temp->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);            
        }
    }

    for(int x = 2; x < edge_args_tid->output_image->cols - 2; x++) {
        for(int y = 2; y < edge_args_tid->output_image->rows - 2; y++) {
            int Gy = 
                    -1 * temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    -2 * temp->at<uchar>(cv::Point(x    , y - 1)) +
                    -1 * temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    1  * temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    2  * temp->at<uchar>(cv::Point(x    , y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y + 1));

            int Gx =
                    -1 * temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    -2 * temp->at<uchar>(cv::Point(x - 1, y    )) +
                    -1 * temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    2  * temp->at<uchar>(cv::Point(x + 1, y    )) +
                    1  * temp->at<uchar>(cv::Point(x + 1, y + 1));

            int G = (int)sqrt(Gx * Gx + Gy * Gy);
            
            if(G > 175) {
                G = 255;
            }
            else {
                G = 0;
            }
            edge_args_tid->output_image->at<uchar>(cv::Point(x, y)) = G;
        }
    }

    delete temp;
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
                //**TODO**
                //Print loaded message
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

            for(int i = 0; i < 4; i++) {
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
                //std::cout << kmeansArgs_tid[i].image_paths_means.size() << std::endl;
                for(image_path_mean_type image_path_mean: kmeansArgs_tid[i].image_paths_means)
                {
                    image_paths_means.push_back(image_path_mean);
                }
            }

            for(int i = 0; i < 4; i++) {
                std::cout << kmeans[i] << " ";
            }
            std::cout << std::endl;

            /*
                At this point the kmeans are calculated based off the seed centroids
                and every image in the vector will have the proper tagging
            */

            //iterate over every image and perform the 3 transforms below
            for(image_path_mean_type image_path_mean: image_paths_means) {
                std::string image_path = image_path_mean.first;
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                cv::Mat output(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

                for(int thread_count = 1; thread_count <= MAX_THREAD_COUNT; thread_count++) {
                    // Edge detection
                    for(int idx = 0; idx < thread_count; idx++) {
                        int cols = image.cols/thread_count;
                        int remainder = 0;
                        if(idx == thread_count - 1 && image.cols%thread_count != 0) {
                            remainder = image.cols%thread_count;
                        }
                        
                        /*
                            Take note that memory is being allocated for the thread images here
                        */
                        edge_args_tid[idx].input_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC3, cv::Scalar(0,0,0));
                        edge_args_tid[idx].output_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC1, cv::Scalar(0));
                        
                        int x_write = 1;
                        for(int x = idx * cols; x < idx * cols + cols + remainder; x++) {
                            int y_write = 1;
                            for(int y = 0; y < image.rows; y++) {
                                if(x == idx * cols && x - 1 >= 0) {
                                    edge_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                                }
                                if(x == idx * cols + cols + remainder - 1 && x + 1 <= image.cols) {
                                    edge_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));    
                                }
                                edge_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                                y_write++;
                            }
                            x_write++;
                        }

                        //Test print if you just want the results of one image
                        //cv::imwrite("/home/nathanjf/test" + std::to_string(idx) + ".JPEG", *edge_args_tid[idx].input_image);
                        
                        pthread_create(&tid[idx], NULL, edgeDetection, (void*)&edge_args_tid[idx]);
                    }
                    
                    // Merge image slices
                    for(int idx = 0; idx < thread_count; idx++) {
                        pthread_join(tid[idx], NULL);
                        int x_write = image.cols/thread_count * idx;
                        for(int x = 1; x < edge_args_tid[idx].output_image->cols - 1; x++) {
                            int y_write = 0;
                            for(int y = 1; y < edge_args_tid[idx].output_image->rows - 1; y++) {
                                output.at<uchar>(cv::Point(x_write, y_write)) = edge_args_tid[idx].output_image->at<uchar>(cv::Point(x, y));
                                y_write++;
                            }
                            x_write++;
                        }

                        /*
                            Take note that memory is being freed here
                        */
                        //cv::imwrite("/home/nathanjf/testOutput" + std::to_string(idx) + ".JPEG", *edge_args_tid[idx].output_image);
                        
                        delete edge_args_tid[idx].input_image;
                        delete edge_args_tid[idx].output_image;
                    }

                    // Moravec corner detection
                    for(int idx = 0; idx < thread_count; idx++) {
                        int cols = image.cols/thread_count;
                        int remainder = 0;
                        if(idx == thread_count - 1 && image.cols%thread_count != 0) {
                            remainder = image.cols%thread_count;
                        }
                        
                        /*
                            Take note that memory is being allocated for the thread images here
                        */
                        moravec_args_tid[idx].input_image = new cv::Mat(image.rows + 2 + 2, cols + remainder + 2 + 2, CV_8UC3, cv::Scalar(0,0,0));
                        moravec_args_tid[idx].output_image = new cv::Mat(image.rows + 2 + 2, cols + remainder + 2 + 2, CV_8UC1, cv::Scalar(0));
                        
                        int x_write = 2;
                        for(int x = idx * cols; x < idx * cols + cols + remainder; x++) {
                            int y_write = 2;
                            for(int y = 0; y < image.rows; y++) {
                                if(x == idx * cols && x - 2 >= 0) {
                                    moravec_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                                    moravec_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 2, y));
                                }
                                if(x == idx * cols + cols + remainder - 1 && x + 2 <= image.cols) {
                                    moravec_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));
                                    moravec_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 2, y));    
                                }
                                moravec_args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                                y_write++;
                            }
                            x_write++;
                        }

                        //Test print if you just want the results of one image
                        cv::imwrite("/home/nathanjf/test" + std::to_string(idx) + ".JPEG", *moravec_args_tid[idx].input_image);
                        
                        //pthread_create(&tid[idx], NULL, edgeDetection, (void*)&edge_args_tid[idx]);
                    }
                    
                    // Merge image slices
                    for(int idx = 0; idx < thread_count; idx++) {
                        pthread_join(tid[idx], NULL);
                        int x_write = image.cols/thread_count * idx;
                        for(int x = 2; x < moravec_args_tid[idx].output_image->cols - 2; x++) {
                            int y_write = 0;
                            for(int y = 2; y < moravec_args_tid[idx].output_image->rows - 2; y++) {
                                output.at<uchar>(cv::Point(x_write, y_write)) = moravec_args_tid[idx].output_image->at<uchar>(cv::Point(x, y));
                                y_write++;
                            }
                            x_write++;
                        }

                        /*
                            Take note that memory is being freed here
                        */
                        //cv::imwrite("/home/nathanjf/testOutput" + std::to_string(idx) + ".JPEG", *edge_args_tid[idx].output_image);
                        
                        delete moravec_args_tid[idx].input_image;
                        delete moravec_args_tid[idx].output_image;
                    }
                }
            }

            //**TODO**
            // HOUGH TRANSFORM

        case TESTING_MODE :
            break;

        case ANIMATION_MODE :
            break;
        
    }
        
    std::cout << "No errors so far" << std::endl;

    return 0;
}