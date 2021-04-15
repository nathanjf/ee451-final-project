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
// Moravec Corner detection specific variables
int edge_optimal_threads[4] = {0, 0, 0 ,0};
int moravec_optimal_threads[4] = {0, 0, 0, 0};
struct args {
    cv::Mat * input_image;
    cv::Mat * output_image;
};
args args_tid[MAX_THREAD_COUNT];

// Hough Transform specific variables
int hough_optimal_threads[4] = {0, 0, 0, 0};
struct args_hough {
    cv::Mat * input_image;
    cv::Mat * output_image;
    int ** accumulator_space;
    int d;
    int idx;
};
args_hough args_hough_tid[MAX_THREAD_COUNT];

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
    args * args_tid = (args *)arg;
    cv::Mat * temp = new cv::Mat(args_tid->output_image->rows, args_tid->output_image->cols, CV_8UC1, cv::Scalar(0));

    // Convert to gray
    for(int x = 0; x < args_tid->input_image->cols; x++) {
        for(int y = 0; y < args_tid->input_image->rows; y++) {
            cv::Vec3b pixel = args_tid->input_image->at<cv::Vec3b>(cv::Point(x, y));
            temp->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);            
        }
    }

    for(int x = 1; x < args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 1; y++) {
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

    for(int x = 1; x < args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 1; y++) {
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
            args_tid->output_image->at<uchar>(cv::Point(x, y)) = G;
        }
    }

    delete temp;
    
    return NULL;
}

void * moravecCorner(void * arg) {
    args * args_tid = (args *)arg;
    cv::Mat * temp = new cv::Mat(args_tid->output_image->rows, args_tid->output_image->cols, CV_8UC1, cv::Scalar(0));
    int ** temp_int = new int*[args_tid->output_image->rows + 6];
    for(int i = 0; i <  args_tid->output_image->rows + 6; i++) {
        temp_int[i] = new int[args_tid->output_image->cols + 6];
        for(int j = 0; j < args_tid->output_image->cols + 6; j++) {
            temp_int[i][j] = 0;
        }
    }
    

    // Convert to gray
    for(int x = 0; x < args_tid->input_image->cols; x++) {
        for(int y = 0; y < args_tid->input_image->rows; y++) {
            cv::Vec3b pixel = args_tid->input_image->at<cv::Vec3b>(cv::Point(x, y));
            temp->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);    
            //args_tid->output_image->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);        
        }
    }

    /*
    for(int x = 2; x < args_tid->output_image->cols - 2; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 2; y++) {
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
    */

    for(int x = 2; x < args_tid->output_image->cols - 2; x++) {
        for(int y = 2; y < args_tid->output_image->rows - 2; y++) {
            //Intensity of local window
            int local_intensity = 
                    temp->at<uchar>(cv::Point(x - 1, y - 1)) +
                    temp->at<uchar>(cv::Point(x    , y - 1)) +
                    temp->at<uchar>(cv::Point(x + 1, y - 1)) +
                    temp->at<uchar>(cv::Point(x - 1, y + 1)) +
                    temp->at<uchar>(cv::Point(x    , y + 1)) +
                    temp->at<uchar>(cv::Point(x + 1, y + 1)) +
                    temp->at<uchar>(cv::Point(x - 1, y    )) +
                    temp->at<uchar>(cv::Point(x    , y    )) +
                    temp->at<uchar>(cv::Point(x + 1, y    ));
            local_intensity = local_intensity / 9;
            int min = INT32_MAX;
            for(int u = -1; u <= 1; u++) {
                for(int v = -1; v <= 1; v++) {
                    if(!(u == 0 && v == 0)) {        
                        int shifted_intensity = 
                            temp->at<uchar>(cv::Point(x - 1 + u, y - 1 + v)) +
                            temp->at<uchar>(cv::Point(x     + u, y - 1 + v)) +
                            temp->at<uchar>(cv::Point(x + 1 + v, y - 1 + v)) +
                            temp->at<uchar>(cv::Point(x - 1 + u, y + 1 + v)) +
                            temp->at<uchar>(cv::Point(x     + u, y + 1 + v)) +
                            temp->at<uchar>(cv::Point(x + 1 + u, y + 1 + v)) +
                            temp->at<uchar>(cv::Point(x - 1 + u, y     + v)) +
                            temp->at<uchar>(cv::Point(x     + u, y     + v)) +
                            temp->at<uchar>(cv::Point(x + 1 + u, y     + v));
                        shifted_intensity = shifted_intensity / 9;
                        int E = (shifted_intensity - local_intensity) * (shifted_intensity - local_intensity);
                        if(E <= min) {
                            min = E;
                        }   
                    } 
                }

            }
            if(min < 150) {
                min = 0;
            }
            temp_int[y + 3][x + 3] = min;
        }
    }
    for(int x = 2; x < args_tid->output_image->cols - 2; x++) {
        for(int y = 2; y < args_tid->output_image->rows - 2; y++) {
            //Intensity of local window
            bool is_extrema = true;
            int intensity = temp_int[y + 3][x + 3];
            for(int u = -5; u <= 5; u++) {
                for(int v = -5; v <= 5; v++) {
                    if(!(u == 0 && v == 0))
                    {        
                        int shifted_intensity = temp_int[y + v + 3][x + u + 3];
                            
                        if(shifted_intensity >= intensity) {
                            is_extrema = false;
                        }   
                    } 
                }
            }
            if(is_extrema == true) {
                args_tid->output_image->at<uchar>(cv::Point(x, y)) = 255;
            }
            else {
                args_tid->output_image->at<uchar>(cv::Point(x, y)) = 0;
            }
        }
    }

    delete temp;
    for(int i = 0; i < args_tid->output_image->rows + 6; i++) {
        delete temp_int[i];
    }
    delete temp_int;

    return NULL;
}

void * houghTransform(void * arg) {
    args_hough * args_tid = (args_hough *)arg;
    cv::Mat * temp = new cv::Mat(args_tid->output_image->rows, args_tid->output_image->cols, CV_8UC1, cv::Scalar(0));

    // Convert to gray
    for(int x = 0; x < args_tid->input_image->cols; x++) {
        for(int y = 0; y < args_tid->input_image->rows; y++) {
            cv::Vec3b pixel = args_tid->input_image->at<cv::Vec3b>(cv::Point(x, y));
            temp->at<uchar>(cv::Point(x, y)) = color2gray(pixel.val[0], pixel.val[1], pixel.val[2]);            
        }
    }

    for(int x = 1; x < args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 1; y++) {
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

    for(int x = 1; x < args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 1; y++) {
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
            args_tid->output_image->at<uchar>(cv::Point(x, y)) = G;
        }
    }

    // Do the accumulator now
    for(int x = 1; x < args_tid->output_image->cols - 1; x++) {
        for(int y = 1; y < args_tid->output_image->rows - 1; y++) {
            if(args_tid->output_image->at<uchar>(cv::Point(x, y)) == 255) {
                //Feature space exploration
                for(int theta = 0; theta < 180; theta++) {
                     
                    int p = (int)((double)(args_tid->idx + (x - 1)) * cos((double)theta * 3.14/180.0) + (double)(y - 1) * sin((double)theta * 3.14/180.0)) + args_tid->d;
                    args_tid->accumulator_space[theta][p] += 1;
                }
            }
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
                int d = (int)sqrt((double)(image.cols * image.cols + image.rows * image.rows));

                cv::imwrite("/home/nathanjf/testInput.JPEG", image);

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
                            args_tid[idx].input_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC3, cv::Scalar(0,0,0));
                            args_tid[idx].output_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC1, cv::Scalar(0));
                            
                            int x_write = 1;
                            for(int x = idx * cols; x < idx * cols + cols + remainder; x++) {
                                int y_write = 1;
                                for(int y = 0; y < image.rows; y++) {
                                    if(y == 0) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }
                                    if(y == image.rows - 1) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));        
                                    }

                                    if(x == idx * cols && x - 1 >= 0) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                                    }
                                    else if (x == idx * cols && !(x - 1 >= 0)){
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }
                                    
                                    if(x == idx * cols + cols + remainder - 1 && x + 1 < image.cols) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));    
                                    }
                                    if(x == idx * cols + cols + remainder - 1 && !(x + 1 < image.cols)) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }

                                    //Top corner
                                    if(y == 0){
                                        if(x == idx * cols) {
                                            args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                        if(x == idx * cols + cols + remainder - 1) {
                                            args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                    } 
                                    if(y == image.rows - 1) {
                                        if(x == idx * cols) {
                                            args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                        if(x == idx * cols + cols + remainder - 1) {
                                            args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                    }

                                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }

                            //Test print if you just want the results of one image
                            //cv::imwrite("/home/nathanjf/test" + std::to_string(idx) + ".JPEG", *args_tid[idx].input_image);
                            
                            pthread_create(&tid[idx], NULL, edgeDetection, (void*)&args_tid[idx]);
                        }
                        
                        // Merge image slices
                        for(int idx = 0; idx < thread_count; idx++) {
                            pthread_join(tid[idx], NULL);
                            int x_write = image.cols/thread_count * idx;
                            for(int x = 1; x < args_tid[idx].output_image->cols - 1; x++) {
                                int y_write = 0;
                                for(int y = 1; y < args_tid[idx].output_image->rows - 1; y++) {
                                    output.at<uchar>(cv::Point(x_write, y_write)) = args_tid[idx].output_image->at<uchar>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }

                            /*
                                Take note that memory is being freed here
                            */
                            cv::imwrite("/home/nathanjf/testOutput" + std::to_string(idx) + ".JPEG", *args_tid[idx].output_image);
                            
                            delete args_tid[idx].input_image;
                            delete args_tid[idx].output_image;
                        }
                        cv::imwrite("/home/nathanjf/testOutputEdge.JPEG", output);

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


                            args_tid[idx].input_image = new cv::Mat(image.rows + 2 + 2, cols + remainder + 2 + 2, CV_8UC3, cv::Scalar(0,0,0));
                            args_tid[idx].output_image = new cv::Mat(image.rows + 2 + 2, cols + remainder + 2 + 2, CV_8UC1, cv::Scalar(0));
                            
                            int x_write = 2;
                            for(int x = idx * cols; x < idx * cols + cols + remainder; x++) {
                                int y_write = 2;
                                for(int y = 0; y < image.rows; y++) {
                                    if(x == idx * cols && x - 2 >= 0) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 2, y));
                                    }
                                    if(x == idx * cols + cols + remainder - 1 && x + 2 < image.cols) {
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));
                                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 2, y));    
                                    }
                                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }
                            //Test print if you just want the results of one image
                            //cv::imwrite("/home/nathanjf/test" + std::to_string(idx) + ".JPEG", *args_tid[idx].input_image);
                            
                            pthread_create(&tid[idx], NULL, moravecCorner, (void*)&args_tid[idx]);
                        }
                        
                        // Merge image slices
                        for(int idx = 0; idx < thread_count; idx++) {
                            pthread_join(tid[idx], NULL);
                            int x_write = image.cols/thread_count * idx;
                            for(int x = 2; x < args_tid[idx].output_image->cols - 2; x++) {
                                int y_write = 0;
                                for(int y = 2; y < args_tid[idx].output_image->rows - 2; y++) {
                                    output.at<uchar>(cv::Point(x_write, y_write)) = args_tid[idx].output_image->at<uchar>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }

                            /*
                                Take note that memory is being freed here
                            */
                            //cv::imwrite("/home/nathanjf/testOutput" + std::to_string(idx) + ".JPEG", *args_tid[idx].output_image);
                        
                            delete args_tid[idx].input_image;
                            delete args_tid[idx].output_image;

                        }

                        cv::imwrite("/home/nathanjf/testOutputMoravec.JPEG", output);

                    // Hough Transform
                        for(int idx = 0; idx < thread_count; idx++) {
                            int cols = image.cols/thread_count;
                            int remainder = 0;
                            if(idx == thread_count - 1 && image.cols%thread_count != 0) {
                                remainder = image.cols%thread_count;
                            }
                            
                            /*
                                Take note that memory is being allocated for the thread images here
                            */

                            
                            args_hough_tid[idx].input_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC3, cv::Scalar(0,0,0));
                            args_hough_tid[idx].output_image = new cv::Mat(image.rows + 2, cols + remainder + 2, CV_8UC1, cv::Scalar(0));
                            args_hough_tid[idx].accumulator_space = new int*[180];
                            for(int i = 0; i <  180; i++) {
                                args_hough_tid[idx].accumulator_space[i] = new int[2*d];
                                for(int j = 0; j < 2*d; j++) {
                                    args_hough_tid[idx].accumulator_space[i][j] = 0;
                                }
                            }
                            args_hough_tid[idx].d = d;
                            args_hough_tid[idx].idx = idx * cols;

                            int x_write = 1;
                            for(int x = idx * cols; x < idx * cols + cols + remainder; x++) {
                                int y_write = 1;
                                for(int y = 0; y < image.rows; y++) {
                                    if(y == 0) {
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }
                                    if(y == image.rows - 1) {
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));        
                                    }
                                    if(x == idx * cols && x - 1 >= 0) {
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                                    }
                                    else if (x == idx * cols && !(x - 1 >= 0)){
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }
                                    if(x == idx * cols + cols + remainder - 1 && x + 1 < image.cols) {
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));    
                                    }
                                    if(x == idx * cols + cols + remainder - 1 && !(x + 1 < image.cols)) {
                                        args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                    }
                                    if(y == 0){
                                        if(x == idx * cols) {
                                            args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                        if(x == idx * cols + cols + remainder - 1) {
                                            args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                    } 
                                    if(y == image.rows - 1) {
                                        if(x == idx * cols) {
                                            args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                        if(x == idx * cols + cols + remainder - 1) {
                                            args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                                        }
                                    }

                                    args_hough_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }
                            //Test print if you just want the results of one image
                            //cv::imwrite("/home/nathanjf/test" + std::to_string(idx) + ".JPEG", *args_tid[idx].input_image);
                            
                            pthread_create(&tid[idx], NULL, houghTransform, (void*)&args_hough_tid[idx]);
                        }
                        
                        // Merge image slices
                        int ** accumulator_space = new int*[180];
                        for(int i = 0; i < 180; i++) {
                            accumulator_space[i] = new int[2*d];
                            for(int j = 0; j < 2*d; j++) {
                                accumulator_space[i][j] = 0;
                            }
                        }
                                                
                        for(int idx = 0; idx < thread_count; idx++) {
                            pthread_join(tid[idx], NULL);
                            int x_write = image.cols/thread_count * idx;
                            for(int x = 1; x < args_hough_tid[idx].output_image->cols - 1; x++) {
                                int y_write = 0;
                                for(int y = 1; y < args_hough_tid[idx].output_image->rows - 1; y++) {
                                    output.at<uchar>(cv::Point(x_write, y_write)) = args_hough_tid[idx].output_image->at<uchar>(cv::Point(x, y));
                                    y_write++;
                                }
                                x_write++;
                            }
                            for(int i = 0; i <  180; i++) {
                                for(int j = 0; j < 2*d; j++) {
                                    accumulator_space[i][j] = accumulator_space[i][j] + args_hough_tid[idx].accumulator_space[i][j];
                                }
                            }

                            /*
                                Take note that memory is being freed here
                            */
                            //cv::imwrite("/home/nathanjf/testOutput" + std::to_string(idx) + ".JPEG", *args_tid[idx].output_image);
                        
                            delete args_hough_tid[idx].input_image;
                            delete args_hough_tid[idx].output_image;
                            for(int i = 0; i <  180; i++) {
                                delete args_hough_tid[idx].accumulator_space[i];
                            }
                            delete args_hough_tid[idx].accumulator_space;
                        }
                        // Turn accumulator into image
                        int max = 0;
                        int max_theta = 0;
                        double max_p = 0;

                        for(int theta = 0; theta < 180; theta++) {
                            for(int p = 0; p < 2*d; p++) {
                                if(accumulator_space[theta][p] > max) {
                                    max = accumulator_space[theta][p];
                                    max_p = p - d + 1;
                                    max_theta = theta;
                                }
                            }
                        }

                        //max_theta = 59;
                        //max_p = 193;

                        double a = cos((double)max_theta * 3.14/180.0);
                        double b = sin((double)max_theta * 3.14/180.0);
                        double x0 = (a * (double)max_p); //+ image.cols/2;
                        double y0 = (b * (double)max_p); //+ image.rows/2;
                        int x1 = (int)(x0 + 1000.0 * (-b));
                        int y1 = (int)(y0 + 1000.0 * (a));
                        int x2 = (int)(x0 - 1000.0 * (-b));
                        int y2 = (int)(y0 - 1000.0 * (a));

                        for(int i = 0; i <  180; i++) {
                            delete accumulator_space[i];
                        }
                        delete accumulator_space;

                        cv::Mat temp = image.clone();
                        cv::line(temp, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 3);
                        //cv::imwrite("/home/nathanjf/testOutputHough.JPEG", houghOutput);
                        cv::imwrite("/home/nathanjf/testOutputHoughVisual.JPEG", temp);
                }
            }

        case TESTING_MODE :
            break;

        case ANIMATION_MODE :
            break;
        
    }
        
    std::cout << "No errors so far" << std::endl;

    return 0;
}