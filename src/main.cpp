#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include <dirent.h>

#include <pthread.h>

#include <chrono>

#include <iostream>
#include <fstream>

// Operating modes
#define NO_MODE         0
#define TRAINING_MODE   1
#define TESTING_MODE    2
#define ANIMATION_MODE  3

// Config values
#define TOTAL_CONFIG_LINES 10
#define UNSET              "NULL"

// Program values
#define MAX_THREAD_COUNT 32  

// <file_path, <pixel_count, cluster_id>>
typedef std::pair<std::string, std::pair<int, int>> image_path_mean_type;

// Global variables
std::vector<std::string> image_paths;
std::vector<image_path_mean_type> image_paths_means;

// Thread id array
pthread_t tid[MAX_THREAD_COUNT];

// kmean clustering specific variables , 1920*1080*3
#define KMEANS_ITERATIONS 100
int kmeans[5] = {256*144*3, 426*240*3, 740*480*3, 1280*720*3, 1920*1080*3};
struct kmeansArgs {
    std::vector<image_path_mean_type> image_paths_means;
};
kmeansArgs kmeansArgs_tid[MAX_THREAD_COUNT];
pthread_mutex_t kmeansMutex;

// Edge detection specific variables
// Moravec Corner detection specific variables
double edge_optimal_threads[5] = {0, 0, 0 ,0, 0};
double edge_optimal_times[5] = {100000.0, 100000.0, 100000.0, 100000.0, 100000.0};

double moravec_optimal_threads[5] = {0, 0, 0 ,0, 0};
double moravec_optimal_times[5] = {100000.0, 100000.0, 100000.0, 100000.0, 100000.0};
struct args {
    cv::Mat * input_image;
    cv::Mat * output_image;
};
args args_tid[MAX_THREAD_COUNT];

// Hough Transform specific variables
double hough_optimal_threads[5] = {0, 0, 0, 0, 0};
double hough_optimal_times[5] = {100000.0, 100000.0, 100000.0, 100000.0, 100000.0};
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
            
            if(G > 150) {
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

void edgeDetectionSpawner(cv::Mat image, cv::Mat &output, int thread_count) {
    // Create threads
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
        pthread_create(&tid[idx], NULL, edgeDetection, (void*)&args_tid[idx]);
    }
    
    // Join threads
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
        delete args_tid[idx].input_image;
        delete args_tid[idx].output_image;
    }
    return;
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

void moravecCornerSpawner(cv::Mat image, cv::Mat &output, int thread_count) {
    // Create threads
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
                if(y == 0) {
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write - 2)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                
                }
                if(y == image.rows - 2) {
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));        
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write + 2)) = image.at<cv::Vec3b>(cv::Point(x, y));        
                }

                if(x == idx * cols && x - 2 >= 0) {
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 1, y));
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x - 2, y));
                }
                else if (x == idx * cols && !(x - 2 >= 0)){
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                }
                
                if(x == idx * cols + cols + remainder - 2 && x + 2 < image.cols) {
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 1, y));    
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x + 2, y));    
                }

                if(x == idx * cols + cols + remainder - 2 && !(x + 2 < image.cols)) {
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                    args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                }

                //Top corner
                if(y == 0){
                    if(x == idx * cols) {
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write - 2)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write - 2)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    
                    }
                    if(x == idx * cols + cols + remainder - 2) {
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write - 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write - 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write - 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    }
                } 
                if(y == image.rows - 2) {
                    if(x == idx * cols) {
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 1, y_write + 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write + 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write - 2, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    }
                    if(x == idx * cols + cols + remainder - 2) {
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 1, y_write + 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write + 2)) = image.at<cv::Vec3b>(cv::Point(x, y));
                        args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write + 2, y_write + 1)) = image.at<cv::Vec3b>(cv::Point(x, y));    
                    }
                }

                args_tid[idx].input_image->at<cv::Vec3b>(cv::Point(x_write, y_write)) = image.at<cv::Vec3b>(cv::Point(x, y));
                y_write++;
            }
            x_write++;
        }
        pthread_create(&tid[idx], NULL, moravecCorner, (void*)&args_tid[idx]);
    }
    
    // Join threads
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
        delete args_tid[idx].input_image;
        delete args_tid[idx].output_image;

    }
    return;
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
            
            if(G > 150) {
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

void houghTransformSpawner(cv::Mat image, cv::Mat &output, int d, int thread_count, int ** &accumulator_space) {
    // Create threads
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
        pthread_create(&tid[idx], NULL, houghTransform, (void*)&args_hough_tid[idx]);
    }
    
    // Join threads
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

        delete args_hough_tid[idx].input_image;
        delete args_hough_tid[idx].output_image;
        for(int i = 0; i <  180; i++) {
            delete args_hough_tid[idx].accumulator_space[i];
        }
        delete args_hough_tid[idx].accumulator_space;
    }
    return;
}

int main(int argc, char * argv[]) {

    bool first = true;
    int count;
    int count_max;

    int MODE = NO_MODE;

    int VALID_CONFIG_LINES = 0;
    std::fstream CONFIG;
    std::fstream training_data;
    std::fstream graph_data;
    std::fstream test_data;

    std::string TRAINING_IMAGE_DIRECTORY    = UNSET;
    std::string TRAINING_DATA_PATH          = UNSET;
    std::string GRAPH_DATA_PATH             = UNSET;

    std::string TEST_IMAGE_DIRECTORY        = UNSET;
    std::string TEST_DATA_PATH              = UNSET;

    std::string TEST_OUTPUT_DIRECTORY       = UNSET;

    std::string ANIMATION_IMAGE_DIRECTORY           = UNSET;
    std::string ANIMATION_OUTPUT_DIRECTORY_EDGE     = UNSET;
    std::string ANIMATION_OUTPUT_DIRECTORY_MORAVEC  = UNSET;
    std::string ANIMATION_OUTPUT_DIRECTORY_HOUGH    = UNSET;
    
    
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
                if(setting_prefix == "GRAPH_DATA_PATH") {
                    GRAPH_DATA_PATH = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "TEST_IMAGE_DIRECTORY") {
                    TEST_IMAGE_DIRECTORY = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "TEST_DATA_PATH") {
                    TEST_DATA_PATH = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "TEST_OUTPUT_DIRECTORY") {
                    TEST_OUTPUT_DIRECTORY = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "ANIMATION_IMAGE_DIRECTORY") {
                    ANIMATION_IMAGE_DIRECTORY = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "ANIMATION_OUTPUT_DIRECTORY_EDGE") {
                    ANIMATION_OUTPUT_DIRECTORY_EDGE = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "ANIMATION_OUTPUT_DIRECTORY_MORAVEC") {
                    ANIMATION_OUTPUT_DIRECTORY_MORAVEC = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                if(setting_prefix == "ANIMATION_OUTPUT_DIRECTORY_HOUGH") {
                    ANIMATION_OUTPUT_DIRECTORY_HOUGH = line.substr(line.find("=") + 1);
                    VALID_CONFIG_LINES++;
                }
                
            }
            if(VALID_CONFIG_LINES != TOTAL_CONFIG_LINES) {
                std::cout << "INVALID COFIG: FOLLOWING ERRORS" << std::endl;
                if(TRAINING_IMAGE_DIRECTORY == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TRAINING_IMAGE_DIRECTORY\" is not set" << std::endl;
                }
                if(TRAINING_DATA_PATH == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TRAINING_DATA_PATH\" is not set" << std::endl;
                }
                if(GRAPH_DATA_PATH == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"GRAPH_DATA_PATH\" is not set" << std::endl;
                }
                if(TEST_IMAGE_DIRECTORY == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TEST_IMAGE_DIRECTORY\" is not set" << std::endl;
                }
                if(TEST_DATA_PATH == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TEST_DATA_PATH\" is not set" << std::endl;
                }
                if(TEST_OUTPUT_DIRECTORY == UNSET) {
                    std::cout << "INVALID CONFIG LINE: \"TEST_OUTPUT_DIRECTORY\" is not set" << std::endl;
                }
            
                //**TODO**
                // Add a check for the remaining settings
                return 0;
            }
            CONFIG.close();

            std::cout << "\"TRAINING_IMAGE_DIRECTORY\" is set to: " << TRAINING_IMAGE_DIRECTORY << std::endl;
            std::cout << "\"TRAINING_DATA_PATH\" is set to: " << TRAINING_DATA_PATH << std::endl;
            std::cout << "\"GRAPH_DATA_PATH\" is set to: " << GRAPH_DATA_PATH << std::endl;
            std::cout << "\"TEST_IMAGE_DIRECTORY\" is set to: " << TEST_IMAGE_DIRECTORY << std::endl;
            std::cout << "\"TEST_DATA_PATH\" is set to: " << TEST_DATA_PATH << std::endl;
            std::cout << "\"TEST_OUTPUT_DIRECTORY\" is set to: " << TEST_DATA_PATH << std::endl;
            
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

            graph_data.open(GRAPH_DATA_PATH.c_str(), std::fstream::out | std::fstream::trunc);
            if(graph_data.is_open()) {
                graph_data.close();
            }
            else {
                /* TODO */
                // Output that there was training data failed
            }
            training_data.open(TRAINING_DATA_PATH.c_str(), std::fstream::out | std::fstream::trunc);
            if(training_data.is_open()) {
                training_data.close();
            }
            else {
                /* TODO */
                // Output that there graph data failed
                return 0;
            }

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
                for(image_path_mean_type image_path_mean: kmeansArgs_tid[i].image_paths_means)
                {
                    image_paths_means.push_back(image_path_mean);
                }
            }

            //iterate over every image and perform the 3 transforms below
            count = 1;
            count_max = image_paths_means.size();
            for(image_path_mean_type image_path_mean: image_paths_means) {
                std::cout << "Working on image " << count << "\tof " << count_max << "\t..." << std::endl;
                std::string image_path = image_path_mean.first;
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                cv::Mat output(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
                int d = (int)sqrt((double)(image.cols * image.cols + image.rows * image.rows));

                std::chrono::high_resolution_clock::time_point start;
                std::chrono::high_resolution_clock::time_point end;
                std::chrono::duration<double> elaps;
                
                double edge_elaps = 0.0;
                double moravec_elaps = 0.0;
                double hough_elaps = 0.0;

                double edge_worst_time = 0.0;
                double edge_best_time = 100.0;

                double moravec_worst_time = 0.0;
                double moravec_best_time = 100.0;

                double hough_worst_time = 0.0;
                double hough_best_time = 100.0;

                double edge_best_threads = 0.0;
                double moravec_best_threads = 0.0;
                double hough_best_threads = 0.0;

                // Run for every thread count
                for(int thread_count = 1; thread_count <= MAX_THREAD_COUNT; thread_count++) {
                    for(int trial = 1; trial <= 3; trial++) {
                        // Edge detection
                            start = std::chrono::high_resolution_clock::now();

                            edgeDetectionSpawner(image, output, thread_count);

                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                edge_elaps = elaps.count();
                            }
                            else {
                                edge_elaps = (edge_elaps + elaps.count()) / 2.0;
                            }
                            
                        // Moravec corner detection
                            start = std::chrono::high_resolution_clock::now();;
                            
                            moravecCornerSpawner(image, output, thread_count);

                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                moravec_elaps = elaps.count();
                            }
                            else {
                                moravec_elaps = (moravec_elaps + elaps.count()) / 2.0;
                            }

                        // Hough Transform
                            start = std::chrono::high_resolution_clock::now();;
                            
                            // Create accumulator
                            int ** accumulator_space = new int*[180];
                            for(int i = 0; i < 180; i++) {
                                accumulator_space[i] = new int[2*d];
                                for(int j = 0; j < 2*d; j++) {
                                    accumulator_space[i][j] = 0;
                                }
                            }
                            
                            houghTransformSpawner(image, output, d, thread_count, accumulator_space);
                            
                            // Free accumulator
                            for(int i = 0; i < 180; i++) {
                                delete accumulator_space[i];
                            }
                            delete accumulator_space;
        
                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                hough_elaps = elaps.count();
                            }
                            else {
                                hough_elaps = (hough_elaps + elaps.count()) / 2.0;
                            }

                    }

                    // Store best and worst times
                    if(edge_elaps > edge_worst_time) {
                        edge_worst_time = edge_elaps;
                    }
                    if(edge_elaps < edge_best_time) {
                        edge_best_time = edge_elaps;
                        edge_best_threads = thread_count;
                    }
                    // Store best and worst times
                    if(moravec_elaps > moravec_worst_time) {
                        moravec_worst_time = moravec_elaps;
                    }
                    if(moravec_elaps < moravec_best_time) {
                        moravec_best_time = moravec_elaps;
                        moravec_best_threads = thread_count;
                    }
                    // Store best and worst times
                    if(hough_elaps > hough_worst_time) {
                        hough_worst_time = hough_elaps;
                    }
                    if(hough_elaps < hough_best_time) {
                        hough_best_time = hough_elaps;
                        hough_best_threads = thread_count;
                    }
                }
                // Store data
                graph_data.open(GRAPH_DATA_PATH.c_str(), std::fstream::app);
                graph_data << image_path_mean.second.first << "\t" << edge_best_threads << "\t" << moravec_best_threads << "\t" << hough_best_threads << "\n";
                graph_data.close();

                if(first) {
                    first = false;
                    edge_optimal_threads[image_path_mean.second.second] = edge_best_threads;
                    edge_optimal_times[image_path_mean.second.second] = edge_best_time;
                    
                    moravec_optimal_threads[image_path_mean.second.second] = moravec_best_threads;
                    moravec_optimal_times[image_path_mean.second.second] = moravec_best_time;

                    hough_optimal_threads[image_path_mean.second.second] = hough_best_threads;
                    hough_optimal_times[image_path_mean.second.second] = hough_best_time;
                }
                else {
                    edge_optimal_threads[image_path_mean.second.second] = (edge_optimal_threads[image_path_mean.second.second] + edge_best_threads) / 2.0;
                    edge_optimal_times[image_path_mean.second.second] = (edge_optimal_times[image_path_mean.second.second] + edge_best_time) / 2.0;
                    
                    moravec_optimal_threads[image_path_mean.second.second] = (moravec_optimal_threads[image_path_mean.second.second] + moravec_best_threads) / 2.0;
                    moravec_optimal_times[image_path_mean.second.second] = (moravec_optimal_times[image_path_mean.second.second] + moravec_best_time) / 2.0;

                    hough_optimal_threads[image_path_mean.second.second] = (hough_optimal_threads[image_path_mean.second.second] + hough_best_threads) / 2.0;
                    hough_optimal_times[image_path_mean.second.second] = (hough_optimal_times[image_path_mean.second.second] + hough_best_time) / 2.0;
                }
                count++;
            }
            training_data.open(TRAINING_DATA_PATH.c_str(), std::fstream::app);
            for(int i = 0; i < sizeof(kmeans)/sizeof(int); i++) {
                training_data << "KMEAN" << i << "=" << kmeans[i] << "\n";
                training_data << "EDGE_OPT" << i << "=" << round(edge_optimal_threads[i]) << "\n";
                training_data << "MORAVEC_OPT" << i << "=" << round(moravec_optimal_threads[i]) << "\n";
                training_data << "HOUGH_OPT" << i << "=" << round(hough_optimal_threads[i]) << "\n";
            }
            training_data.close();
            break;

        case TESTING_MODE :
            test_data.open(TEST_DATA_PATH.c_str(), std::fstream::out | std::fstream::trunc);
            if(test_data.is_open()) {
                test_data.close();
            }
            else {
                /* TODO */
                return 0;
            }

            DIR *               test_dir;
            struct dirent *     test_dirent;
            /*
                Load all the images into a vector of image paths and error check
            */
            test_dir = opendir(TEST_IMAGE_DIRECTORY.c_str());
            if(test_dir) {
                while((test_dirent = readdir(test_dir)) != NULL) {
                    std::string filename(test_dirent->d_name);
                    if(filename.find(".JPEG") != std::string::npos || filename.find(".jpeg") != std::string::npos) {
                        image_paths.push_back(TEST_IMAGE_DIRECTORY + filename);
                    }
                }
                closedir(test_dir);
            }
            else {
                //**TODO**
                //Not a valid directory
                return 0;
            }

            // Assign each image to its closest kmean
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

            training_data.open(TRAINING_DATA_PATH, std::fstream::in);
            if(training_data.is_open()){
                int VALID_TRAINING_LINES = 0;
                int TOTAL_TRAINING_LINES = 20;
                std::string line;
                while(std::getline(training_data, line)) {
                    std::string setting_prefix = line.substr(0, line.find("="));
                    if(setting_prefix == "KMEAN0" || setting_prefix == "KMEAN1" || setting_prefix == "KMEAN2" || setting_prefix == "KMEAN3" || setting_prefix == "KMEAN4") {
                        int idx = std::atoi(setting_prefix.substr(5).c_str());
                        kmeans[idx] = std::atoi(line.substr(line.find("=") + 1).c_str());
                        VALID_TRAINING_LINES++;
                    }
                    if(setting_prefix == "EDGE_OPT0" || setting_prefix == "EDGE_OPT1" || setting_prefix == "EDGE_OPT2" || setting_prefix == "EDGE_OPT3" || setting_prefix == "EDGE_OPT4") {
                        int idx = std::atoi(setting_prefix.substr(8).c_str());
                        edge_optimal_threads[idx] = std::atoi(line.substr(line.find("=") + 1).c_str());
                        VALID_TRAINING_LINES++;
                    }
                    if(setting_prefix == "MORAVEC_OPT0" || setting_prefix == "MORAVEC_OPT1" || setting_prefix == "MORAVEC_OPT2" || setting_prefix == "MORAVEC_OPT3" || setting_prefix == "MORAVEC_OPT4") {
                        int idx = std::atoi(setting_prefix.substr(11).c_str());
                        moravec_optimal_threads[idx] = std::atoi(line.substr(line.find("=") + 1).c_str());
                        VALID_TRAINING_LINES++;
                    }
                    if(setting_prefix == "HOUGH_OPT0" || setting_prefix == "HOUGH_OPT1" || setting_prefix == "HOUGH_OPT2" || setting_prefix == "HOUGH_OPT3" || setting_prefix == "HOUGH_OPT4") {
                        int idx = std::atoi(setting_prefix.substr(9).c_str());
                        hough_optimal_threads[idx] = std::atoi(line.substr(line.find("=") + 1).c_str());
                        VALID_TRAINING_LINES++;
                    }
                }
                if(VALID_TRAINING_LINES != TOTAL_TRAINING_LINES) {
                    std::cout << "INVALID TRAINING DATA!" << std::endl;
                    //**TODO**
                    // Add a check for the remaining settings
                    return 0;
                }
                training_data.close();
            } else {
                std::cout << "NO TRAINING DATA!" << std::endl;
            }

            count = 1;
            count_max = image_paths_means.size();
            for(image_path_mean_type image_path_mean: image_paths_means) {
                std::cout << "Working on image " << count << "\tof " << count_max << "\t..." << std::endl;
                std::string image_path = image_path_mean.first;
                cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                cv::Mat output(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
                int d = (int)sqrt((double)(image.cols * image.cols + image.rows * image.rows));

                std::chrono::high_resolution_clock::time_point start;
                std::chrono::high_resolution_clock::time_point end;
                std::chrono::duration<double> elaps;
                
                double edge_elaps = 0.0;
                double moravec_elaps = 0.0;
                double hough_elaps = 0.0;

                int edge_predicted_min = edge_optimal_threads[image_path_mean.second.second];
                int moravec_predicted_min = moravec_optimal_threads[image_path_mean.second.second];
                int hough_predicted_min = hough_optimal_threads[image_path_mean.second.second];

                double edge_predicted_min_time = 100.0;
                double moravec_predicted_min_time = 100.0;
                double hough_predicted_min_time = 100.0;

                double edge_worst_time = 0.0;
                double edge_best_time = 100.0;

                double moravec_worst_time = 0.0;
                double moravec_best_time = 100.0;

                double hough_worst_time = 0.0;
                double hough_best_time = 100.0;

                double edge_best_threads = 0.0;
                double moravec_best_threads = 0.0;
                double hough_best_threads = 0.0;

                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "INPUT.JPEG", image);

                // Run for every thread count
                for(int thread_count = 1; thread_count <= MAX_THREAD_COUNT; thread_count++) {
                    for(int trial = 1; trial <= 3; trial++) {
                        // Edge detection
                            start = std::chrono::high_resolution_clock::now();

                            edgeDetectionSpawner(image, output, thread_count);

                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                if(thread_count == edge_predicted_min) {
                                    edge_predicted_min_time = elaps.count();
                                }
                                edge_elaps = elaps.count();
                            }
                            else {
                                if(thread_count == edge_predicted_min) {
                                    edge_predicted_min_time = (edge_predicted_min_time + elaps.count()) / 2.0;
                                }
                                edge_elaps = (edge_elaps + elaps.count()) / 2.0;
                            }

                            if(trial == 1 && thread_count == MAX_THREAD_COUNT - 1) {
                                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "EDGE.JPEG", output);
                            }

                        // Moravec corner detection
                            start = std::chrono::high_resolution_clock::now();;
                            
                            moravecCornerSpawner(image, output, thread_count);

                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                if(thread_count == moravec_predicted_min) {
                                    moravec_predicted_min_time = elaps.count();
                                }
                                moravec_elaps = elaps.count();
                            }
                            else {
                                if(thread_count == moravec_predicted_min) {
                                    moravec_predicted_min_time = (moravec_predicted_min_time + elaps.count()) / 2.0;
                                }
                                moravec_elaps = (moravec_elaps + elaps.count()) / 2.0;
                            }

                            if(trial == 1 && thread_count == MAX_THREAD_COUNT - 1) {
                                cv::Mat temp = image.clone();
                                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "MORAVEC_RAW.JPEG", output);
                                for(int x = 0 ; x < image.cols; x++) {
                                    for(int y = 0; y < image.rows; y++) {
                                        if(output.at<uchar>(cv::Point(x, y)) == 255) {
                                            cv::circle(temp, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                                        }
                                    }
                                }
                                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "MORAVEC_POINT.JPEG", temp);
                            }

                        // Hough Transform
                            start = std::chrono::high_resolution_clock::now();;
                            
                            // Create accumulator
                            int ** accumulator_space = new int*[180];
                            for(int i = 0; i < 180; i++) {
                                accumulator_space[i] = new int[2*d];
                                for(int j = 0; j < 2*d; j++) {
                                    accumulator_space[i][j] = 0;
                                }
                            }
                            
                            houghTransformSpawner(image, output, d, thread_count, accumulator_space);
                            
                            if(trial == 1 && thread_count == MAX_THREAD_COUNT - 1) {
                                // Turn accumulator into image
                                cv::Mat accumulator(2 * d, 180, CV_8UC3, cv::Scalar(0,0,0));
                                for(int x = 0; x < 180; x++) {
                                    for(int y = 0; y < 2 * d; y++) {
                                        accumulator.at<cv::Vec3b>(cv::Point(x,y)).val[0] = accumulator_space[x][y]/2;
                                        accumulator.at<cv::Vec3b>(cv::Point(x,y)).val[1] = accumulator_space[x][y]/2;
                                        accumulator.at<cv::Vec3b>(cv::Point(x,y)).val[2] = accumulator_space[x][y]/2;
                                    }
                                } 
                                
                                int max = 0;
                                int max_theta = 0;
                                int max_p = 0;

                                cv::Mat temp = image.clone();
                                for(int theta = 0; theta < 180; theta++) {
                                    for(int p = 0; p < 2*d; p++) {
                                        if(accumulator_space[theta][p] > max) {
                                            max = accumulator_space[theta][p];
                                            max_p = p - d + 1;
                                            max_theta = theta;
                                        }
                                    }
                                }
                            
                                double a = cos((double)max_theta * 3.14/180.0);
                                double b = sin((double)max_theta * 3.14/180.0);
                                double x0 = (a * (double)max_p); //+ image.cols/2;
                                double y0 = (b * (double)max_p); //+ image.rows/2;
                                int x1 = (int)(x0 + 1000.0 * (-b));
                                int y1 = (int)(y0 + 1000.0 * (a));
                                int x2 = (int)(x0 - 1000.0 * (-b));
                                int y2 = (int)(y0 - 1000.0 * (a));
                                cv::line(temp, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255), 3);
                                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "HOUGH_LINE.JPEG", temp);
                                cv::imwrite(TEST_OUTPUT_DIRECTORY + std::to_string(count) + "_" + "HOUGH_RAW.JPEG", accumulator);
                            }

                            // Free accumulator
                            for(int i = 0; i < 180; i++) {
                                delete accumulator_space[i];
                            }
                            delete accumulator_space;
        
                            end = std::chrono::high_resolution_clock::now();;
                            elaps = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
                            if(trial == 1) {
                                if(thread_count == hough_predicted_min) {
                                        hough_predicted_min_time = elaps.count();
                                    }
                                hough_elaps = elaps.count();
                            }
                            else {
                                if(thread_count == moravec_predicted_min) {
                                        hough_predicted_min_time = (hough_predicted_min_time + elaps.count()) / 2.0;
                                    }
                                hough_elaps = (hough_elaps + elaps.count()) / 2.0;
                            }
                    }
                    // Store best and worst times
                    if(edge_elaps > edge_worst_time) {
                        edge_worst_time = edge_elaps;
                    }
                    if(edge_elaps < edge_best_time) {
                        edge_best_time = edge_elaps;
                        edge_best_threads = thread_count;
                    }
                    // Store best and worst times
                    if(moravec_elaps > moravec_worst_time) {
                        moravec_worst_time = moravec_elaps;
                    }
                    if(moravec_elaps < moravec_best_time) {
                        moravec_best_time = moravec_elaps;
                        moravec_best_threads = thread_count;
                    }
                    // Store best and worst times
                    if(hough_elaps > hough_worst_time) {
                        hough_worst_time = hough_elaps;
                    }
                    if(hough_elaps < hough_best_time) {
                        hough_best_time = hough_elaps;
                        hough_best_threads = thread_count;
                    }
                }
                test_data.open(TEST_DATA_PATH, std::fstream::app);
                test_data << image_path_mean.second.first << "\t" << 
                    edge_predicted_min << "\t" << edge_best_threads << "\t" << 
                    edge_worst_time - edge_predicted_min_time << "\t" << edge_worst_time - edge_best_time << "\t" << edge_worst_time << "\t" <<
                    moravec_predicted_min << "\t" << moravec_best_threads << "\t" << 
                    moravec_worst_time - moravec_predicted_min_time << "\t" << moravec_worst_time - moravec_best_time << "\t" << moravec_worst_time << "\t" <<
                    hough_predicted_min << "\t" << hough_best_threads << "\t" <<
                    hough_worst_time - hough_predicted_min_time << "\t" << hough_worst_time - hough_best_time << "\t" << hough_worst_time << "\n";
                test_data.close();

                count++;
            }
            //Load in training data
            //Load test images
            //Test test images depending on which kmean they are closest to
            //Log best and worst times
            //Log best and worst thread_counts
            break;

        case ANIMATION_MODE :
            std::string image_path = ANIMATION_IMAGE_DIRECTORY;
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            cv::Mat output_internal(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat output_external(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
            for(int x = 0; x < image.rows; x++) {
                for(int y = 0; y < image.cols; y++) {

                }

            }


            //Render example outputs
            //Render example
            break;
        
    }

    return 0;
}