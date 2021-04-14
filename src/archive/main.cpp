/*

    Main thread management, command line arg parsing, output and input management

    Will be responsible for spawning the kmeans threads and data if the program is set to training
    Will be responsible for running the tests if the program is set to testing

*/

#include <iostream>
#include <string.h>
#include <vector>

using namespace std;


//Expected command line args: training/testing (0 for training, 1 for testing), filepath for training/testing image set
int main(int argc, char** argv)
{
 	bool training; //Stores whether this is a training run or not
 	string filePath; //Stores file path of image set


 	//Parse command line args
	if(argc < 3){
		cout << "Expected two command line arguments:" << endl;
		cout << "1. training/testing (0 for training, 1 for testing)" << endl;
		cout << "2. filepath for training/testing image set" << endl;
		return 0;
	}
	else{
		strcmp(argv[1], "0") == 0 ? training = true : training = false;
		filePath = argv[2];
	}


	if(training){ //Training phase
		//Perform k-means grouping on images, result is a vector of pairs <string imagePath, int kmeansIndex> 
		//and a vector of pixel counts for the kmeans groups
		pair<vector<pair<string, int> >, vector<int> > groupings = kmeans(filePath);
		vector<pair<string, int> > groupedImages = groupings.first; //<string imageFile, int kmeansIndex>
		vector<int> kmeansPixelCounts = groupings.second;


		
		//Vectors to store optimal thread counts for each operation
		//Format: [[int kmeansPixelCount, int numPointsUsed, int totalOptimalThreads], [...]]
		vector<vector<int> > edgeDetectionOptimal;
		vector<vector<int> > cornerDetectionOptimal;
		vector<vector<int> > hughTransformOptimal;

		//Initialize vectors with kmeans pixel counts info:
		for(int i = 0; i < kmeansPixelCounts.size(); i++){
			vector<int> toAdd = {kmeansPixelCounts[i], 0, 0};
			edgeDetectionOptimal.push_back(toAdd);
			cornerDetectionOptimal.push_back(toAdd);
			hughTransformOptimal.push_back(toAdd);
		}


		for(int i = 0; i < groupedImages.size(); i++){ //For each image in list:

			int bestEdgeTime = 9999999999;
			int optimalEdgeCount = 0;
			int bestCornerTime = 9999999999;
			int optimalCornerCount = 0;
			int bestHughTime = 9999999999;
			int optimalHughCount = 0;

			for(int j = 1; j <= 32; j++){ //For each thread count 1-32

				int edgeTime = //Perform edge detection
				if(bestEdgeTime > edgeTime){
					bestEdgeTime = edgeTime;
					optimalEdgeCount = j;
				}

				int cornerTime = //Perform corner detection
				if(bestCornerTime > cornerTime){
					bestCornerTime = cornerTime;
					optimalCornerCount = j;
				}

				int hughTime = //Perform Hugh transoform
				if(bestHughTime > hughTime){
					bestHughTime = hughTime;
					optimalHughCount = j;
				} 

				//Log execution times for this thread count
			}

			int kmeansIndex = groupedImages[i].second;

			//Update result vectors with this image's info
			edgeDetectionOptimal[kmeansIndex][1]++; //Increment num points used for this group
			edgeDetectionOptimal[kmeansIndex][2] += optimalEdgeCount; //Add optimal thread count to total

			cornerDetectionOptimal[kmeansIndex][1]++;
			cornerDetectionOptimal[kmeansIndex][2] += optimalCornerCount;

			hughTransformOptimal[kmeansIndex][1]++;
			hughTransformOptimal[kmeansIndex][2] += optimalHughCount;

			//Log kmeans pixel counts and average optimal thread counts (total / numPointsUsed) for each algorithm

			return 0;
	}
	else{ //Testing phase
		//Read in kmeans pixel counts and optimal thread counts for each algorithm
		//Format: [[pixelCount, optimalThreads], [...]]
		vector<vector<int>> edgeOptimal;
		vector<vector<int>> cornerOptimal;
		vector<vector<int>> hughOptimal;

		//Collect images from filePath into vector of strings representing the image files
		//Format: <string filename, int pixelCount>
		vector<pair<string, int> > imageFiles;

		//Determine closest matching pixelCount for each image
		//Set the int in imageFiles equal to this closest pixelCount
		for(int i = 0; i < imageFiles.size(); i++){
			int smallestDiff = 999999999;
			int closestMatch = 0;
			for(int j = 0; j < edgeOptimal.size(); j++){
				if(abs(edgeOptimal[j][0] - imageFiles[i].second) < smallestDiff){
					closestMatch = edgeOptimal[j][0];
					smallestDiff = abs(edgeOptimal[j][0] - imageFiles[i].second); 
				}
			}
			imageFiles[i].second = closestMatch;
		}


		//Deterime optimal thread count for each image and algorithm. Place results in vector
		//Format: [[optimalEdge, optimalCorner, optimalHugh], [...]]
		vector<vector<int> > optimalCounts;
		for(int i = 0; i < imageFiles.size(); i++){
			vector<int> optimal;
			for(int j = 0; j < edgeOptimal.size(); j++){
				if(edgeOptimal[j][0] == imageFiles[i].second){
					optimal.push_back(edgeOptimal[j][1]);
					break;
				}
			}
			for(int j = 0; j < cornerOptimal.size(); j++){
				if(cornerOptimal[j][0] == imageFiles[i].second){
					optimal.push_back(cornerOptimal[j][1]);
					break;
				}
			}
			for(int j = 0; j < hughOptimal.size(); j++){
				if(hughOptimal[j][0] == imageFiles[i].second){
					optimal.push_back(hughOptimal[j][1]);
					break;
				}
			}
			optimalCounts.push_back(optimal);
		}


		vector<vector<int> > optimalTimes; //Format: [[edgetime, cornerTime, hughTime], [...]]
		//Calculate time to complete algorithms using our expected optimal counts
		for(int i = 0; i < imageFiles.size(); i++){
			int edgeTime = edgeDetection(imageFiles[i], optimalCounts[i][0]);
			int cornerTime = cornerDetection(imageFiles[i], optimalCounts[i][1]);
			int hughTime = hughDetection(imageFiles[i], optimalCounts[i][2]);
			vector<int> times = {edgetime, cornerTime, hughTime};
			optimalTimes.push_back(times);
		}


		vector<vector<int> > realOptimalTimes;
		vecotr<vector<int> > realOptimalCounts;
		//Calculate real optimal times by trying with all thread counts
		for(int i = 0; i < imageFiles.size(); i++){

			int bestEdgeTime = optimalTimes[i][0];
			int optimalEdgeCount = optimalCounts[i][0];
			int bestCornerTime = optimalTimes[i][1];
			int optimalCornerCount = optimalCounts[i][1];
			int bestHughTime = optimaltimes[i][2];
			int optimalHughCount = optimalCounts[i][2];

			for(int j = 1; j <= 32; j++){ //For each thread count 1-32

				int edgeTime = edgeDetection(imageFiles[i], optimalCounts[i][0]);
				if(bestEdgeTime > edgeTime){
					bestEdgeTime = edgeTime;
					optimalEdgeCount = j;
				}

				int cornerTime = //Perform corner detection
				if(bestCornerTime > cornerTime){
					bestCornerTime = cornerTime;
					optimalCornerCount = j;
				}

				int hughTime = //Perform Hugh transoform
				if(bestHughTime > hughTime){
					bestHughTime = hughTime;
					optimalHughCount = j;
				} 
			}
			vector<int> times = {bestEdgeTime, bestCornerTime, bestHughTime};
			vector<int> counts = {optimalEdgeCount, optimalCornerCount, optimalHughCount};
			realOptimalTimes.push_back(times);
			realOptimalCounts.push_back(counts);
		}

		for(int i = 0; i < imageFiles.size(); i++){
			bool optimalMatch;
			optimalCounts[i][0] == realOptimalCounts[i][0] ? optimalMatch = true : optimalMatch = false;
			//Log result
		}


	}

  
    


    return 0;
}