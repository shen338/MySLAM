#include <opencv2/core.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}


int main(int ac, char** av){
    if (ac != 2)
    {
        help(av);
        return 1;
    }

    // Open YAML file
    // Run the write part, all read part is commented
    string filename = "../test.yaml";
    FileStorage fs(filename, FileStorage::WRITE);
    // fs.open(filename, FileStorage::READ);
    
    // Use << operator to output data
    fs << "iterationN" << 100;
    // int itNr;
    // fs["iterationNr"] >> itNr;
    // itNr = (int) fs["iterationNr"];

    Mat R = Mat_<uchar >::eye  (3, 3),
        T = Mat_<double>::zeros(3, 1);
    fs << "R" << R;                                      // Write cv::Mat
    fs << "T" << T;

    // fs["R"] >> R;                                      // Read cv::Mat
    // fs["T"] >> T;
    
    // Store a sequence
    fs << "strings" << "[";                              // text - string sequence
    fs << "image1.jpg" << "Awesomeness" << "baboon.jpg";
    fs << "]";                                           // close sequence
    
    // Store a mapping (dict)
    fs << "Mapping";                              // text - mapping
    fs << "{" << "One" << 1;
    fs <<        "Two" << 2 << "}";

    // Read from sequence and Dict
    // FileNode n = fs["strings"];
    // if(n.type() != FileNode::SEQ){
    //     cerr << "strings is not a sequence" << endl;
    // }

    // FileNodeIterator iter = n.begin(), iter_end = n.end();
    // for(;iter != iter_end;iter++){
    //     cout << (string)*iter << endl;
    // }

    // FileNode n = fs["Mapping"];                                // Read mappings from a sequence
    // cout << "Two  " << (int)(n["Two"]) << "; ";
    // cout << "One  " << (int)(n["One"]) << endl << endl;

    fs.release();

    return 0;
}
