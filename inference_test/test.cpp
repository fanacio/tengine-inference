/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: fanacio
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#define __GNUC__
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H        416
#define DEFAULT_IMG_W        416
#define DEFAULT_IMG_C        3
#define DEFAULT_SCALE1       0.229f * 255
#define DEFAULT_SCALE2       0.224f * 255
#define DEFAULT_SCALE3       0.225f * 255
#define DEFAULT_MEAN1        0.485 * 255
#define DEFAULT_MEAN2        0.456 * 255
#define DEFAULT_MEAN3        0.406 *255
#define DEFAULT_LOOP_COUNT   1
#define DEFAULT_THREAD_COUNT 1
//下面的宏用于后处理
#define OUTPUT_N             1
#define OUTPUT_C             3549
#define OUTPUT_HW            6
#define BOX_CORNER_HW        4
#define GRIDS_HW             2
#define STRIDES_HW           1
#define NUM_CLASSES          1
#define CONF_THRE            0.25
#define DETECTIONS_DIMS      7
#define CLASS_AGNOSTIC       true
#define NMS_THRE             0.45
#define LEGACY               false
#define ACLM                 true

using namespace std;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

int getBinSize(std::string path)
{
    int size = 0;
    std::ifstream infile(path, std::ifstream::binary);

    infile.seekg(0, infile.end);
    size = infile.tellg();
    infile.seekg(0,infile.beg);

    infile.close();
    printf("\npath=%s, size=%d \n",path , size);
    return size;
}

void readBin(std::string path, char *buf , int size)
{
    std::ifstream infile(path, std::ifstream::binary);

    infile.read(static_cast<char *>(buf), size);
    infile.close();
}

float *grids_strides(std::string filePath)
{
    int size = getBinSize(filePath);
    char *buf = new char [size];
    readBin(filePath, buf, size);
    float *fbuf = reinterpret_cast<float *>(buf);
    return fbuf;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("yolox-nano-shoot.jpg", image);
}


void get_input_data_shoot(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, const float* scale, float ratio)
{
    cv::Mat sample = cv::imread(image_file, 1);
    int pic_h = sample.rows;
    int pic_w = sample.cols;

    // if (sample.channels() == 1)
    //     cv::cvtColor(sample, sample, cv::COLOR_GRAY2RGB);
    // else
    //     cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);

    /* resize process */
    cv::resize(sample, sample, cv::Size(pic_w * ratio, pic_h * ratio));
    sample.convertTo(sample, CV_32FC3);
    //Fill size
    cv::copyMakeBorder(sample, sample, 0, img_h - pic_h * ratio, 0, img_w - pic_w * ratio, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    float* img_data = (float*)sample.data;

    /* nhwc to nchw */
    if (LEGACY)
    {
        
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int in_index = h * img_w * 3 + w * 3 + c;
                }
            }
        }
    }
    else
    {
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int in_index = h * img_w * 3 + w * 3 + c;          
                    input_data[out_index] = (img_data[in_index]) ;
                }
            }
        }     
    }
}

static void nms_sorted_bboxes(const std::vector<Object>& yoloxobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    
    const int n = yoloxobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = yoloxobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = yoloxobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = yoloxobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
            {
                picked[j] = a.prob > b.prob ? i : picked[j];
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

int post_run(float* output_data, float* grids, float* strides, std::vector<Object>& o_objects, float ratio, int raw_w, int raw_h)
{
    for (size_t i = 0; i < OUTPUT_C; i++)
    {
        output_data[OUTPUT_HW*i] = (output_data[OUTPUT_HW*i]+grids[GRIDS_HW*i])*strides[i];
        output_data[OUTPUT_HW*i+1] = (output_data[OUTPUT_HW*i+1]+grids[GRIDS_HW*i+1])*strides[i];
        output_data[OUTPUT_HW*i+2] = exp(output_data[OUTPUT_HW*i+2])*strides[i];
    }
    
    /*Get score and class with highest confidence*/
    float *class_conf = new float[OUTPUT_N * OUTPUT_C];
    int *class_pred = new int[OUTPUT_N * OUTPUT_C];

    for (size_t i = 0; i < OUTPUT_C; i++)
    {
        float maxconf = 0.f ;
        int max_pred = 0 ;
        for (size_t j = 5; j < 5 + NUM_CLASSES; j++)
        {
            if (output_data[OUTPUT_HW*i+j] > maxconf)
            {
                maxconf = output_data[OUTPUT_HW*i+j];
                max_pred = j - 5;
            }
        }
        class_conf[i] = maxconf;
        class_pred[i] = max_pred;
    }

    bool *conf_mask = new bool[OUTPUT_N * OUTPUT_C];
    vector<int> index ;
    for (size_t i = 0; i < OUTPUT_C; i++)
    {
        conf_mask[i]  = output_data[OUTPUT_HW*i+4] * (float)class_conf[i] >= CONF_THRE ? true : false;
        if (conf_mask[i] == true)
        {
            index.push_back(i);
        }
    }
    int count = index.size();

    /* Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred) */
    float *detections = new float[DETECTIONS_DIMS * OUTPUT_C];
    for (size_t i = 0; i < OUTPUT_C; i++)
    {
        for (size_t j = 0; j < 5; j++)
        {
            detections[DETECTIONS_DIMS * i+j] = output_data[OUTPUT_HW * i + j];
        }
        detections[DETECTIONS_DIMS * i+5] = class_conf[i];
    }

    float *o_detections = new float[count * DETECTIONS_DIMS];

    for (size_t i = 0; i < count; i++)
    {
        for (size_t j = 0; j < DETECTIONS_DIMS; j++)
        {
            o_detections[DETECTIONS_DIMS * i + j] = detections[DETECTIONS_DIMS * index[i] + j];
        }    
    }

    if (count != 0)
    {
        if (CLASS_AGNOSTIC)
        {
            std::vector<int> picked;
            std::vector<Object> objects;
            for (size_t i = 0; i < count; i++)
            {
                Object obj;
                obj.rect.x = o_detections[DETECTIONS_DIMS * i]- o_detections[DETECTIONS_DIMS * i + 2]/2;
                obj.rect.y = o_detections[DETECTIONS_DIMS * i + 1]- o_detections[DETECTIONS_DIMS * i + 3]/2;
                obj.rect.width = o_detections[DETECTIONS_DIMS * i + 2];
                obj.rect.height = o_detections[DETECTIONS_DIMS * i + 3];
                obj.label = (int)o_detections[DETECTIONS_DIMS * i + 6];
                obj.prob = o_detections[DETECTIONS_DIMS * i + 4] * o_detections[DETECTIONS_DIMS * i + 5];
                objects.push_back(obj);
            }
            /* NMS */
            nms_sorted_bboxes(objects, picked, NMS_THRE);

            int objects_num = picked.size();
            o_objects.resize(objects_num);
            for (int i = 0; i < objects_num; i++)
            {
                o_objects[i] = objects[picked[i]];
                float x1 = (o_objects[i].rect.x);
                float y1 = (o_objects[i].rect.y);
                float x2 = (o_objects[i].rect.x + o_objects[i].rect.width);
                float y2 = (o_objects[i].rect.y + o_objects[i].rect.height);

                x1 = x1 / ratio;
                y1 = y1 / ratio;
                x2 = x2 / ratio;
                y2 = y2 / ratio;

                x1 = std::max(std::min(x1, (float)(raw_w - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(raw_h - 1)), 0.f);
                x2 = std::max(std::min(x2, (float)(raw_w - 1)), 0.f);
                y2 = std::max(std::min(y2, (float)(raw_h - 1)), 0.f);

                o_objects[i].rect.x = x1;
                o_objects[i].rect.y = y1;
                o_objects[i].rect.width = x2 - x1;
                o_objects[i].rect.height = y2 - y1;
            }
        }
        else
        {
            fprintf(stderr, "This feature needs to be improved.\n");
        }

    }
    else
    {
        fprintf(stderr, "no target : shoot.\n");
        return -1;
    }

    return 0;

}

int yolox_shoot(const char* model_file, const char* image_file, int img_h, int img_w, int img_c, float* mean, float* scale,
                     int loop_count, int num_thread)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_BIG;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }

    int raw_h = img.rows;
    int raw_w = img.cols;
    float ratio = (float)img_w/raw_w > (float)img_h/raw_h ? (float)img_h/raw_h : (float)img_w/raw_w;

    //读取bin文件
    float* grids = grids_strides("../../lib/common/grids.bin");
    float* strides = grids_strides("../../lib/common/strides.bin");

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create arm acl backend */
    context_t aclm_context = create_context("aclm", 1);
    int rtt = add_context_device(aclm_context, "ACLM");
    if (0 > rtt)
    {
        fprintf(stderr, "add_context_device NVDEVICE failed.\n");
        return -1;
    }

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph;
    if (ACLM)
    {
        graph = create_graph(aclm_context, "tengine", model_file);
    }
    else
    {
        graph = create_graph(NULL, "tengine", model_file);
    }

    //graph_t graph = create_graph(NULL, "tengine", model_file); //不适用GPU加速
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * img_c;
    int dims[] = {1, img_c, img_h, img_w};  // nchw
    std::vector<float> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_shoot(image_file, input_data.data(), img_h, img_w, mean, scale, ratio);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < loop_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "\nmodel file : %s\n", model_file);
    fprintf(stderr, "image file : %s\n", image_file);
    fprintf(stderr, "img_h, img_w, scale[3], mean[3] : %d %d , %.3f %.3f %.3f, %.3f %.3f %.3f\n", img_h, img_w,
            scale[0], scale[1], scale[2], mean[0], mean[1], mean[2]);
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of shoot */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = (float*)get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor);

    /* postprocess */
    std::vector<Object> o_objects;
    post_run(output_data, grids, strides, o_objects, ratio, raw_w, raw_h);

    /* yolox-nano draw the result */
    draw_objects(img, o_objects);

    fprintf(stderr, "--------------------------------------\n");

    /* release tengine */
    //free(input_data);
    //free(output_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file]\n [-g img_h,img_w] [-s scale[0],scale[1],scale[2]] [-w "
        "mean[0],mean[1],mean[2]] [-r loop_count] [-t thread_count]\n");
    fprintf(
        stderr,
        "\nmobilenet example: \n    ./shoot -m /path/to/shootmodel_caffe_uint8.tmfile -i /path/to/img.jpg -g 3,416,416 -s "
        "0.017,0.017,0.017 -w 104.007,116.669,122.679\n");
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = NULL;
    char* image_file = NULL;
    float img_hw[3] = {0.f};
    int img_h = 0;
    int img_w = 0;
    int img_c = 0;
    float mean[3] = {-1.0, -1.0, -1.0};
    float scale[3] = {0.f, 0.f, 0.f};

    int res;
    while ((res = getopt(argc, argv, "m:i:l:g:s:w:r:t:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'g':
            split(img_hw, optarg, ",");
            img_c = (int)img_hw[0];
            img_h = (int)img_hw[1];
            img_w = (int)img_hw[2];
            break;
        case 's':
            split(scale, optarg, ",");
            break;
        case 'w':
            split(mean, optarg, ",");
            break;
        case 'r':
            loop_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == NULL)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    if (img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        fprintf(stderr, "Image height not specified, use default %d\n", img_h);
    }

    if (img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        fprintf(stderr, "Image width not specified, use default  %d\n", img_w);
    }
    if (img_c == 0)
    {
        img_c = DEFAULT_IMG_C;
        fprintf(stderr, "Image channel not specified, use default  %d\n", img_c);
    }

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        scale[0] = DEFAULT_SCALE1;
        scale[1] = DEFAULT_SCALE2;
        scale[2] = DEFAULT_SCALE3;
        fprintf(stderr, "Scale value not specified, use default  %.3f, %.3f, %.3f\n", scale[0], scale[1], scale[2]);
    }

    if (mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        fprintf(stderr, "Mean value not specified, use default   %.3f, %.3f, %.3f\n", mean[0], mean[1], mean[2]);
    }

    if (yolox_shoot(model_file, image_file, img_h, img_w, img_c, mean, scale, loop_count, num_thread) < 0)
        return -1;

    return 0;
}
