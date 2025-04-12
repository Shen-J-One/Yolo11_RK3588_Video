// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <chrono>

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

/*-------------------------------------------
            Global Variables
-------------------------------------------*/
std::atomic<bool> signal_received(false);

// Performance optimization settings
#define MAX_QUEUE_SIZE 128       // Maximum frames to buffer in queue (reduced from 256)
#define WORKER_THREADS 1        // Number of inference threads
#define DISPLAY_SCALE_FACTOR 1  // Scale display window (higher = smaller window)
#define PROCESSING_SCALE 1      // Scale factor for input frames (1.0 = 100% of original size)
#define CONFIDENCE_THRESHOLD 0.2 // Detection confidence threshold
#define USE_OPENMP 1            // Use OpenMP for parallel processing
#define DISPLAY_DELAY_MS 15     // Delay between frames to control display speed (ms) (0 = disable, 1 = enable)

// Frame structure to pass between threads
struct Frame {
    cv::Mat raw;               // Original frame
    cv::Mat processed;         // Processed frame with detections
    double timestamp;          // Capture timestamp
    bool ready;                // Whether the frame is processed and ready for display
};

// Threading resources
std::queue<Frame> frame_queue;
std::vector<Frame> display_frames(MAX_QUEUE_SIZE);
std::mutex queue_mutex, display_mutex;
std::condition_variable queue_cond, display_cond;
int next_frame_idx = 0;
int next_display_idx = 0;
std::atomic<bool> processing_complete(false);

// Extended image buffer with additional tracking
struct extended_buffer_t {
    image_buffer_t buf;
    bool in_use;
};

// Image buffer pool for reuse
std::vector<extended_buffer_t> buffer_pool;
std::mutex buffer_mutex;

// Pre-allocated buffers for zero-copy when possible
image_buffer_t preallocated_buf;

// Adaptive FPS and performance tracking
double target_fps = 30.0;
double current_fps = 0.0;
double processing_time_ms = 0.0;
double avg_processing_time = 0.0;
int processed_frames = 0;

// Display timing control
std::chrono::time_point<std::chrono::steady_clock> last_display_time;

/*-------------------------------------------
            Signal Handler
-------------------------------------------*/
void signal_handler(int signal) {
    printf("\nReceived signal %d, exiting...\n", signal);
    signal_received = true;
}

/*-------------------------------------------
    Optimized Mat to Image Buffer Conversion
-------------------------------------------*/
int cv_mat_to_image_buffer(cv::Mat& cv_img, image_buffer_t* img_buf) {
    if (cv_img.empty() || !img_buf) {
        return -1;
    }
    
    // Setup buffer properties
    img_buf->width = cv_img.cols;
    img_buf->height = cv_img.rows;
    img_buf->format = IMAGE_FORMAT_RGB888;
    img_buf->size = cv_img.total() * 3;
    
    // Reuse buffer if possible
    if (img_buf->virt_addr == NULL) {
        img_buf->virt_addr = (unsigned char*)malloc(img_buf->size);
        if (img_buf->virt_addr == NULL) {
            printf("Failed to allocate memory for image buffer\n");
            return -1;
        }
    }
    
    // Fast BGR to RGB conversion
    int width = cv_img.cols;
    int height = cv_img.rows;
    unsigned char* src = cv_img.data;
    unsigned char* dst = img_buf->virt_addr;
    
#if USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < height; i++) {
        unsigned char* src_row = src + i * cv_img.step;
        unsigned char* dst_row = dst + i * width * 3;
        
        for (int j = 0; j < width; j++) {
            int src_idx = j * 3;
            int dst_idx = j * 3;
            dst_row[dst_idx] = src_row[src_idx+2];      // R = B
            dst_row[dst_idx+1] = src_row[src_idx+1];    // G = G
            dst_row[dst_idx+2] = src_row[src_idx];      // B = R
        }
    }
    
    return 0;
}

/*-------------------------------------------
    Get Buffer from Pool or Create New
-------------------------------------------*/
image_buffer_t* get_buffer_from_pool(int width, int height) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    // Search for available buffer of correct size
    for (auto& item : buffer_pool) {
        if (item.buf.width == width && item.buf.height == height && !item.in_use) {
            item.in_use = true;
            return &item.buf;
        }
    }
    
    // Create new buffer if none available
    extended_buffer_t new_item;
    new_item.in_use = true;
    new_item.buf.width = width;
    new_item.buf.height = height;
    new_item.buf.format = IMAGE_FORMAT_RGB888;
    new_item.buf.size = width * height * 3;
    new_item.buf.virt_addr = (unsigned char*)malloc(new_item.buf.size);
    
    if (new_item.buf.virt_addr == NULL) {
        return NULL;
    }
    
    buffer_pool.push_back(new_item);
    return &buffer_pool.back().buf;
}

/*-------------------------------------------
    Return Buffer to Pool
-------------------------------------------*/
void return_buffer_to_pool(image_buffer_t* buf) {
    if (!buf) return;
    
    std::lock_guard<std::mutex> lock(buffer_mutex);
    for (auto& item : buffer_pool) {
        if (item.buf.virt_addr == buf->virt_addr) {
            item.in_use = false;
            break;
        }
    }
}

/*-------------------------------------------
    Inference Worker Thread
-------------------------------------------*/
void inference_worker(rknn_app_context_t* app_ctx, int thread_id) {
    printf("Inference worker %d started\n", thread_id);
    
    while (!signal_received && !processing_complete) {
        Frame frame;
        bool has_frame = false;
        
        // Get a frame from the queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cond.wait(lock, [&]() { 
                return !frame_queue.empty() || signal_received || processing_complete; 
            });
            
            if (signal_received || processing_complete) {
                break;
            }
            
            if (!frame_queue.empty()) {
                frame = frame_queue.front();
                frame_queue.pop();
                has_frame = true;
                
                // Notify that we've removed a frame, in case main thread is waiting
                queue_cond.notify_one();
            }
        }
        
        if (!has_frame) continue;
        
        // Start timing
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        // Prepare image buffer from the frame
        image_buffer_t* frame_buf = get_buffer_from_pool(frame.raw.cols, frame.raw.rows);
        if (!frame_buf) {
            printf("Failed to get buffer from pool\n");
            continue;
        }
        
        // Convert frame to image buffer
        cv_mat_to_image_buffer(frame.raw, frame_buf);
        
#if defined(RV1106_1103)
        // RV1106 rga requires DMA memory allocation
        int ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, frame_buf->size, &app_ctx->img_dma_buf.dma_buf_fd, 
                               (void **) & (app_ctx->img_dma_buf.dma_buf_virt_addr));
        memcpy(app_ctx->img_dma_buf.dma_buf_virt_addr, frame_buf->virt_addr, frame_buf->size);
        dma_sync_cpu_to_device(app_ctx->img_dma_buf.dma_buf_fd);
        frame_buf->fd = app_ctx->img_dma_buf.dma_buf_fd;
        app_ctx->img_dma_buf.size = frame_buf->size;
#endif
        
        // Run inference
        object_detect_result_list od_results;
        int ret = inference_yolo11_model(app_ctx, frame_buf, &od_results);
        
        if (ret != 0) {
            printf("Inference failed with error %d\n", ret);
#if defined(RV1106_1103)
            dma_buf_free(app_ctx->img_dma_buf.size, &app_ctx->img_dma_buf.dma_buf_fd,
                        app_ctx->img_dma_buf.dma_buf_virt_addr);
#endif
            return_buffer_to_pool(frame_buf);
            continue;
        }
        
        // Draw bounding boxes on the processed frame
        cv::Mat result_frame = frame.raw.clone();
        char text[256];
        
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // Only draw detections above threshold
            if (det_result->prop >= CONFIDENCE_THRESHOLD) {
                // Draw rectangle with thickness proportional to confidence
                int thickness = std::max(1, (int)(det_result->prop * 5));
                cv::rectangle(result_frame, 
                            cv::Point(det_result->box.left, det_result->box.top),
                            cv::Point(det_result->box.right, det_result->box.bottom),
                            cv::Scalar(0, 0, 255), thickness);
                
                // Draw label with background for better visibility
                sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(result_frame, 
                            cv::Point(det_result->box.left, det_result->box.top - text_size.height - 5),
                            cv::Point(det_result->box.left + text_size.width, det_result->box.top),
                            cv::Scalar(0, 0, 0), -1);
                cv::putText(result_frame, text, 
                        cv::Point(det_result->box.left, det_result->box.top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // Measure inference time
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
        
        processing_time_ms = elapsed * 1000;
        
        // Update the processed frame in the display buffer
        {
            std::lock_guard<std::mutex> lock(display_mutex);
            frame.processed = result_frame;
            frame.ready = true;
            display_frames[next_display_idx] = frame;
            next_display_idx = (next_display_idx + 1) % MAX_QUEUE_SIZE;
            
            // Update FPS statistics
            processed_frames++;
            avg_processing_time = (avg_processing_time * (processed_frames - 1) + elapsed) / processed_frames;
            current_fps = 1.0 / avg_processing_time;
        }
        
        // Signal that a frame is ready for display
        display_cond.notify_one();
        
        // Free resources
#if defined(RV1106_1103)
        dma_buf_free(app_ctx->img_dma_buf.size, &app_ctx->img_dma_buf.dma_buf_fd,
                   app_ctx->img_dma_buf.dma_buf_virt_addr);
#endif
        return_buffer_to_pool(frame_buf);
    }
    
    printf("Inference worker %d stopped\n", thread_id);
}

/*-------------------------------------------
                Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <video_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_path = argv[2];

    // Register signal handler
    signal(SIGINT, signal_handler);
    
    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // Initialize post-processing
    init_post_process();

    // Initialize YOLO model
    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model failed! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    // Open video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        printf("Failed to open video file: %s\n", video_path);
        deinit_post_process();
        release_yolo11_model(&rknn_app_ctx);
        return -1;
    }
    
    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    
    // Calculate processing dimensions
    int proc_width = frame_width * PROCESSING_SCALE;
    int proc_height = frame_height * PROCESSING_SCALE;
    
    printf("Video properties: %dx%d, %.2f fps, %d frames\n", 
           frame_width, frame_height, video_fps, total_frames);
    printf("Processing at resolution: %dx%d\n", proc_width, proc_height);
    
    // Initialize display frames buffer
    for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
        display_frames[i].ready = false;
    }
    
    // Pre-allocate image buffer for zero-copy when possible
    memset(&preallocated_buf, 0, sizeof(image_buffer_t));
    preallocated_buf.width = proc_width;
    preallocated_buf.height = proc_height;
    preallocated_buf.format = IMAGE_FORMAT_RGB888;
    preallocated_buf.size = preallocated_buf.width * preallocated_buf.height * 3;
    preallocated_buf.virt_addr = (unsigned char*)malloc(preallocated_buf.size);
    
    // Start inference worker threads
    std::vector<std::thread> workers;
    for (int i = 0; i < WORKER_THREADS; i++) {
        workers.push_back(std::thread(inference_worker, &rknn_app_ctx, i));
    }
    
    // Create display window
    cv::namedWindow("YOLO Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO Detection", frame_width/DISPLAY_SCALE_FACTOR, frame_height/DISPLAY_SCALE_FACTOR);
    
    // Initialize display timing
    last_display_time = std::chrono::steady_clock::now();
    
    // Frame reading and display
    int frame_count = 0;
    int frames_since_last_fps_update = 0;
    int skip_frames = 0; // Fixed frame skipping (no skipping)
    struct timespec fps_timer_start, fps_timer_now;
    clock_gettime(CLOCK_MONOTONIC, &fps_timer_start);
    
    while (!signal_received) {
        // Read frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            printf("Reached end of video\n");
            break;
        }
        
        frame_count++;
        
        // Skip frames if needed
        if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 1) {
            continue;
        }
        
        // Resize frame to improve performance
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(proc_width, proc_height));
        
        // Add frame to queue for processing
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Wait until there's space in the queue
            queue_cond.wait(lock, [&]() {
                return frame_queue.size() < MAX_QUEUE_SIZE || signal_received;
            });
            
            if (signal_received) {
                break;
            }
            
            // Now we know there's space in the queue
            Frame new_frame;
            new_frame.raw = resized_frame;
            new_frame.ready = false;
            
            // Get current timestamp
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            new_frame.timestamp = ts.tv_sec + ts.tv_nsec / 1000000000.0;
            
            frame_queue.push(new_frame);
            
            // Signal workers that new frame is available
            queue_cond.notify_one();
        }
        
        // Check for processed frames to display
        bool got_frame = false;
        cv::Mat display_frame;
        
        {
            std::unique_lock<std::mutex> lock(display_mutex);
            display_cond.wait_for(lock, std::chrono::milliseconds(1), [&]() {
                for (const auto& frame : display_frames) {
                    if (frame.ready) return true;
                }
                return false;
            });
            
            // Find a ready frame to display
            for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
                if (display_frames[i].ready) {
                    display_frame = display_frames[i].processed.clone();
                    display_frames[i].ready = false; // Mark as used
                    got_frame = true;
                    break;
                }
            }
        }
        
        // Display frame if available
        if (got_frame) {
            // Add FPS and frame count to display
            char fps_text[64];
            sprintf(fps_text, "FPS: %.1f  Processing: %.1f ms  Frame: %d/%d", 
                    current_fps, processing_time_ms, frame_count, total_frames);
            
            cv::putText(display_frame, fps_text, cv::Point(10, 30), 
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Display result
            cv::imshow("YOLO Detection", display_frame);
            
            // Add delay to control display speed
            cv::waitKey(DISPLAY_DELAY_MS);
        }
        
        // Update FPS counter every 30 frames
        frames_since_last_fps_update++;
        if (frames_since_last_fps_update >= 30) {
            clock_gettime(CLOCK_MONOTONIC, &fps_timer_now);
            double elapsed = (fps_timer_now.tv_sec - fps_timer_start.tv_sec) + 
                           (fps_timer_now.tv_nsec - fps_timer_start.tv_nsec) / 1000000000.0;
            
            double fps = frames_since_last_fps_update / elapsed;
            
            // Reset counter
            frames_since_last_fps_update = 0;
            clock_gettime(CLOCK_MONOTONIC, &fps_timer_start);
            
            // Update current FPS (this is display FPS, not inference FPS)
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                printf("Display FPS: %.2f, Inference FPS: %.2f, Processing: %.1f ms/frame, Queue size: %zu/%d\n", 
                       fps, current_fps, processing_time_ms, frame_queue.size(), MAX_QUEUE_SIZE);
            }
        }
        
        // Check for ESC key (no need for additional waitKey since we added delay above)
        int key = cv::waitKey(1);
        if (key == 27) { // ESC key
            printf("ESC pressed, exiting...\n");
            signal_received = true;
        }
        
        // Add a small sleep if no frame was processed to avoid CPU spinning
        if (!got_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        // Exit if we've processed all frames
        if (frame_count >= total_frames) {
            break;
        }
    }
    
    // Signal threads to exit and wait for completion
    processing_complete = true;
    queue_cond.notify_all();
    for (auto& worker : workers) {
        worker.join();
    }
    
    printf("Finished processing video.\n");
    
    // Clean up buffer pool
    for (auto& item : buffer_pool) {
        if (item.buf.virt_addr != NULL) {
            free(item.buf.virt_addr);
            item.buf.virt_addr = NULL;
        }
    }
    
    if (preallocated_buf.virt_addr != NULL) {
        free(preallocated_buf.virt_addr);
    }
    
    // Release video resources
    cap.release();
    cv::destroyAllWindows();

    // Cleanup YOLO resources
    deinit_post_process();
    release_yolo11_model(&rknn_app_ctx);

    return 0;
}
