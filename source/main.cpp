//
// Copyright (C) YuqiaoZhang(HanetakaChou)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>
#include "../thirdparty/mediapipe/include/mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "../thirdparty/mediapipe/include/mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#define CV_IGNORE_DEBUG_BUILD_GUARD 1
#include <opencv2/opencv.hpp>

// https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
// https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
extern uint8_t const *const face_landmarker_task_base;
extern size_t const face_landmarker_task_size;

// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
// https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
// https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
// https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
extern uint8_t const *const pose_landmarker_task_base;
extern size_t const pose_landmarker_task_size;

#define ENABLE_DEBUG_DISPLAY 1
#define ENABLE_FPS_OUTPUT 1
#define ENABLE_DEBUG_OUTPUT 0
#define ENABLE_FACE_LANDMARKER 1
#define ENABLE_POSE_LANDMARKER 1

int main(int argc, char *argv[])
{
    cv::VideoCapture video_capture;

    bool video_capture_open;
    bool is_camera_video_capture;
    if (argc >= 2)
    {
        video_capture_open = video_capture.open(argv[1], cv::CAP_ANY);
        is_camera_video_capture = false;
    }
    else
    {
        video_capture_open = video_capture.open(0, cv::CAP_ANY);
        is_camera_video_capture = true;
    }

    if (!(video_capture_open && video_capture.isOpened()))
    {
        std::cout << "fail to open video capture " << std::endl;
        return -1;
    }
    else
    {
        cv::String backend_name = video_capture.getBackendName();
        std::cout << "video capture backend name: " << backend_name << std::endl;
    }

    if (is_camera_video_capture)
    {
        // Too high resolution may reduce FPS
        video_capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        video_capture.set(cv::CAP_PROP_FPS, 60);
    }

    std::cout << "!!!!!!!" << std::endl;
    std::cout << "ATTENTION: you may set the environment variable TFLITE_FORCE_GPU=1 to force OpenCL inference" << std::endl;
    std::cout << "!!!!!!!" << std::endl;

#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
    void *face_landmarker = NULL;
    {
        FaceLandmarkerOptions options;
        options.base_options.model_asset_buffer = reinterpret_cast<char const *>(face_landmarker_task_base);
        options.base_options.model_asset_buffer_count = static_cast<unsigned int>(face_landmarker_task_size);
        options.base_options.model_asset_path = NULL;
        options.running_mode = VIDEO;
        options.num_faces = 1;
        // options.min_face_detection_confidence = 0.5F;
        // options.min_face_presence_confidence = 0.5F;
        // options.min_tracking_confidence = 0.5F;
        options.output_face_blendshapes = true;
        options.output_facial_transformation_matrixes = true;
        options.result_callback = NULL;

        char *error_msg_face_landmarker_create = NULL;
        face_landmarker = face_landmarker_create(&options, &error_msg_face_landmarker_create);
        assert(NULL == error_msg_face_landmarker_create);
    }
    assert(NULL != face_landmarker);
#endif

#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
    void *pose_landmarker = NULL;
    {
        PoseLandmarkerOptions options;
        options.base_options.model_asset_buffer = reinterpret_cast<char const *>(pose_landmarker_task_base);
        options.base_options.model_asset_buffer_count = static_cast<unsigned int>(pose_landmarker_task_size);
        options.base_options.model_asset_path = NULL;
        options.running_mode = VIDEO;
        options.num_poses = 1;
        // options.min_pose_detection_confidence = 0.5F;
        // options.min_pose_presence_confidence = 0.5F;
        // options.min_tracking_confidence = 0.5F;
        options.output_segmentation_masks = false;
        options.result_callback = NULL;

        char *error_msg_pose_landmarker_create = NULL;
        pose_landmarker = pose_landmarker_create(&options, &error_msg_pose_landmarker_create);
        assert(NULL == error_msg_pose_landmarker_create);
    }
#endif

#if defined(ENABLE_DEBUG_DISPLAY) && ENABLE_DEBUG_DISPLAY
    constexpr char const k_window_name[] = {"Press Any Key To Exit"};

    cv::namedWindow(k_window_name, cv::WINDOW_AUTOSIZE);
#endif

    std::vector<uint8_t> pixel_data;
    double const tick_frequency = cv::getTickFrequency();
#if defined(ENABLE_FPS_OUTPUT) && ENABLE_FPS_OUTPUT
    int64 tick_count_previous = cv::getTickCount();
#endif
    bool running = true;
    while (running)
    {
        cv::Mat raw_video_image_matrix;
        if (video_capture.read(raw_video_image_matrix) && (!raw_video_image_matrix.empty()))
        {
            cv::Mat input_image_matrix;
            {
                cv::cvtColor(raw_video_image_matrix, input_image_matrix, cv::COLOR_BGR2RGB);
            }

            MpImage input_image;
            {
                // mediapipe/examples/desktopdemo_run_graph_main.cc
                // mediapipe/framework/formats/image_frame_opencv.h
                // mediapipe/framework/formats/image_frame_opencv.cc

                constexpr ImageFormat const k_format = SRGB;
                constexpr int const k_number_of_channels_for_format = 3;
                constexpr int const k_channel_size_for_format = sizeof(uint8_t);
                constexpr int const k_mat_type_for_format = CV_8U;

                constexpr uint32_t const k_default_alignment_boundary = 16U;

                input_image.type = MpImage::IMAGE_FRAME;
                input_image.image_frame.format = k_format;
                input_image.image_frame.width = input_image_matrix.cols;
                input_image.image_frame.height = input_image_matrix.rows;

                int const type = CV_MAKETYPE(k_mat_type_for_format, k_number_of_channels_for_format);
                int const width_step = (((input_image.image_frame.width * k_number_of_channels_for_format * k_channel_size_for_format) - 1) | (k_default_alignment_boundary - 1)) + 1;
                assert(type == input_image_matrix.type());
                assert(width_step == input_image_matrix.step[0]);
                input_image.image_frame.image_buffer = static_cast<uint8_t *>(input_image_matrix.data);
                assert(0U == (reinterpret_cast<uintptr_t>(input_image.image_frame.image_buffer) & (k_default_alignment_boundary - 1)));
            }

            int64 const tick_count_current = cv::getTickCount();

#if defined(ENABLE_FPS_OUTPUT) && ENABLE_FPS_OUTPUT
            double const fps = tick_frequency / static_cast<double>(tick_count_current - tick_count_previous);
            tick_count_previous = tick_count_current;
            std::cout << "FPS: " << fps << std::endl;
#endif

            size_t const frame_timestamp_ms = static_cast<size_t>((static_cast<double>(tick_count_current) * 1000.0) / tick_frequency);

#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
            FaceLandmarkerResult face_landmarker_result;
            {
                char *error_msg_face_landmarker_detect_for_video = NULL;
                int status_face_landmarker_detect_for_video = face_landmarker_detect_for_video(face_landmarker, &input_image, frame_timestamp_ms, &face_landmarker_result, &error_msg_face_landmarker_detect_for_video);
                assert(NULL == error_msg_face_landmarker_detect_for_video);
                assert(0 == status_face_landmarker_detect_for_video);
            }
#endif

#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
            PoseLandmarkerResult pose_landmarker_result;
            {
                char *error_msg_pose_landmarker_detect_for_video = NULL;
                int status_pose_landmarker_detect_for_video = pose_landmarker_detect_for_video(pose_landmarker, &input_image, frame_timestamp_ms, &pose_landmarker_result, &error_msg_pose_landmarker_detect_for_video);
                assert(NULL == error_msg_pose_landmarker_detect_for_video);
                assert(0 == status_pose_landmarker_detect_for_video);
            }
#endif

#if defined(ENABLE_DEBUG_DISPLAY) && ENABLE_DEBUG_DISPLAY
            cv::Mat debug_display_image_matrix;
            {
                cv::Mat temp_image_matrix = input_image_matrix;

#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
                if (face_landmarker_result.face_landmarks_count >= 1U)
                {
                    for (uint32_t landmarks_index = 0U; landmarks_index < face_landmarker_result.face_landmarks[0].landmarks_count; ++landmarks_index)
                    {
                        NormalizedLandmark const *normalized_landmark = &face_landmarker_result.face_landmarks[0].landmarks[landmarks_index];

                        if (((!normalized_landmark->has_visibility) || (normalized_landmark->visibility > 0.5F)) && ((!normalized_landmark->has_presence) || (normalized_landmark->presence > 0.5F)))
                        {
                            cv::Point point(static_cast<int>(normalized_landmark->x * temp_image_matrix.cols), static_cast<int>(normalized_landmark->y * temp_image_matrix.rows));
                            cv::circle(temp_image_matrix, point, 1, cv::Scalar(0, 255, 0), -1);
                        }
                    }
                }
#endif

#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
                if (pose_landmarker_result.pose_landmarks_count >= 1U)
                {
                    if (pose_landmarker_result.pose_landmarks[0].landmarks_count > 0)
                    {
                        NormalizedLandmark const *normalized_landmark_1 = &pose_landmarker_result.pose_landmarks[0].landmarks[0];

                        if (((!normalized_landmark_1->has_visibility) || (normalized_landmark_1->visibility > 0.5F)) && ((!normalized_landmark_1->has_presence) || (normalized_landmark_1->presence > 0.5F)))
                        {
                            cv::Point point_1(static_cast<int>(normalized_landmark_1->x * temp_image_matrix.cols), static_cast<int>(normalized_landmark_1->y * temp_image_matrix.rows));

                            for (uint32_t landmarks_index = 1U; landmarks_index < pose_landmarker_result.pose_landmarks[0].landmarks_count; ++landmarks_index)
                            {
                                NormalizedLandmark const *normalized_landmark_2 = &pose_landmarker_result.pose_landmarks[0].landmarks[landmarks_index];

                                if (((!normalized_landmark_2->has_visibility) || (normalized_landmark_2->visibility > 0.5F)) && ((!normalized_landmark_2->has_presence) || (normalized_landmark_2->presence > 0.5F)))
                                {

                                    cv::Point point_2(static_cast<int>(normalized_landmark_2->x * temp_image_matrix.cols), static_cast<int>(normalized_landmark_2->y * temp_image_matrix.rows));

                                    cv::line(temp_image_matrix, point_1, point_2, cv::Scalar(255, 0, 0), 1);
                                }
                            }
                        }
                    }
                }
#endif

                if (is_camera_video_capture)
                {
                    // Left <-> Right
                    cv::Mat temp_image_matrix_2;
                    cv::flip(temp_image_matrix, temp_image_matrix_2, 1);

                    cv::cvtColor(temp_image_matrix_2, debug_display_image_matrix, cv::COLOR_RGB2BGR);
                }
                else
                {
                    cv::cvtColor(temp_image_matrix, debug_display_image_matrix, cv::COLOR_RGB2BGR);
                }
            }

            cv::imshow(k_window_name, debug_display_image_matrix);
#endif

#if defined(ENABLE_DEBUG_OUTPUT) && ENABLE_DEBUG_OUTPUT
#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
            if (face_landmarker_result.face_blendshapes_count >= 1U)
            {
                for (uint32_t blend_shape_index = 0U; blend_shape_index < face_landmarker_result.face_blendshapes[0].categories_count; ++blend_shape_index)
                {
                    Category const *category = &face_landmarker_result.face_blendshapes[0].categories[blend_shape_index];

                    std::cout << category->category_name << ": " << category->score << std::endl;
                }
            }
#endif
#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
            if (pose_landmarker_result.pose_world_landmarks_count >= 1U)
            {
                for (uint32_t world_landmark_index = 0U; world_landmark_index < pose_landmarker_result.pose_world_landmarks[0].landmarks_count; ++world_landmark_index)
                {
                    Landmark const *landmark = &pose_landmarker_result.pose_world_landmarks[0].landmarks[world_landmark_index];

                    if (((!landmark->has_visibility) || (landmark->visibility > 0.5F)) && ((!landmark->has_presence) || (landmark->presence > 0.5F)))
                    {
                        std::cout << "x: " << landmark->x << " y: " << landmark->y << " z: " << landmark->z << std::endl;
                    }
                }
            }
#endif
#endif

#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
            face_landmarker_close_result(&face_landmarker_result);
#endif

#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
            pose_landmarker_close_result(&pose_landmarker_result);
#endif
        }
        else
        {
            // reuse the last successful result
        }

#if defined(ENABLE_DEBUG_DISPLAY) && ENABLE_DEBUG_DISPLAY
        int const pressed_key = cv::waitKey(1);
        if (pressed_key >= 0 && pressed_key != 255)
        {
            running = false;
        }
#endif
    }

#if defined(ENABLE_DEBUG_DISPLAY) && ENABLE_DEBUG_DISPLAY
    cv::destroyAllWindows();
#endif

#if defined(ENABLE_FACE_LANDMARKER) && ENABLE_FACE_LANDMARKER
    {
        char *error_msg_face_landmarker_close = NULL;
        face_landmarker_close(face_landmarker, &error_msg_face_landmarker_close);
        assert(NULL == error_msg_face_landmarker_close);
        face_landmarker = NULL;
    }
#endif

#if defined(ENABLE_POSE_LANDMARKER) && ENABLE_POSE_LANDMARKER
    {
        char *error_msg_pose_landmarker_close = NULL;
        pose_landmarker_close(pose_landmarker, &error_msg_pose_landmarker_close);
        assert(NULL == error_msg_pose_landmarker_close);
        pose_landmarker = NULL;
    }
#endif

    video_capture.release();

    return 0;
}