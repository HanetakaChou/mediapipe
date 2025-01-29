
# Bazel  

release v0.10.20  
0f808998a178edde401b3af2c944441a9c0a3b85  

[Installing on Windows](https://ai.google.dev/edge/mediapipe/framework/getting_started/install#installing_on_windows)  

> Enable OpenCL Delegate  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\BUILD  
remove the "android_hardware_buffer" after the "select" within the "_DELEGATE_NO_GL_DEPS" 
>
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\api.h  
remove portable_gl31 header  
change GLuint to cl_GLuint  
change GLenum to cl_GLenum  
change GL_INVALID_INDEX to 0xFFFFFFFFu  
change GL_INVALID_ENUM to 0x0500  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\api.cc  
change GL_INVALID_INDEX to 0xFFFFFFFFu  
change GL_INVALID_ENUM to 0x0500  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\delegate.cc  
remove android_hardware_buffer header  
~~remove log "GpuDelegate invoke thread != prepare thread"~~  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\cl\\api.h  
>> remove EGL header  
>> IsGlAware always return false  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\cl\\opencl_wrapper.cc  
>> change LoadLibraryA to LoadLibraryW  
>  
>> C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\org_tensorflow\\tensorflow\\lite\\delegates\\gpu\\cl\\cl_device.cc  
>> check "CL_DEVICE_HOST_UNIFIED_MEMORY" to make sure the Discrete GPU is used  
>> "CL_DEVICE_LOCAL_MEM_TYPE" not reliable  
>  
>> mediapipe\mediapipe\tasks\cc\core\mediapipe_builtin_op_resolver.h  
the base class of "MediaPipeBuiltinOpResolver" should be "BuiltinOpResolverWithoutDefaultDelegates" instead of "BuiltinOpResolver"  
otherwise, tflite will create XNNPACK delegate when the GPU delegate has already been created  (this is NOT allowed, which causes the pose landmarker NOT working)  
>  
>> mediapipe\mediapipe\calculators\tensor\BUILD  
mediapipe\mediapipe\calculators\tflite\BUILD  
add "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate" for "inference_calculator_cpu", "inference_calculator_xnnpack" and "tflite_inference_calculator" 
>  
>> mediapipe\mediapipe\calculators\tensor\inference_calculator_cpu.cc  
mediapipe\mediapipe\calculators\tensor\inference_calculator_xnnpack.cc  
mediapipe\mediapipe\calculators\tflite\tflite_inference_calculator.cc  
add "GPU V2" support  
note: the "return nullptr (Default tflite inference requested - no need to modify graph)" within "InferenceCalculatorCpuImpl::MaybeCreateDelegate" should also be modified
>  

```cpp
absl::Status CreateDefaultGPUDevice(CLDevice *result)
{
  cl_uint num_platforms;
  cl_int status_get_platform_id_1 = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (CL_SUCCESS != status_get_platform_id_1)
  {
    return absl::UnknownError(absl::StrFormat("clGetPlatformIDs returned %d", status_get_platform_id_1));
  }

  if (!(num_platforms > 0))
  {
    return absl::UnknownError("No supported OpenCL platform.");
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  cl_int status_get_platform_id_2 = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (CL_SUCCESS != status_get_platform_id_2)
  {
    return absl::UnknownError(absl::StrFormat("clGetPlatformIDs returned %d", status_get_platform_id_2));
  }

  cl_platform_id selected_platform_id = 0;
  cl_device_id selected_device_id = 0;
  bool device_found = false;

  for (cl_uint platform_index = 0; platform_index < num_platforms; ++platform_index)
  {
    cl_platform_id platform_id = platforms[platform_index];

    cl_uint num_devices;
    cl_int status_get_device_id_1 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (CL_SUCCESS == status_get_device_id_1)
    {
      if (num_devices > 0)
      {
        std::vector<cl_device_id> devices(num_devices);
        cl_int status_get_device_id_2 = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        if (CL_SUCCESS == status_get_device_id_2)
        {
          for (cl_uint device_index = 0; device_index < num_devices; ++device_index)
          {
            cl_device_id device_id = devices[device_index];

            cl_bool host_unified_memory;
            cl_int status_get_device_host_unified_memory = clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(host_unified_memory), &host_unified_memory, nullptr);
            if (CL_SUCCESS == status_get_device_host_unified_memory)
            {
              if (CL_FALSE == host_unified_memory)
              {
                selected_platform_id = platform_id;
                selected_device_id = device_id;
                device_found = true;
                break;
              }
              else if (!device_found)
              {
                selected_platform_id = platform_id;
                selected_device_id = device_id;
                device_found = true;
              }
            }
            else
            {
              ABSL_RAW_LOG(WARNING, "clGetDeviceInfo returned %d", (int)status_get_device_host_unified_memory);
            }
          }
        }
        else
        {
          ABSL_RAW_LOG(ERROR, "clGetDeviceIDs returned %d", (int)status_get_device_id_2);
        }
      }
      else
      {
        ABSL_RAW_LOG(WARNING, "No GPU on %d platform.", (int)platform_id);
      }
    }
    else
    {
      ABSL_RAW_LOG(ERROR, "clGetDeviceIDs returned %d", (int)status_get_device_id_1);
    }
  }

  if (!device_found)
  {
    return absl::UnknownError("No GPU on all platforms.");
  }

  *result = CLDevice(selected_device_id, selected_platform_id);

  LoadOpenCLFunctionExtensions(selected_platform_id);

  return absl::OkStatus();
}
```

```cpp
extern "C" unsigned long __stdcall GetEnvironmentVariableW(const wchar_t* lpName, wchar_t* lpBuffer, unsigned long nSize);

    wchar_t environment_variable_buffer[2];
    unsigned long size = GetEnvironmentVariableW(L"TFLITE_FORCE_GPU", environment_variable_buffer, 2U);
    if ((1U == size || 2U == size) && L'1' == environment_variable_buffer[0])
    {
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        gpu_opts.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

        return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(&gpu_opts), &TfLiteGpuDelegateV2Delete);
    }
    else
    {
        TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
        // if (use_xnnpack)
        // {
        //   xnnpack_opts.num_threads = GetXnnpackNumThreads(opts_has_delegate, opts_delegate);
        // }
        return TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts), &TfLiteXNNPackDelegateDelete);
    }
```

```cpp
extern "C" unsigned long __stdcall GetEnvironmentVariableW(const wchar_t* lpName, wchar_t* lpBuffer, unsigned long nSize);

    {
        wchar_t environment_variable_buffer[2];
        unsigned long size = GetEnvironmentVariableW(L"TFLITE_FORCE_GPU", environment_variable_buffer, 2U);
        if ((1U == size || 2U == size) && L'1' == environment_variable_buffer[0])
        {
            TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
            gpu_opts.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
            delegate_ = TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(&gpu_opts), &TfLiteGpuDelegateV2Delete);
        }
        else
        {
            TfLiteXNNPackDelegateOptions xnnpack_opts = TfLiteXNNPackDelegateOptionsDefault();
            xnnpack_opts.num_threads = GetXnnpackNumThreads(calculator_opts);
            delegate_ = TfLiteDelegatePtr(TfLiteXNNPackDelegateCreate(&xnnpack_opts), &TfLiteXNNPackDelegateDelete);
        }
        RET_CHECK_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);
        return absl::OkStatus();
    }
```

```cmd
REM use Bazel 6.5.0
REM use LLVM 18

SET BAZEL_SH=C:\Program Files\Git\bin\bash.exe
SET BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community
SET BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC
SET BAZEL_VC_FULL_VERSION=14.29.30133
SET BAZEL_WINSDK_FULL_VERSION=10.0.20348.0
SET BAZEL_LLVM=C:\Program Files\LLVM

REM which python3
REM DEL /F/Q "C:\Users\<Your name>\AppData\Local\Microsoft\WindowsApps\python.exe"
REM DEL /F/Q "C:\Users\<Your name>\AppData\Local\Microsoft\WindowsApps\python3.exe"

REM Python.exe --version
SET TF_PYTHON_VERSION=Python 3.9.13

REM C:\\users\\\<Your username\>\\\_bazel\_\\<Your username\>\\\<ID\>\\external\\com_github_glog_glog_windows\\bazel\\glog.bzl  
REM remove the "-DGLOG_EXPORT=__declspec(dllexport)" from windows_only_copts

cd <Your mediapipe repository root directory>
REM bazel.exe clean --expunge
REM -c dbg
REM --copt=/std:c++20
REM --define xnn_enable_avxvnniint8=false
REM mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
bazel.exe build -c opt --compiler=clang-cl --define MEDIAPIPE_DISABLE_GPU=1 --copt=-DMP_EXPORT="" --copt=-D_WIN32_WINNT=_WIN32_WINNT_WIN7 --copt=-DTFL_STATIC_LIBRARY_BUILD --copt=-DTFLITE_GPU_BINARY_RELEASE --copt=-DCL_DELEGATE_NO_GL mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu
```

[Model](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)  

```cmd
curl https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite -o mediapipe/modules/face_detection/face_detection_short_range.tflite  
curl https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite -o mediapipe/modules/face_landmark/face_landmark.tflite  
curl https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite -o mediapipe/modules/iris_landmark/iris_landmark.tflite  
curl https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite -o mediapipe/modules/pose_detection/pose_detection.tflite
curl https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite -o mediapipe/modules/pose_landmark/pose_landmark_full.tflite
```  

```cmd
REM https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md#desktop  
bazel-bin\mediapipe\examples\desktop\iris_tracking\iris_tracking_cpu.exe --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt

REM https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#desktop
bazel-bin\mediapipe\examples\desktop\pose_tracking\pose_tracking_cpu.exe --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
```

> Build MediaPipe C Shared Library  
>  
>> mediapipe\\BUILD  
define "mediapipe_c" shared library to reference "face_landmarker_lib" and "pose_landmarker_lib"  
>  

```bazel
cc_binary(
    name = "mediapipe_c",
    srcs  = select({
        "@bazel_tools//src/conditions:windows": [
            "dllmain.cpp",
            ],
        "//conditions:default": [],
    }),
    deps = [
        "//mediapipe/tasks/c/vision/face_landmarker:face_landmarker_lib",
        "//mediapipe/tasks/c/vision/pose_landmarker:pose_landmarker_lib",
        ],
    linkshared = 1,
    win_def_file = select({
        "@bazel_tools//src/conditions:windows": "mediapipe_c.def",
        "//conditions:default": None,
    }),
    visibility = ["//visibility:public"],
)
```

```def
EXPORTS
	face_landmarker_create
	face_landmarker_close
	face_landmarker_close_result
	face_landmarker_create
	face_landmarker_detect_async
	face_landmarker_detect_for_video
	face_landmarker_detect_image
	pose_landmarker_close
	pose_landmarker_close_result
	pose_landmarker_create
	pose_landmarker_detect_async
	pose_landmarker_detect_for_video
	pose_landmarker_detect_image
```

```cpp
#define WIN32_LEAN_AND_MEAN 1
#include <sdkddkver.h>
#include <Windows.h>
#include <assert.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        DisableThreadLibraryCalls(hModule);
    }
    break;
    case DLL_PROCESS_DETACH:
    {
    }
    break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    default:
    {
        // we have disabled the "Thread Library Calls"
        // assert(false);
    }
    }
    return TRUE;
}
```

```cmd
bazel.exe build -c opt --compiler=clang-cl --define MEDIAPIPE_DISABLE_GPU=1 --copt=-DMP_EXPORT=__declspec(dllexport) --copt=-D_WIN32_WINNT=_WIN32_WINNT_WIN7 --copt=-DTFL_STATIC_LIBRARY_BUILD --copt=-DTFLITE_GPU_BINARY_RELEASE --copt=-DCL_DELEGATE_NO_GL mediapipe:mediapipe_c
```