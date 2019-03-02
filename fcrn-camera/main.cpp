#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "NvInfer.h"

#include <signal.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>

#include "cudaResize.h"
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"
#include "cudaRGB.h"
#include "cudaOverlay.h"
#include "cudaUtility.h"
#include "../imageNet.h"

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

bool signalRecieved = false;

void signalHandler( int sigNo )
{
  if( sigNo == SIGINT )
  {
    signalRecieved = true;
  }
}

class Logger : public nvinfer1::ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
    std::cout << msg << std::endl;
  }
} gLogger;

cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
                 float* output, size_t outputWidth, size_t outputHeight,
              cudaStream_t stream );

void getColor(float x, float&r, float&g, float&b) 
  {
    x = x * 6;
    r = 0.0f; g = 0.0f; b = 0.0f;
    if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
    else if (4 <= x && x <= 5) r = x - 4;
    else if (1 <= x && x <= 2) r = 1.0f - (x - 1);
    if (1 <= x && x <= 3) g = 1.0f;
    else if (0 <= x && x <= 1) g = x - 0;
    else if (3 <= x && x <= 4) g = 1.0f - (x - 3);
    if (3 <= x && x <= 5) b = 1.0f;
    else if (2 <= x && x <= 3) b = x - 2;
    else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
  }


int main( int argc, char** argv )
{
  const int   DEFAULT_CAMERA = 0;
  const int   MAX_BATCH_SIZE = 1;
  const char* MODEL_NAME     = "/home/kirill/Code/CNN_Depth_Reconstruction_VSLAM/trt_resnet_engine_320x240.trt";
  const char* INPUT_BLOB     = "tf/Placeholder";
  const char* OUTPUT_BLOB    = "tf/Reshape";
  const int IMG_WIDTH = 320;
  const int IMG_HEIGHT = 240;

  if( signal(SIGINT, signalHandler) == SIG_ERR )
  { 
    std::cout << "\ncan't catch SIGINT" <<std::endl;
  }
 
  gstCamera* camera = gstCamera::Create(IMG_WIDTH, IMG_HEIGHT, DEFAULT_CAMERA);
	
  if( !camera )
  {
    std::cout << "\nsegnet-camera:  failed to initialize video device" << std::endl;
    return 0;
  }
	
  std::cout << "\nsegnet-camera:  successfully initialized video device" << std::endl;
  std::cout << "    width:  " << camera->GetWidth() << std::endl;
  std::cout << "   height:  " << camera->GetHeight() << std::endl;
  std::cout << "    depth:  "<< camera->GetPixelDepth() << std::endl;

  float* outCPU  = nullptr;
  float* outCUDA = nullptr;

  if( !cudaAllocMapped( (void**)&outCPU, (void**)&outCUDA, camera->GetWidth() * camera->GetHeight() * sizeof(float) * 4 ) )
  {
    std::cout << "Failed to allocate CUDA memory" << std::endl;
    return 0;
  }
  
//Read model from file, deserialize it and create runtime, engine and exec context
  std::ifstream model( MODEL_NAME, std::ios::binary );

  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(model), {});
  std::size_t modelSize = buffer.size() * sizeof( unsigned char );

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
  nvinfer1::IExecutionContext* context = engine->createExecutionContext(); 

  if( !context )
  {
    std::cout << "Failed to create execution context" << std::endl;
    return 0;
  }


//Set input info + alloc memory
  const int inputIndex = engine->getBindingIndex( INPUT_BLOB );
  nvinfer1::Dims inputDims = engine->getBindingDimensions( inputIndex );
  std::cout << "-  Input binding index: " << inputIndex << std::endl;
  std::cout << "- Number of dimensions: " << inputDims.nbDims << std::endl;
  std::cout << "- c: " << DIMS_C(inputDims) << " h: " << DIMS_H(inputDims) << " w: " << DIMS_W(inputDims) << std::endl;

  std::size_t inputSize = MAX_BATCH_SIZE * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);

  void* inputCPU  = nullptr;
  void* inputCUDA = nullptr;

  if( !cudaAllocMapped( (void**)&inputCPU, (void**)&inputCUDA, inputSize) )
  {
    std::cout << "Failed to alloc CUDA memory for input" << std::endl;
    return 0;
  }

//Set output info + alloc memory 
  void* outputCPU  = nullptr;
  void* outputCUDA = nullptr;


  const int outputIndex = engine->getBindingIndex( OUTPUT_BLOB );
  nvinfer1::Dims outputDims = engine->getBindingDimensions( outputIndex );
  std::size_t outputSize = MAX_BATCH_SIZE * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);

  std::cout << "- Output binding index: " << outputIndex << std::endl;
  std::cout << "- Number of dimensions: " << outputDims.nbDims << std::endl;
  std::cout << "- c: " << DIMS_C(outputDims) << " h: " << DIMS_H(outputDims) << " w: " << DIMS_W(outputDims) << std::endl;

  if( !cudaAllocMapped( (void**)&outputCPU, (void**)&outputCUDA, outputSize) )
  {
    std::cout << "Failed to alloc CUDA memory for output" << std::endl;
    return 0;
  }

//Create CUDA stream for GPU inner data sync
  cudaStream_t stream = nullptr;
  CUDA_FAILED( cudaStreamCreateWithFlags(&stream, cudaStreamDefault ) );

  void* imgRGBCPU = NULL;
  void* imgRGB = NULL;
  if( !cudaAllocMapped((void**)&imgRGBCPU, (void**)&imgRGB, inputSize) )
  {
    printf("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n", inputSize);
    return false;
  }


  if( !camera->Open() )
  {
    std::cout << "Failed to open camera" << std::endl;
  }

  std::cout << "Camera is open for streaming" << std::endl;

  glDisplay* display = glDisplay::Create();
  glTexture* texture = NULL;
  glTexture* texture2 = NULL;
  
  if( !display ) {
    printf("\nsegnet-camera:  failed to create openGL display\n");
  }
  else
  {
    texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGB32F_ARB);

    if( !texture )
    {
      printf("segnet-camera:  failed to create first openGL texture\n");
    }

    texture2 = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGB32F_ARB);

    if( !texture2 )
    {
      printf("segnet-camera:  failed to create second openGL texture\n");
    }

  }

  void* coloredDepth = NULL;
  void* coloredDepthCPU = NULL;
  if( !cudaAllocMapped((void**)&coloredDepthCPU, (void**)&coloredDepth, inputSize) )
  {
    printf("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n", inputSize);
    return false;
  }


// Main loop
  while( !signalRecieved )
  {
    auto start = std::chrono::high_resolution_clock::now();

    void* imgCPU = NULL;
    void* imgCUDA = NULL;

    if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
    {
      std::cout << "Failed to capture frame" << std::endl;
    }
    auto moment_after_capture = std::chrono::high_resolution_clock::now();

    void* imgRGBA = NULL;

    if( !camera->ConvertRGBA(imgCUDA, &imgRGBA, true) )
    {
      std::cout << "Failed to convert from NV12 to RGBA" << std::endl;
    }

    const float2& inputRange = {1.0, 1.0};
    if( CUDA_FAILED(cudaPreImageNet((float4*)imgRGBA, IMG_WIDTH, IMG_HEIGHT, (float*)imgRGB, IMG_WIDTH, IMG_HEIGHT, stream)) )
    {
      printf("cudaPreImageNet failed\n");
    }
    auto moment_after_preprocess = std::chrono::high_resolution_clock::now();

    void* bindings[] = {imgRGB, outputCUDA};
    context->execute( MAX_BATCH_SIZE, bindings );
 
    auto finish = std::chrono::high_resolution_clock::now();

    //measure execution time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time to process single frame: " << elapsed.count() << std::endl;
    elapsed = moment_after_capture - start;
    std::cout << "Time to capture: " << elapsed.count() << std::endl;
    elapsed = moment_after_preprocess - moment_after_capture;
    std::cout << "Time to preprocess: " << elapsed.count() << std::endl;
    elapsed = finish - moment_after_preprocess;
    std::cout << "Time to run engine: " << elapsed.count() << std::endl;

    // update display
    if( display != NULL )
    {
      char str[256];
      display->SetTitle(str); 
  
      // next frame in the window
      display->UserEvents();
      display->BeginRender();

      if( texture != NULL )
      {

        float* depth = (float*)outputCUDA;
        auto before_fors = std::chrono::high_resolution_clock::now();

        float max_depth = 0;
        float min_depth = 10;
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
        {
          if (depth[i] > max_depth)
            max_depth = depth[i];
          if (depth[i] < min_depth)
            min_depth = depth[i];
        }

        //test for loop speed
        float3* poop = new float3[IMG_WIDTH * IMG_HEIGHT];
        auto before_poop = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
        {
          poop[i].x = 1;
          poop[i].y = 1;
          poop[i].z = 1;
        }
        auto after_poop = std::chrono::high_resolution_clock::now();
        elapsed = after_poop - before_poop;
        std::cout << "time to poop: " << elapsed.count() << std::endl;
        delete poop;

        float3* coloredDepthFloat = (float3*)coloredDepth;

        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
        {
          depth[i] = depth[i] / 10;
          float r, g, b;
          getColor(depth[i], r, g, b);
          coloredDepthFloat[i].x = r;
          coloredDepthFloat[i].y = g;
          coloredDepthFloat[i].z = b;
        }
        auto after_fors = std::chrono::high_resolution_clock::now();
        std::cout << "min depth: " << min_depth << std::endl;
        std::cout << "max depth: " << max_depth << std::endl;
        elapsed = after_fors - before_fors;
        std::cout << "Time to process for loops: " << elapsed.count() << std::endl;

        // map from CUDA to openGL using GL interop
        void* tex_map = texture->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, coloredDepth, texture->GetSize(), cudaMemcpyDeviceToDevice);
          texture->Unmap();
        }
        else
        {
          std::cout << "tex_map = NULL" << std::endl;
        }

        // draw the texture
        texture->Render(10, 10);
      }
      float3* imgRGBFloat3 = (float3*)imgRGB;
      float4* imgRGBAFloat4 = (float4*)imgRGBA;

      if( texture2 != NULL )
      {
        // rescale image pixel intensities for display
        CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
                   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
                   camera->GetWidth(), camera->GetHeight()));
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
        {
          imgRGBFloat3[i].x = imgRGBAFloat4[i].x;
          imgRGBFloat3[i].y = imgRGBAFloat4[i].y;
          imgRGBFloat3[i].z = imgRGBAFloat4[i].z;
        }

        // map from CUDA to openGL using GL interop
        void* tex_map = texture2->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, imgRGB, texture2->GetSize(), cudaMemcpyDeviceToDevice);
          texture2->Unmap();
        }
        else
        {
          std::cout << "tex_map2 = NULL" << std::endl;
        }

        // draw the texture
        texture2->Render(10 + IMG_WIDTH, 10);
      }

      display->EndRender();
    }
    //measure rendering time
    auto after_rendering = std::chrono::high_resolution_clock::now();
    elapsed = after_rendering - finish;
    std::cout << "Time to postprocess and render: " << elapsed.count() << std::endl << std::endl;
  }

  if( camera != nullptr )
  {
    camera->Close();
    delete camera;
  }

  if( display != NULL )
  {
    delete display;
    display = NULL;
  }

  //delete imgRGBAResized;
  //delete imgRGB;
  return 0;
}
