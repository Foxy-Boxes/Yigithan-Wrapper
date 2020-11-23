#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_timer.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <thread>
#include <assert.h>
#include <set>
#include <vector>
#include "lfqueue.h"


//using namespace cv;

enum WaitingState:uint32_t {
    Min_Waiting = 0x01,
    Min_Ready = 0x02,
    Min_Found = 0x04,
    Min_Dont = 0x08,
    Min_Acknowladged = 0x10,
};

LFQueue<uint32_t> Points;
LFQueue<uint32_t> Hull;
LFQueue<uint64_t> Segments;
std::set<uint64_t> NoDup;
LFQueue<uint64_t> NoDupQueue;
typedef struct Angle{
    std::atomic_uint32_t minCheck;
    float angle;
}Angle;

void SobelFilter(void* texture, uint16_t height, uint16_t width);
void SwingArm(uint16_t startPos, uint16_t height, uint16_t width);
void extractSegments(uint16_t width);
bool insideHull(uint16_t pointX,uint16_t pointY,uint16_t width, uint16_t height);
int main()
{

    // retutns zero on success else non-zero
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        printf("error initializing SDL: %s\n", SDL_GetError());
    }
    SDL_Window* win = SDL_CreateWindow("GAME", // creates a window
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       800, 600, 0);

    // triggers the program that controls
    // your graphics hardware and sets flags
    Uint32 render_flags = SDL_RENDERER_ACCELERATED;

    // creates a renderer to render our images
    SDL_Renderer* rend = SDL_CreateRenderer(win, -1, render_flags);

    // creates a surface to load an image into the main memory
    SDL_Surface* surface;

    // please provide a path for your image
    surface = IMG_Load("../boi.png");
    SDL_Surface* hawai = IMG_Load("../venus-hawaii.png");

    // loads image to our graphics hardware memory.
    SDL_Texture* tex = SDL_CreateTextureFromSurface(rend, surface);
    SDL_Surface * image = SDL_ConvertSurfaceFormat(surface,SDL_PIXELFORMAT_ARGB8888,0);
    SDL_Texture * gray = SDL_CreateTexture(rend,
                                              SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC,
                                              image->w, image->h);
    SDL_Texture * hawaiTex = SDL_CreateTextureFromSurface(rend,hawai);
    SDL_SetWindowResizable(win,SDL_TRUE);

    Uint32* pixels = (Uint32 *)image->pixels;
    Uint32 copyBuffer[image->w*image->h];
    uint8_t grayPixels[image->w*image->h];
    for (int y = 0; y < image->h; y++)
    {
        for (int x = 0; x < image->w; x++)
        {
            Uint32 pixel = pixels[y * image->w + x];
            // TODO convert pixel to grayscale here
            Uint8 r = pixel >> 16 & 0xFF;
            Uint8 g = pixel >> 8 & 0xFF;
            Uint8 b = pixel & 0xFF;
            grayPixels[y*image->w+x] = 0.212671f * r + 0.715160f * g + 0.072169f * b;
        }
    }

    SobelFilter(grayPixels,image->h,image->w);
    printf("%d, %d\n", Points.get_elem(0) % image->w,Points.get_elem(0) / image->w);
    printf("%d\n", grayPixels[Points.get_elem(0)]);
    printf("%d\n", grayPixels[0]);

    uint16_t iter = Points.get_count();
    uint16_t minx[2] = {SDL_MAX_UINT16,0};
    for (uint16_t i = 0; i < iter; ++i) {
        auto x = Points.get_elem(i)%image->w;
        minx[0] = minx[0] ^ ((x ^ minx[0]) & -(x < minx[0]));
        minx[1] += (minx[0] == x) * (i - minx[1]);
//        pixels[Points.get_elem(i)] = 0xFF00FF00;
    }
    SwingArm(minx[1],image->h,image->w);
    extractSegments(image->w);
    printf("Hull : %d, Seg: %d", Hull.get_count(), NoDup.size());
    for(uint16_t i = 0; i < NoDupQueue.get_count();i++){
        uint64_t segment = NoDupQueue.get_elem(i);
        uint16_t segone_x = segment >> 48;
        uint16_t segone_y = (segment >> 32) & 0xFFFF;
        uint16_t segtwo_x = (segment >> 16) & 0xFFFF;
        uint16_t segtwo_y = (segment) & 0xFFFF;
        auto fromx = segone_x ^ ((segtwo_x ^ segone_x) & -(segtwo_x < segone_x)),tox = segone_x+segtwo_x-fromx;
        auto fromy = segone_y ^ ((segtwo_y ^ segone_y) & -(segtwo_y < segone_y)),toy = segone_y+segtwo_y-fromy;
        printf("%d %d\n",segone_x,segtwo_x);
        printf("%d %d\n",segone_y,segtwo_y);
        for(uint16_t j = fromx; j < tox; j++){
            for (uint16_t k = fromy; k < toy; ++k) {
//                pixels[j+k*image->w] = 0xFF00FFFF;
            }
        }
    }
//    assert(Hull.get_count() == Segments.get_count());
    for(uint16_t i = 0; i < Hull.get_count(); ++i){
        pixels[Hull.get_elem(i)] = 0xFF0000FF;
    }
    for (uint16_t i = 0; i < image->w*image->h; ++i) {
        uint32_t point = i;
        uint16_t pointX = point%image->w,pointY = point/image->w;
        if(insideHull(pointX,pointY,image->w,image->h)){
            copyBuffer[point] = pixels[point];
        } else{
            copyBuffer[point] = 0;
        }
    }
    for(uint16_t i = 0; i < image->w; ++i){
        for(uint16_t j = 0; j < image->h; j++){
            if(copyBuffer[i+image->w*j] != 0) {
                ((Uint32 *) hawai->pixels)[(200 + i) + (200 + j) * hawai->w] = copyBuffer[i + image->w * j];
            }
        }
    }
    // clears main-memory
    SDL_FreeSurface(surface);


    SDL_Rect dest;

    // connects our texture with dest to control position
    SDL_QueryTexture(tex, NULL, NULL, &dest.w, &dest.h);

    // adjust height and width of our image box.
//    dest.w <<= 2;
//    dest.h <<= 2;

    // sets initial x-position of object
    dest.x = (1000 - dest.w) / 2;

    // sets initial y-position of object
    dest.y = (1000 - dest.h) / 2;

    uint8_t counter = SDL_MAX_UINT8;
//    SDL_UpdateTexture(gray, NULL, copyBuffer,
//                      image->w * sizeof(Uint32));
    SDL_UpdateTexture(hawaiTex,NULL,hawai->pixels
    ,hawai->w*sizeof(Uint32));
    while (counter){
        counter--;
        SDL_RenderClear(rend);
        SDL_RenderCopy(rend, hawaiTex,NULL,NULL);
//        SDL_RenderCopy(rend, gray, NULL, &dest);

        // triggers the double buffers
        // for multiple rendering
        SDL_RenderPresent(rend);

        // calculates to 60 fps
        SDL_Delay(1000 / 60);
    }
//    VideoCapture cap(0); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;
//
//    Mat edges;
//    namedWindow("edges",1);
//    for(;;)
//    {
//        Mat frame;
//        cap >> frame; // get a new frame from camera
//        cvtColor(frame, edges, CV_BGR2GRAY);
//        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
//        Canny(edges, edges, 0, 30, 3);
//        imshow("edges", edges);
//        if(waitKey(30) >= 0) break;
//    }
    SDL_FreeSurface(image);
    SDL_DestroyTexture(tex);
    SDL_DestroyTexture(gray);

    // destroy renderer
    SDL_DestroyRenderer(rend);

    // destroy window
    SDL_DestroyWindow(win);

    SDL_Quit();

    return 0;
}


void SobelFilter(void* texture, uint16_t height, uint16_t width){
    static int8_t SxB[16] = {-1,0,1,-2,0,2,-1,0,1};
    static int8_t SyB[16] = {-1,-2,-1,0,0,0,1,2,1};
    static const __m128i Sx = _mm_load_si128(reinterpret_cast<__m128i *>(SxB));
    static const __m128i Sy = _mm_load_si128(reinterpret_cast<__m128i *>(SyB));
    static const __m128i Zeros = _mm_setzero_si128();
#pragma omp parallel for
    for(uint32_t i = width + 1; i < height*width - width - 1 ; i++){
        uint8_t buf[16]{};
        buf[0] = ((uint8_t*)texture)[i - width - 1];
        buf[1] = ((uint8_t*)texture)[i - width];
        buf[2] = ((uint8_t*)texture)[i - width + 1];
        buf[3] = ((uint8_t*)texture)[i - 1];
        buf[4] = ((uint8_t*)texture)[i];
        buf[5] = ((uint8_t*)texture)[i + 1];
        buf[6] = ((uint8_t*)texture)[i + width - 1];
        buf[7] = ((uint8_t*)texture)[i + width];
        buf[8] = ((uint8_t*)texture)[i + width + 1];
        const __m128i matrixbuffer = _mm_loadu_si128(reinterpret_cast<__m128i *>(buf));
        const __m128i Gx_16 = _mm_maddubs_epi16(matrixbuffer,Sx);
        const __m128i Gy_16 = _mm_maddubs_epi16(matrixbuffer,Sy);

        const __m128i Gx = _mm_hadd_epi16(_mm_hadd_epi16(_mm_hadd_epi16(Gx_16,Zeros),Zeros),Zeros);
        const __m128i Gy = _mm_hadd_epi16(_mm_hadd_epi16(_mm_hadd_epi16(Gy_16,Zeros),Zeros),Zeros);
        uint64_t GxFinal = _mm_extract_epi16(Gx , 0);
        uint64_t GyFinal = _mm_extract_epi16(Gy , 0);
        if (GxFinal*GxFinal+GyFinal*GyFinal > 1000000000){
            if(GyFinal*GyFinal > GxFinal*GxFinal+1000000){
                continue;
            }
            //have a random
            Points.push_back(i);
        }
    }

}

void angleSorter(Angle* angles, uint16_t count,uint16_t* selectedIndex, std::atomic_uint16_t* done);

int crossProduct(uint32_t a, uint32_t b, uint32_t c,uint16_t width) {
    int y1 = a/width - b/width;
    int y2 = a/width - c/width;
    int x1 = a%width - b%width;
    int x2 = a%width - c%width;
    return y2*x1 - y1*x2;          //if result < 0, c in the left, > 0, c in the right, = 0, a,b,c are collinear
}

int distance(uint32_t a, uint32_t b, uint32_t c,uint16_t width) {
    int y1 = a/width - b/width;
    int y2 = a/width - c/width;
    int x1 = a%width - b%width;
    int x2 = a%width - c%width;

    int item1 = (y1*y1 + x1*x1);
    int item2 = (y2*y2 + x2*x2);

    if(item1 == item2)
        return 0;             //when b and c are in same distance from a
    else if(item1 < item2)
        return -1;          //when b is closer to a
    return 1;              //when c is closer to a
}

void SwingArm(uint16_t startPos, uint16_t height, uint16_t width){
    //Do it again DUNKOF
    const uint16_t Count = Points.get_count();
    uint32_t current = Points.get_elem(startPos);
    std::set<uint32_t> result;                 //set is used to avoid entry of duplicate uint32_ts
    result.insert(current);
    std::vector<uint32_t> *collinearPoints = new std::vector<uint32_t>;
    uint16_t comp = 0;
    while(true) {
        uint32_t nextTarget = Points.get_elem(0);

        for(int i = 1; i<Count; i++) {
            if(Points.get_elem(i) == current)       //when selected uint32_t is current uint32_t, ignore rest part
                continue;
            int val = crossProduct(current, nextTarget,Points.get_elem(i),width);
            int64_t diff_y=(current/width - Points.get_elem(i)/width);
            int64_t diff_x=(current%width - Points.get_elem(i)%width);
            if(diff_x*diff_x+diff_y*diff_y > 200000000){
                continue;
            }
            if(val > 0) {            //when ith uint32_t is on the left side

                nextTarget = Points.get_elem(i);

                delete collinearPoints;
                collinearPoints = new std::vector<uint32_t>;      //reset collinear uint32_ts

            }else if(val == 0) {          //if three uint32_ts are collinear
                if(distance(current, nextTarget, Points.get_elem(i),width) < 0) { //add closer one to collinear list
                    collinearPoints->push_back(nextTarget);
                    nextTarget = Points.get_elem(i);
                }else{
                    collinearPoints->push_back( Points.get_elem(i)); //when ith uint32_t is closer or same as nextTarget
                }
            }
        }
        std::vector<uint32_t>::iterator it;

        for(it = collinearPoints->begin(); it != collinearPoints->end(); it++) {

//            result.insert(*it);     //add alluint32_ts in collinear uint32_ts to result set
        }

        if(nextTarget == Points.get_elem(startPos))        //when next uint32_t is start it means, the area covered
            break;
        comp++;
        if(comp > Count){
            break;
        }
        uint32_t pos = current;
        uint16_t posX = pos % width, posY = pos / width;
        uint32_t anchor = nextTarget;
        uint16_t anchorX = anchor%width, anchorY = anchor/width;
        uint64_t segment;
        segment = ((uint64_t)posX << 48) | ((uint64_t)posY << 32) | ((uint64_t)anchorX << 16) | (anchorY);
        NoDup.insert(segment);
        result.insert(nextTarget);
        current = nextTarget;
    }
    for(auto res:result){
        Hull.push_back(res);
    }
}

void angleSorter(Angle* angles, uint16_t count,uint16_t* selectedIndex,std::atomic_uint16_t* done){
    LFQueue<uint16_t> indices;
    float minimum = MAXFLOAT;
    uint16_t middle_count;
    assert(count > 0);
#pragma omp parallel for
    for(uint16_t i = 0; i < (count >> 1); i++){
        uint32_t WhatToDo = Min_Dont|Min_Ready;
        uint32_t Acknowladge;
        while (!((Acknowladge = angles[i].minCheck.load(std::memory_order_release)) & WhatToDo ));
        if (Acknowladge & Min_Dont){
            continue;
        } else{
            indices.push_back(i);
        }
    }
    minimum = angles[indices.get_elem(0)].angle;
    for(uint16_t j = 0; j < (middle_count = indices.get_count()); j++){
        uint16_t currIndex = indices.get_elem(j);
        if(angles[currIndex].angle < minimum){
            minimum = angles[currIndex].angle;
            *selectedIndex = currIndex;
        }
    }
#pragma omp parallel for
    for(uint16_t i = (count >> 1); i < count; i++){
        uint32_t WhatToDo = Min_Dont|Min_Ready;
        uint32_t Acknowladge;
        while (!((Acknowladge = angles[i].minCheck.load(std::memory_order_release)) & WhatToDo ));
        if (Acknowladge & Min_Dont){
            continue;
        } else{
            indices.push_back(i);
        }
    }
    for(uint16_t j = middle_count; j < indices.get_count(); j++){
        uint16_t currIndex = indices.get_elem(j);
        if(angles[currIndex].angle < minimum){
            minimum = angles[currIndex].angle;
            *selectedIndex = currIndex;
        }
    }
    assert(minimum != MAXFLOAT);
    printf("%f, %d\n",minimum, *selectedIndex);
    done->store(Min_Found,std::memory_order_acquire);
}
void extractSegments(uint16_t width){
    for(int i = 0; i < Segments.get_count(); i++){
        NoDup.insert(Segments.get_elem(i));
    }

    for(auto no :NoDup){
        NoDupQueue.push_back(no);
    }
}
inline int32_t cross(int32_t x1,int32_t y1, int32_t x2, int32_t y2){
    return x1*y2-x2*y1;
}

inline int32_t dot(int32_t x1,int32_t y1, int32_t x2, int32_t y2){
    return x1*x2+y2*y1;
}
bool insideHull(uint16_t pointX,uint16_t pointY,uint16_t width, uint16_t height){
    static const uint32_t relPoint = (height >> 1) * width + (width >> 1);
    static const uint16_t relPointX = relPoint%width;
    static const uint16_t relPointY = relPoint/width;
    std::atomic<uint32_t> count{0};

    for(uint16_t i = 0; i < NoDupQueue.get_count(); i++){
        uint64_t segment = NoDupQueue.get_elem(i);
        uint16_t segone_x = segment >> 48;
        uint16_t segone_y = (segment >> 32) & 0xFFFF;
        uint16_t segtwo_x = (segment >> 16) & 0xFFFF;
        uint16_t segtwo_y = (segment) & 0xFFFF;
        int32_t rxs = cross(relPointX-pointX,relPointY-pointY,segtwo_x-segone_x,segtwo_y-segone_y);
        int32_t q_pxr = cross(segone_x-pointX,segone_y-pointY,relPointX-pointX,relPointY-pointY);
        if(rxs == 0){
            if(q_pxr != 0){
                continue;
            }
            else{

                float t0,t1;
                uint32_t sor = dot(segtwo_x-segone_x,segtwo_y-segone_y,relPointX-pointX,relPointY-pointY);
                uint32_t ror = dot(relPointX-pointX,relPointY-pointY,relPointX-pointX,relPointY-pointY);
                t0 = (float)dot(segone_x-pointX,segone_y-pointY,relPointX-pointX,relPointY-pointY)/ror;
                t1 = t0 + (float)sor/ror;
                if(sor > 0){
                    //continue;
                    if((t0 < 0 && t1 >= 0)||(t1 > 1 && t0 <= 1)||(t1 >= 0 && t0 <= 1)){
                        count.fetch_add(1,std::memory_order_acquire);
                    }
                }else{
                    //continue;
                    if((t1 < 0 && t0 >= 0)||(t0 > 1 && t1 <= 1)||(t0 >= 0 && t1 <= 1)){
                        count.fetch_add(1,std::memory_order_acquire);
                    }
                }
            }
        } else{
            float u,t;
            t = (float)cross(segone_x-pointX,segone_y-pointY,segtwo_x-segone_x,segtwo_y-segone_y)/rxs;
            u = (float)q_pxr/rxs;
            if(u > 1 || u < 0 || t > 1|| t < 0){
                continue;
            } else{
                //continue;
                count.fetch_add(1,std::memory_order_acquire);
            }
        }

    }
    return (count.load(std::memory_order_release) & 1) == 0;
}