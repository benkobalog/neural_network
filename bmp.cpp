//  Bitmap.cpp
//

#include <iostream>
#include <stdio.h>

#include "bmp.h"

Bitmap::Bitmap(const char* filename) {
    FILE* file;
    file = fopen(filename, "rb");

    std::cout << sizeof(BITMAPFILEHEADER) << std::endl;

    if(file != NULL) { // file opened
        BITMAPFILEHEADER h;
        size_t x = fread(&h, sizeof(BITMAPFILEHEADER), 1, file); //reading the FILEHEADER

        std::cout << x;
        fread(&this->ih, sizeof(BITMAPINFOHEADER), 1, file);

        int byte;
        int count = 0;
        for(int i=0;i<84;i++) byte = getc(file); //magic number 84 offset why?

        int image[3];
        int i = 0;

        int img_size = (52*52);
        RGBQUAD pixel[img_size];
        int line_num = 0;
        while ( i < img_size )
        {
        //while ( ! feof(file) ){    // foreach pixel
            fread(&pixel[i], sizeof(RGBQUAD), 1, file);


            printf("pixel %d: [%d,%d,%d]\n",i, pixel[i].rgbRed,pixel[i].rgbGreen,pixel[i].rgbBlue);
            i++;
            if (i%52 == 0 )
            {
                printf("----line: %d ---\n", line_num);
                line_num++;
            }
        }
        fclose(file);
    }
}

Bitmap::~Bitmap() {}

void Bitmap::print_pixels(){

    //std::cout << pixels << std::endl;
}
