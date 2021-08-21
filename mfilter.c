#include <stdio.h>
#include <stdlib.h>

struct Shape{
    int y;
    int x;
};
struct Frame{
    int y1;
    int y2;
    int x1;
    int x2;
};

float mean(int** arr, struct Frame* f){
    /* Return mean value of elements of 2d array 'arr' 
       inside the frame 'f'.*/
    int i, j;
    unsigned int sum = 0;
    unsigned int count = 0;
    for (i = f->y1; i < f->y2; i++)
        for (j = f->x1; j < f->x2; j++){
            sum += arr[i][j];
            count++;
        }
    float res = (float)sum / count;
    return res;
}

void replace(int** from, int** to, struct Frame* f){
    /* Replace values of array 'to' inside of the frame 'f'. */
    int i, j;
    for (i = f->y1; i < f->y2; i++)
        for (j = f->x1; j < f->x2; j++)
            to[i][j] = from[i][j];
}

int any(int** arr, struct Frame* f){
    /* Return true if any value inside the frame is true. */
    int i, j;
    for (i = f->y1; i < f->y2; i++)
        for (j = f->x1; j < f->x2; j++)
            if (arr[i][j] > 0)
                return 1;
    return 0;
}

int all(int** arr, struct Frame* f){
    /* Return true if all values inside the frame are true. */
    int i, j;
    for (i = f->y1; i < f->y2; i++)
        for (j = f->x1; j < f->x2; j++)
            if (arr[i][j] < 1)
                return 0;
    return 1;
}

void mfilter(int** arr,
             int** prev_arr,
             int** out_arr,
             struct Shape* size,
             struct Shape* conv,
             float threshold){
    /* Replace all subarrays of 'prev_arr' that have changed more
       than 'threshold' with values of new array 'arr'. */

    // Generate a difference map between arr and prev_arr
    // (0 is differing pixel, 1 -- identical)
    //
    // Allocate 2d array
    int** map;
    int col;
    map = malloc(size->y * sizeof(int*));
    for(col = 0; col < size->y; col++)
        map[col] = malloc(size->x * sizeof(int));
    // Calc the map
    int i, j;
    for(i=0; i < size->y; i++){
        for (j=0; j < size->x; j++){
            map[i][j] = abs(arr[i][j] + prev_arr[i][j] - 1);
        }
    }
    // Fill the out_arr
    int y1 = 0;
    int x1 = 0;
    int y2 = size->y % conv->y;
    if (y2 == 0)
        y2 = conv->y;
    int x2 = size->x % conv->x;
    if (x2 == 0)
        x2 = conv->x;
    int x_start = x2;
    struct Frame i_fr;
    struct Frame* i_fr_ptr = &i_fr;
    for (; y2 <= size->y; y1 = y2, y2 += conv->y){
        for (; x2 <= size->x; x1 = x2, x2 += conv->x){
            i_fr.y1 = y1;
            i_fr.y2 = y2;
            i_fr.x1 = x1;
            i_fr.x2 = x2;
            if ((mean(map, i_fr_ptr) < threshold)
                || (!any(arr, i_fr_ptr))
                || all(arr, i_fr_ptr)
            )
                replace(arr, out_arr, i_fr_ptr);
            else
                replace(prev_arr, out_arr, i_fr_ptr);
        }
        x1 = 0;
        x2 = x_start;
    }
}

/*
void replace_medianblur(int** from, int** to, struct Frame* f){
    int i, j;
    int pixel = (mean(from, f) < 0.5) ? 0 : 1;
    for (i = f->y1; i < f->y2; i++){
        for (j = f->x1; j < f->x2; j++){
            to[i][j] = pixel;
        }
    }
}

void get_inner_frame(int y,
                     int x,
                     int tb,
                     int lr,
                     struct Shape* size,
                     struct Frame* out){
    // Top coord
    out->y1 = y - tb;
    if (out->y1 < 0)
        out->y1 = 0;
    // Bottom coord
    out->y2 = y + tb;
    if (out->y2 > size->y)
        out->y2 = size->y;
    // Left coord
    out->x1 = x - lr;
    if (out->x1 < 0)
        out->x1 = 0;
    // Right coord
    out->x2 = x + lr;
    if (out->x2 > size->x)
        out->x2 = size->x;
}

// per-pixel
void mfilter(int** arr,
             int** prev_arr,
             int** out_arr,
             struct Shape* size,
             struct Shape* conv,
             float threshold){

    // Generate a difference map between arr and prev_arr
    // (0 is differing pixel, 1 -- identical)
    //
    // Allocate 2d array
    int** map;
    int col;
    map = malloc(size->y * sizeof(int*));
    for(col = 0; col < size->y; col++)
        map[col] = malloc(size->x * sizeof(int));
    // Calc the map
    int i, j;
    for(i=0; i < size->y; i++){
        for (j=0; j < size->x; j++){
            map[i][j] = abs(arr[i][j] + prev_arr[i][j] - 1);
        }
    }
    // Fill the out_arr
    int tb = conv->y / 2;
    int lr = conv->x / 2;
    int y, x;
    struct Frame i_fr = {0, 0, 0, 0};
    struct Frame* i_fr_ptr = &i_fr;
    for (y=0; y < size->y; y++){
        for(x=0; x < size->x; x++){
            get_inner_frame(y, x, tb, lr, size, i_fr_ptr);
            if ((mean(map, i_fr_ptr) < threshold)
                || (!any(arr, i_fr_ptr))
                || all(arr, i_fr_ptr)
            )
                replace(arr, out_arr, i_fr_ptr);
            else
                replace(prev_arr, out_arr, i_fr_ptr);
        }
    }
}

// high range
void mfilter(int** arr,
             int** prev_arr,
             int** out_arr,
             struct Shape* size,
             struct Shape* conv,
             float threshold){

    // Generate a difference map between arr and prev_arr
    // (0 is differing pixel, 1 -- identical)
    //
    // Allocate 2d array
    int** map;
    int col;
    map = malloc(size->y * sizeof(int*));
    for(col = 0; col < size->y; col++)
        map[col] = malloc(size->x * sizeof(int));
    // Calc the map
    int i, j;
    for(i=0; i < size->y; i++){
        for (j=0; j < size->x; j++){
            map[i][j] = abs(arr[i][j] + prev_arr[i][j] - 1);
        }
    }
    // Fill the out_arr
    int y1 = 0;
    int x1 = 0;
    int y2, x2;
    int y_c, x_c;
    int tb = conv->y;
    int lr = conv->x;
    struct Frame big = {0,0,0,0};
    struct Frame i_fr;
    for (y2 = conv->y; y2 <= size->y; y2 += conv->y, y1 += conv->y){
        for (x2 = conv->x; x2 <= size->x; x2 += conv->x, x1 += conv->x){
            i_fr.y1 = y1;
            i_fr.y2 = y2;
            i_fr.x1 = x1;
            i_fr.x2 = x2;
            y_c = y2 - conv->y / 2;
            x_c = x2 - conv->x / 2;
            get_inner_frame(y_c, x_c, tb, lr, size, &big);
            if ((mean(map, &big) < threshold)
                || (!any(arr, &big))
                || all(arr, &big)
            )
                replace(arr, out_arr, &i_fr);
            else
                replace(prev_arr, out_arr, &i_fr);
            //printf("%d %d %d %d\n", y1, y2, x1, x2);
        }
        x1 = 0;
    }
}
*/