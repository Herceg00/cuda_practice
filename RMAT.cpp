#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

#include <stdio.h>
#include <math.h>
#include <vector>

using namespace std;

#ifndef uint32_t
#define uint32_t int
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T rand_uniform_val(int _upper_border)
{
    return (_T)(rand() % _upper_border);
}

template <>
int rand_uniform_val(int _upper_border)
{
    return (int)(rand() % _upper_border);
}

template <>
float rand_uniform_val(int _upper_border)
{
    return (float)(rand() % _upper_border) / _upper_border;
}

template <>
double rand_uniform_val(int _upper_border)
{
    return (double)(rand() % _upper_border) / _upper_border;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void uniform_random(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _omp_threads,  bool _directed, bool _weighted)
{
    int n = (int)log2(_vertices_count);

    cout << "using " << _omp_threads << " threads" << endl;

    // generate and add edges to graph
    unsigned int seed = 0;
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++)
        {
            int from = rand() % _vertices_count;
            int to = rand() % _vertices_count;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);

            if(_directed)
            {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;

                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }

            if(!_directed)
            {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void R_MAT(int *src_ids, int *dst_ids, float *weights, int _vertices_count, long _edges_count, int _a_prob,  int _b_prob,
           int _c_prob, int _d_prob, int _omp_threads,  bool _directed, bool _weighted)
{
    int n = (int)log2(_vertices_count);

    cout << "using " << _omp_threads << " threads" << endl;

    // generate and add edges to graph
    unsigned int seed = 0;
#pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();

#pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;

                int step = (int)pow(2, n - (i + 1));

                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;

            int from = x_middle;
            int to = y_middle;
            float edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);

            if(_directed)
            {
                src_ids[cur_edge] = from;
                dst_ids[cur_edge] = to;

                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }

            if(!_directed)
            {
                src_ids[cur_edge] = min(to, from);
                dst_ids[cur_edge] = max(to, from);
                if(_weighted)
                    weights[cur_edge] = edge_weight;
            }
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        int threads = omp_get_max_threads();

        int vertices_count = pow(2.0, 4);
        long long edges_count = 3 * (long long)vertices_count;

        int *src_ids = new int[edges_count];
        int *dst_ids = new int[edges_count];
        float *weights = new float[edges_count];

        R_MAT(src_ids, dst_ids, weights, vertices_count, edges_count, 45, 20, 20, 15, threads, true, true);
        uniform_random(src_ids, dst_ids, weights, vertices_count, edges_count, threads, true, true);


        delete[] src_ids;
        delete[] dst_ids;
        delete[] weights;
    }
    catch (const char *error)
    {
        cout << error << endl;
        getchar();
        return 1;
    }
    catch (...)
    {
        cout << "unknown error" << endl;
    }

    cout << "press any key to exit..." << endl;
    //getchar();
    return 0;
}