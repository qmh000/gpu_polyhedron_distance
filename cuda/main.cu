#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include <vector>
#include <algorithm>
#include "helper_cuda.h"
#include <float.h>
#include <thrust/host_vector.h>

#define BLOCK_SIZE 128
#define NUM_REPEATS 10

struct Triangle {
    Eigen::Vector3f v[3];

    void setVertex(int ind, Eigen::Vector3f ver) { v[ind] = ver; }
};


__device__ float TriDist(Eigen::Vector3f& P, Eigen::Vector3f& Q,
        const Eigen::Vector3f S[3], const Eigen::Vector3f T[3]);

__device__ void SegPoints(Eigen::Vector3f& VEC,
        Eigen::Vector3f& X, Eigen::Vector3f& Y,
        const Eigen::Vector3f P, const Eigen::Vector3f A,
        const Eigen::Vector3f Q, const Eigen::Vector3f B);

__global__ void PolyDist(Triangle* A, Triangle* B, int size1, int size2, float* dist);

Triangle*  read_off(const std::string filename, int &size){
    Triangle *obj = NULL;
    if (filename.empty()) {
        return NULL;
    }
    std::ifstream fin;
    fin.open(filename);

    if (!fin)
    {
        printf("File Error\n");
        return NULL;
    }
    else
    {
        printf("File opened successfully\n");

        int nVertices = 0;
        int nFaces = 0;
        int nEdges = 0;

        std::vector<Eigen::Vector3f> vertices;
        std::string str;
        fin >> str;

        fin >> nVertices >> nFaces >> nEdges;
        obj = (Triangle*)malloc(sizeof(Triangle) * nFaces);

        float x, y, z;
        for (int i = 0; i < nVertices; i++) {
            fin >> x >> y >> z;
            vertices.push_back({x, y, z});
        }

        unsigned int n, a, b, c;
        for (int i = 0; i < nFaces; i++) {
            fin >> n >> a >> b >> c;
            Triangle face;
            face.setVertex(0, vertices[a]);
            face.setVertex(1, vertices[b]);
            face.setVertex(2, vertices[c]);
            obj[i] = face;
        }
        size = nFaces;
    }
    fin.close();
    return obj;
}


int main()
{
    Triangle* h_obj1 = NULL;
    Triangle* h_obj2 = NULL;
    int size1, size2;
    h_obj1 = read_off("nuclei.off", size1);
    h_obj2 = read_off("vessel.off", size2);

    Triangle* d_obj1 = NULL;
    Triangle* d_obj2 = NULL;
    int memsize1 = sizeof(Triangle) * size1, memsize2 = sizeof(Triangle) * size2;

    float *d_dist = NULL, *h_dist = NULL;

    checkCudaErrors(cudaMalloc((void**) &d_obj1, memsize1));
    checkCudaErrors(cudaMalloc((void**) &d_obj2, memsize2));
    h_dist = (float*)malloc(sizeof(float) * size1 * size2);
    checkCudaErrors(cudaMalloc((void**) &d_dist, sizeof(float) * size1 * size2));

    checkCudaErrors(cudaMemcpy(d_obj1, h_obj1, memsize1, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_obj2, h_obj2, memsize2, cudaMemcpyHostToDevice));

    const int grid_size_x = (size1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_size_y = (size2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; repeat ++) {
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        checkCudaErrors(cudaEventRecord(start));
        cudaEventQuery(start);

        PolyDist<<<block_size, grid_size>>>(d_obj1, d_obj2, size1, size2, d_dist);

        checkCudaErrors(cudaMemcpy(h_dist, d_dist, sizeof(float) * size1 * size2, cudaMemcpyDeviceToHost));

        float *dist = thrust::min_element(thrust::host, h_dist, h_dist + (size1 * size2));
        printf("%f\n", *dist);

        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float elapsed_time;
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
    }
    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    free(h_obj1);
    free(h_obj2);
    free(h_dist);
    checkCudaErrors(cudaFree(d_obj1));
    checkCudaErrors(cudaFree(d_obj2));
    checkCudaErrors(cudaFree(d_dist));

    return 0;
}


__global__ void PolyDist(Triangle* A, Triangle* B, int size1, int size2, float* dist) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    Eigen::Vector3f p, q;
    if(x < size1 && y < size2) {
        float t = TriDist(p, q, A[x].v, B[y].v);
        dist[x * size2 + y] = t;
    }
    return;
}


__device__ void SegPoints(Eigen::Vector3f& VEC,
        Eigen::Vector3f& X, Eigen::Vector3f& Y,
        const Eigen::Vector3f P, const Eigen::Vector3f A,
        const Eigen::Vector3f Q, const Eigen::Vector3f B)
{
    float A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;
    Eigen::Vector3f T;
    Eigen::Vector3f TMP;

    T = Q - P;
    A_dot_A = A.dot(A);
    B_dot_B = B.dot(B);
    A_dot_B = A.dot(B);
    A_dot_T = A.dot(T);
    B_dot_T = B.dot(T);

    float t, u;

    float denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

    t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

    if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

    u = (t * A_dot_B - B_dot_T) / B_dot_B;

    if ((u <= 0) || isnan(u)) {

        Y = Q;

        t = A_dot_T / A_dot_A;

        if ((t <= 0) || isnan(t)) {
            X = P;
            VEC = Q - P;
        }
        else if (t >= 1) {
            X = P + A;
            VEC = Q - X;
        }
        else {
            X = P + A * t;
            TMP = T.cross(A);
            VEC = A.cross(TMP);
        }
    }
    else if (u >= 1) {

        Y = Q + B;

        t = (A_dot_B + A_dot_T) / A_dot_A;

        if ((t <= 0) || isnan(t)) {
            X = P;
            VEC = Y - P;
        }
        else if (t >= 1) {
            X = P + A;
            VEC = Y - X;
        }
        else {
            X = P + A * t;
            T = Y - P;
            TMP = T.cross(A);
            VEC = A.cross(TMP);
        }
    }
    else {

        Y = Q + B * u;

        if ((t <= 0) || isnan(t)) {
            X = P;
            TMP = T.cross(B);
            VEC = B.cross(TMP);
        }
        else if (t >= 1) {
            X = P + A;
            T = Q - X;
            TMP = T.cross(B);
            VEC = B.cross(TMP);
        }
        else {
            X = P + A * t;
            VEC = A.cross(B);
            if (VEC.dot(T) < 0) {
                VEC = VEC * -1;
            }
        }
    }
}


__device__ float TriDist(Eigen::Vector3f& P, Eigen::Vector3f& Q,
        const Eigen::Vector3f S[3], const Eigen::Vector3f T[3])
{

    Eigen::Vector3f Sv[3], Tv[3];
    Eigen::Vector3f VEC;

    Sv[0] = S[1] - S[0];
    Sv[1] = S[2] - S[1];
    Sv[2] = S[0] - S[2];

    Tv[0] = T[1] - T[0];
    Tv[1] = T[2] - T[1];
    Tv[2] = T[0] - T[2];

    Eigen::Vector3f V;
    Eigen::Vector3f Z;
    Eigen::Vector3f minP, minQ;
    float mindd;
    int shown_disjoint = 0;

    mindd = (S[0] - T[0]).squaredNorm() + 1;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {

            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            V = Q - P;
            float dd = V.dot(V);

            if (dd <= mindd)
            {
                minP = P;
                minQ = Q;
                mindd = dd;

                Z = S[(i + 2) % 3] - P;
                float a = Z.dot(VEC);
                Z = T[(j + 2) % 3] - Q;
                float b = Z.dot(VEC);

                if ((a <= 0) && (b >= 0)) return sqrt(dd);

                float p = V.dot(VEC);

                if (a < 0) a = 0;
                if (b > 0) b = 0;
                if ((p - a + b) > 0) shown_disjoint = 1;
            }
        }
    }

    Eigen::Vector3f Sn;
    float Snl;

    Sn = Sv[0].cross(Sv[1]);
    Snl = Sn.dot(Sn);

    if (Snl > 1e-15)
    {

        Eigen::Vector3f Tp;

        V = S[0] - T[0];
        Tp[0] = V.dot(Sn);

        V = S[0] - T[1];
        Tp[1] = V.dot(Sn);

        V = S[0] - T[2];
        Tp[2] = V.dot(Sn);

        int point = -1;
        if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
        {
            if (Tp[0] < Tp[1]) point = 0; else point = 1;
            if (Tp[2] < Tp[point]) point = 2;
        }
        else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
        {
            if (Tp[0] > Tp[1]) point = 0; else point = 1;
            if (Tp[2] > Tp[point]) point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;

            V = T[point] - S[0];
            Z = Sn.cross(Sv[0]);
            if (V.dot(Z) > 0)
            {
                V = T[point] - S[1];
                Z = Sn.cross(Sv[1]);
                if (V.dot(Z) > 0)
                {
                    V = T[point] - S[2];
                    Z = Sn.cross(Sv[2]);
                    if (V.dot(Z) > 0)
                    {

                        P = T[point] + Sn * Tp[point] / Snl;
                        Q = T[point];
                        return sqrt((P - Q).squaredNorm());
                    }
                }
            }
        }
    }

    float Tnl;
    Eigen::Vector3f Tn;

    Tn = Tv[0].cross(Tv[1]);
    Tnl = Tn.dot(Tn);

    if (Tnl > 1e-15)
    {
        float Sp[3];

        V = T[0] - S[0];
        Sp[0] = V.dot(Tn);

        V = T[0] - S[1];
        Sp[1] = V.dot(Tn);

        V = T[0] - S[2];
        Sp[2] = V.dot(Tn);

        int point = -1;
        if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
        {
            if (Sp[0] < Sp[1]) point = 0; else point = 1;
            if (Sp[2] < Sp[point]) point = 2;
        }
        else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
        {
            if (Sp[0] > Sp[1]) point = 0; else point = 1;
            if (Sp[2] > Sp[point]) point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;

            V = S[point] - T[0];
            Z = Tn.cross(Tv[0]);
            if (V.dot(Z) > 0)
            {
                V = S[point] - T[1];
                Z = Tn.cross(Tv[1]);
                if (V.dot(Z) > 0)
                {
                    V = S[point] - T[2];
                    Z = Tn.cross(Tv[2]);
                    if (V.dot(Z) > 0)
                    {
                        P = S[point];
                        Q = S[point] + Tn * Sp[point] / Tnl;
                        return sqrt((P - Q).squaredNorm());
                    }
                }
            }
        }
    }

    if (shown_disjoint)
    {
        P = minP;
        Q = minQ;
        return sqrt(mindd);
    }
    else return 0;
}
