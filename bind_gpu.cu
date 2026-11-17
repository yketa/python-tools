#include <complex>
#include <math.h>
#include <numbers>
#include <vector>

using namespace std::complex_literals;  // using comlex numbers

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "assert.hpp"
#include "bind.hpp"

/*
 *  Fourier transforms
 *
 */

__global__ void ft2D(
    double* fourier_real, double* fourier_imag, // Fourier transform arrays
    int N, double* positions, double* values,   // positions and values
    int n, double* wavevectors) {               // wavevectors
/*
Compute 2D Fourier transofrm of `values' associated to 2D `positions' at 2D
`wavevectors'.
*/

    int k = blockIdx.x*blockDim.x + threadIdx.x;    // global index used to access elements of the arrays
    if (k >= n) return;

    fourier_real[k] = 0;                    // initialise real part
    fourier_imag[k] = 0;                    // initialise imaginary part
    for (int i=0; i < N; i++) {
        fourier_real[k] += values[i]*cos(   // real Fourier term
            wavevectors[2*k + 0]*positions[2*i + 0]
            + wavevectors[2*k + 1]*positions[2*i + 1]);
        fourier_imag[k] -= values[i]*sin(   // imaginary Fourier term
            wavevectors[2*k + 0]*positions[2*i + 0]
            + wavevectors[2*k + 1]*positions[2*i + 1]);
    }
}

pybind11::array_t<std::complex<double>> getAllFT2D_GPU(
    pybind11::array_t<double> const& positions,
    pybind11::array_t<double> const& L,
    pybind11::array_t<double> const& values,
    double const& qmin, double const& qmax, int const& nmax=0) {
/*
Return 2D Fourier transform of `values' associated to 2D `positions' at 2D wave
vectors whose norms belong to [`qmin', `qmax'].
*/

    // wave vector norms
    pybind11::array_t<double> qARR = getAllWaveVectors2D(L, qmin, qmax, nmax);
    if (qARR.request().size == 0)
        { return pybind11::array_t<std::complex<double>>(0); }
    auto _q = qARR.unchecked<2>();      // direct access to wave vectors
    long int const n = _q.shape(0);

    // check positions and values arrays
    auto r = positions.unchecked<2>();  // direct access to positions
    assert(r.ndim() == 2);
    assert(r.shape(1) == 2);
    long int const N = r.shape(0);
    auto v = values.unchecked<>();      // direct access to first values
    if (v.shape(0) != N) {
        throw std::invalid_argument("Positions and values must have = sizes.");
    }
    if (v.ndim() > 1) {
        throw std::invalid_argument("Values must be 1-dimensional.");
    }

    // Fourier transform
    pybind11::array_t<std::complex<double>> fourier({n});
    auto ft = fourier.mutable_unchecked<1>();
    std::vector<double> fourier_real(n, 0);
    std::vector<double> fourier_imag(n, 0);

    // allocate memory to copy arguments to device
    double* d_fourier_real;
    cudaMalloc(&d_fourier_real, n*sizeof(double));
    cudaMemcpy(d_fourier_real, &(fourier_real[0]),
        n*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_fourier_imag;
    cudaMalloc(&d_fourier_imag, n*sizeof(double));
    cudaMemcpy(d_fourier_imag, &(fourier_imag[0]),
        n*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_positions;
    cudaMalloc(&d_positions, 2*N*sizeof(double));
    cudaMemcpy(d_positions, positions.request().ptr,
        2*N*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_values;
    cudaMalloc(&d_values, N*sizeof(double));
    cudaMemcpy(d_values, values.request().ptr,
        N*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_wavevectors;
    cudaMalloc(&d_wavevectors, 2*n*sizeof(double));
    cudaMemcpy(d_wavevectors, qARR.request().ptr,
        2*n*sizeof(double),
        cudaMemcpyHostToDevice);

    // launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;
    ft2D<<<numBlocks, blockSize>>>(
        d_fourier_real, d_fourier_imag, // Fourier transform array
        N, d_positions, d_values,       // positions and values
        n, d_wavevectors);              // wavevectors
    cudaMemcpy(&(fourier_real[0]), d_fourier_real,
        n*sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&(fourier_imag[0]), d_fourier_imag,
        n*sizeof(double),
        cudaMemcpyDeviceToHost);
    for (int k=0; k < n; k++) ft(k) = fourier_real[k] + 1i*fourier_imag[k];

    // free memory
    cudaFree(d_fourier_real);
    cudaFree(d_fourier_imag);
    cudaFree(d_positions);
    cudaFree(d_values);
    cudaFree(d_wavevectors);
    
    // return array
    return fourier;
}

pybind11::array_t<std::complex<double>> getFT2D_GPU(
    pybind11::array_t<double> const& positions,
    pybind11::array_t<double> const& L,
    pybind11::array_t<double> const& values,
    double const& q, double const& dq=0.1)
    { return getAllFT2D_GPU(positions, L, values, q - dq/2, q + dq/2); }
/*
Return 2D Fourier transform of `values' associated to 2D `positions' at 2D wave
vectors whose norms belong to [`q' - `dq'/2, `q' + `dq'/2].
*/

/*
 *  Correlations
 *
 */

__global__ void corr2D(
    double* height, long int* occupancy,                                    // height and occupancy arrays
    int N, double* positions, double* L, double* values1, double* values2,  // positions and values
    bool periodic_boundary_conditions,                                      // periodic boundary conditions
    double rmin, double rmax, double dbin, int nBins) {                     // histogram parameters
/*
Compute radial correlations between `values1' and `values2'.
*/

    int i = blockIdx.x*blockDim.x + threadIdx.x;                // global index used to access elements of the arrays
    if (i >= N) return;

    for (int j=i; j < N; j++) {                                 // consider all pairs only once
        double dx = positions[2*j + 0] - positions[2*i + 0];    // x-dimension separation
        double dy = positions[2*j + 1] - positions[2*i + 1];    // y-dimension separation
        if (periodic_boundary_conditions) {                     // wrap around periodic boundary condition
            dx = remainder(dx, L[0]);
            dy = remainder(dy, L[1]);
        }
        double dist = sqrt(dx*dx + dy*dy);                      // distance corresponding to difference vector
        if (dist < rmin || dist >= rmax) continue;              // distance out of bins
        int bin = (dist - rmin)/dbin;                           // bin corresponding to distance
        height[i*nBins + bin] +=                                // product
            (values1[i]*values2[j] + values1[j]*values2[i])/2.;
        occupancy[i*nBins + bin] += 1;
    }
}

pybind11::array_t<double> getRadialCorrelations2D_GPU(
    pybind11::array_t<double> const& positions,
    pybind11::array_t<double> const& L,
    pybind11::array_t<double> const& values1,
    pybind11::array_t<double> const& values2_,
    int const& nBins,
    bool const& periodic_boundary_conditions=true,
    double const& rmin=0, double const& rmax_=0,
    bool const& rescale_pair_distribution=false) {
/*
Return 2D radial correlations of `values1' and `values2' (1D or 2D).
*/

    // check system size
    std::vector<double> systemSize = _get2DL(L);

    // system area (for normalisation)
    double const area = [&systemSize]() {
        if (systemSize[0]*systemSize[1] == 0) {
            return 1.;
        }
        else {
            return systemSize[0]*systemSize[1];
        }
    }();

    // check maximum radius
    double const rmax = rmax_ == 0 ?                                // is rmax_ == 0?
        *std::min_element(systemSize.begin(), systemSize.end())/2   // then rmax = min(L)/2
        : rmax_;                                                    // otherwise rmax = rmax_
    if (rmin == rmax) {
        throw std::invalid_argument("Min- and maximum radii cannot be equal.");
    }

    // check positions and values arrays
    pybind11::array_t<double> const values2 =
        [&values1, &values2_]() {
            if (values2_.unchecked<>().size() == 0) {   // is second values an empty array?
                return values1;                         // then copy it from first values
            }
            else {
                return values2_;                        // otherwise use input second values
            }
    }();
    auto r = positions.unchecked<2>();                  // direct access to positions
    auto v1 = values1.unchecked<>();                    // direct access to first values
    auto v2 = values2.unchecked<>();                    // direct access to second values
    if (v1.shape(0) != r.shape(0) ||
        v1.shape(0) != v2.shape(0) || v1.size() != v2.size()) {
        throw std::invalid_argument("Values' sizes are not consistent.");
    }
    if (v1.ndim() > 1 || v2.ndim() > 1) {
        throw std::invalid_argument("Values must be 1-dimensional.");
    }
    long int const N = r.shape(0);                      // number of particles

    // initialise returned histograms and computation quantities
    pybind11::array_t<double> correlations({nBins, 2});
    auto c = correlations.mutable_unchecked<2>();               // direct access to correlation histogram
    for (int bin=0; bin < nBins; bin++) {
        c(bin, 0) = rmin + (bin + 1./2.)*(rmax - rmin)/nBins;   // centre of correlation histogram bins
        c(bin, 1) = 0;                                          // height of correlation histogram bins
    }
    std::vector<double> height(N*nBins, 0);                     // height of correlation histogram bins
    std::vector<long int> occupancy(N*nBins, 0);                // occupancy of histogram bins (for normlaisation)
    long int nPairs = N*(N - 1)/2;                              // number of unique pairs of points (for normalisation)

    // allocate memory to copy arguments to device
    double* d_height;
    cudaMalloc(&d_height, N*nBins*sizeof(double));
    cudaMemcpy(d_height, &(height[0]),
        N*nBins*sizeof(double),
        cudaMemcpyHostToDevice);
    long int* d_occupancy;
    cudaMalloc(&d_occupancy, N*nBins*sizeof(long int));
    cudaMemcpy(d_occupancy, &(occupancy[0]),
        N*nBins*sizeof(int),
        cudaMemcpyHostToDevice);
    double* d_positions;
    cudaMalloc(&d_positions, 2*N*sizeof(double));
    cudaMemcpy(d_positions, positions.request().ptr,
        2*N*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_L;
    cudaMalloc(&d_L, 2*sizeof(double));
    cudaMemcpy(d_L, &(systemSize[0]),
        2*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_values1;
    cudaMalloc(&d_values1, N*sizeof(double));
    cudaMemcpy(d_values1, values1.request().ptr,
        N*sizeof(double),
        cudaMemcpyHostToDevice);
    double* d_values2;
    cudaMalloc(&d_values2, N*sizeof(double));
    cudaMemcpy(d_values2, values2.request().ptr,
        N*sizeof(double),
        cudaMemcpyHostToDevice);

    // launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1)/blockSize;
    corr2D<<<numBlocks, blockSize>>>(
        d_height, d_occupancy,                      // height and occupancy arrays
        N, d_positions, d_L, d_values1, d_values2,  // positions and values
        periodic_boundary_conditions,               // periodic boundary conditions
        rmin, rmax, (rmax - rmin)/nBins, nBins);    // histogram parameters
    cudaMemcpy(&(height[0]), d_height,
        N*nBins*sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&(occupancy[0]), d_occupancy,
        N*nBins*sizeof(long int),
        cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_height);
    cudaFree(d_occupancy);
    cudaFree(d_positions);
    cudaFree(d_L);
    cudaFree(d_values1);
    cudaFree(d_values2);
    
    // sum and normalise
    for (int bin=0; bin < nBins; bin++) {
        long int occup = 0;
        for (int i=0; i < N; i++) {
            c(bin, 1) += height[i*nBins + bin];
            occup += occupancy[i*nBins + bin];
        }
        if (occup > 0) {
            // mean over computed values
            c(bin, 1) /= occup;
            // correction by pair distribution function
            if (!rescale_pair_distribution) { continue; }
            if (bin == 0 && rmin == 0) { continue; }    // do not consider 0th bin
            c(bin, 1) /=
                ((double) occup/nPairs)                 // histogram value
                *area/((rmax - rmin)/nBins)             // normalisation
                /(2*std::numbers::pi*c(bin, 0));        // radial projection
        }
    }

    return correlations;
}

/*
 *  Binding
 *
 */

PYBIND11_MODULE(bind_gpu, m) {

    m.def("getAllWaveVectors2D", &getAllWaveVectors2D,
        "Return wave vectors associated to rectangular box.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "qmin : float\n"
        "    Minimum wave vector norm.\n"
        "qmax : float\n"
        "    Maximum wave vector norm.\n"
        "nmax : int\n"
        "    Maximum number of wave vectors to return. (default: 0)\n"
        "    NOTE: if nmax == 0 returns all wave vectors.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "wv : (*, 2) float numpy array\n"
        "    Array of (2\\pi/L nx, 2\\pi/L ny) wave vectors corresponding to\n"
        "    to the target interval [`qmin', `qmax'].\n"
        "    NOTE: Only a single vector of each pair of opposite wave\n"
        "          vectors is returned. Here it is chosen such that ny >= 0.",
        pybind11::arg("L"),
        pybind11::arg("qmin"),
        pybind11::arg("qmax"),
        pybind11::arg("nmax")=0);

    m.def("getWaveVectors2D", &getWaveVectors2D,
        "Return wave vectors associated to rectangular box.\n"
        "(see getAllWaveVectors2D)\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "q : float\n"
        "    Target wave vector norm.\n"
        "dq : float\n"
        "    Width of wave vector norm interval. (default: 0.1)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "wv : (*, 2) float numpy array\n"
        "    Array of (2\\pi/L nx, 2\\pi/L ny) wave vectors corresponding to\n"
        "    to the target interval [`q' - `dq'/2, `q' + `dq'/2].\n"
        "    NOTE: Only a single vector of each pair of opposite wave\n"
        "          vectors is returned. Here it is chosen such that ny >= 0.",
        pybind11::arg("L"),
        pybind11::arg("q"),
        pybind11::arg("dq")=0.1);

    m.def("getAllFT2D_GPU", &getAllFT2D_GPU,
        "Return 2D Fourier transform of delta-peaked values.\n"
        "\n"
        ".. math::"
        "V(k_l) = \\sum_i \\exp(-1i k_l \\cdot r_i) v_i\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "positions : (*, 2) float array-like\n"
        "    Positions r_i of delta-peaked values.\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "values : (*,) float array-like\n"
        "    Delta-peaked values v_i.\n"
        "qmin : float\n"
        "    Minimum wave vector norm.\n"
        "qmax : float\n"
        "    Maximum wave vector norm.\n"
        "nmax : int\n"
        "    Maximum number of wave vectors to use. (default: 0)\n"
        "    NOTE: if nmax == 0 uses all wave vectors.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ft : (**,) complex numpy array\n"
        "    Fourier transform of `values' for each wave vector in the\n"
        "    target norm interval [`qmin', `qmax'].\n"
        "    NOTE: These are given by getWaveVectors2D.\n",
        pybind11::arg("positions"),
        pybind11::arg("L"),
        pybind11::arg("values"),
        pybind11::arg("qmin"),
        pybind11::arg("qmax"),
        pybind11::arg("nmax")=0);

    m.def("getFT2D_GPU", &getFT2D_GPU,
        "Return 2D Fourier transform of delta-peaked values.\n"
        "(see getAllFT2D)\n"
        "\n"
        ".. math::"
        "V(k_l) = \\sum_i \\exp(-1i k_l \\cdot r_i) v_i\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "positions : (*, 2) float array-like\n"
        "    Positions r_i of delta-peaked values.\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the box.\n"
        "values : (*,) float array-like\n"
        "    Delta-peaked values v_i.\n"
        "q : float\n"
        "    Target wave vector norm.\n"
        "dq : float\n"
        "    Width of wave vector norm interval. (default: 0.1)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ft : (**,) complex numpy array\n"
        "    Fourier transform of `values' for each wave vector in the\n"
        "    target norm interval [`q' - `dq'/2, `q' + `dq'/2].\n"
        "    NOTE: These are given by getWaveVectors2D.\n",
        pybind11::arg("positions"),
        pybind11::arg("L"),
        pybind11::arg("values"),
        pybind11::arg("q"),
        pybind11::arg("dq")=0.1);

    m.def("getRadialCorrelations2D_GPU", &getRadialCorrelations2D_GPU,
        "Compute two-dimensional (2D) radial correlations between `values1'\n"
        "(and `values2') associated to each of the positions of a 2D system\n"
        "with linear size(s) `L'.\n"
        "\n"
        ".. math::"
        "C(|r1 - r0|) ~ \\langle values1(r0) \\cdot values2(r1)^* \\rangle\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "positions : (*, 2) float array-like\n"
        "    Positions associated to values.\n"
        "L : float or (1,)- or (2,) float array-like\n"
        "    Size of the system box in each dimension.\n"
        "values1 : (*,) float array-like\n"
        "    Values to compute the correlations of `values2' with.\n"
        "values2 : (*,) float array-like\n"
        "    Values to compute the correlations of `values1' with.\n"
        "    NOTE: if `values2' is empty then it is replaced with `values1'.\n"
        "nBins : int\n"
        "    Number of intervals of distances on which to compute the\n"
        "    correlations.\n"
        "periodic_boundary_conditions : bool\n"
        "    Use periodic boundary conditions when computing distances\n"
        "    between points. (default: True)\n"
        "rmin : float\n"
        "    Minimum distance (included) at which to compute the\n"
        "    correlations. (default: 0)\n"
        "rmax : float\n"
        "    Maximum distance (excluded) at which to compute the\n"
        "    correlations. (default: 0)\n"
        "    NOTE: if max == 0 then max = min(L)/2.\n"
        "rescale_pair_distribution : bool\n"
        "    Rescale correlations by pair distribution function.\n"
        "    (default: False)\n"
        "\n"
        "Returns\n"
        "-------\n"
        "correlations : (nBins, 2) float numpy array\n"
        "    Array of (r, C(r)) where r is the centre of the correlation\n"
        "    histogram bin and C(r) is the height of the correlation\n"
        "    histogram bin.",
        pybind11::arg("positions"),
        pybind11::arg("L"),
        pybind11::arg("values1"),
        pybind11::arg("values2"),
        pybind11::arg("nBins"),
        pybind11::arg("periodic_boundary_conditions")=true,
        pybind11::arg("rmin")=0,
        pybind11::arg("rmax")=0,
        pybind11::arg("rescale_pair_distribution")=false);
}

