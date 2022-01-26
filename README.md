HLRcompress
===========

Data compression library for general applications based on hierarchical low-rank 
methods combined with floating point number compression schemes.

## Installation

HLRcompress uses *cmake* for compiling and installation. It also supports
[TBB](https://threadingbuildingblocks.org) or *OpenMP* for parallelization
and [ZFP](https://zfp.io) for additional floating point number compression.

```sh
mkdir build
cd build
cmake ..
make
```

If support for *TBB* or *ZFP* is wanted and both are not available in standard paths, you
may configure their installation directories via

```sh
cmake -DTBB_DIR=<...> -DZFP_DIR=<...> ..
```

### BLAS/LAPACK

Via *cmake* you can also define the BLAS/LAPACK implementation to be used with
HLRcompress. It is **important** to use a **sequential** version of BLAS/LAPACK as all
parallelization is performed within HLRcompress. Any further parallelization on the block
level usually results in a drastic drop of performance.

With Intel MKL, the sequential version can be selected via

```sh
cmake -DBLA_VENDOR=Intel10_64lp_seq ..
```

or

```sh
cmake -DBLA_VENDOR=Intel10_64ilp_seq -DHLRCOMPRESS_USE_ILP64=ON ..
```

## Examples

### logmatrix

This example generates a matrix with entries $`a_{ij} = log |x_i - x_j|`$ with $`x_i`$ being
uniformly distributed on the unit circle in $`R^2`$.

The first command line argument defines the dimension of the matrix, the second argument
specifies the compression error and the third argument the (maximal) size of tiles. With
ZFP support enabled, the fourth argument defines the compression rate of ZFP.

For a computation with 1024 rows/columns, $`10^{-4}`$ compression error, a tile size of 32
and with a ZFP rate of 16 the full command line would be

```sh
./logmatrix 1024 1e-4 32 16
```

With CUDA available, it will use the experimental CUDA based compression instead of 

### h5compress

This example program needs additional HDF5 support, i.e., the HDF5 library installed
(*cmake* should be able to detect correct paths).

It will read dense data from HDF5 files and compress it afterwards. Aside from the first
command line argument, which specifies the file to read, the remaining arguments are
identical to the *logmatrix* example:

```sh
./h5compress data.h5 1e-6 32 24
```
