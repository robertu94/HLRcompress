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

### hlrcompress

This program reads raw data or data from HDF5 files (assuming HDF5 support configured via **cmake**) and applies
HLRcompression algorithm.

| Option               | Description          |
|----------------------|----------------------|
| `-i file`            | define data file (HDF5 or raw data) |
| `-d "type dim0 dim1"`| define data type for raw data (*float* or *double* and dimensions) |
| `-e eps`             | set relative accuracy |
| `-l apx`             | low-rank approximation scheme (svd,rrqr,randsvd) |
| `-t size`            | set tile size |
| `-p eps`             | set ZFP compression accuracy (default) |
| `-r rate`            | set ZFP compression rate |
| `-a eps`             | set ZFP compression adaptive accuracy |
| `-n`                 | no ZFP compression |
| `-b num`             | benchmark compression |
  
Please note that for fixed or adaptive ZFP accuracy, the argument is a factor to the actual precision, 
which is automatically chosen based on the compression accuracy and the matrix norm. By default this 
should be set to **1**.

```sh
./h5compress -i data.h5 -e 1e-6 -t 32 -p 1.0
```

### logmatrix

This example generates a matrix with entries $`a_{ij} = log |x_i - x_j|`$ with $`x_i`$ being
uniformly distributed on the unit circle in $`R^2`$.

| Option    | Description          |
|-----------|----------------------|
| `-n size` | set matrix size |
| `-e eps`  | set relative accuracy |
| `-l apx`  | low-rank approximation scheme (svd,rrqr,randsvd) |
| `-t size` | set tile size |
| `-p eps`  | set ZFP compression accuracy (default) |
| `-r rate` | set ZFP compression rate |
| `-a eps`  | set ZFP compression adaptive accuracy |
| `-b num`  | benchmark compression |

As for **hlrcompress** the argument to `-p` and `-p` are factors for the actual accuracy
and should be kept close to 1.

For a computation with 1024 rows/columns, $`10^{-4}`$ compression error, a tile size of 32
and with a ZFP rate of 16 the full command line would be

```sh
./logmatrix -i 1024 -e 1e-4 -t 32 -r 16
```

With CUDA available, it will use the experimental CUDA based compression instead of the
CPU implementation.

