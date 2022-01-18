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

If support for *TBB* or *ZFP* is wanted and both are not available in standard path, you
may configure their installation directories via

```sh
cmake -DTBB_DIR=<...> -DZFP_DIR=<...> ..
```
