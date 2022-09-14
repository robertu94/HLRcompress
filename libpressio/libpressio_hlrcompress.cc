#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

#if HLRCOMPRESS_USE_HDF5 == 1
#  include <hdf5.h>
#endif

#include <hlrcompress/compress.hh>
#include <hlrcompress/approx/svd.hh>
#include <hlrcompress/approx/rrqr.hh>
#include <hlrcompress/approx/randsvd.hh>
#include <hlrcompress/hlr/error.hh>

using namespace hlrcompress;

namespace libpressio { namespace hlrcompress_ns {

class hlrcompress_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:abs", acc);
    set(options, "hlrcompress:apx", apx);
    set(options, "hlrcompress:acc", acc);
    set(options, "hlrcompress:ntile", ntile);

    set_type(options, "hlrcompress:z_reversible", pressio_option_bool_type);
    set_type(options, "hlrcompress:z_fixed_rate", pressio_option_uint32_type);
    set_type(options, "hlrcompress:z_fixed_accuracy", pressio_option_double_type);
    set_type(options, "hlrcompress:z_adaptive", pressio_option_double_type);
    switch(zconf.mode) {
      case compress_reversible:
        set(options, "hlrcompress:z_reversible", true);
        break;
      case compress_fixed_rate:
        set(options, "hlrcompress:z_fixed_rate", zconf.rate);
        break;
      case compress_fixed_accuracy:
        set(options, "hlrcompress:z_fixed_accuracy", zconf.accuracy);
        break;
      case compress_adaptive:
        set(options, "hlrcompress:z_adaptive", zconf.accuracy);
        break;
    }
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"()");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pressio:abs", &acc);
    get(options, "hlrcompress:apx", &apx);
    get(options, "hlrcompress:acc", &acc);
    get(options, "hlrcompress:ntile", &ntile);
    {
      bool tmp;
      if(get(options, "hlrcompress:z_reversible", &tmp) == pressio_options_key_set) {
        zconf = reversible();
      }
    }
    {
      int32_t rate ;
      if(get(options, "hlrcompress:z_fixed_rate", &rate) == pressio_options_key_set) {
        zconf = fixed_rate(rate);
      }
    }
    {
      double accuracy ;
      if(get(options, "hlrcompress:z_fixed_accuracy", &accuracy) == pressio_options_key_set) {
        zconf = fixed_accuracy(accuracy);
      }
    }
    {
      double adaptive;
      if(get(options, "hlrcompress:z_adaptive", &adaptive) == pressio_options_key_set) {
        zconf = fixed_accuracy(adaptive);
      }
    }
    return 0;
  }

  template <class T>
  int compress_impl_typed(const pressio_data* input,
                    struct pressio_data* output) {
    auto dims = input->normalized_dims(2, 1);
    auto  M = blas::matrix< T >( dims[0] , dims[1] );

    auto zM = std::unique_ptr<block<T>>();
    if      ( apx == "default" ) zM = ::hlrcompress::compress<T>( M, acc, hlrcompress::default_approx(), ntile, zconf );
    else if ( apx == "svd"     ) zM = ::hlrcompress::compress<T>( M, acc, SVD(), ntile, zconf );
    else if ( apx == "rrqr"    ) zM = ::hlrcompress::compress<T>( M, acc, RRQR(), ntile, zconf );
    else if ( apx == "randsvd" ) zM = ::hlrcompress::compress<T>( M, acc, RandSVD(), ntile, zconf );
    else throw std::runtime_error("unexpected approx method " + apx);

    /*TODO set the compressed data*/

    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      switch(input->dtype()) {
        case pressio_float_dtype:
          return compress_impl_typed<float>(input, output);
        case pressio_double_dtype:
          return compress_impl_typed<double>(input, output);
        default:
          throw std::runtime_error("unsupported datatype");
      }
    } catch (std::exception const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  template <class T>
  int decompress_impl_typed(const pressio_data* input, struct pressio_data* output) {

    return set_error(1, "decompression implemented error");

    /*
    auto dims = output->normalized_dims(2, 1);
    auto  M = blas::matrix< T >( dims[0] , dims[1] );

    auto zM = std::unique_ptr<block<T>>();
    if      ( apx == "default" ) zM = ::hlrcompress::compress<T>( M, acc, hlrcompress::default_approx(), ntile, zconf );
    else if ( apx == "svd"     ) zM = ::hlrcompress::compress<T>( M, acc, SVD(), ntile, zconf );
    else if ( apx == "rrqr"    ) zM = ::hlrcompress::compress<T>( M, acc, RRQR(), ntile, zconf );
    else if ( apx == "randsvd" ) zM = ::hlrcompress::compress<T>( M, acc, RandSVD(), ntile, zconf );
    else throw std::runtime_error("unexpected approx method " + apx);

    // TODO we need an api to set the compressed data here ...
    zM->uncompress();

    *output = pressio_data::copy(
        output->dtype(),
        zM.data(),
        dims
        );
    */
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
      switch(output->dtype()) {
        case pressio_float_dtype:
          return compress_impl_typed<float>(input, output);
        case pressio_double_dtype:
          return compress_impl_typed<double>(input, output);
        default:
          throw std::runtime_error("unsupported datatype");
      }
    } catch (std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "hlrcompress"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<hlrcompress_compressor_plugin>(*this);
  }

  double acc = 1e-4;
  size_t ntile = 32;
  std::string apx = "default";
  zconfig_t zconf = fixed_accuracy( 1.0 );
  
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "hlrcompress", []() {
  return compat::make_unique<hlrcompress_compressor_plugin>();
});

} }


extern "C" void  register_libpressio_hlrcompress() {
}
