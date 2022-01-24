#ifndef __HLRCOMPRESS_APPROX_ACCURACY_HH
#define __HLRCOMPRESS_APPROX_ACCURACY_HH
//
// Project     : HLRcompress
// Module      : approx/accuracy
// Description : defines truncation accuracy for low-rank blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlrcompress/hlr/indexset.hh>
#include <hlrcompress/blas/vector.hh>

namespace hlrcompress
{

//
// defines accuracy for truncation of low rank blocks.
//
struct accuracy
{
private:
    // relative and absolute truncation accuracy
    double  _rel_eps, _abs_eps;

    // upper limit for rank (negative means no limit; 0 = not set)
    int     _max_rank;
    
public:
    /////////////////////////////////////////////////
    //
    // constructor and destructor
    //

    //
    // construct accuracy object for exact truncation
    //
    accuracy ()
            : _rel_eps(0.0)
            , _max_rank(-1)
            , _abs_eps(0.0)
    {}

    //
    // construct accuracy object for fixed accuracy truncation
    //
    accuracy ( const double  relative_eps,
               const double  absolute_eps = 0.0 )
            : _rel_eps(relative_eps)
            , _max_rank(-1)
            , _abs_eps(absolute_eps)
    {}

    //
    // copy constructor
    //
    accuracy ( const accuracy &  acc )
            : _rel_eps(0.0)
            , _max_rank(-1)
            , _abs_eps(0.0)
    {
        *this = acc;
    }

    // explicit virtual destructor
    virtual ~accuracy () {}
    
    /////////////////////////////////////////////////
    //
    // truncation rank computation
    //

    // return truncation rank based on given singular values
    template < typename value_t >
    size_t  trunc_rank ( const blas::vector< value_t > &  sv ) const
    {
        // initialise with either fixed rank or fixed accuracy
        auto  eps = value_t( rel_eps() ) * std::abs( sv(0) );
        auto  k   = idx_t( sv.length() );

        // apply absolute lower limit for singular values
        eps = std::max( eps, value_t( abs_eps() ) );

        // apply maximal rank
        if ( has_max_rank() )
            k = std::min( k, idx_t( max_rank() ) );

        // compare singular values and stop, if truncation rank was reached
        for ( idx_t  i = 0; i < k; ++i )
        {
            if ( std::abs( sv(i) ) < eps )
            {
                k = i;
                break;
            }// if
        }// for
    
        return k;
    }
    
    /////////////////////////////////////////////////
    //
    // accuracy management
    //

    // return accuracy description for individual subblock
    virtual const accuracy    acc ( const indexset &  /* rowis */,
                                    const indexset &  /* colis */ ) const 
    {
        return *this;
    }

    // return accuracy description for individual submatrix
    const accuracy    operator () ( const indexset &  rowis,
                                    const indexset &  colis ) const
    {
        return acc( rowis, colis );
    }

    /////////////////////////////////////////////////
    //
    // access accuracy data
    //

    // return maximal rank (nonnegative!)
    size_t  max_rank       () const { return std::max( 0, _max_rank ); }

    // return true if maximal truncation rank was defined
    bool    has_max_rank   () const { return _max_rank >= 0; }
    
    // return relative accuracy
    double  rel_eps        () const { return _rel_eps; }

    // return absolute accuracy
    double  abs_eps        () const { return _abs_eps; }

    // return true if accuracy is "exact"
    bool    is_exact       () const { return (_rel_eps == 0.0) && (_abs_eps == 0.0); }

    // set maximal rank in truncation
    void    set_max_rank   ( const int  k )
    {
        _max_rank = std::max( 0, k );
    }

    // copy operator
    accuracy & operator = ( const accuracy & ta )
    {
        _rel_eps  = ta._rel_eps;
        _abs_eps  = ta._abs_eps;
        _max_rank = ta._max_rank;

        return *this;
    }
};

/////////////////////////////////////////////////
//
// functional ctors
//

//
// create accuracy object with fixed (relative) precision relative_eps
//
inline
accuracy
relative_prec ( const double  relative_eps )
{
    return accuracy( relative_eps, 0.0 );
}

//
// create accuracy object with fixed absolute precision absolute_eps
//
inline
accuracy
absolute_prec ( const double  absolute_eps )
{
    return accuracy( 0.0, absolute_eps );
}

/////////////////////////////////////////////////
//
// per block adaptive accuracy
//
struct adaptive_accuracy : public accuracy
{
    adaptive_accuracy ( const double  abs_eps )
            : accuracy( 0.0, abs_eps )
    {}
    
    virtual const accuracy  acc ( const indexset &  rowis,
                                  const indexset &  colis ) const
    {
        return absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
    }
};

}// namespace hlrcompress

#endif  // __HLRCOMPRESS_APPROX_ACCURACY_HH
