#!/usr/bin/env python
#
#       math.py
#       
#       Copyright 2010 Enrico Avventi <avventi@Lonewolf>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License version 2 as 
#       published by the Free Software Foundation.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

from slycot import _wrapper
import numpy as np

def mc01td(dico,dp,p):
    """ dp,stable,nz = mc01td(dico,dp,p)
    
    To determine whether or not a given polynomial P(x) with real
    coefficients is stable, either in the continuous-time or discrete-
    time case.

    A polynomial is said to be stable in the continuous-time case
    if all its zeros lie in the left half-plane, and stable in the
    discrete-time case if all its zeros lie inside the unit circle.

    
    Required arguments:
        dico : input string(len=1)
            Indicates whether the stability test to be applied to P(x) is in 
            the continuous-time or discrete-time case as follows:
            = 'C':  continuous-time case;
            = 'D':  discrete-time case.
        dp : input int
            The degree of the polynomial P(x).  dp >= 0.
        p : input rank-1 array('d') with bounds (dp + 1)
            This array must contain the coefficients of P(x) in increasing 
            powers of x.
    Return objects:
        dp : int
            If P(dp+1) = 0.0 on entry, then dp contains the index of the highest 
            power of x for which P(dp+1) <> 0.0.
        stable : int
            Equal to 1 if P(x) if stable, 0 otherwise.
        nz : int
            The number of unstable zeros.
    """
    hidden = ' (hidden by the wrapper)' 
    arg_list = ['dico', 'dp', 'P', 'stable', 'nz', 'DWORK', 'IWARN'+hidden, 
        'INFO'+hidden]
    out = _wrapper.mc01td(dico,dp,p)
    if out[-1] < 0:
        error_text = "The following argument had an illegal value: "+arg_list[-out[-1]-1]
        e = ValueError(error_text)
        e.info = out[-1]
        raise e
    if out[-1] == 1:
        warings.warn('entry P(x) is the zero polynomial.')
    if out[-1] == 2:
        warings.warn('P(x) may have zeros very close to stability boundary.')
    if out[-2] > 0:
        warnings.warn('The degree of P(x) has been reduced to %i' %(dp-k))
    return out[:-2]


def mb03vd(a,n=None,ilo=1,ihi=None,ldtau=None):
    """ a,tau = mb03vd(a,n=None,ilo=1,ihi=None,ldtau=None)
    
      
    To reduce a product of p real general matrices A = A_1*A_2*...*A_p
    to upper Hessenberg form, H = H_1*H_2*...*H_p, where H_1 is
    upper Hessenberg, and H_2, ..., H_p are upper triangular, by using
    orthogonal similarity transformations on A,

          Q_1' * A_1 * Q_2 = H_1,
          Q_2' * A_2 * Q_3 = H_2,
                 ...
          Q_p' * A_p * Q_1 = H_p.
          
    Required arguments
    ------------------
        a : list of matrices
            The matrices of factors to be reduced. The length of this list is p.
        
        
    Optional arguments
    ------------------
        n : int
            The order of the square matrices A_1, A_2, ..., A_p., n >= 0
            
        ilo : int
        ihi : int
              It is assumed that all matrices A_j, j = 2, ..., p, are
              already upper triangular in rows and columns 1:ILO-1 and
              IHI+1:N, and A_1 is upper Hessenberg in rows and columns
              1:ILO-1 and IHI+1:N, with A_1(ILO,ILO-1) = 0 (unless
              ILO = 1), and A_1(IHI+1,IHI) = 0 (unless IHI = N).
              If this is not the case, ILO and IHI should be set to 1
              and N, respectively.
              1 <= ILO <= max(1,N); min(ILO,N) <= IHI <= N.
              
        ldtau : int
                The leading dimension of the array tau.
                LDTAU >= max(1,N-1).
                    
                The leading N-1 elements in the j-th column contain the
                scalar factors of the elementary reflectors used to form
                the matrix Q_j, j = 1, ..., P. See FURTHER COMMENTS.
    
    Returns
    -------
        h : List of matrices
            The leading N-by-N upper triangle and the first
            subdiagonal of A(*,*,1) contain the upper Hessenberg
            matrix H_1, and the elements below the first subdiagonal,
            with the first column of the array TAU represent the
            orthogonal matrix Q_1 as a product of elementary
            reflectors. See FURTHER COMMENTS.
            For j > 1, the leading N-by-N upper triangle of A(*,*,j)
            contains the upper triangular matrix H_j, and the elements
            below the diagonal, with the j-th column of the array TAU
            represent the orthogonal matrix Q_j as a product of
            elementary reflectors. See FURTHER COMMENTS.
            
        tau : matrix ldtau-by-p
              The leading N-1 elements in the j-th column contain the
              scalar factors of the elementary reflectors used to form
              the matrix Q_j, j = 1, ..., P. See FURTHER COMMENTS.
          
    Further comments
    ----------------
      Each matrix Q_j is represented as a product of (ihi-ilo)
      elementary reflectors,

         Q_j = H_j(ilo) H_j(ilo+1) . . . H_j(ihi-1).

      Each H_j(i), i = ilo, ..., ihi-1, has the form

         H_j(i) = I - tau_j * v_j * v_j',

      where tau_j is a real scalar, and v_j is a real vector with
      v_j(1:i) = 0, v_j(i+1) = 1 and v_j(ihi+1:n) = 0; v_j(i+2:ihi)
      is stored on exit in A_j(i+2:ihi,i), and tau_j in TAU(i,j).

      The contents of A_1 are illustrated by the following example
      for n = 7, ilo = 2, and ihi = 6:

      on entry                         on exit

      ( a   a   a   a   a   a   a )    ( a   h   h   h   h   h   a )
      ( 0   a   a   a   a   a   a )    ( 0   h   h   h   h   h   a )
      ( 0   a   a   a   a   a   a )    ( 0   h   h   h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  h   h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  v3  h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  v3  v4  h   h   h )
      ( 0   0   0   0   0   0   a )    ( 0   0   0   0   0   0   a )

      where a denotes an element of the original matrix A_1, h denotes
      a modified element of the upper Hessenberg matrix H_1, and vi
      denotes an element of the vector defining H_1(i).

      The contents of A_j, j > 1, are illustrated by the following
      example for n = 7, ilo = 2, and ihi = 6:

      on entry                         on exit

      ( a   a   a   a   a   a   a )    ( a   h   h   h   h   h   a )
      ( 0   a   a   a   a   a   a )    ( 0   h   h   h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  h   h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  v3  h   h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  v3  v4  h   h   h )
      ( 0   a   a   a   a   a   a )    ( 0   v2  v3  v4  v5  h   h )
      ( 0   0   0   0   0   0   a )    ( 0   0   0   0   0   0   a )

      where a denotes an element of the original matrix A_j, h denotes
      a modified element of the upper triangular matrix H_j, and vi
      denotes an element of the vector defining H_j(i). (The element
      (1,2) in A_p is also unchanged for this example.)

      Note that for P = 1, the LAPACK Library routine DGEHRD could be
      more efficient on some computer architectures than this routine
      (a BLAS 2 version).
      
      
    Example
    -------
        
    >>> import numpy as np
    >>> import slycot
    >>> 
    >>> n = 5
    >>> p = 3
    >>> 
    >>> np.random.seed(1)
    >>> 
    >>> 
    >>> print "n = %d, p = %d" % (5,3)
    >>> 
    >>> #Generate system dynamics
    >>> As = [np.random.random((n,n)) for i in range(p)]
    >>> 
    >>> # The magic slicot routine
    >>> a,tau = slycot.mb03vd(As)
    >>> 
    >>> Qs = slycot.mb03vy(a,tau)
    >>> 
    >>> print "Q_1..Q_p"
    >>> for q in Qs:
    >>>   print q
    >>>   # Assert Qs are orthonormal
    >>>   assert(np.max(np.fabs(np.eye(n)-np.dot(q.T,q)))<1e-8)
    >>>   
    >>> 
    >>> print "H_1..H_p"
    >>> # Construct H from Q*A*Q  and check for consistency
    >>> for i,(ql,A, qr) in enumerate(zip(Qs,As,Qs[1:]+[Qs[0]])):
    >>>   H_s = np.array(a[i])
    >>>   if i==0:
    >>>     for i in range(1,n):
    >>>       for j in range(i-1):
    >>>         H_s[i,j] = 0
    >>>   else:
    >>>     for i in range(n):
    >>>       for j in range(i):
    >>>         H_s[i,j] = 0
    >>>     
    >>>   print H_s
    >>>   
    >>>   H = np.dot(np.dot(ql.T,A),qr)
    >>>   
    >>>   assert(np.max(np.fabs(H-H_s))<1e-8)
    >>>   
      
    """
    
    if isinstance(a,list):
      a=np.dstack(a)
      
    if n is None:  n = a.shape[0]
    
    if ihi is None: ihi = n
    if ldtau is None: ldtau = n-1
    lda1,lda2,p = a.shape
    
    hidden = ' (hidden by the wrapper)' 
    arg_list = ['n','p','ilo','ihi','a','lda1','lda2','tau','ldtau','INFO'+hidden]
    out = _wrapper.mb03vd(n,p,ilo,ihi,a,lda1,lda2,ldtau)
    
    
    if out[-1] < 0:
        error_text = "The following argument had an illegal value: "+arg_list[-out[-1]-1]
        e = ValueError(error_text)
        e.info = out[-1]
        raise e
    return ([out[0][:,:,i] for i in range(p)],) + out[1:-1]


def mb03vy(a,tau,n=None,ilo=1,ihi=None):
    """ q = mb03vy(a,tau,n=None,ilo=0,ihi=None)
    
    To generate the real orthogonal matrices Q_1, Q_2, ..., Q_p,
    which are defined as the product of ihi-ilo elementary reflectors
    of order n, as returned by SLICOT Library routine MB03VD:

      Q_j = H_j(ilo) H_j(ilo+1) . . . H_j(ihi-1).

    Required arguments
    ------------------
        a : List of  matrices
            The leading N-by-N strictly lower triangular
            part of A(*,*,j) must contain the vectors which define the
            elementary reflectors used for reducing A_j, as returned
            by SLICOT Library routine MB03VD, j = 1, ..., p.
            On exit, the leading N-by-N part of A(*,*,j) contains the
            N-by-N orthogonal matrix Q_j, j = 1, ..., p.
          
        tau : matrix
              The leading N-1 elements in the j-th column must contain
              the scalar factors of the elementary reflectors used to
              form the matrix Q_j, as returned by SLICOT Library routine
              MB03VD.
      
    Optional arguments
    ------------------
        n : integer
            The order of the matrices Q_1, Q_2, ..., Q_p.  N >= 0.
      
        
        ilo : integer
        ihi : integer
              The values of the indices ilo and ihi, respectively, used
              in the previous call of the SLICOT Library routine MB03VD.
              1 <= ILO <= max(1,N); min(ILO,N) <= IHI <= N.
      
    Example
    -------
      See mb03vd
    
      
    """
    
    if isinstance(a,list):
      a=np.dstack(a)
    
    if n is None:  n = a.shape[0]  
    if ihi is None: ihi = n
  
    lda1,lda2,p = a.shape
    ldtau = tau.shape[0]
    
    ldwork = 4*n
    
    hidden = ' (hidden by the wrapper)' 
    arg_list = ['n', 'p', 'ilo', 'ihi', 'a', 'lda1', 'lda2', 'tau', 'ldtau', 'ldwork','INFO'+hidden]
    out = _wrapper.mb03vy(n, p, ilo, ihi, a, lda1, lda2, tau, ldtau, ldwork)
    
    
    if out[-1] < 0:
        error_text = "The following argument had an illegal value: "+arg_list[-out[-1]-1]
        e = ValueError(error_text)
        e.info = out[-1]
        raise e
    return [out[0][:,:,i] for i in range(p)]


def mb03wd(h, z=None,n = None, job="S", compz="I", ilo=1, ihi=None, iloz=1, ihiz=None, ldtau=None, ldwork=None):
    """ t, z, wr, wi = mb03wd(h,z=None,n=None, job="S", compz="I", ilo=1, ihi=None, iloz=1, ihiz=None, ldtau=None, ldwork=None
    
    Required arguments
    ------------------
        h : List of  matrices
            The leading N-by-N part of H(*,*,1) must contain
            the upper Hessenberg matrix H_1 and the leading N-by-N
            part of H(*,*,j) for j > 1 must contain the upper
            triangular matrix H_j, j = 2, ..., p.
          
    
    Optional arguments
    ------------------
        n : integer
            The order of the matrix H.  n >= 0
    
        job : {'E','S'}
              Indicates whether the user wishes to compute the full
              Schur form or the eigenvalues only, as follows:
              = 'E':  Compute the eigenvalues only;
              = 'S':  Compute the factors T_1, ..., T_p of the full
                      Schur form, T = T_1*T_2*...*T_p.

        compz : {'N','I','V'}
              Indicates whether or not the user wishes to accumulate
              the matrices Z_1, ..., Z_p, as follows:
              = 'N':  The matrices Z_1, ..., Z_p are not required;
              = 'I':  Z_i is initialized to the unit matrix and the
                      orthogonal transformation matrix Z_i is returned,
                      i = 1, ..., p;
              = 'V':  Z_i must contain an orthogonal matrix Q_i on
                      entry, and the product Q_i*Z_i is returned,
                      i = 1, ..., p.

  
        p : integer
            The number of matrices in the product H_1*H_2*...*H_p.
            P >= 1.
    
    
        ilo : integer
        ihi : integer
              It is assumed that all matrices H_j, j = 2, ..., p, are
              already upper triangular in rows and columns 1:ILO-1 and
              IHI+1:N, and H_1 is upper quasi-triangular in rows and
              columns 1:ILO-1 and IHI+1:N, with H_1(ILO,ILO-1) = 0
              (unless ILO = 1), and H_1(IHI+1,IHI) = 0 (unless IHI = N).
              The routine works primarily with the Hessenberg submatrix
              in rows and columns ILO to IHI, but applies the
              transformations to all the rows and columns of the
              matrices H_i, i = 1,...,p, if JOB = 'S'.
              1 <= ILO <= max(1,N); min(ILO,N) <= IHI <= N.
              
        iloz : integer
        ihiz : integer
               Specify the rows of Z to which the transformations must be
               applied if COMPZ = 'I' or COMPZ = 'V'.
               1 <= ILOZ <= ILO; IHI <= IHIZ <= N.


            
        z: List of matrices
           If COMPZ = 'V', the leading N-by-N-by-P part of
           this array must contain the current matrix Q of
           transformations accumulated by SLICOT Library routine
           MB03VY.
           If COMPZ = 'I', Z need not be set on entry.


    Returns
    -------
          
        t : list of triangular matrices         
            if JOB = 'S', the leading N-by-N part of T(*,*,1)
            is upper quasi-triangular in rows and columns ILO:IHI,
            with any 2-by-2 diagonal blocks corresponding to a pair of
            complex conjugated eigenvalues, and the leading N-by-N
            part of T(*,*,j) for j > 1 contains the resulting upper
            triangular matrix T_j.
            If JOB = 'E', the contents of T are unspecified
            
        z:  list of orthonormal matrices 
            If COMPZ = 'V', or COMPZ = 'I', the leading
            N-by-N-by-P part of this array contains the transformation
            matrices which produced the Schur form; the
            transformations are applied only to the submatrices
            Z_j(ILOZ:IHIZ,ILO:IHI), j = 1, ..., P.
            If COMPZ = 'N', Z is not referenced.

        wr :  array of length n
        wi :  array of length n
              The real and imaginary parts, respectively, of the
              computed eigenvalues ILO to IHI are stored in the
              corresponding elements of WR and WI. If two eigenvalues
              are computed as a complex conjugate pair, they are stored
              in consecutive elements of WR and WI, say the i-th and
              (i+1)th, with WI(i) > 0 and WI(i+1) < 0. If JOB = 'S', the
              eigenvalues are stored in the same order as on the
              diagonal of the Schur form returned in H.
              
    Example
    -------
      
    >>> import numpy as np
    >>> import slycot
    >>> 
    >>> n = 5
    >>> p = 3
    >>> 
    >>> np.random.seed(1)
    >>> print "n = %d, p = %d" % (5,3)
    >>> 
    >>> #Generate system dynamics
    >>> As = [np.random.random((n,n)) for i in range(p)]
    >>> 
    >>> a,tau = slycot.mb03vd(As)
    >>> Qs = slycot.mb03vy(a,tau)
    >>> 
    >>> t,z,wr,wi = slycot.mb03wd(a,z=Qs,compz="V")
    >>> 
    >>> print "T1..Tp"
    >>> print t
    >>> 
    >>> print "Z1..Zp"
    >>> print z
    >>> 
    >>> # Construct T from Q*A*Q  and check for consistency
    >>> for i,(zl,A, zr) in enumerate(zip(z,As,z[1:]+[z[0]])):
    >>>   T_s = np.array(t[i])
    >>>   
    >>>   T = np.dot(np.dot(zl.T,A),zr)
    >>>   
    >>>   print T
    >>>   
    >>>   assert(np.max(np.fabs(T-T_s))<1e-8)
    >>>   
                  
    """

    if isinstance(h,list):
      h=np.dstack(h)
    if isinstance(z,list):
      z=np.dstack(z) 
    
    if n is None:  n = h.shape[0]
        
    if ihi is None: ihi = n
    if ldtau is None: ldtau = n-1
    if ihiz is None: ihiz = n
    ldh1,ldh2,p = h.shape

    if ldwork is None:
      ldwork = ihi-ilo+p-1
      
    if z is None:
      if compz=="I":
        z = np.zeros((n,n,p))
      else :
        raise Exception("Z is required for compz N or V")
    
    ldz1,ldz2 = z.shape[:2]
    
    hidden = ' (hidden by the wrapper)' 
    arg_list = ['job', 'compz', 'n', 'p', 'ilo', 'ihi', 'iloz', 'ihiz', 'h', 'ldh1', 'ldh2', 'Z', 'ldz1', 'ldz2', 'ldwork', 'INFO'+hidden]
    print "h", type(h), h.shape
    out = _wrapper.mb03wd(job, compz, n, p, ilo, ihi, iloz, ihiz, h, ldh1, ldh2, z, ldz1, ldz2, ldwork)
    
    
    if out[-1] < 0:
        error_text = "The following argument had an illegal value: "+arg_list[-out[-1]-1]
        e = ValueError(error_text)
        e.info = out[-1]
        raise e
    return ([out[0][:,:,i] for i in range(p)],) + ([out[1][:,:,i] for i in range(p)],) + out[2:-1]




# to be replaced by python wrappers
