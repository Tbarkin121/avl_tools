 ---------------------------------------------------------------
 Vortex Lattice Output -- Total Forces

 Configuration: atail_100g                                                  
     # Surfaces =   4
     # Strips   =  60
     # Vortices = 600

  Sref = 0.70000E-01   Cref = 0.80000E-01   Bref = 0.85000    
  Xref = 0.77403E-01   Yref =  0.0000       Zref = 0.36177E-02

 Standard axis orientation,  X fwd, Z down         

 Run case: level flight, Xcg=3.0                   

  Alpha =  -4.41597     pb/2V =  -0.00000     p'b/2V =  -0.00000
  Beta  =   0.00000     qc/2V =   0.00000
  Mach  =     0.000     rb/2V =  -0.00000     r'b/2V =  -0.00000

  CXtot =  -0.03227     Cltot =  -0.00000     Cl'tot =  -0.00000
  CYtot =  -0.00000     Cmtot =  -0.00000
  CZtot =  -0.09781     Cntot =   0.00000     Cn'tot =   0.00000

  CLtot =   0.10000
  CDtot =   0.02464
  CDvis =   0.02000     CDind = 0.0046427
  CLff  =   0.10231     CDff  = 0.0044747    | Trefftz
  CYff  =  -0.00000         e =    0.0721    | Plane  

   ruddervator     =  -2.00633

 ---------------------------------------------------------------

 Stability-axis derivatives...

                             alpha                beta
                  ----------------    ----------------
 z' force CL |    CLa =   5.284239    CLb =  -0.000000
 y  force CY |    CYa =   0.000000    CYb =  -0.137010
 x' mom.  Cl'|    Cla =  -0.000000    Clb =  -0.000742
 y  mom.  Cm |    Cma =  -0.479110    Cmb =   0.000000
 z' mom.  Cn'|    Cna =   0.000000    Cnb =   0.023026

                     roll rate  p'      pitch rate  q'        yaw rate  r'
                  ----------------    ----------------    ----------------
 z' force CL |    CLp =  -0.000000    CLq =   7.858454    CLr =  -0.000000
 y  force CY |    CYp =   0.007961    CYq =   0.000000    CYr =   0.060623
 x' mom.  Cl'|    Clp =  -0.516166    Clq =   0.000000    Clr =   0.037650
 y  mom.  Cm |    Cmp =   0.000000    Cmq =  -4.444967    Cmr =  -0.000000
 z' mom.  Cn'|    Cnp =  -0.011077    Cnq =  -0.000000    Cnr =  -0.012533

                  ruddervator  d01 
                  ----------------
 z' force CL |   CLd01 =   0.006726
 y  force CY |   CYd01 =   0.000000
 x' mom.  Cl'|   Cld01 =  -0.000000
 y  mom.  Cm |   Cmd01 =  -0.012986
 z' mom.  Cn'|   Cnd01 =  -0.000000
 Trefftz drag| CDffd01 =  -0.000746
 span eff.   |    ed01 =   0.021210



 Neutral point  Xnp =   0.084657

 Clb Cnr / Clr Cnb  =   0.010731    (  > 1 if spirally stable )

CG = 0.0774
Cref = 0.80000E-01

staticMargin = 0.1 = (x_NP - x_CG)/c_ref <=> x_CG = x_NP - 0.1 * c_ref
staticMargin = (0.084657 - 0.0774)/0.08 = 0.0907

Close enough... 

x_CG = 0.084657 - 0.1 * 0.08 = 0.07667