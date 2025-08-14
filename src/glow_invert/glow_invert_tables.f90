program glow_invert_tables

! Adapted from glowdriver by Stan Solomon, 2/2016

! Runs through a range of Q and Ec to create height integrated
! intensities of each of four emission lines:
! 4278, 5577, 6300, 8446  
! Uses MSIS/IRI for input.
! Runs GLOW for designated inputs multiple times.
! MPI and netCDF libraries not required.

! 2023-08-31 update to take
! For definitions of use-associated variables, see subroutine GLOW and module CGLOW.

! Other definitions:
! f107p   Solar 10.7 cm flux for previous day
! ap      Ap index of geomagnetic activity
! z       altitude array, km

! Array dimensions:
! jmax    number of altitude levels
! nbins   number of energetic electron energy bins
! lmax    number of wavelength intervals for solar flux
! nmaj    number of major species
! nst     number of states produced by photoionization/dissociation
! nei     number of states produced by electron impact
! nex     number of ionized/excited species
! nw      number of airglow emission wavelengths
! nc      number of component production terms for each emission

  use cglow,only: jmax,nbins,lmax,nmaj,nei,nex,nw,nc,nst
  use cglow,only: idate,ut,glat,glong,f107a,f107,f107p,ap,ef,ec
  use cglow,only: iscale,jlocal,kchem,xuvfac
  use cglow,only: sza,dip,efrac,ierr
  use cglow,only: zz,zo,zn2,zo2,zns,znd,zno,ztn,ze,zti,zte
  use cglow,only: ener,del,phitop,wave1,wave2,sflux,pespec,sespec,uflx,dflx,sion
  use cglow,only: photoi,photod,phono,aglw,tei,tpi,tir,ecalc,zxden,zeta,zceta,zlbh
  use cglow,only: cglow_init
  use cglow,only: data_dir

  implicit none

  character(len=1024) :: iri90_dir
  character(len=1024) :: out_dir

  real,allocatable :: z(:)                    ! glow height coordinate in km (jmax)
  real,allocatable :: zun(:), zvn(:)          ! neutral wind components (not in use)
  real,allocatable :: pedcond(:), hallcond(:) ! Pederson and Hall conductivities in S/m (mho)
  real,allocatable :: outf(:,:)               ! iri output (11,jmax)
  real,allocatable :: ped2d(:,:,:), hall2d(:,:,:), edens3d(:,:,:)
  real :: rz12,stl,fmono,emono
  real :: d(8), t(2), sw(25), oarr(30)
  integer :: l,j,jj,ijf,jmag,iday,mmdd,i,ii,n,k,ix,itail
  integer :: instance,iostatus
  logical :: jf(12)
  data sw/25*1./
!------------
  ! variables for the tables
  integer, parameter :: nec = 396, nq = 501
  integer, parameter :: nec0 = 200, nec1 = 150, nec2 = 46
  integer :: iec, iq, iut
  real, parameter :: ec0 = 2.5, ec1 = 510, ec2 = 2500
  real, parameter :: q0 = 0.5
  real, parameter :: dec0 = 2.5, dec1 = 10, dec2 = 500
  real, parameter :: dq = 0.5
  real :: ecvec( nec ), qvec( nq )
  real, allocatable :: B4278(:,:), B5577(:,:), B6300(:,:), B8446(:,:)
  real, allocatable :: eta4278(:,:,:), eta5577(:,:,:), eta6300(:,:,:), eta8446(:,:,:)
  real, allocatable :: dzz( : )
  ! static MSIS and ionsophere parameters for now
  real, parameter :: inglat = 65, inglong = -147
  real, parameter :: inf107 = 141.9, inf107A = 158.2, inf107p = 138.2, inap = 17.0
  integer, parameter :: indate = 23073, inut = 24300
  ! variables for timing
  real :: clock_rate, outparams(20)
  integer :: t1, t2, clock_max
  character(len=1024) :: fout
  ! Make the vectors
  DO iec = 1, nec0, 1
     ecvec( iec ) = (iec - 1) * dec0 + ec0
  ENDDO

  DO iec = nec0 + 1 , nec0 + nec1, 1
     ecvec( iec ) = (iec - nec0) * dec1 + ec1
  ENDDO

  DO iec = nec0 + nec1 + 1, nec0 + nec1 + nec2, 1
     ecvec( iec ) = (iec - (nec0 + nec1 + 1) ) * dec2 + ec2
  ENDDO

  qvec( 1 ) = 0.25
  DO iq  = 2,  nq, 1
     qvec ( iq ) = (iq - 2) * dq + q0
  ENDDO
!
! Initialize standard switches:
!
  iscale=1
  xuvfac=3.
  kchem=4
  jlocal=0
  itail=0
  fmono=0.
  emono=0.
!
! Set data directories:
!
  data_dir    = 'data/'
  iri90_dir   = 'data/iri90/'
!
! Set number of altitude levels:
!
  jmax = 102
!
! Allocate local arrays:
!
  allocate(z(jmax))
  allocate(zun(jmax))
  allocate(zvn(jmax))
  allocate(pedcond(jmax))
  allocate(hallcond(jmax))
  allocate(outf(11,jmax))
  allocate(dzz(jmax))
  allocate(B4278( nq, nec))
  allocate(B5577( nq, nec))
  allocate(B6300( nq, nec))
  allocate(B8446( nq, nec))
  allocate(eta4278( nq, nec, jmax))
  allocate(eta5577( nq, nec, jmax))
  allocate(eta6300( nq, nec, jmax))
  allocate(eta8446( nq, nec, jmax))
  allocate(hall2d(nq, nec, jmax))
  allocate(ped2d( nq, nec, jmax ))
  allocate(edens3d( nq, nec, jmax ))
!  ut = inut
!  idate = indate
!  glat = inglat
!  glong = inglong
!  f107 = inf107
!  f107a = inf107a
!  f107p = inf107p
!  ap = inap
!
! Call CGLOW_INIT (module CGLOW) to set array dimensions and allocate use-associated variables:
! (This was formerly done using common blocks, including common block /cglow/.)
!
  call cglow_init
!
! Call EGRID to set up electron energy grid:
!
  call egrid (ener, del, nbins)
!
! Loop to call GLOW for designated inputs until end-of-file or any character on standard input:
!
! set the time tick for timing
  call system_clock( t1, clock_rate, clock_max ) 
!
! Get input values:
!
!    write(6,"('Enter date, UT, lat, lon, F107a, F107, F107p, Ap, Ef, Ec')")
!    read(5,*,iostat=iostatus) idate,ut,glat,glong,f107a,f107,f107p,ap
    read(5,*,iostat=iostatus) idate,ut,glat,glong,f107a,f107,f107p,ap,out_dir
    if (iostatus /= 0) stop
    iut = INT( ut )
    outparams(1) = FLOAT( idate )
    outparams(2) = ut
    outparams(3) = glat
    outparams(4) = glong
    outparams(5) = f107a
    outparams(6) = f107
    outparams(7) = f107p
    outparams(8) = ap
    do i = 9, 20, 1
       outparams(i) = 0.0
    enddo
! Calculate local solar time:
  !
  stl = ut/3600. + glong/15.
  if (stl < 0.) stl = stl + 24.
  if (stl >= 24.) stl = stl - 24.
  !
  ! Call MZGRID to use MSIS/NOEM/IRI inputs on default altitude grid:
  !
  call mzgrid (jmax,nex,idate,ut,glat,glong,stl,f107a,f107,f107p,ap,iri90_dir, &
       z,zo,zo2,zn2,zns,znd,zno,ztn,zun,zvn,ze,zti,zte,zxden)
  !
  ! Call MAXT to put auroral electron flux specified by namelist input into phitop array:
  !
  do iec = 1, nec, 1
     IF ( MOD(iec,100) == 0 ) THEN
        open( unit=41, file='invert_table_stat_v3.txt', status='replace', action='write')
        WRITE(41, '(I5.5)') iec
        close(41)
     ENDIF
     write(*,*) iec, 'of', nec
     
     do iq = 1, nq, 1
        ef = qvec( iq )
        ec = ecvec( iec )
        
        phitop(:) = 0.
        if (ef>.001 .and. ec>1.) call maxt (ef,ec,ener,del,nbins,itail,fmono,emono,phitop)
        !
! Fill altitude array, converting to cm:
!
        zz(:) = z(:) * 1.e5     ! km to cm at all jmax levels

        do j = 1, jmax - 1
           dzz( j ) = zz( j + 1 ) - zz( j )
        enddo
        dzz( jmax ) = dzz( jmax - 1 )
!
! Call GLOW to calculate ionized and excited species, airglow emission rates,
! and vertical column brightnesses:
!
        call glow
        !
        ! Call CONDUCT to calculate Pederson and Hall conductivities:
        !
        do j=1,jmax
           call conduct (glat, glong, z(j), zo(j), zo2(j), zn2(j), &
                zxden(3,j), zxden(6,j), zxden(7,j), ztn(j), zti(j), zte(j), &
                pedcond(j), hallcond(j))
        enddo
        !
        ! Output section:
        !
        !    write(6,"(1x,i7,9f8.1)") idate,ut,glat,glong,f107a,f107,f107p,ap,ef,ec
        !    write(6,"('   Z     Tn       O        N2        NO      Ne(in)    Ne(out)  Ionrate      O+       O2+      NO+       N(2D)    Pederson   Hall')")
        !    write(6,"(1x,0p,f5.1,f6.0,1p,12e10.2)") (z(j),ztn(j),zo(j),zn2(j),zno(j),ze(j), &
        !      ecalc(j),tir(j),zxden(3,j),zxden(6,j),zxden(7,j),zxden(10,j),pedcond(j),hallcond(j),j=1,jmax)
        !    write(6,"('   Z      3371    4278    5200    5577    6300    7320   10400    3644    7774    8446    3726    LBH     1356    1493    1304')")
        !    write(6,"(1x,f5.1,15f9.2)")(z(j),(zeta(ii,j),ii=1,15),j=1,jmax)
        
        ! Calculate brightnesses (rayleighs)
        
        B4278( iq, iec ) = SUM( zeta( 2, : ) * dzz( : ) ) / 1.0E6
        B5577( iq, iec ) = SUM( zeta( 4, : ) * dzz( : ) ) / 1.0E6
        B6300( iq, iec ) = SUM( zeta( 5, : ) * dzz( : ) ) / 1.0E6
        B8446( iq, iec ) = SUM( zeta( 9, : ) * dzz( : ) ) / 1.0E6

        eta4278(iq, iec, : ) = zeta( 2, : )
        eta5577(iq, iec, : ) = zeta( 4, : )
        eta6300(iq, iec, : ) = zeta( 5, : )
        eta8446(iq, iec, : ) = zeta( 9, : )
        
        hall2d( iq, iec, : ) = hallcond(: )
        ped2d ( iq, iec, : ) = pedcond( : )
        edens3d(iq, iec, : ) = ecalc( : )
        !    call emout("data/emout.txt",28)
     enddo
  enddo
  
! write the brigthness files out.
  !write( fout, "('output/v3/I4278_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'I4278_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=42, file=trim(fout), access='stream', &
       status='replace', action='write')
  WRITE(42) outparams
  WRITE(42) FLOAT( nq )
  WRITE(42) FLOAT( nec )
  write(42) qvec
  write(42) ecvec
  write(42) B4278
  CLOSE(42)

  !write( fout, "('output/v3/I5577_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'I5577_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=55, file=trim(fout), access='stream', &
       status='replace', action='write')
  write(55) outparams
  write(55) float(nq)
  write(55) float(nec)
  write(55) qvec
  write(55) ecvec
  write(55) B5577
  CLOSE(55)

  !write( fout, "('output/v3/I6300_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'I6300_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=63, file=trim(fout), access='stream', &
       status='replace', action='write')
  write(63) outparams
  write(63) float(nq)
  write(63) float(nec)
  write(63) qvec
  write(63) ecvec
  write(63) B6300
  CLOSE(63)

  !write( fout, "('output/v3/I8446_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'I8446_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=84, file=trim(fout), access='stream', &
       status='replace', action='write')
  write(84) outparams
  write(84) float(nq)
  write(84) float(nec)
  write(84) qvec
  write(84) ecvec
  write(84) B8446
  CLOSE(84)

  !write( fout, "('output/v3/eta4278_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'eta4278_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=142, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(142) outparams
  write(142) float(nq)
  write(142) float(nec)
  write(142) float(jmax)
  write(142) qvec
  write(142) ecvec
  write(142) zz
  write(142) (eta4278( : , :, j ), j=1,jmax)
  CLOSE(142)

  !write( fout, "('output/v3/eta5577_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'eta5577_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=155, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(155) outparams
  write(155) float(nq)
  write(155) float(nec)
  write(155) float(jmax)
  write(155) qvec
  write(155) ecvec
  write(155) zz
  write(155) (eta5577( : , :, j ), j=1,jmax)
  CLOSE(155)  

  !write( fout, "('output/v3/eta6300_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'eta6300_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=163, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(163) outparams
  write(163) float(nq)
  write(163) float(nec)
  write(163) float(jmax)
  write(163) qvec
  write(163) ecvec
  write(163) zz
  write(163) (eta6300( : , :, j ), j=1,jmax)
  CLOSE(163)

  !write( fout, "('output/v3/eta8446_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'eta8446_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=184, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(184) outparams
  write(184) float(nq)
  write(184) float(nec)
  write(184) float(jmax)
  write(184) qvec
  write(184) ecvec
  write(184) zz
  write(184) (eta8446( : , :, j ), j=1,jmax)
  CLOSE(184)  

  !write( fout, "('output/v3/hall3d_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'hall3d_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=85, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(85) outparams
  write(85) float(nq)
  write(85) float(nec)
  write(85) float(jmax)
  write(85) qvec
  write(85) ecvec
  write(85) zz
  write(85) (hall2d( : , :, j ), j=1,jmax)
  CLOSE(85)
  
  !write( fout, "('output/v3/ped3d_',I5.5,'_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'ped3d_',I5.5,'_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=86, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(86) outparams
  write(86) float(nq)
  write(86) float(nec)
  write(86) float(jmax)
  write(86) qvec
  write(86) ecvec
  write(86) zz
  write(86) (ped2d( : , :, j ), j=1,jmax)
  CLOSE(86)

  !write( fout, "('output/v3/edens3d_',I5.5, '_',I5.5,'.bin')"), idate, iut
  write( fout, "(A,'edens3d_',I5.5, '_',I5.5,'.bin')"), trim(out_dir), idate, iut
  OPEN( unit=87, file=trim(fout), access='stream', &
       status = 'replace', action='write')
  write(87) outparams
  write(87) float(nq)
  write(87) float(nec)
  write(87) float(jmax)
  write(87) qvec
  write(87) ecvec
  write(87) zz
  write(87) (edens3d( :, :, j ), j=1,jmax)
  CLOSE(87)
  
  call system_clock( t2, clock_rate, clock_max)
  
  write(*,*) 'elapsed time = ', (t2 - t1) / clock_rate
  open( unit=41, file='invert_table_stat_v3.txt', status='replace', action='write')
  WRITE(41,* ) 'elapsed time = ', (t2 - t1) / clock_rate
  close(41)
  stop
  
end program glow_invert_tables
