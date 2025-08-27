    SUBROUTINE emout(ROFILE,LUN)
      use cglow,only: jmax,idate
      use cglow,only: ut,glat,glong,f107,f107p,f107a
      use cglow,only: zz,aglw,sza,dip,xuvfac,ztn,zti,zte,zo,zo2,zns,zn2,ecalc,zxden,zeta

      implicit none
      
      integer,intent(in) :: lun
      character(len=40),intent(in) :: rofile
      integer :: j, ii

!      write(6, "(' lun, rofile ')") lun, "  ", rofile      
      open(unit=lun,file=rofile,status='unknown')

      write(lun,"('   Z      3371    4278    5200    5577    6300    7320   10400    3644    7774    8446    3726    LBH     1356    1493    1304')")
      write(lun,"(1x,f10.1,15f10.2)")(zz(j),(zeta(ii,j),ii=1,15),j=1,jmax)

      close(lun)
      return

    end subroutine emout
