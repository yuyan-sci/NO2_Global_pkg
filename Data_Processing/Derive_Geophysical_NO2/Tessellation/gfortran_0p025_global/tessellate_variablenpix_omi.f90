! =========================================================================!
! Main Interface to Rob Spurr's's Tesslation Software                      !
! Xiong Liu, December 1, 2004                                              !
! Modified by Caroline Nowlan, October 26 2012: increased maximum values   !
! allowed for grid boxes and variables, and cleaned up code.               !
! Purpose: Map satellite measurements (irregular grids) to regular grids   !
!                                                                          !
! Input filename for setting up grid has contents:                         !
! nlon, nlat, irgrid_flag: dimension of regular grids; irregular grid flag !  
! infname: input data file                                                 !                   
! outfname: output file with tessellated data                              !
! max_area: max area of irregular grid box, set to dummy val for reg grid  !
!                                                                          !
! Input data file named in grid setup file has contents:                   !
! nval: number of data elements to be mapped (e.g., # of layers for ozone  !
!       profiles, 1 for a single vertical column density                   !
! ncor: number of corners to define a ground pixel (ncor must be 4)        ! 
! coords(4, 2): lon1,lat1,lon2,lat2,lon3,lat3,lon4,lat4                    !
! pixdat(1:nval) : data elements to be mapped                              !
!                                                                          !
! output:                                                                  !
!       gridded data [180W, 90S, 180E, 90N]                                !
! **** Limitation ***:                                                     !
! a pixel should be able to be arranged such that 4 corners:               !
!   1: west most  2: north most 3:  east most 4: south most                !                    
! =========================================================================! 

PROGRAM TESSELATE

  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(8.0D0)
  USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
  DOUBLE PRECISION, PARAMETER :: TINY_A = 1.0D-300
  INTEGER, PARAMETER :: IHI = HUGE(0), ILO = -HUGE(0)


! For 0.05x0.05
  INTEGER, PARAMETER :: MAXLON = 7201, MAXLON1=10801, MAXLAT = 3601
! For 0.1x01
!  INTEGER, PARAMETER :: MAXLON = 3601, MAXLON1=5401, MAXLAT = 2000
! For 0.025x0.025
! INTEGER, PARAMETER :: MAXLON = 14401, MAXLON1=21601, MAXLAT = 7201
! For 0.01x0.01
!  INTEGER, PARAMETER :: MAXLON = 36001, MAXLON1=54001, MAXLAT = 14001

  INTEGER, PARAMETER :: MVAL = 6, MAXCORN = 4, INLUN = 21, OUTLUN = 22


  INTEGER            :: nlon, nlat, nval, i, j, k, nxdim, nydim, ymin, ymax, jfinis, &
       istart1, istart2, istart3, istart4, xmin, xmax, nlon1, inerrstat, outerrstat, &
       orient, npix, ncor, fidx, lidx, nactpix

  INTEGER,        DIMENSION(MAXLON1)                :: xind, ylimit_lower, ylimit_upper
  REAL (KIND=dp), DIMENSION(MAXLAT)                 :: lats
  REAL (KIND=dp), DIMENSION(MAXLON1)                :: lons
  REAL (KIND=dp), DIMENSION(MAXLAT, MAXLON1)        :: area, carea

  REAL (KIND=dp), DIMENSION(MAXLAT, MAXLON)         :: totarea, narea
  REAL (KIND=dp), DIMENSION(MVAL, MAXLAT, MAXLON)   :: gridmean!, gridstd
  REAL (KIND=dp), DIMENSION(MVAL)                   :: pixdat
  REAL (KIND=dp), DIMENSION(MAXCORN, 2)             :: coords
  REAL (KIND=dp), DIMENSION(6, 4)                   :: side_params
  REAL (KIND=dp), DIMENSION(2)                      :: temploc
  REAL (KIND=dp)                                    :: delt_lon, delt_lat, sumarea, lim1, &
       lim2, max_area, small_area, big_area
  CHARACTER (LEN=256)                               :: gridfile, outfname, infname
  INTEGER                                           :: irgrid


  CALL GETARG(1 , gridfile)
  OPEN (UNIT = INLUN, FILE = TRIM(ADJUSTL(gridfile)), STATUS = 'OLD', IOSTAT = inerrstat)
  IF (inerrstat /= 0) THEN
     WRITE(*, *) 'Error in opening ', TRIM(ADJUSTL(gridfile)); STOP
  ENDIF
  READ (INLUN, *) nlon, nlat, irgrid
  READ (INLUN, '(A)') infname
  READ (INLUN, '(A)') outfname


  ! Check data dimension
  nlon1 = CEILING(nlon * 1.5)

  IF (nlon >= MAXLON) THEN
     WRITE(*, *) 'Dimension in longitude >= MAXLON. Please increase MAXLON!!!'; STOP
  ENDIF
  IF (nlon1 >= MAXLON1) THEN
     WRITE(*, *) 'Dimension in longitutde >= MAXLON1. Please increase MAXLON1!!!'; STOP
  ENDIF

  IF (nlat >= MAXLAT) THEN
     WRITE(*, *) 'Dimension in latitude >= MAXLAT. Please increase MAXLAT!!!';   STOP
  ENDIF


  ! Note the longitude/latitude coordinates start from -180/-90
  IF (irgrid == 0) THEN
     delt_lon = 360.0 / nlon; delt_lat = 180.0 / nlat
     max_area = delt_lon * delt_lat
     DO i = 1, nlon1 + 1           ! extra grids for dealing with across dateline
        lons(i) = -180.0 + (i - 1) * delt_lon
     ENDDO
     DO i = 1, nlat + 1
        lats(i) = -90.0 + (i - 1) * delt_lat
     ENDDO
  ELSE
     max_area = 0.0
     READ(INLUN, *) lons(1:nlon1+1)
     READ(INLUN, *) lats(1:nlat+1)
     READ(INLUN, *) max_area 
  ENDIF
  CLOSE (INLUN)


  OPEN (UNIT = INLUN, FILE = TRIM(ADJUSTL(infname)), STATUS = 'OLD', IOSTAT = inerrstat)
  IF (inerrstat /= 0) THEN
     WRITE(*, *) 'Error in opening ', TRIM(ADJUSTL(infname)); STOP
  ENDIF
  READ (INLUN, *) nval, ncor

  IF (nval > MVAL) THEN
     WRITE(*, *) 'Dimension in data points > MAXVAL. Please increase MAXVAL!!!'; STOP
  ENDIF
  IF (ncor /= MAXCORN) THEN
     WRITE(*, *) 'One pixel to be represented by 4 corners!!!'; STOP
  ENDIF

  nactpix = 0; area = 0.0; totarea = 0.0; gridmean = 0.0 !; gridstd = 0.0   ! Initialize data

  npix = 0

  DO


     ! Read coordinates and data for one ground pixel
     READ(INLUN, *, IOSTAT=inerrstat) ((coords(j, k), k = 1, 2), j = 1, 4), pixdat(1:nval)

     ! Check for end of data
     IF (inerrstat < 0) EXIT

     npix = npix + 1

     ! Convert longitude to [-180, 180]
     WHERE(coords(:, 1) > 180.0) 
        coords(:, 1) = coords(:, 1) - 360.0
     ENDWHERE

     ! check for special case: cross date line
     IF (MAXVAL(coords(:, 1)) - MINVAL(coords(:, 1)) > 180.) THEN
        WHERE (coords(:, 1) < 0) 
           coords(:, 1) = coords(:, 1) + 360.0
        ENDWHERE
     ENDIF

     IF (MAXVAL(coords(:, 2)) >= lats(nlat+1)  .OR. MINVAL(coords(:, 2)) <= lats(1)) CYCLE

     !Exact coordinate values not allowed, make small displacement ???
     DO j = 1, ncor
        IF (ANY(coords(j, 1) - lons(1:nlon1+1) == 0.) ) THEN
           coords(j, 1) = coords(j, 1) + 1.0D-3
           IF (coords(j, 1) == lons(nlon1+1)) coords(j, 1) = coords(j, 1) - 1.0D-3
        ENDIF

        IF (ANY(coords(j, 2) - lats(1:nlat+1) == 0.) ) THEN
           coords(j, 2) = coords(j, 2) + 1.0D-3
           IF (coords(j, 2) == lats(nlat+1)) coords(j, 2) = coords(j, 2) - 1.0D-3
        ENDIF
     ENDDO

     ! Need to sort the four corners, so that
     ! lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4
     ! 1: most west 3: most east, 2: most top 4: most bottom of a pixel
     fidx = MINVAL(MINLOC(coords(:, 1))); lidx = MAXVAL(MAXLOC(coords(:, 1)))

     IF (fidx /= 1) THEN
        temploc = coords(1, :); coords(1, :) = coords(fidx, :); coords(fidx, :) = temploc
     ENDIF

     IF (lidx /= 3 .AND. lidx /= 1) THEN
        temploc = coords(3, :); coords(3, :) = coords(lidx, :); coords(lidx, :) = temploc
     ENDIF

     IF (coords(2, 2) < coords(4, 2)) THEN
        temploc = coords(4, :); coords(4, :) =  coords(2, :); coords(2, :) =  temploc
     ENDIF

     ! Slightly modify the corners so that they do not have the same lon and lat
     IF (coords(1, 1) >= coords(2, 1)) coords(2, 1) = coords(1, 1) + 0.01
     IF (coords(2, 1) >= coords(3, 1)) coords(3, 1) = coords(2, 1) + 0.01
     IF (coords(3, 1) <= coords(4, 1)) coords(4, 1) = coords(3, 1) - 0.01
     IF (coords(4, 1) <= coords(1, 1)) coords(1, 1) = coords(4, 1) - 0.01

     IF (coords(1, 2) >= coords(2, 2)) coords(2, 2) = coords(1, 2) + 0.01
     IF (coords(2, 2) <= coords(3, 2)) coords(3, 2) = coords(2, 2) - 0.01
     IF (coords(3, 2) <= coords(4, 2)) coords(4, 2) = coords(3, 2) - 0.01
     IF (coords(4, 2) >= coords(1, 2)) coords(1, 2) = coords(4, 2) + 0.01

     ! check for corner 2 and 4 to see if top most and bottom most
     IF (coords(2, 2) /= MAXVAL(coords(:, 2)) .OR. &
          coords(4, 2) /= MINVAL(coords(:, 2))) THEN
        PRINT *, i, ' irregular pixel '
        WRITE(*, '(8f12.4)') coords
        CYCLE
     ENDIF

     ! set up parameters to describe each side 
     CALL PARSETUP ( coords, side_params, orient )


     IF (irgrid == 0) THEN
        ! index at latitude direction (index start from 1)
        ymin = INT    ((coords(4, 2) + 90.0 ) / delt_lat) + 1 
        ymax = CEILING((coords(2, 2) + 90.0 ) / delt_lat) + 1

        ! index at longitude direction (index start from 1)
        istart1 = INT((coords(1, 1) + 180.0 ) / delt_lon) + 1
        istart2 = INT((coords(2, 1) + 180.0 ) / delt_lon) + 1
        istart3 = INT((coords(3, 1) + 180.0 ) / delt_lon) + 1
        istart4 = INT((coords(4, 1) + 180.0 ) / delt_lon) + 1
     ELSE
        IF (.NOT. IEEE_IS_FINITE(delt_lat) .OR. ABS(delt_lat) <= TINY_A) THEN
          WRITE(*,*) 'Bad delt_lat=',delt_lat,' nlat=',nlat; STOP 11
        END IF
        IF (.NOT. IEEE_IS_FINITE(delt_lon) .OR. ABS(delt_lon) <= TINY_A) THEN
          WRITE(*,*) 'Bad delt_lon=',delt_lon,' nlon=',nlon; STOP 12
        END IF
        IF (.NOT. ALL(IEEE_IS_FINITE(coords))) THEN        
          WRITE(*,*) 'coords has NaN/Inf at pixel ', npix; CYCLE
        END IF

        ! index at latitude direction (index start from 1)
        ymin = MINVAL(MAXLOC(lats(1:nlat+1), MASK = (lats(1:nlat+1) <= coords(4, 2))))
        ymax = MINVAL(MINLOC(lats(1:nlat+1), MASK = (lats(1:nlat+1) >= coords(2, 2))))

        istart1 = MINVAL(MAXLOC(lons(1:nlon1+1), MASK = (lons(1:nlon1+1) <= coords(1, 1))))
        istart2 = MINVAL(MAXLOC(lons(1:nlon1+1), MASK = (lons(1:nlon1+1) <= coords(2, 1))))
        istart3 = MINVAL(MAXLOC(lons(1:nlon1+1), MASK = (lons(1:nlon1+1) <= coords(3, 1))))
        istart4 = MINVAL(MAXLOC(lons(1:nlon1+1), MASK = (lons(1:nlon1+1) <= coords(4, 1))))
     ENDIF
     nydim  = ymax - ymin + 1
     jfinis = ymax - ymin
     xmin  = istart1; xmax = istart3 + 1
     nxdim = xmax - xmin + 1   


     ! get relative offset in longitude direction
     istart2 = istart2 - istart1 + 1
     istart3 = istart3 - istart1 + 1
     istart4 = istart4 - istart1 + 1
     istart1 = 1

     area(ymin:ymax, xmin:xmax) = 0.0
     carea(ymin:ymax, xmin:xmax) = 0.0

     CALL TESSELATE_AREAMASTER (lons(xmin:xmax), lats(ymin:ymax), nxdim, nydim,    &
          coords, side_params, orient, istart1, istart2, istart3, istart4, jfinis, &
          area(ymin:ymax, xmin:xmax), sumarea, ylimit_lower(1:nxdim),ylimit_upper(1:nxdim)) 
      IF (ANY(.NOT. IEEE_IS_FINITE(area(ymin:ymax, xmin:xmax)))) THEN
        CYCLE
      END IF


     ! extra boundary where area is not computed
     xmax = xmax - 1; ymax = ymax - 1

     DO j = xmin, xmax
        xind(j) = MOD(j, nlon)
        IF (xind(j) == 0) xind(j) = nlon
     ENDDO

     big_area   = MAXVAL(area(ymin:ymax, xmin:xmax))
     small_area = MINVAL(area(ymin:ymax, xmin:xmax))

     IF ((ANY(area(ymin:ymax, xmin:xmax) < -1.0d-2*max_area) .OR. &
          ANY(area(ymin:ymax, xmin:xmax) > max_area))) THEN
        !PRINT *, i, 'Bad output!!!', MINVAL(area(ymin:ymax, xmin:xmax)), &
        !     MAXVAL(area(ymin:ymax, xmin:xmax)), max_area
        CYCLE
     ENDIF

     WHERE (area(ymin:ymax, xmin:xmax) > 0.0)
        carea(ymin:ymax, xmin:xmax) = 1.0
     ENDWHERE

     totarea(ymin:ymax, xind(xmin:xmax)) = totarea(ymin:ymax, xind(xmin:xmax))  &
          + area(ymin:ymax, xmin:xmax)
     narea(ymin:ymax, xind(xmin:xmax)) = narea(ymin:ymax, xind(xmin:xmax))  &
          + carea(ymin:ymax, xmin:xmax)

     DO j = 1, nval
        gridmean(j, ymin:ymax, xind(xmin:xmax)) = gridmean(j, ymin:ymax, xind(xmin:xmax)) &
             + area(ymin:ymax, xmin:xmax) * pixdat(j)

     ENDDO

     nactpix = nactpix + 1
  ENDDO
  CLOSE (INLUN)

  INTEGER :: ii, jj, kk
  DO ii = 1, nval
  DO jj = 1, nlat
     DO kk = 1, nlon
        IF (IEEE_IS_FINITE(totarea(jj,kk)) .AND. totarea(jj,kk) > TINY_A) THEN
        IF (IEEE_IS_FINITE(gridmean(ii,jj,kk))) THEN
           gridmean(ii,jj,kk) = gridmean(ii,jj,kk) / totarea(jj,kk)
        ELSE
           gridmean(ii,jj,kk) = 0.0D0
        END IF
        ELSE
        gridmean(ii,jj,kk) = 0.0D0
        END IF
     END DO
  END DO
  END DO

     !   WHERE (narea > 1) 
     !      gridstd(i, :, :) = SQRT( (gridstd(i, :, :) / totarea(:, :) - gridmean(i, :, :) ** 2.0)  &
     !           * narea(:, :) / (narea(:, :) - 1.0) )
     !   ENDWHERE   

     !   WHERE (narea == 1) 
     !      gridstd(i, :, :) = 0.0
     !   ENDWHERE  
  ENDDO



  OPEN (UNIT = OUTLUN, FILE = TRIM(ADJUSTL(outfname)), STATUS = 'UNKNOWN', IOSTAT = outerrstat)
  IF (outerrstat /= 0) THEN
     print *, outfname
     WRITE(*, *) 'Error in opening output filename!!!', outerrstat; STOP
  ENDIF
  DO i=1,nval
     DO j=1, nlat
        WRITE (OUTLUN, '(7201D14.5)') gridmean(i, j, 1:nlon)
     ENDDO
  ENDDO

  DO j = 1, nlat
     WRITE (OUTLUN, '(7201D14.5)') totarea(j, 1:nlon)
  ENDDO

  INTEGER :: jj, tmpi
  DO j = 1, nlat
    DO jj = 1, nlon
      IF (.NOT. IEEE_IS_FINITE(narea(j,jj))) THEN
        tmpi = 0
      ELSE IF (narea(j,jj) >= DBLE(IHI)) THEN
        tmpi = IHI
      ELSE IF (narea(j,jj) <= DBLE(ILO)) THEN
        tmpi = ILO
      ELSE
        tmpi = INT(narea(j,jj))
      END IF
      ! You can buffer these into an integer row if you want to keep the single WRITE
      ! rowbuf(jj) = tmpi
    END DO
    ! WRITE(OUTLUN, '(7201I5)') rowbuf(1:nlon)
  END DO
  CLOSE (OUTLUN)

  WRITE(*, *) 'Total number of input pixels = ', npix
  WRITE(*, *) 'Number of successful pixels  = ', nactpix

END PROGRAM TESSELATE
