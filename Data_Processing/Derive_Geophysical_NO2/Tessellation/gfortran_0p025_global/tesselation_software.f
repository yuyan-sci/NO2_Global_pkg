	SUBROUTINE TESSELATE_AREAMASTER
     I    ( XGRID, YGRID, NXDIM, NYDIM,
     I      CORNER_COORDS, SIDE_PARAMS, ORIENT,
     I      ISTART1, ISTART2, ISTART3, ISTART4, JFINIS,
     O      AREA, SUM, YLIMIT_LOWER, YLIMIT_UPPER )

C    ********************************************************************
C    *   Robert Spurr, November 1998					*
C    *   SAO, 60 Garden Street, Cambridge, MA 02138, USA		*
C    *   +1 (617) 496 7819; email rspurr@cfa.harvard.edu		*
C    *									*
C    *   Algorithm MAY be subject of licensing agreement		*
C    ********************************************************************

C  Master module Inputs
C  ====================

C  Corenr Coordinates and Side curves
C  ----------------------------------

C  We have 4 corner coordinates CORNER_COORDS, and parameters SIDE_PARAMS 
C  describing the 4 curves joining these corners, and an orientation (ORIENT) 
C  for the footprint as follows :

C        2
C        /\
C       /  \
C      /    \
C     1\     \3		Orientation = -1 (Corner 2 before Corner 4)
C       \    /
C        \  /
C         \/4

C          2
C         /\
C        /  \
C       /    \3
C      /     / 
C     1\    /		Orientation = +1 (Corner 4 before Corner 2)
C       \  /
C        \/4

C  note that the actual area routines do not need to know the curve
C  equations - all information is contained in the SIDE_PARAMS array.

C  Offset and Box limits
C  ---------------------

C  The following inputs are all Offsets in the X-Y grid which contains
C  the footprint defined by CORNER_COORDS. The User should ensure that any 
C  original grid of values be reduced to cover the footprint (avoids
C  unnecessary computation inside the tessellation module)

C		ISTART1, ISTART2, ISTART3, ISTART4, JFINIS

C   JFINIS is the offset in Y-space of corner 2, (the offset for corner 4
C   should be set to 1).
C   ISTART1, ISTART2, ISTART3, ISTART4 are the offsets in the X-direction
C   for corners 1 through 4 respectively. ISTART1 = 1 upon input.

C  Grid and Data input
C  -------------------

C  Integers NXDIM and NYDIM are just dimensioning values for the X and Y
C  grid values XGRID and YGRID. The array of data bins ALBUSED covers
C  only the box containing the footprint, and this should be prepared
C  before entry to the tessellation master.

C  Master module Output
C  ====================

C  The outputs include the tessellated area bins AREA which cover the pixel
C  and the total area of the pixel (SUM), the area-weighted albedo (ALBEDO)

C  Also output (for reference) are the ranges over which grid bins in the
C  Y direction are required. These are dependent on the X-slice.

C  Subroutine declarations
C  =======================

C  input

	INTEGER		NXDIM, NYDIM, ORIENT
	REAL*8		XGRID(NXDIM), YGRID(NYDIM)
	INTEGER		ISTART1, ISTART2, ISTART3, ISTART4, JFINIS
	REAL*8		CORNER_COORDS(4,2), SIDE_PARAMS(6,4)

C  output

	REAL*8		AREA(NYDIM, NXDIM), SUM
	INTEGER		YLIMIT_LOWER(NXDIM)
	INTEGER		YLIMIT_UPPER(NXDIM)

C  Local variables
C  ===============

C  Options for pathway throught the tessellation

	LOGICAL		DOUBLE_FIRST, DOUBLE_LAST
	LOGICAL		DO_GROUP1, DO_GROUP2
	LOGICAL		TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE

C  counters

	INTEGER		I, J

C  Selection module, chooses path through the X-slicing
C  ====================================================

	CALL TESSELATE_OPTIONS_CHOOSER
     I    ( XGRID, YGRID, NXDIM, NYDIM,
     I      ISTART1, ISTART2, ISTART3, ISTART4, JFINIS, ORIENT,
     O      DO_GROUP1, DOUBLE_FIRST, DOUBLE_LAST,
     O      DO_GROUP2, TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE )

C  initialise output
C  =================

	DO I = ISTART1, ISTART3
	  DO J = 1, JFINIS
	    AREA(J,I) = 0.0
	  ENDDO
	ENDDO

C  Tessellate for group 1 options
C  ==============================

	IF ( DO_GROUP1 ) THEN

	  CALL TESSELATE_OPTIONS_1
     I     ( XGRID, YGRID, NXDIM, NYDIM, CORNER_COORDS, ORIENT, 
     I       DOUBLE_FIRST, DOUBLE_LAST, SIDE_PARAMS,
     I       ISTART1, ISTART2, ISTART3, ISTART4, JFINIS,
     O       AREA, YLIMIT_LOWER, YLIMIT_UPPER )

C  Tessellate for group 2 options
C  ==============================

	ELSE IF ( DO_GROUP2 ) THEN

	  CALL TESSELATE_OPTIONS_2
     I   ( XGRID, YGRID, NXDIM, NYDIM, CORNER_COORDS, ORIENT, 
     I     TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE, SIDE_PARAMS,  
     I     ISTART1, ISTART2, ISTART3, ISTART4, JFINIS,
     O     AREA, YLIMIT_LOWER, YLIMIT_UPPER )

	ENDIF

C  Area sum and weighted albedo computation
C  ========================================

C	  PRINT *, YLIMIT_LOWER(I), YLIMIT_UPPER(I), ISTART1, ISTART3,
C	1	 NYDIM, NXDIM 


	SUM = 0.0
	DO I = ISTART1, ISTART3
	  DO J = YLIMIT_LOWER(I), YLIMIT_UPPER(I)
	    SUM = SUM + AREA(J,I)
	  ENDDO
	ENDDO

C  Finish

	END

C

	SUBROUTINE TESSELATE_OPTIONS_CHOOSER
     I    ( XGRID, YGRID, NXDIM, NYDIM,
     I      ISTART1, ISTART2, ISTART3, ISTART4,
     I      JFINIS, ORIENT,
     O      DO_GROUP1, DOUBLE_FIRST, DOUBLE_LAST,
     O      DO_GROUP2, TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE )

C  input

	INTEGER		NXDIM, NYDIM
	REAL*8		XGRID(NXDIM)
	REAL*8		YGRID(NYDIM)
	INTEGER		ISTART1, ISTART2, ISTART3, ISTART4
	INTEGER		JFINIS, ORIENT

C  output (pathway options

	LOGICAL		DOUBLE_FIRST, DOUBLE_LAST
	LOGICAL		DO_GROUP1, DO_GROUP2
	LOGICAL		TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE

C   Initialise options

	DOUBLE_FIRST = .FALSE.
	DOUBLE_LAST = .FALSE.
	TRIPLE_FIRST = .FALSE.
	TRIPLE_LAST = .FALSE.
	DOUBLE_MIDDLE = .FALSE.
	DO_GROUP1 = .FALSE.
	DO_GROUP2 = .FALSE.

C  Group 1
C  =======

C  determining factor: Corners 2 and 4 are NOT in the same X-slice

	IF ( ISTART2.NE.ISTART4 ) THEN
	  DO_GROUP1 = .TRUE.
	  DOUBLE_FIRST = ((ISTART1.EQ.ISTART2).OR.(ISTART1.EQ.ISTART4))
	  DOUBLE_LAST  = ((ISTART3.EQ.ISTART2).OR.(ISTART3.EQ.ISTART4))
	  RETURN
	ENDIF

C  Group 2
C  =======

C  determining factor: Corners 2 and 4 ARE in the same X-slice

	IF ( ISTART2.EQ.ISTART4 ) THEN
	  DO_GROUP2 = .TRUE.
	  TRIPLE_FIRST = ((ISTART1.EQ.ISTART2).AND.DO_GROUP2)
	  TRIPLE_LAST  = ((ISTART3.EQ.ISTART2).AND.DO_GROUP2)
	  DOUBLE_MIDDLE= ((.NOT.TRIPLE_FIRST).AND.(.NOT.TRIPLE_LAST))
	  RETURN
	ENDIF

C  finish

	END
C  List of Modules in Tessellation package :

C	tesselate_options_1
C	tesselate_options_2
C	singleside_slicer
C	doubleside_slicer
C	single_left_slicer
C	single_right_slicer
C	single_topbot_slicer
C	double_left_slicer
C	double_right_slicer
C	double_middle_slicer
C	triple_left_slicer
C	triple_right_slicer
C	quadruple_slicer
C	offset

C    ********************************************************************
C    *   Robert Spurr, November 1998					*
C    *   SAO, 60 Garden Street, Cambridge, MA 02138, USA		*
C    *   +1 (617) 496 7819; email rspurr@cfa.harvard.edu		*
C    *									*
C    *   Algorithm MAY be subject of licensing agreement		*
C    ********************************************************************

C  function parameters are stored in array PARS which should
C  be pre-computed. The modules do not know what function is being
C  used in the area and line intercept determinations.

	SUBROUTINE TESSELATE_OPTIONS_1
     I     ( XGRID, YGRID, NXDIM, NYDIM,
     I       CC, ORIENT, DOUBLE_FIRST, DOUBLE_LAST, PARS,
     I       ISTART1, ISTART2, ISTART3, ISTART4, JFINIS,
     O       AREA, YLIMIT_LOWER, YLIMIT_UPPER )

C  input 
C  =====

C  grid values

	INTEGER		NXDIM, NYDIM
	REAL*8		XGRID(NXDIM), YGRID(NYDIM)

C  Corner coordinates

	REAL*8		CC(4,2)

C  Offsets

	INTEGER		ISTART1, ISTART2, ISTART4, ISTART3, JFINIS

C  function parameters

	REAL*8		PARS(6,4)

C  control of options

	INTEGER		ORIENT
	LOGICAL		DOUBLE_FIRST, DOUBLE_LAST

C  output
C  ======

	REAL*8		AREA(NYDIM,NXDIM)
	INTEGER		YLIMIT_LOWER(NXDIM)
	INTEGER		YLIMIT_UPPER(NXDIM)

C  local variables
C  ===============

	REAL*8		XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U
	INTEGER		IS, IS_NEXT, IS_FIRST, IS_LAST, IS_BEG, IS_END
	INTEGER		I_NEXT, JC1, JC2, JOLD_L, JOLD_U, JNEW_L, JNEW_U
	INTEGER		SIDE_L, SIDE_U, SIDE_C, SIDE_A, ISTART_DUMMY

C  parameter values and indices

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	INTEGER		ASCT, PARALLEL, PERPEN
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )
	PARAMETER	( ASCT = 1, PARALLEL = 1, PERPEN = -1 )

C  external functions

	REAL*8		SIDEFUNCTION
	INTEGER		OFFSET
	EXTERNAL	OFFSET, SIDEFUNCTION

C  DOUBLE FIRST OPTION
C  ###################

	IF ( DOUBLE_FIRST ) THEN

	  IS_FIRST = 1
	  I_NEXT = ISTART1 + 1
	  IS_NEXT = IS_FIRST
	  XNEW = XGRID(I_NEXT)

C  Orientation = 1  (Corners 4 and 1; upper/lower sides are 1 and 3)
C  Orientation = -1 (Corners 2 and 1; upper/lower sides are 2 and 4)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_C = SIDE_4
	    SIDE_U = SIDE_1
	    SIDE_L = SIDE_3
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_C = SIDE_2
	    SIDE_U = SIDE_2
	    SIDE_L = SIDE_4
	  ENDIF

C  Find where upper/lower sides cut next x-level

	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_U )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_L )
	  JNEW_L  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JC1 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_1,2) )
	  JC2 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_C,2) )
	  JNEW_U  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1
	  

C  get double corner areas

	  CALL DOUBLE_LEFT_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JNEW_L, JC1, JC2, JNEW_U, XNEW, YNEW_L, YNEW_U,
     I       CC(SIDE_1,1), CC(SIDE_1,2),
     I       CC(SIDE_C,1), CC(SIDE_C,2),
     O       AREA(1,IS_FIRST) )

C  Assign limits

	  IF ( ORIENT .EQ. 1 ) THEN
	    YLIMIT_LOWER(IS_FIRST) = JC2
	    YLIMIT_UPPER(IS_FIRST) = JNEW_U
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    YLIMIT_LOWER(IS_FIRST) = JNEW_L
	    YLIMIT_UPPER(IS_FIRST) = JC2
	  ENDIF

	ENDIF

C  SINGLE FIRST OPTION
C  ###################

	IF ( .NOT. DOUBLE_FIRST ) THEN

C  First slice (includes corner 1)
C  -------------------------------

	  IS_FIRST = 1
	  I_NEXT = ISTART1 + 1
	  IS_NEXT = IS_FIRST

C  Find where sides 1 and 4 cut next x-level, 

	  XNEW   = XGRID(I_NEXT)
	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_1 )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_4 )
	  JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JC1    = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_1,2) )
	  JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1

C  assign Corner area

	  CALL SINGLE_LEFT_SLICER
     I     ( YGRID, NYDIM, NYDIM, PARS,
     I       JNEW_L, JC1, JNEW_U,
     I       XNEW, YNEW_L, YNEW_U,
     I       CC(SIDE_1,1), CC(SIDE_1,2),
     O       AREA(1,IS_FIRST) )

C  Assign limits

	  YLIMIT_LOWER(IS_FIRST) = JNEW_L
	  YLIMIT_UPPER(IS_FIRST) = JNEW_U

C  Slices between Corners 1 and 2/4 (depends on orientation)
C  --------------------------------

C  How many X-slices 

	  IS_BEG = IS_FIRST + 1
	  IF ( ORIENT.EQ.1 ) THEN
	    ISTART_DUMMY = ISTART4
	  ELSE IF ( ORIENT.EQ.-1 ) THEN
	    ISTART_DUMMY = ISTART2
	  ENDIF
	  IS_END = ISTART_DUMMY - ISTART1

C  Start loop over X-slices (not done if IS_BEG > IS_END)

	  DO IS = IS_BEG, IS_END

C  update left slice boundary values

	    JOLD_L = JNEW_L
	    JOLD_U = JNEW_U
	    YOLD_L = YNEW_L
	    YOLD_U = YNEW_U
	    XOLD   = XNEW

C  Find where sides borders cut next x-level

	    I_NEXT = I_NEXT + 1
	    XNEW   = XGRID(I_NEXT)
	    YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_1 )
	    YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_4 )
	    JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	    JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

		IF (JNEW_L .EQ. 0) JNEW_L = 1
		IF (JOLD_L .EQ. 0) JOLD_L = 1

C  Assign areas for each X-slice

	    CALL DOUBLESIDE_SLICER
     I     ( YGRID, NYDIM, NYDIM, SIDE_4, SIDE_1, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PERPEN, ORIENT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA(1,IS) )

C  Assign limits

	    YLIMIT_LOWER(IS) = JNEW_L
	    YLIMIT_UPPER(IS) = JNEW_U

	  ENDDO

C  Slice including Corner 2/4 (depends on orientation)
C  --------------------------

C  update left slice boundary values

 	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD   = XNEW

	  IS_NEXT = IS_END + 1
	  I_NEXT = I_NEXT + 1
	  XNEW   = XGRID(I_NEXT)

C  Orientation = 1  (Corner 4; upper/lower sides are 1 and 3)
C  Orientation = -1 (Corner 2; upper/lower sides are 2 and 4)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_C = SIDE_4
	    SIDE_U = SIDE_1
	    SIDE_L = SIDE_3
	    SIDE_A = SIDE_U
	    JC1 = 1
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_C = SIDE_2
	    SIDE_U = SIDE_2
	    SIDE_L = SIDE_4
	    SIDE_A = SIDE_L
	    JC1 = JFINIS
	  ENDIF

C  Find where upper/lower sides cut next x-level

	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_U )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_L )
	  JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1
C  Assign area

	  CALL SINGLE_TOPBOT_SLICER
     I   ( YGRID, NYDIM, NYDIM, SIDE_A, PARS, -ORIENT, ORIENT,
     I     JOLD_L, JOLD_U, JC1, JNEW_L, JNEW_U,
     I     XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     I     CC(SIDE_C,1), CC(SIDE_C,2), 
     O     AREA(1,IS_NEXT) )

C  Assign limits

	  IF ( ORIENT .EQ. 1 ) THEN
	    YLIMIT_LOWER(IS_NEXT) = JC1
	    YLIMIT_UPPER(IS_NEXT) = JNEW_U
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    YLIMIT_LOWER(IS_NEXT) = JNEW_L
	    YLIMIT_UPPER(IS_NEXT) = JC1
	  ENDIF

	ENDIF

C  CENTRAL SECTION
C  ###############

C  Slices between corner 2/4 and corner 4/2  (depends on orientation)
C  ----------------------------------------

C  How many slices

	IS_BEG = IS_NEXT + 1
	IF ( ORIENT.EQ.1 ) THEN
	  ISTART_DUMMY = ISTART2
	  SIDE_U = SIDE_1
	  SIDE_L = SIDE_3
	ELSE IF ( ORIENT.EQ.-1 ) THEN
	  ISTART_DUMMY = ISTART4
	  SIDE_U = SIDE_2
	  SIDE_L = SIDE_4
	ENDIF
	IS_END = ISTART_DUMMY - ISTART1

C  Start loop over X-slices (not done if IS_BEG > IS_END)

	DO IS = IS_BEG, IS_END

C  update left slice boundary values

	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD  = XNEW

C  Find where upper and lower sides cut next x-level

	  I_NEXT = I_NEXT + 1
	  XNEW = XGRID(I_NEXT)
	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_U )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_L )
	  JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign areas for each X-slice

	  CALL DOUBLESIDE_SLICER
     I     ( YGRID, NYDIM, NYDIM, SIDE_L, SIDE_U, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PARALLEL, ORIENT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA(1,IS) )

C	  PRINT *, IS, AREA(1, IS)
C	  PRINT *, ' '

C  Assign limits

	  IF ( ORIENT .EQ. 1 ) THEN
	    YLIMIT_LOWER(IS) = JOLD_L
	    YLIMIT_UPPER(IS) = JNEW_U
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    YLIMIT_LOWER(IS) = JNEW_L
	    YLIMIT_UPPER(IS) = JOLD_U
	  ENDIF

	ENDDO

C  DOUBLE LAST OPTION
C  ##################

	IF ( DOUBLE_LAST ) THEN

C  Final slice (includes corners 2/4 and 3)
C  ----------------------------------------

C  update left slice boundary values

	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD  = XNEW
	  IS_LAST = IS_END + 1

C  Orientation = 1  (Corners 2 and 3; upper/lower sides are 1 and 3)
C  Orientation = -1 (Corners 4 and 3; upper/lower sides are 2 and 4)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_C = SIDE_2
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_C = SIDE_4
	  ENDIF

C  Corner limits

	  JC1 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_C,2) )
	  JC2 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_3,2) )

C  get double corner

	  CALL DOUBLE_RIGHT_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JOLD_L, JC1, JC2, JOLD_U, XOLD, YOLD_L, YOLD_U,
     I       CC(SIDE_C,1), CC(SIDE_C,2), CC(SIDE_3,1), CC(SIDE_3,2),
     O       AREA(1,IS_LAST) )

C  Assign limits

	  IF ( ORIENT .EQ. 1 ) THEN
	    YLIMIT_LOWER(IS_LAST) = JOLD_L
	    YLIMIT_UPPER(IS_LAST) = JC1
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    YLIMIT_LOWER(IS_LAST) = JC1
	    YLIMIT_UPPER(IS_LAST) = JOLD_U
	  ENDIF

	ENDIF

C  SINGLE LAST OPTION
C  ##################

	IF ( .NOT. DOUBLE_LAST ) THEN

C  Slice including Corner 2/4 (depends on orientation)
C  --------------------------

C  update left slice boundary values

 	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD   = XNEW
	  IS_NEXT = IS_END + 1
	  I_NEXT = I_NEXT + 1
	  XNEW   = XGRID(I_NEXT)

C  Orientation = 1  (Corner 2; upper/lower sides are 2 and 3)
C  Orientation = -1 (Corner 4; upper/lower sides are 2 and 3)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_C = SIDE_2
	    SIDE_U = SIDE_2
	    SIDE_L = SIDE_3
	    SIDE_A = SIDE_3
	    JC1 = JFINIS
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_C = SIDE_4
	    SIDE_U = SIDE_2
	    SIDE_L = SIDE_3
	    SIDE_A = SIDE_2
	    JC1 = 1
	  ENDIF

C  Find where upper/lower sides cut next x-level

	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_U )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_L )
	  JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	   IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign area (ORIENT controls Top or Bottom)

	  CALL SINGLE_TOPBOT_SLICER
     I   ( YGRID, NYDIM, NYDIM, SIDE_A, PARS, ORIENT, ORIENT,
     I     JOLD_L, JOLD_U, JC1, JNEW_L, JNEW_U,
     I     XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     I     CC(SIDE_C,1), CC(SIDE_C,2), 
     O     AREA(1,IS_NEXT) )

C  Assign limits

	  IF ( ORIENT .EQ. 1 ) THEN
	    YLIMIT_LOWER(IS_NEXT) = JOLD_L
	    YLIMIT_UPPER(IS_NEXT) = JC1
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    YLIMIT_LOWER(IS_NEXT) = JC1
	    YLIMIT_UPPER(IS_NEXT) = JOLD_U
	  ENDIF

C  Slices between corner 2 and corner 3
C  ------------------------------------

	  IS_BEG = IS_NEXT + 1
	  IS_END = ISTART3 - ISTART1

	  DO IS = IS_BEG, IS_END

C  update left slice boundary values

	    JOLD_L = JNEW_L
	    JOLD_U = JNEW_U
	    YOLD_L = YNEW_L
	    YOLD_U = YNEW_U
	    XOLD  = XNEW

C  Find where sides 2 and 3 cut next x-level

	    I_NEXT = I_NEXT + 1
	    XNEW = XGRID(I_NEXT)
	    YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_2 )
	    YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_3 )
	    JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	    JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

		IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign area

	    CALL DOUBLESIDE_SLICER
     I     ( YGRID, NYDIM, NYDIM, SIDE_3, SIDE_2, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PERPEN, ORIENT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA(1,IS) )

C  Assign limits

	    YLIMIT_LOWER(IS) = JOLD_L
	    YLIMIT_UPPER(IS) = JOLD_U

	  ENDDO

C  Last slice (including corner 3)
C  -------------------------------

C  update left slice boundary values; Find Y-limit for this corner

	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD   = XNEW
	  IS_LAST = IS_END + 1
	  JC2 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_3,2) )

	  CALL SINGLE_RIGHT_SLICER
     I     ( YGRID, NYDIM, NYDIM, PARS,
     I       JOLD_L, JC2, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U, CC(SIDE_3,1), CC(SIDE_3,2),
     O       AREA(1,IS_LAST) )

C  Assign limits

	    YLIMIT_LOWER(IS_LAST) = JOLD_L
	    YLIMIT_UPPER(IS_LAST) = JOLD_U

	ENDIF

C  Finish

	END

C


C  function parameters are stored in array PARS which should
C  be pre-computed. The modules do not know what function is being
C  used in the area and line intercept determinations.

	SUBROUTINE TESSELATE_OPTIONS_2
     I   ( XGRID, YGRID, NXDIM, NYDIM,
     I     CC, ORIENT, TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE, PARS,  
     I     ISTART1, ISTART2, ISTART3, ISTART4, JFINIS,
     O     AREA, YLIMIT_LOWER, YLIMIT_UPPER )

C  input 
C  =====

C  grid values

	INTEGER		NXDIM, NYDIM
	REAL*8		XGRID(NXDIM), YGRID(NYDIM)

C  Corner coordinates

	REAL*8		CC(4,2)

C  Offsets

	INTEGER		ISTART1, ISTART2, ISTART4, ISTART3, JFINIS

C  function parametaers

	REAL*8		PARS(6,4)

C  control of options

	INTEGER		ORIENT
	LOGICAL		TRIPLE_FIRST, TRIPLE_LAST, DOUBLE_MIDDLE

C  output
C  ======

	REAL*8		AREA(NYDIM,NXDIM)
	INTEGER		YLIMIT_LOWER(NXDIM)
	INTEGER		YLIMIT_UPPER(NXDIM)

C  local variables
C  ===============

	REAL*8		XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U
	INTEGER		IS, IS_NEXT, IS_FIRST, IS_LAST, IS_BEG, IS_END
	INTEGER		JC1, JC2, JC3, JC4, JOLD_L, JOLD_U, JNEW_L, JNEW_U
	INTEGER		SIDE_F, SIDE_D, I_NEXT, IS_SINGLE, ISTART_DUMMY

C  parameter values and indices

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	INTEGER		ASCT, PARALLEL, PERPEN
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )
	PARAMETER	( ASCT = 1, PARALLEL = 1, PERPEN = -1 )

C  external functions

	REAL*8		SIDEFUNCTION
	INTEGER		OFFSET
	EXTERNAL	OFFSET, SIDEFUNCTION

C  QUADRUPLE OPTION (all 4 corners in one slice)
C  ################

	IF ( TRIPLE_FIRST .AND. TRIPLE_LAST ) THEN

C   Orientation = +1, in order 1-4-2-3
C   Orientation = -1, in order 1-2-4-3

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_F = SIDE_4
	    SIDE_D = SIDE_2
	    JC2 = 1
   	    JC3 = JFINIS
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_F = SIDE_2
	    SIDE_D = SIDE_4
	    JC2 = JFINIS
   	    JC3 = 1
	  ENDIF

C  get area

	  IS_SINGLE = 1
	  JC1 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_1,2) )
	  JC4 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_3,2) )
	  CALL QUADRUPLE_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JC1, JC2, JC3, JC4,
     I       CC(SIDE_1,1), CC(SIDE_1,2),
     I       CC(SIDE_F,1), CC(SIDE_F,2),
     I       CC(SIDE_D,1), CC(SIDE_D,2),
     I       CC(SIDE_3,1), CC(SIDE_3,2),
     O       AREA(1,IS_SINGLE) )

C  Assign limits

	  YLIMIT_LOWER(IS_SINGLE) = 1
	  YLIMIT_UPPER(IS_SINGLE) = JFINIS

C  return after this option - no more to do

	  RETURN

	ENDIF

C  TRIPLE FIRST OPTION
C  ###################

	IF ( TRIPLE_FIRST ) THEN

	  IS_FIRST = 1
	  I_NEXT = ISTART1 + 1
	  IS_NEXT = IS_FIRST
	  XNEW = XGRID(I_NEXT)

C  Orientation = 1  (Corners 1-4-2; upper/lower sides are 2 and 3)
C  Orientation = -1 (Corners 1-2-4; upper/lower sides are 2 and 3)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_F = SIDE_4
	    SIDE_D = SIDE_2
	    JC2 = 1
   	    JC3 = JFINIS
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_F = SIDE_2
	    SIDE_D = SIDE_4
	    JC2 = JFINIS
   	    JC3 = 1
	  ENDIF

C  Find where upper/lower sides cut next x-level

	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_2 )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_3 )
	  JNEW_L  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JC1 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_1,2) )
	  JNEW_U  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1

C  get triple corner

	  CALL TRIPLE_LEFT_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JC1, JC2, JC3, JNEW_L, JNEW_U,
     I       XNEW, YNEW_L, YNEW_U,
     I       CC(SIDE_1,1), CC(SIDE_1,2),
     I       CC(SIDE_F,1), CC(SIDE_F,2),
     I       CC(SIDE_D,1), CC(SIDE_D,2),
     O       AREA(1,IS_FIRST) )

C  Assign limits

	  YLIMIT_LOWER(IS_FIRST) = 1
	  YLIMIT_UPPER(IS_FIRST) = JFINIS

	ENDIF

C  SINGLE FIRST OPTION
C  ###################

	IF ( .NOT. TRIPLE_FIRST ) THEN

C  First slice (includes corner 1)
C  -------------------------------

	  IS_FIRST = 1
	  I_NEXT = ISTART1 + 1
	  IS_NEXT = IS_FIRST

C  Find where sides 1 and 4 cut next x-level, assign area and check

	  XNEW   = XGRID(I_NEXT)
	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_1 )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_4 )
	  JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JC1    = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_1,2) )
	  JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )
	  IF (JNEW_L .EQ. 0) JNEW_L = 1
		
	  CALL SINGLE_LEFT_SLICER
     I     ( YGRID, NYDIM, NYDIM, PARS,
     I       JNEW_L, JC1, JNEW_U,
     I       XNEW, YNEW_L, YNEW_U,
     I       CC(SIDE_1,1), CC(SIDE_1,2),
     O       AREA(1,IS_FIRST) )

C  Assign limits

	  YLIMIT_LOWER(IS_FIRST) = JNEW_L
	  YLIMIT_UPPER(IS_FIRST) = JNEW_U

C  Slices between Corners 1 and 2/4 (depends on orientation)
C  --------------------------------

C  How many X-slices 

	  IS_BEG = IS_FIRST + 1
	  IF ( ORIENT.EQ.1 ) THEN
	    ISTART_DUMMY = ISTART4
	  ELSE IF ( ORIENT.EQ.-1 ) THEN
	    ISTART_DUMMY = ISTART2
	  ENDIF
	  IS_END = ISTART_DUMMY - ISTART1

C  Start loop over X-slices (not done if IS_BEG > IS_END)

	  DO IS = IS_BEG, IS_END

C  update left slice boundary values

	    JOLD_L = JNEW_L
	    JOLD_U = JNEW_U
	    YOLD_L = YNEW_L
	    YOLD_U = YNEW_U
	    XOLD   = XNEW

C  Find where sides borders cut next x-level

	    I_NEXT = I_NEXT + 1
	    XNEW   = XGRID(I_NEXT)
	    YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_1 )
	    YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_4 )
	    JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	    JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

		IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign area

	    CALL DOUBLESIDE_SLICER
     I     ( YGRID, NYDIM, NYDIM, SIDE_4, SIDE_1, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PERPEN, ORIENT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA(1,IS) )

C  Assign limits

	    YLIMIT_LOWER(IS) = JNEW_L
	    YLIMIT_UPPER(IS) = JNEW_U

	  ENDDO

	ENDIF

C  DOUBLE MIDDLE OPTION
C  ####################

	IF ( DOUBLE_MIDDLE ) THEN

C  Slice including corners 2 and 4 (together)
C  -------------------------------

	  IS_NEXT = IS_END + 1

C  update left slice boundary values

 	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD  = XNEW

C  Orientation = 1  (Corners 4-2; upper/lower sides are 2 and 3)
C  Orientation = -1 (Corners 2-4; upper/lower sides are 2 and 3)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_F = SIDE_4
	    SIDE_D = SIDE_2
	    JC1 = 1
   	    JC2 = JFINIS
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_F = SIDE_2
	    SIDE_D = SIDE_4
	    JC1 = JFINIS
   	    JC2 = 1
	  ENDIF

C  Find where upper/lower sides cut next x-level

	  I_NEXT = I_NEXT + 1
	  XNEW   = XGRID(I_NEXT)
	  YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_2 )
	  YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_3 )
	  JNEW_L  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	  JNEW_U  = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

	  IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign area

	  CALL DOUBLE_MIDDLE_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JOLD_L, JOLD_U, JC1, JC2, JNEW_L, JNEW_U,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     I       CC(SIDE_F,1), CC(SIDE_F,2),
     I       CC(SIDE_D,1), CC(SIDE_D,2),
     O       AREA(1, IS_NEXT) )

C  Assign limits

	  YLIMIT_LOWER(IS_NEXT) = 1
	  YLIMIT_UPPER(IS_NEXT) = JFINIS

	ENDIF

C  TRIPLE LAST OPTION
C  ##################

	IF ( TRIPLE_LAST ) THEN

C  Final slice (includes corners 2/4 and 3)
C  ----------------------------------------

C  update left slice boundary values

	  JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD  = XNEW
	  IS_LAST = IS_END + 1

C  Orientation = 1  (Corners 4,2,3; upper/lower sides are 1 and 4)
C  Orientation = -1 (Corners 2,4,3; upper/lower sides are 1 and 4)

	  IF ( ORIENT .EQ. 1 ) THEN
	    SIDE_F = SIDE_4
	    SIDE_D = SIDE_2
	    JC1 = 1
   	    JC2 = JFINIS
	  ELSE IF ( ORIENT .EQ. -1 ) THEN
	    SIDE_F = SIDE_2
	    SIDE_D = SIDE_4
	    JC1 = JFINIS
   	    JC2 = 1
	  ENDIF

C  Corner limits

	  JC3 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_3,2) )

	  CALL TRIPLE_RIGHT_SLICER
     I     ( YGRID, NYDIM, NYDIM, ORIENT, PARS,
     I       JC1, JC2, JC3, JOLD_L, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U,
     I       CC(SIDE_F,1), CC(SIDE_F,2),
     I       CC(SIDE_D,1), CC(SIDE_D,2),
     I       CC(SIDE_3,1), CC(SIDE_3,2),
     O       AREA(1, IS_LAST) )

C  Assign limits

	    YLIMIT_LOWER(IS_LAST) = 1
	    YLIMIT_UPPER(IS_LAST) = JFINIS

	ENDIF

C  SINGLE LAST OPTION
C  ##################

	IF ( .NOT. TRIPLE_LAST ) THEN

C  Slices between corner 2 and corner 3
C  ------------------------------------

	  IS_BEG = IS_NEXT + 1
	  IS_END = ISTART3 - ISTART1

	  DO IS = IS_BEG, IS_END

C  update left slice boundary values

	    JOLD_L = JNEW_L
	    JOLD_U = JNEW_U
	    YOLD_L = YNEW_L
	    YOLD_U = YNEW_U
	    XOLD  = XNEW

C  Find where sides 2 and 3 cut next x-level

	    I_NEXT = I_NEXT + 1
	    XNEW = XGRID(I_NEXT)
	    YNEW_U = SIDEFUNCTION ( PARS, XNEW, SIDE_2 )
	    YNEW_L = SIDEFUNCTION ( PARS, XNEW, SIDE_3 )
	    JNEW_L = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_L )
	    JNEW_U = OFFSET ( 1, YGRID, NYDIM, ASCT, YNEW_U )

		IF (JNEW_L .EQ. 0) JNEW_L = 1

C  Assign area

	    CALL DOUBLESIDE_SLICER
     I     ( YGRID, NYDIM, NYDIM, SIDE_3, SIDE_2, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PERPEN, ORIENT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA(1,IS) )

C  Assign limits

	    YLIMIT_LOWER(IS) = JOLD_L
	    YLIMIT_UPPER(IS) = JOLD_U

	  ENDDO

C  Last slice (including corner 3)
C  -------------------------------

C  update left slice boundary values; Find Y-limit for this corner

          JOLD_L = JNEW_L
	  JOLD_U = JNEW_U
	  YOLD_L = YNEW_L
	  YOLD_U = YNEW_U
	  XOLD   = XNEW
	  IS_LAST = IS_END + 1
	  JC2 = OFFSET ( 1, YGRID, NYDIM, ASCT, CC(SIDE_3,2) )

	  CALL SINGLE_RIGHT_SLICER
     I     ( YGRID, NYDIM, NYDIM, PARS,
     I       JOLD_L, JC2, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U, CC(SIDE_3,1), CC(SIDE_3,2),
     O       AREA(1,IS_LAST) )

C  Assign limits

	    YLIMIT_LOWER(IS_LAST) = JOLD_L
	    YLIMIT_UPPER(IS_LAST) = JOLD_U

	ENDIF

C  Finish

	END

	SUBROUTINE SINGLESIDE_SLICER
     I     ( YG, LOCAL_NYDIM, LOCAL_NYBOX, SIDE, PARS,
     I       JOLD, JNEW, XOLD, YOLD, XNEW, YNEW,
     O       AREA )

C  input/output

	INTEGER		LOCAL_NYDIM, LOCAL_NYBOX, SIDE
	REAL*8		YG(LOCAL_NYDIM), PARS(6,4)
	INTEGER		JOLD, JNEW
	REAL*8		XOLD, XNEW, YOLD, YNEW
	REAL*8		AREA(LOCAL_NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4, J
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables (including external functions)

	REAL*8		X1, Y1, X2, Y2, QUAD, CORNER
	REAL*8		INV_SIDEFUNCTION, GRID_AREA, CORNER_AREA
	EXTERNAL	INV_SIDEFUNCTION, GRID_AREA, CORNER_AREA

C  TRIVIAL CASE WHEN J_OLD = J_NEW
C  ===============================

	IF ( JOLD.EQ.JNEW ) THEN
  	  AREA(JOLD) = CORNER_AREA(PARS,XOLD,XNEW,YOLD,YNEW,SIDE)
	  RETURN
	ENDIF

C  FOR SIDE 1
C  ==========

	IF ( SIDE .EQ. SIDE_1 ) THEN
C  first corner situation
	  X1 = XOLD
	  Y1 = YOLD
	  Y2 = YG(JOLD+1)
	  X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	  QUAD   = GRID_AREA (PARS,X2,XNEW,Y1,Y2)
	  CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	  AREA(JOLD) = CORNER + QUAD

C	  WRITE(*, *) 'SIDE1: ', CORNER, QUAD
C  line sides situation
	  DO J = JOLD + 1, JNEW - 1
	    Y1 = Y2
	    X1 = X2
	    Y2 = YG(J+1)
	    X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	    QUAD   = GRID_AREA (PARS,X2,XNEW,Y1,Y2)
	    CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	    AREA(J) = CORNER + QUAD
	  ENDDO
C  last corner situation
	  Y1 = Y2
	  X1 = X2
	  Y2 = YNEW
	  X2 = XNEW
	  AREA(JNEW) = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)

C  FOR SIDE 2
C  ==========

	ELSE IF ( SIDE .EQ. SIDE_2 ) THEN
C  first corner situation
	  X1 = XOLD
	  Y1 = YOLD
	  Y2 = YG(JOLD)
	  X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	  AREA(JOLD) = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
C	  WRITE(*, *) 'SIDE2: ', AREA(JOLD)

C  line sides situation
	  DO J = JOLD - 1, JNEW + 1, -1
	    Y1 = Y2
	    X1 = X2
	    Y2 = YG(J)
	    X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	    QUAD   = GRID_AREA (PARS,XOLD,X1,Y2,Y1)
	    CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	    AREA(J) = CORNER + QUAD
	  ENDDO
C  last corner situation
	  Y1 = Y2
	  X1 = X2
	  Y2 = YNEW
	  X2 = XNEW
	  QUAD   = GRID_AREA (PARS,XOLD,X1,Y2,Y1)
	  CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	  AREA(JNEW) = CORNER + QUAD

C  FOR SIDE 3
C  ==========

	ELSE IF ( SIDE .EQ. SIDE_3 ) THEN
C  first corner situation
	  X1 = XOLD
	  Y1 = YOLD
	  Y2 = YG(JOLD+1)
	  X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	  AREA(JOLD) = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
C	  WRITE(*, *) 'SIDE3: ', AREA(JOLD)
C  line sides situation
	  DO J = JOLD + 1, JNEW - 1
	    Y1 = Y2
	    X1 = X2
	    Y2 = YG(J+1)
	    X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	    QUAD   = GRID_AREA (PARS,XOLD,X1,Y1,Y2)
	    CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	    AREA(J) = CORNER + QUAD
	  ENDDO
C  last corner situation
	  Y1 = Y2
	  X1 = X2
	  Y2 = YNEW
	  X2 = XNEW
	  QUAD   = GRID_AREA (PARS,XOLD,X1,Y1,Y2)
	  CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	  AREA(JNEW) = CORNER + QUAD

C  FOR SIDE 4
C  ==========

	ELSE IF ( SIDE .EQ. SIDE_4 ) THEN
C  first corner situation
	  X1 = XOLD
	  Y1 = YOLD
	  Y2 = YG(JOLD)
	  X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	  QUAD   = GRID_AREA (PARS,X2,XNEW,Y2,Y1)
	  CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	  AREA(JOLD) = CORNER + QUAD
C	  WRITE(*, *) 'SIDE4: ', CORNER, QUAD
C  line sides situation
	  DO J = JOLD - 1, JNEW + 1, -1
	    Y1 = Y2
	    X1 = X2
	    Y2 = YG(J)
	    X2 = INV_SIDEFUNCTION ( PARS, Y2, SIDE )
	    QUAD   = GRID_AREA (PARS,X2,XNEW,Y2,Y1)
	    CORNER = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	    AREA(J) = CORNER + QUAD
	  ENDDO
C  last corner situation
	  Y1 = Y2
	  X1 = X2
	  Y2 = YNEW
	  X2 = XNEW

	  IF (NEW /= 0) AREA(JNEW) = CORNER_AREA(PARS,X1,X2,Y1,Y2,SIDE)
	
	ENDIF

C  Finish

	END

C

	SUBROUTINE DOUBLESIDE_SLICER
     I     ( YG, NYDIM, NYBOX, SIDE_L, SIDE_U, PARS,
     I       JOLD_L, JOLD_U, JNEW_L, JNEW_U, PARALLEL, ORNT,
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     O       AREA )

C  input/output

	INTEGER		NYBOX, NYDIM, SIDE_L, SIDE_U, ORNT, PARALLEL
	REAL*8		YG(NYDIM), PARS(6,4)
	INTEGER		JOLD_L, JOLD_U, JNEW_L, JNEW_U
	REAL*8		XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables (including external function)

	INTEGER		J, JMIN, JMAX, LOCAL_NYBOX
	PARAMETER	( LOCAL_NYBOX = 5000 )
	REAL*8		AREA_L(LOCAL_NYBOX), AREA_U(LOCAL_NYBOX)
	REAL*8		GRID, GRID_AREA
	EXTERNAL	GRID_AREA

C  initialise

	JMIN = MIN ( JOLD_L, JNEW_L )
	JMAX = MAX ( JOLD_U, JNEW_U )
	DO J = JMIN, JMAX
	   IF (j .gt. 0) THEN
		  AREA_L(J) = 0.0
		  AREA_U(J) = 0.0
	   ENDIF
	ENDDO

C  SIDE SLICING
C  ============

C  slicing Lower side (labelled L = either 3 or 4)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_L, PARS,
     I       JOLD_L, JNEW_L, XOLD, YOLD_L, XNEW, YNEW_L,
     O       AREA_L )
C	  WRITE(*, *) 'AREA_L = ', AREA_L(1:10)

C  slicing Upper side (labelled U = either 1 or 2)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_U, PARS,
     I       JOLD_U, JNEW_U, XOLD, YOLD_U, XNEW, YNEW_U,
     O       AREA_U )

C	  WRITE(*, *) 'AREA_U = ', AREA_U(1:10)
C	  WRITE(*, *) 'PARALLEL = ', PARALLEL

C  Parallel lines (or at least running together)
C  =============================================

	IF ( PARALLEL .EQ. 1 ) THEN

C  Orientation 1
C  #############

	  IF ( ORNT .EQ. 1 ) THEN

C  overlapping case

	    IF ( JNEW_L .GT. JOLD_U ) THEN

	      DO J = JOLD_L, JOLD_U-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      DO J = JNEW_L+1, JNEW_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YNEW_L,YG(JNEW_L+1))
	      AREA_L(JNEW_L) = AREA_L(JNEW_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JOLD_U),YOLD_U)
	      AREA_U(JOLD_U) = AREA_U(JOLD_U) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JNEW_L),YG(JNEW_L+1))
	      AREA(JNEW_L) = AREA_L(JNEW_L) + AREA_U(JNEW_L) - GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JOLD_U),YG(JOLD_U+1))
	      AREA(JOLD_U) = AREA_L(JOLD_U) + AREA_U(JOLD_U) - GRID
	      DO J = JOLD_U+1, JNEW_L-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

C  same grid cases

	    ELSE IF ( JNEW_L .EQ. JOLD_U ) THEN

	      DO J = JOLD_L, JNEW_L-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      DO J = JOLD_U+1, JNEW_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YNEW_L,YOLD_U)
	      AREA(JOLD_U) = AREA_L(JNEW_L) + AREA_U(JOLD_U) + GRID

C non-overlapping case :
C    Complete lower and upper, assigning directly and fill in grids

	    ELSE

	      DO J = JOLD_L, JNEW_L-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      DO J = JOLD_U+1, JNEW_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YNEW_L,YG(JNEW_L+1))
	      AREA(JNEW_L) = AREA_L(JNEW_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JOLD_U),YOLD_U)
	      AREA(JOLD_U) = AREA_U(JOLD_U) + GRID
	      DO J = JNEW_L+1, JOLD_U-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

	    ENDIF

C  Orientation -1
C  ##############

	  ELSE IF ( ORNT .EQ. -1 ) THEN

C  overlapping case

	    IF ( JNEW_U .LT. JOLD_L ) THEN

	      DO J = JOLD_L+1, JOLD_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      DO J = JNEW_L, JNEW_U-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YOLD_L,YG(JOLD_L+1))
	      AREA_L(JOLD_L) = AREA_L(JOLD_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JNEW_U),YNEW_U)
	      AREA_U(JNEW_U) = AREA_U(JNEW_U) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JOLD_L),YG(JOLD_L+1))
	      AREA(JOLD_L) = AREA_L(JOLD_L) + AREA_U(JOLD_L) - GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JNEW_U),YG(JNEW_U+1))
	      AREA(JNEW_U) = AREA_L(JNEW_U) + AREA_U(JNEW_U) - GRID
	      DO J = JNEW_U+1, JOLD_L-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

C  same grid cases

	    ELSE IF ( JNEW_U .EQ. JOLD_L ) THEN

	      DO J = JNEW_L, JOLD_L-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      DO J = JNEW_U+1, JOLD_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YOLD_L,YNEW_U)
	      AREA(JOLD_L) = AREA_L(JOLD_L) + AREA_U(JOLD_L) + GRID

C non-overlapping case :
C    Complete lower and upper, assigning directly and fill in grids

	    ELSE

	      DO J = JNEW_L, JOLD_L-1
	        AREA(J) = AREA_L(J)
	      ENDDO
	      DO J = JNEW_U+1, JOLD_U
	        AREA(J) = AREA_U(J)
	      ENDDO
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YOLD_L,YG(JOLD_L+1))
	      AREA(JOLD_L) = AREA_L(JOLD_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JNEW_U),YNEW_U)
	      AREA(JNEW_U) = AREA_U(JNEW_U) + GRID
	      DO J = JOLD_L+1, JNEW_U-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

	    ENDIF

	  ENDIF

C  Non-parallel cases
C  ==================

	ELSE IF ( PARALLEL .EQ. -1 ) THEN

C  Lower 4 upper 1
C  ###############

	  IF ( SIDE_L .EQ. SIDE_4 ) THEN

C  Assignation

	    DO J = JNEW_L, JOLD_L-1 
	      AREA(J) = AREA_L(J)
	    ENDDO
	    DO J = JOLD_U+1, JNEW_U
	      AREA(J) = AREA_U(J)
	    ENDDO

C  Add grids as required

	    IF ( JOLD_L .EQ. JOLD_U ) THEN

	      GRID = GRID_AREA(PARS,XOLD,XNEW,YOLD_L,YOLD_U)
	      AREA(JOLD_L) = AREA_L(JOLD_L) + AREA_U(JOLD_L) + GRID

	    ELSE

	      GRID = GRID_AREA(PARS,XOLD,XNEW,YOLD_L,YG(JOLD_L+1))
	      AREA(JOLD_L) = AREA_L(JOLD_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JOLD_U),YOLD_U)
	      AREA(JOLD_U) = AREA_U(JOLD_U) + GRID
	      DO J = JOLD_L+1, JOLD_U-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

	    ENDIF

C  Lower 3 upper 2
C  ###############

	  ELSE IF ( SIDE_L .EQ. SIDE_3 ) THEN

C  Assignation

	    DO J = JOLD_L, JNEW_L-1
	      AREA(J) = AREA_L(J)
	    ENDDO
	    DO J = JNEW_U+1, JOLD_U
	      AREA(J) = AREA_U(J)
	    ENDDO

C  Add grids as required

	    IF ( JNEW_L .EQ. JNEW_U ) THEN

	      GRID = GRID_AREA(PARS,XOLD,XNEW,YNEW_L,YNEW_U)
	      AREA(JNEW_L) = AREA_L(JNEW_L) + AREA_U(JNEW_L) + GRID

	    ELSE

	      GRID = GRID_AREA(PARS,XOLD,XNEW,YNEW_L,YG(JNEW_L+1))
	      AREA(JNEW_L) = AREA_L(JNEW_L) + GRID
	      GRID = GRID_AREA(PARS,XOLD,XNEW,YG(JNEW_U),YNEW_U)
	      AREA(JNEW_U) = AREA_U(JNEW_U) + GRID
	      DO J = JNEW_L+1, JNEW_U-1
	        AREA(J) = GRID_AREA(PARS,XOLD,XNEW,YG(J),YG(J+1))
	      ENDDO

	    ENDIF

	  ENDIF

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, NYBOX, PARS,
     I       JNEW_L, JCNR, JNEW_U,
     I       XNEW, YNEW_L, YNEW_U, XCNR, YCNR,
     O       AREA )

C  input/output

	INTEGER		NYDIM, NYBOX
	REAL*8		YG(NYDIM), PARS(6,4)
	INTEGER		JCNR, JNEW_L, JNEW_U
	REAL*8		XNEW,  YNEW_L, YNEW_U, XCNR, YCNR
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX
	PARAMETER	( LOCAL_NYBOX = 5000 )
	REAL*8		AREA_L(LOCAL_NYBOX), AREA_U(LOCAL_NYBOX)

C  Side slicing from corner to next lower boundary (side = 4)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, PARS,
     I       JCNR, JNEW_L, XCNR, YCNR, XNEW, YNEW_L,
     O       AREA_L )
	DO J = JNEW_L, JCNR - 1
	  AREA(J) = AREA_L(J)
	ENDDO

C  Side slicing from corner to next upper boundary (side = 1)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_1, PARS,
     I       JCNR, JNEW_U, XCNR, YCNR, XNEW, YNEW_U,
     O       AREA_U )
	DO J = JCNR+1, JNEW_U
	  AREA(J) = AREA_U(J)
	ENDDO

C  Add contributions in corner square

	AREA(JCNR) = AREA_L(JCNR) + AREA_U(JCNR)

C  Finish

	END

C	  

	SUBROUTINE SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, NYBOX, PARS,
     I       JOLD_L, JCNR, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U, XCNR, YCNR,
     O       AREA )

C  input/output

	INTEGER		NYDIM, NYBOX
	REAL*8		YG(NYDIM), PARS(6,4)
	INTEGER		JCNR, JOLD_L, JOLD_U
	REAL*8		XOLD,  YOLD_L, YOLD_U, XCNR, YCNR
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX
	PARAMETER	( LOCAL_NYBOX = 5000 )
	REAL*8		AREA_L(LOCAL_NYBOX), AREA_U(LOCAL_NYBOX)

C  Side slicing from lower boundary to corner (side = 3)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, PARS,
     I       JOLD_L, JCNR, XOLD, YOLD_L, XCNR, YCNR,
     O       AREA_L )
	DO J = JOLD_L, JCNR - 1
	  AREA(J) = AREA_L(J)
	ENDDO

C  Side slicing from upper boundary to corner (side = 2)

	CALL SINGLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_2, PARS,
     I       JOLD_U, JCNR, XOLD, YOLD_U, XCNR, YCNR,
     O       AREA_U )
	DO J = JCNR+1, JOLD_U
	  AREA(J) = AREA_U(J)
	ENDDO

C  Add contributions in corner square

	AREA(JCNR) = AREA_L(JCNR) + AREA_U(JCNR)

C  Finish

	END

C

	SUBROUTINE SINGLE_TOPBOT_SLICER
     I   ( YG, NYDIM, NYBOX, SIDE_A, PARS, DO_TOP, ORNT,
     I     JOLD_L, JOLD_U, JCNR, JNEW_L, JNEW_U,
     I     XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U, XCNR, YCNR,
     O     AREA )

C  input/output

	INTEGER		NYDIM, NYBOX, SIDE_A, DO_TOP, ORNT
	REAL*8		YG(NYDIM), PARS(6,4)
	INTEGER		JOLD_L, JOLD_U, JCNR, JNEW_L, JNEW_U
	REAL*8		XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U
	REAL*8		XCNR, YCNR
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables (including external functions)

	INTEGER		J, JMAX, JMIN, LOCAL_NYBOX, ASCT, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCT = +1 )
        INTEGER		JNEW_L_F, JNEW_U_F, JOLD_L_D, JOLD_U_D
	REAL*8		AREA_F(LOCAL_NYBOX), AREA_D(LOCAL_NYBOX)
	REAL*8		XNEW_F, YNEW_L_F, YNEW_U_F
	REAL*8		XOLD_D, YOLD_L_D, YOLD_U_D, SIDEFUNCTION
	EXTERNAL	OFFSET, SIDEFUNCTION

C  Bottom slicer
C  #############

	IF ( DO_TOP .EQ. -1 ) THEN

c  Initialise

	  JMAX = MAX(JOLD_U,JNEW_U)
	  DO J = JCNR, JMAX
	    AREA_F(J) = 0.0
	    AREA_D(J) = 0.0
	  ENDDO

C  first slice ( lower side = 4 )

	  XNEW_F   = XCNR
	  YNEW_L_F = YCNR
	  YNEW_U_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_A )
	  JNEW_L_F = JCNR
	  JNEW_U_F = OFFSET ( 1, YG, NYDIM, ASCT, YNEW_U_F )
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_A, PARS,
     I       JOLD_L, JOLD_U, JNEW_L_F, JNEW_U_F, -ORNT, ORNT,
     I       XOLD, XNEW_F, YOLD_L, YOLD_U, YNEW_L_F, YNEW_U_F,
     O       AREA_F )

C  second slice ( lower side = 3 )

	  JOLD_L_D = JCNR
	  JOLD_U_D = JNEW_U_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_A, PARS,
     I       JOLD_L_D, JOLD_U_D, JNEW_L, JNEW_U, ORNT, ORNT,
     I       XOLD_D, XNEW, YOLD_L_D, YOLD_U_D, YNEW_L, YNEW_U,
     O       AREA_D )

C  put the areas together (straightforward)

	  JMAX = MAX(JOLD_U,JNEW_U)
	  DO J = JCNR, JMAX
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

	ENDIF

C  Top slicer
C  ##########

	IF ( DO_TOP .EQ. +1 ) THEN

c  Initialise

	  JMIN = MIN(JOLD_L,JNEW_L)
	  DO J = JMIN, JCNR
	    AREA_F(J) = 0.0
	    AREA_D(J) = 0.0
	  ENDDO

C  first slice ( upper side = 1 )

	  XNEW_F   = XCNR
	  YNEW_U_F = YCNR
	  YNEW_L_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_A )
	  JNEW_U_F = JCNR
	  JNEW_L_F = OFFSET ( 1, YG, NYDIM, ASCT, YNEW_L_F ) 
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_A, SIDE_1, PARS,
     I       JOLD_L, JOLD_U, JNEW_L_F, JNEW_U_F, ORNT, ORNT,
     I       XOLD, XNEW_F, YOLD_L, YOLD_U, YNEW_L_F, YNEW_U_F,
     O       AREA_F )

C  second slice ( upper side = 2 )

	  JOLD_U_D = JCNR
	  JOLD_L_D = JNEW_L_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_A, SIDE_2, PARS,
     I       JOLD_L_D, JOLD_U_D, JNEW_L, JNEW_U, -ORNT, ORNT,
     I       XOLD_D, XNEW, YOLD_L_D, YOLD_U_D, YNEW_L, YNEW_U,
     O       AREA_D )

C  put the areas together (straightforward)

	  JMIN = MIN(JOLD_L,JNEW_L)
	  DO J = JMIN, JCNR
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE DOUBLE_LEFT_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JNEW_L, JCNR_F, JCNR_D, JNEW_U, XNEW, YNEW_L, YNEW_U,
     I       XCNR_F, YCNR_F, XCNR_D, YCNR_D,
     O       AREA )

C  input/output

	INTEGER		NYDIM, NYBOX, ORNT
	REAL*8		YG(NYDIM)
	INTEGER		JNEW_L, JCNR_F, JCNR_D, JNEW_U
	REAL*8		XNEW, YNEW_L, YNEW_U, PARS(6,4)
	REAL*8		XCNR_F, YCNR_F, XCNR_D, YCNR_D
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
        INTEGER		JNEW_L_F, JNEW_U_F, JOLD_L_D, JOLD_U_D
	REAL*8		AREA_F(LOCAL_NYBOX), AREA_D(LOCAL_NYBOX)
	REAL*8		XNEW_F, YNEW_L_F, YNEW_U_F, SIDEFUNCTION
	REAL*8		XOLD_D, YOLD_L_D, YOLD_U_D
	EXTERNAL	OFFSET, SIDEFUNCTION

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JCNR_D, JNEW_U
	    AREA_F(J) = 0.0
	    AREA_D(J) = 0.0
	  ENDDO

C  first slice containing corner

	  XNEW_F   = XCNR_D
	  YNEW_L_F = YCNR_D
	  YNEW_U_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_1 )
	  JNEW_L_F = JCNR_D
	  JNEW_U_F = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_F)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_F, JCNR_F, JNEW_U_F,
     I       XNEW_F, YNEW_L_F, YNEW_U_F, XCNR_F, YCNR_F, 
     O       AREA_F )

C  Second  slice between lines

	  JOLD_L_D = JCNR_D
	  JOLD_U_D = JNEW_U_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L_D, JOLD_U_D, JNEW_L, JNEW_U, ORNT, ORNT,
     I       XOLD_D, XNEW, YOLD_L_D, YOLD_U_D, YNEW_L, YNEW_U,
     O       AREA_D )

C  put the areas together (straightforward)

          DO J = JCNR_D, JNEW_U
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JNEW_L, JCNR_D
		 IF ( j /= 0.0) THEN
			AREA_F(J) = 0.0
			AREA_D(J) = 0.0
	     ENDIF
	  ENDDO

C  first slice containing corner

	  XNEW_F   = XCNR_D
	  YNEW_U_F = YCNR_D
	  YNEW_L_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_4 )
	  JNEW_U_F = JCNR_D
	  JNEW_L_F = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_F)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_F, JCNR_F, JNEW_U_F,
     I       XNEW_F, YNEW_L_F, YNEW_U_F, XCNR_F, YCNR_F,
     O       AREA_F )

C  Second  slice between lines

	  JOLD_U_D = JCNR_D
	  JOLD_L_D = JNEW_L_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L_D, JOLD_U_D, JNEW_L, JNEW_U, -ORNT, ORNT,
     I       XOLD_D, XNEW, YOLD_L_D, YOLD_U_D, YNEW_L, YNEW_U,
     O       AREA_D )

C  put the areas together (straightforward)

          DO J = JNEW_L, JCNR_D
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE DOUBLE_RIGHT_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JOLD_L, JCNR_F, JCNR_D, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U, XCNR_F, YCNR_F, XCNR_D, YCNR_D,
     O       AREA )

C  input/output

	INTEGER		NYDIM, NYBOX, ORNT
	REAL*8		YG(NYDIM)
	INTEGER		JOLD_L, JCNR_F, JCNR_D, JOLD_U
	REAL*8		XOLD, YOLD_L, YOLD_U, PARS(6,4)
	REAL*8		XCNR_F, YCNR_F, XCNR_D, YCNR_D
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
        INTEGER		JNEW_L_F, JNEW_U_F, JOLD_L_D, JOLD_U_D
	REAL*8		AREA_F(LOCAL_NYBOX), AREA_D(LOCAL_NYBOX)
	REAL*8		XNEW_F, YNEW_L_F, YNEW_U_F, SIDEFUNCTION
	REAL*8		XOLD_D, YOLD_L_D, YOLD_U_D
	EXTERNAL	OFFSET, SIDEFUNCTION

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JOLD_L, JCNR_F
	    AREA_F(J) = 0.0
	    AREA_D(J) = 0.0
	  ENDDO

C  first slice for lines 1 and 3

	  XNEW_F   = XCNR_F
	  YNEW_U_F = YCNR_F
	  YNEW_L_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_3 )
	  JNEW_U_F = JCNR_F
	  JNEW_L_F = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_F)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L, JOLD_U, JNEW_L_F, JNEW_U_F, ORNT, ORNT,
     I       XOLD, XNEW_F, YOLD_L, YOLD_U, YNEW_L_F, YNEW_U_F,
     O       AREA_F )

C  Second  slice to final corner (lines 2 and 3)

	  JOLD_L_D = JNEW_L_F
	  JOLD_U_D = JNEW_U_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_D, JCNR_D, JOLD_U_D,
     I       XOLD_D, YOLD_L_D, YOLD_U_D, XCNR_D, YCNR_D,
     O       AREA_D )

C  put the areas together (straightforward)

          DO J = JOLD_L, JCNR_F
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JCNR_F, JOLD_U
	    AREA_F(J) = 0.0
	    AREA_D(J) = 0.0
	  ENDDO

C  first slice for lines 2 and 4

	  XNEW_F   = XCNR_F
	  YNEW_L_F = YCNR_F
	  YNEW_U_F = SIDEFUNCTION ( PARS, XNEW_F, SIDE_2 )
	  JNEW_L_F = JCNR_F
	  JNEW_U_F = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_F)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L, JOLD_U, JNEW_L_F, JNEW_U_F, -ORNT, ORNT,
     I       XOLD, XNEW_F, YOLD_L, YOLD_U, YNEW_L_F, YNEW_U_F,
     O       AREA_F )

C  Second  slice to final corner (lines 2 and 3)

	  JOLD_L_D = JNEW_L_F
	  JOLD_U_D = JNEW_U_F
	  XOLD_D   = XNEW_F
	  YOLD_L_D = YNEW_L_F
	  YOLD_U_D = YNEW_U_F
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_D, JCNR_D, JOLD_U_D,
     I       XOLD_D, YOLD_L_D, YOLD_U_D, XCNR_D, YCNR_D,
     O       AREA_D )

C  put the areas together (straightforward)

          DO J = JCNR_F, JOLD_U
	    AREA(J) = AREA_F(J) + AREA_D(J)
	  ENDDO

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE DOUBLE_MIDDLE_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JOLD_L, JOLD_U, JCNR_F, JCNR_D, JNEW_L, JNEW_U, 
     I       XOLD, XNEW, YOLD_L, YOLD_U, YNEW_L, YNEW_U,
     I       XCNR_F, YCNR_F, XCNR_D, YCNR_D,
     O       AREA )

C  input/output

	INTEGER		NYDIM, NYBOX, ORNT
	REAL*8		YG(NYDIM)
	INTEGER		JCNR_F, JCNR_D
	INTEGER		JOLD_L, JNEW_L, JOLD_U, JNEW_U
	REAL*8		XOLD, YOLD_L, YOLD_U
	REAL*8		XCNR_F, YCNR_F, XCNR_D, YCNR_D
	REAL*8		XNEW, YNEW_L, YNEW_U, PARS(6,4)
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
	REAL*8		AREA_S1(LOCAL_NYBOX), AREA_S2(LOCAL_NYBOX)
	REAL*8		AREA_S3(LOCAL_NYBOX), SIDEFUNCTION
	EXTERNAL	OFFSET, SIDEFUNCTION

        INTEGER		JNEW_L_S1, JNEW_U_S1, JOLD_L_S1, JOLD_U_S1
        INTEGER		JNEW_L_S2, JNEW_U_S2, JOLD_L_S2, JOLD_U_S2
        INTEGER		JNEW_L_S3, JNEW_U_S3, JOLD_L_S3, JOLD_U_S3

	REAL*8		XOLD_S1, YOLD_L_S1, YOLD_U_S1
	REAL*8		XNEW_S1, YNEW_L_S1, YNEW_U_S1
	REAL*8		XOLD_S2, YOLD_L_S2, YOLD_U_S2
	REAL*8		XNEW_S2, YNEW_L_S2, YNEW_U_S2
	REAL*8		XOLD_S3, YOLD_L_S3, YOLD_U_S3
	REAL*8		XNEW_S3, YNEW_L_S3, YNEW_U_S3

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JCNR_F, JCNR_D
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (left edge to first corner)

	  XOLD_S1   = XOLD
	  YOLD_L_S1 = YOLD_L
	  YOLD_U_S1 = YOLD_U
	  JOLD_L_S1 = JOLD_L
	  JOLD_U_S1 = JOLD_U
	  XNEW_S1   = XCNR_F
	  YNEW_L_S1 = YCNR_F
	  YNEW_U_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_1 )
	  JNEW_L_S1 = JCNR_F
	  JNEW_U_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S1)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_1, PARS,
     I       JOLD_L_S1, JOLD_U_S1, JNEW_L_S1, JNEW_U_S1, -ORNT, ORNT,
     I       XOLD_S1, XNEW_S1, YOLD_L_S1,
     I       YOLD_U_S1, YNEW_L_S1, YNEW_U_S1,
     O       AREA_S1 )

C  second slice (first corner to second corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XCNR_D
	  YNEW_U_S2 = YCNR_D
	  YNEW_L_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_3 )
	  JNEW_U_S2 = JCNR_D
	  JNEW_L_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (second corner to right edge)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  XNEW_S3   = XNEW
	  YNEW_U_S3 = YNEW_U
	  YNEW_L_S3 = YNEW_L
	  JNEW_U_S3 = JNEW_U
	  JNEW_L_S3 = JNEW_L
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_2, PARS,
     I       JOLD_L_S3, JOLD_U_S3, JNEW_L_S3, JNEW_U_S3, -ORNT, ORNT,
     I       XOLD_S3, XNEW_S3, YOLD_L_S3,
     I       YOLD_U_S3, YNEW_L_S3, YNEW_U_S3,
     O       AREA_S3 )

C  put the areas together (straightforward)

          DO J = JCNR_F, JCNR_D
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JCNR_D, JCNR_F
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (left edge to first corner)

	  XOLD_S1   = XOLD
	  YOLD_L_S1 = YOLD_L
	  YOLD_U_S1 = YOLD_U
	  JOLD_L_S1 = JOLD_L
	  JOLD_U_S1 = JOLD_U
	  XNEW_S1   = XCNR_F
	  YNEW_U_S1 = YCNR_F
	  YNEW_L_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_4 )
	  JNEW_U_S1 = JCNR_F
	  JNEW_L_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S1)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_1, PARS,
     I       JOLD_L_S1, JOLD_U_S1, JNEW_L_S1, JNEW_U_S1, ORNT, ORNT,
     I       XOLD_S1, XNEW_S1, YOLD_L_S1,
     I       YOLD_U_S1, YNEW_L_S1, YNEW_U_S1,
     O       AREA_S1 )

C  second slice (first corner to second corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XCNR_D
	  YNEW_L_S2 = YCNR_D
	  YNEW_U_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_2 )
	  JNEW_L_S2 = JCNR_D
	  JNEW_U_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, -ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (second corner to right edge)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  XNEW_S3   = XNEW
	  YNEW_U_S3 = YNEW_U
	  YNEW_L_S3 = YNEW_L
	  JNEW_U_S3 = JNEW_U
	  JNEW_L_S3 = JNEW_L
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_2, PARS,
     I       JOLD_L_S3, JOLD_U_S3, JNEW_L_S3, JNEW_U_S3, ORNT, ORNT,
     I       XOLD_S3, XNEW_S3, YOLD_L_S3,
     I       YOLD_U_S3, YNEW_L_S3, YNEW_U_S3,
     O       AREA_S3 )

C  put the areas together (straightforward)

	  DO J = JCNR_D, JCNR_F
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE TRIPLE_LEFT_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JC1, JC2, JC3, JNEW_L, JNEW_U,
     I       XNEW, YNEW_L, YNEW_U, XC1, YC1, XC2, YC2, XC3, YC3,
     O       AREA )

C  Order of corners is 1-4-2 (orient=1), 1-2-4 (orient=-1)

C  input/output

	INTEGER		NYDIM, NYBOX
	REAL*8		YG(NYDIM)
	INTEGER		JC1, JC2, JC3, ORNT
	INTEGER		JNEW_L, JNEW_U
	REAL*8		XNEW, YNEW_L, YNEW_U, PARS(6,4)
	REAL*8		XC1, YC1, XC2, YC2, XC3, YC3
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
	REAL*8		AREA_S1(LOCAL_NYBOX), AREA_S2(LOCAL_NYBOX)
	REAL*8		AREA_S3(LOCAL_NYBOX), SIDEFUNCTION
	EXTERNAL	OFFSET, SIDEFUNCTION

        INTEGER		JNEW_L_S1, JNEW_U_S1
        INTEGER		JNEW_L_S2, JNEW_U_S2, JOLD_L_S2, JOLD_U_S2
        INTEGER		JNEW_L_S3, JNEW_U_S3, JOLD_L_S3, JOLD_U_S3
	REAL*8		XNEW_S1, YNEW_L_S1, YNEW_U_S1
	REAL*8		XOLD_S2, YOLD_L_S2, YOLD_U_S2
	REAL*8		XNEW_S2, YNEW_L_S2, YNEW_U_S2
	REAL*8		XOLD_S3, YOLD_L_S3, YOLD_U_S3
	REAL*8		XNEW_S3, YNEW_L_S3, YNEW_U_S3

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JC2, JC3
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (first corner to second corner)

	  XNEW_S1   = XC2
	  YNEW_L_S1 = YC2
	  YNEW_U_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_1 )
	  JNEW_L_S1 = JC2
	  JNEW_U_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S1)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_S1, JC1, JNEW_U_S1,
     I       XNEW_S1, YNEW_L_S1, YNEW_U_S1, XC1, YC1,
     O       AREA_S1 )

C  second slice (second corner to third corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC3
	  YNEW_U_S2 = YC3
	  YNEW_L_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_3 )
	  JNEW_U_S2 = JC3
	  JNEW_L_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (third corner to right edge)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  XNEW_S3   = XNEW
	  YNEW_U_S3 = YNEW_U
	  YNEW_L_S3 = YNEW_L
	  JNEW_U_S3 = JNEW_U
	  JNEW_L_S3 = JNEW_L
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_2, PARS,
     I       JOLD_L_S3, JOLD_U_S3, JNEW_L_S3, JNEW_U_S3, -ORNT, ORNT,
     I       XOLD_S3, XNEW_S3, YOLD_L_S3,
     I       YOLD_U_S3, YNEW_L_S3, YNEW_U_S3,
     O       AREA_S3 )

C  put the areas together (straightforward)

          DO J = JC2, JC3
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JC3, JC2
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (first corner to second corner)

	  XNEW_S1   = XC2
	  YNEW_U_S1 = YC2
	  YNEW_L_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_4 )
	  JNEW_U_S1 = JC2
	  JNEW_L_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S1)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_S1, JC1, JNEW_U_S1,
     I       XNEW_S1, YNEW_L_S1, YNEW_U_S1, XC1, YC1,
     O       AREA_S1 )

C  second slice (second corner to third corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC3
	  YNEW_L_S2 = YC3
	  YNEW_U_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_2 )
	  JNEW_L_S2 = JC3
	  JNEW_U_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, -ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (third corner to right edge)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  XNEW_S3   = XNEW
	  YNEW_U_S3 = YNEW_U
	  YNEW_L_S3 = YNEW_L
	  JNEW_U_S3 = JNEW_U
	  JNEW_L_S3 = JNEW_L
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_2, PARS,
     I       JOLD_L_S3, JOLD_U_S3, JNEW_L_S3, JNEW_U_S3, ORNT, ORNT,
     I       XOLD_S3, XNEW_S3, YOLD_L_S3,
     I       YOLD_U_S3, YNEW_L_S3, YNEW_U_S3,
     O       AREA_S3 )

C  put the areas together (straightforward)

	  DO J = JC3, JC2
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

	ENDIF

C  Finish

	END

C  

	SUBROUTINE TRIPLE_RIGHT_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JC1, JC2, JC3, JOLD_L, JOLD_U,
     I       XOLD, YOLD_L, YOLD_U, XC1, YC1, XC2, YC2, XC3, YC3,
     O       AREA )

C  Order of corners is 4-2-3 (orient=1), 2-4-3 (orient=-1)

C  input/output

	INTEGER		NYDIM, NYBOX
	REAL*8		YG(NYDIM)
	INTEGER		JC1, JC2, JC3, ORNT
	INTEGER		JOLD_L, JOLD_U
	REAL*8		XOLD, YOLD_L, YOLD_U, PARS(6,4)
	REAL*8		XC1, YC1, XC2, YC2, XC3, YC3
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
	REAL*8		AREA_S1(LOCAL_NYBOX), AREA_S2(LOCAL_NYBOX)
	REAL*8		AREA_S3(LOCAL_NYBOX), SIDEFUNCTION
	EXTERNAL	OFFSET, SIDEFUNCTION

        INTEGER		JNEW_L_S1, JNEW_U_S1, JOLD_L_S1, JOLD_U_S1
        INTEGER		JNEW_L_S2, JNEW_U_S2, JOLD_L_S2, JOLD_U_S2
        INTEGER		JOLD_L_S3, JOLD_U_S3
	REAL*8		XOLD_S1, YOLD_L_S1, YOLD_U_S1
	REAL*8		XNEW_S1, YNEW_L_S1, YNEW_U_S1
	REAL*8		XOLD_S2, YOLD_L_S2, YOLD_U_S2
	REAL*8		XNEW_S2, YNEW_L_S2, YNEW_U_S2
	REAL*8		XOLD_S3, YOLD_L_S3, YOLD_U_S3

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JC1, JC2
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (left edge to first corner)

	  XOLD_S1   = XOLD
	  YOLD_L_S1 = YOLD_L
	  YOLD_U_S1 = YOLD_U
	  JOLD_L_S1 = JOLD_L
	  JOLD_U_S1 = JOLD_U
	  XNEW_S1   = XC1
	  YNEW_L_S1 = YC1
	  YNEW_U_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_1 )
	  JNEW_L_S1 = JC1
	  JNEW_U_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S1)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_1, PARS,
     I       JOLD_L_S1, JOLD_U_S1, JNEW_L_S1, JNEW_U_S1, -ORNT, ORNT,
     I       XOLD_S1, XNEW_S1, YOLD_L_S1,
     I       YOLD_U_S1, YNEW_L_S1, YNEW_U_S1,
     O       AREA_S1 )

C  second slice (first corner to second corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC2
	  YNEW_U_S2 = YC2
	  YNEW_L_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_3 )
	  JNEW_U_S2 = JC2
	  JNEW_L_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (second corner to third corner)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_S3, JC3, JOLD_U_S3,
     I       XOLD_S3, YOLD_L_S3, YOLD_U_S3, XC3, YC3,
     O       AREA_S3 )

C  put the areas together (straightforward)

          DO J = JC1, JC2
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JC2, JC1
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (left edge to first corner)

	  XOLD_S1   = XOLD
	  YOLD_L_S1 = YOLD_L
	  YOLD_U_S1 = YOLD_U
	  JOLD_L_S1 = JOLD_L
	  JOLD_U_S1 = JOLD_U
	  XNEW_S1   = XC1
	  YNEW_U_S1 = YC1
	  YNEW_L_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_4 )
	  JNEW_U_S1 = JC1
	  JNEW_L_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S1)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_1, PARS,
     I       JOLD_L_S1, JOLD_U_S1, JNEW_L_S1, JNEW_U_S1, ORNT, ORNT,
     I       XOLD_S1, XNEW_S1, YOLD_L_S1,
     I       YOLD_U_S1, YNEW_L_S1, YNEW_U_S1,
     O       AREA_S1 )

C  second slice (first corner to second corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC2
	  YNEW_L_S2 = YC2
	  YNEW_U_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_2 )
	  JNEW_L_S2 = JC2
	  JNEW_U_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, -ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (second corner to third corner)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_S3, JC3, JOLD_U_S3,
     I       XOLD_S3, YOLD_L_S3, YOLD_U_S3, XC3, YC3,
     O       AREA_S3 )

C  put the areas together (straightforward)

	  DO J = JC2, JC1
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

	ENDIF

C  Finish

	END

C	  

	SUBROUTINE QUADRUPLE_SLICER
     I     ( YG, NYDIM, NYBOX, ORNT, PARS,
     I       JC1, JC2, JC3, JC4,
     I       XC1, YC1, XC2, YC2, XC3, YC3, XC4, YC4,
     O       AREA )

C  Order of corners is 1-4-2-3 (orient=1), 1-2-4-3 (orient=-1)

C  input/output

	INTEGER		NYDIM, NYBOX
	REAL*8		YG(NYDIM), PARS(6,4)
	INTEGER		JC1, JC2, JC3, JC4, ORNT
	REAL*8		XC1, YC1, XC2, YC2, XC3, YC3, XC4, YC4
	REAL*8		AREA(NYBOX)

C  side parameters

	INTEGER		SIDE_1, SIDE_2, SIDE_3, SIDE_4
	PARAMETER	( SIDE_1 = 1, SIDE_2 = 2 )
	PARAMETER	( SIDE_3 = 3, SIDE_4 = 4 )

C  local variables

	INTEGER		J, LOCAL_NYBOX, ASCENDING, OFFSET
	PARAMETER	( LOCAL_NYBOX = 5000, ASCENDING = +1 )
	REAL*8		AREA_S1(LOCAL_NYBOX), AREA_S2(LOCAL_NYBOX)
	REAL*8		AREA_S3(LOCAL_NYBOX), SIDEFUNCTION
	EXTERNAL	OFFSET, SIDEFUNCTION

        INTEGER		JNEW_L_S1, JNEW_U_S1
        INTEGER		JNEW_L_S2, JNEW_U_S2, JOLD_L_S2, JOLD_U_S2
        INTEGER		JOLD_L_S3, JOLD_U_S3
	REAL*8		XNEW_S1, YNEW_L_S1, YNEW_U_S1
	REAL*8		XOLD_S2, YOLD_L_S2, YOLD_U_S2
	REAL*8		XNEW_S2, YNEW_L_S2, YNEW_U_S2
	REAL*8		XOLD_S3, YOLD_L_S3, YOLD_U_S3

C  Orientation = 1
C  ===============

	IF ( ORNT .EQ. 1 ) THEN

C  Initialise

	  DO J = JC2, JC3
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (first corner to second corner)

	  XNEW_S1   = XC2
	  YNEW_L_S1 = YC2
	  YNEW_U_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_1 )
	  JNEW_L_S1 = JC2
	  JNEW_U_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S1)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_S1, JC1, JNEW_U_S1,
     I       XNEW_S1, YNEW_L_S1, YNEW_U_S1, XC1, YC1,
     O       AREA_S1 )

C  second slice (second corner to third corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC3
	  YNEW_U_S2 = YC3
	  YNEW_L_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_3 )
	  JNEW_U_S2 = JC3
	  JNEW_L_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_3, SIDE_1, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (third corner to fourth corner)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_S3, JC4, JOLD_U_S3,
     I       XOLD_S3, YOLD_L_S3, YOLD_U_S3, XC4, YC4,
     O       AREA_S3 )

C  put the areas together (straightforward)

          DO J = JC2, JC3
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

C  Orientation = -1
C  ================

	ELSE IF ( ORNT .EQ. -1 ) THEN

C  Initialise

	  DO J = JC3, JC2
	    AREA_S1(J) = 0.0
	    AREA_S2(J) = 0.0
	    AREA_S3(J) = 0.0
	  ENDDO

C  first slice (first corner to second corner)

	  XNEW_S1   = XC2
	  YNEW_U_S1 = YC2
	  YNEW_L_S1 = SIDEFUNCTION ( PARS, XNEW_S1, SIDE_4 )
	  JNEW_U_S1 = JC2
	  JNEW_L_S1 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_L_S1)
	  CALL SINGLE_LEFT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JNEW_L_S1, JC1, JNEW_U_S1,
     I       XNEW_S1, YNEW_L_S1, YNEW_U_S1, XC1, YC1,
     O       AREA_S1 )

C  second slice (second corner to third corner)

	  XOLD_S2   = XNEW_S1
	  YOLD_L_S2 = YNEW_L_S1
	  YOLD_U_S2 = YNEW_U_S1
	  JOLD_L_S2 = JNEW_L_S1
	  JOLD_U_S2 = JNEW_U_S1
	  XNEW_S2   = XC3
	  YNEW_L_S2 = YC3
	  YNEW_U_S2 = SIDEFUNCTION ( PARS, XNEW_S2, SIDE_2 )
	  JNEW_L_S2 = JC3
	  JNEW_U_S2 = OFFSET(1,YG,NYDIM,ASCENDING,YNEW_U_S2)
	  CALL DOUBLESIDE_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, SIDE_4, SIDE_2, PARS,
     I       JOLD_L_S2, JOLD_U_S2, JNEW_L_S2, JNEW_U_S2, -ORNT, ORNT,
     I       XOLD_S2, XNEW_S2, YOLD_L_S2,
     I       YOLD_U_S2, YNEW_L_S2, YNEW_U_S2,
     O       AREA_S2 )

C  third slice (third corner to fourth corner)

	  XOLD_S3   = XNEW_S2
	  YOLD_L_S3 = YNEW_L_S2
	  YOLD_U_S3 = YNEW_U_S2
	  JOLD_L_S3 = JNEW_L_S2
	  JOLD_U_S3 = JNEW_U_S2
	  CALL SINGLE_RIGHT_SLICER
     I     ( YG, NYDIM, LOCAL_NYBOX, PARS,
     I       JOLD_L_S3, JC4, JOLD_U_S3,
     I       XOLD_S3, YOLD_L_S3, YOLD_U_S3, XC4, YC4,
     O       AREA_S3 )

C  put the areas together (straightforward)

	  DO J = JC3, JC2
	    AREA(J) = AREA_S1(J) + AREA_S2(J) + AREA_S3(J)
	  ENDDO

	ENDIF

C  Finish

	END

C

C  Offset function
	
!xliu	INTEGER FUNCTION OFFSET1
!xliu     C    ( START, GRID, NGRID, ASCENDING, VALUE )
!xliu
!xliu	INTEGER		START, NGRID, N, ASCENDING
!xliu	REAL*8		GRID(NGRID), VALUE
!xliu	LOGICAL		LOOP
!xliu
!xliu	OFFSET = 0
!xliu	IF ( ASCENDING. EQ.1 ) THEN
!xliu	  LOOP = .TRUE.
!xliu	  N = START
!xliu	  !loop1: DO WHILE (LOOP)
!xliu	  !  IF ( VALUE > GRID(N) ) THEN
!xliu	  !    OFFSET = N
!xliu	  !	  EXIT loop1
!xliu	  !  ENDIF
!xliu	  !  N = N + 1
!xliu	  !ENDDO loop1
!xliu	  !print *, N, NGRID
!xliu	  !WRITE(*, '(6f10.4)') GRID(1:NGRID), VALUES
!xliu	  DO WHILE (LOOP)
!xliu	    N = N + 1
!xliu	    IF (N .LT. NGRID .AND. VALUE .LT.GRID(N)) THEN
!xliu		   LOOP = .FALSE.
!xliu		   OFFSET = N - 1
!xliu	    ELSE IF (N .EQ. NGRID) THEN
!xliu		   LOOP = .FALSE.
!xliu		   OFFSET = NGRID
!xliu		ENDIF
!xliu	  ENDDO
!xliu	ELSE
!xliu	  LOOP = .TRUE.
!xliu	  N = START
!xliu	  DO WHILE (LOOP)
!xliu	    N = N - 1
!xliu	    IF (VALUE.GT.GRID(N)) THEN
!xliu	      LOOP = .FALSE.
!xliu	      OFFSET = N + 1
!xliu	    ENDIF
!xliu	  ENDDO
!xliu	ENDIF
!xliu
!xliu	END
!xliu
	INTEGER FUNCTION OFFSET ( START, GRID, NGRID, ASCENDING, VALUE )

	INTEGER		START, NGRID, N, ASCENDING
	REAL*8		GRID(NGRID), VALUE
	LOGICAL		LOOP

	OFFSET = 0

	!IF ( ASCENDING. EQ.1 ) THEN
        !     offset = MAXVAL( MAXLOC( grid(1:ngrid), MASK=(grid(1:ngrid) <= value ) ) )
        !ELSE
        !     offset = MAXVAL( MAXLOC( grid(1:start), MASK=(grid(1:start) <= value ) ) )
        !END IF
        !RETURN

	IF ( ASCENDING .EQ.1 ) THEN
	   offset = MAXVAL( MAXLOC( grid(1:ngrid), 
	1	MASK=(grid(1:ngrid) < value ) ) )
	ELSE
	  LOOP = .TRUE.
	  N = START
	  DO WHILE (LOOP)
	    N = N - 1
	    IF (VALUE.GT.GRID(N)) THEN
	      LOOP = .FALSE.
	      OFFSET = N + 1
	    ENDIF
	  ENDDO
	ENDIF

	END

C  Tessellation_sphericalfunc.f
C  ============================

C  Compilation of functions and modules required to carry out the
C  tessellation algorithm in spherical geometry. Comprises

C	SIDEFUNCTION
C	Great circle function expressing Latitude on GC, given Longitude

C	INV_SIDEFUNCTION
C	Great circle inverse function expressing Longitude on GC, given
C       Latitude, allowing for phase differences, and taking care that
C       Longitude value lies between appropriate corner limits.

C	CORNER_AREA
C	Returns the area of Spherical triangle bound by one pixel side,
C       one latitude ordinate and one longitude ordinate.

C	GRID_AREA
C	Returns the area of single latitude/longitude bin on a sphere.

C       PARAMETER_SETUP
C       Calculates the parameters for each of the 4 sides of the footprint
C          PARS(1,SIDE) = K (see equation in text)
C          PARS(2,SIDE) = PHI_S (see equation in text)
C          PARS(3,SIDE) = Q (see text), only required for Corner_area
C          PARS(4,SIDE) = degrees to radains conversion (same for all sides)
C          PARS(5,SIDE) = Lower longitude limit (INV_SIDEFUNCTION)
C          PARS(6,SIDE) = Upper longitude limit (INV_SIDEFUNCTION)

C        Also returns the orientation of the pixel

C    ********************************************************************
C    *   Robert Spurr, November 1998					*
C    *   SAO, 60 Garden Street, Cambridge, MA 02138, USA		*
C    *   +1 (617) 496 7819; email rspurr@cfa.harvard.edu		*
C    *									*
C    *   Algorithm MAY be subject of licensing agreement		*
C    ********************************************************************

C  great circle Side function

	REAL*8 FUNCTION SIDEFUNCTION ( PARS, PHI, SIDE )
	REAL*8		PARS(6,4), PHI, DTR, HELP
	INTEGER		SIDE
	DTR  = PARS(4,SIDE)
	HELP = DSIN ( DTR * ( PHI - PARS(2,SIDE) ) )
	SIDEFUNCTION = DATAN ( PARS(1,SIDE) * HELP ) / DTR
	END

C  great circle inverse Side function

	REAL*8 FUNCTION INV_SIDEFUNCTION ( PARS, LAM, SIDE )
	REAL*8		PARS(6,4), LAM, DTR, HELP, PS, X, XR
	INTEGER		SIDE
	DTR  = PARS(4,SIDE)
	PS = PARS(2,SIDE)
	HELP = DTAN ( DTR * LAM ) / PARS(1,SIDE)
	HELP = DASIN(HELP)/DTR
	X = PS + HELP
	XR = X + 360.0D0

C	WRITE(*, *) '*** ', X, XR, SIDE, PARS(5, SIDE), PARS(6, SIDE),
C     * PS, HELP, LAM, DTR, PARS(1, SIDE)

	IF ( X.GE.PARS(5,SIDE).AND.X.LE.PARS(6,SIDE) ) THEN
	  INV_SIDEFUNCTION = X
	ELSE IF ( XR.GE.PARS(5,SIDE).AND.XR.LE.PARS(6,SIDE) ) THEN
	  INV_SIDEFUNCTION = XR
C  
	ELSE IF (X .LE. PARS(5, SIDE)) THEN
	  INV_SIDEFUNCTION = PS + 180.0 - HELP
	ELSE
	  INV_SIDEFUNCTION = PS - 180.0 - HELP
	ENDIF
	END

C  Corner area function

	REAL*8 FUNCTION CORNER_AREA (PARS,X1,X2,Y1,Y2,SIDE)
	REAL*8 		PARS(6,4),X1,X2,Y1,Y2
	INTEGER		SIDE
	REAL*8		DTR, Q1, Q2, ASTAR, DX, CORNER
	DTR = PARS(4,SIDE)
	Q1 = PARS(3,SIDE)*DCOS((X1-PARS(2,SIDE))*DTR)
	Q2 = PARS(3,SIDE)*DCOS((X2-PARS(2,SIDE))*DTR)
	ASTAR = DASIN(Q1)-DASIN(Q2)
	DX = ( X2 - X1 ) * DTR
	IF ( SIDE.EQ.1 ) THEN
	  CORNER = ASTAR - DX*DSIN(Y1*DTR)
	ELSE IF  ( SIDE.EQ.2 ) THEN
	  CORNER = ASTAR - DX*DSIN(Y2*DTR)
	ELSE IF  ( SIDE.EQ.3 ) THEN
	  CORNER = DX*DSIN(Y2*DTR) - ASTAR
	ELSE IF  ( SIDE.EQ.4 ) THEN
	  CORNER = DX*DSIN(Y1*DTR) - ASTAR
	ENDIF
	CORNER_AREA = DABS(CORNER) / DTR / DTR

C	IF (CORNER_AREA .GT. 1000.) THEN
C		 PRINT *, 'Bad Case', SIDE
C		 PRINT *,  X1, X2, Y1, Y2, DTR, Q1, Q2, ASTAR, DX, CORNER, CORNER_AREA
C	ENDIF
	END

C  Grid area function ( convention x2 > x1, Y2 > Y1 )

	REAL*8 FUNCTION GRID_AREA (PARS,X1,X2,Y1,Y2)
	REAL*8 		PARS(6,4),X1,X2,Y1,Y2,DTR
	DTR = PARS(4,1)
	GRID_AREA = (X2-X1)*(DSIN(Y2*DTR)-DSIN(Y1*DTR))/DTR
	END

C  parameter setup for great circles

	SUBROUTINE PARSETUP ( CC, PARS, ORIENT )

C  input/output

	REAL*8		CC(4,2),PARS(6,4)
	INTEGER		ORIENT

C  local variables

	REAL*8		L,K,L1(4),L2(4),P1(4),P2(4),DTR
	REAL*8		A,B,C,Q1,Q2,PS,Q,K0,L0
	INTEGER		SIDE

C  degrees to radians conversion

	DTR = DATAN(1.0D0)/45.0D0

C  Limiting latitude and Longitude values at corners

	SIDE = 1
	P2(SIDE) = CC(2,1)
	P1(SIDE) = CC(1,1)
	L2(SIDE) = CC(2,2)
	L1(SIDE) = CC(1,2)
	SIDE = 2
	P2(SIDE) = CC(3,1)
	P1(SIDE) = CC(2,1)
	L2(SIDE) = CC(3,2)
	L1(SIDE) = CC(2,2)
	SIDE = 3
	P2(SIDE) = CC(4,1)
	P1(SIDE) = CC(3,1)
	L2(SIDE) = CC(4,2)
	L1(SIDE) = CC(3,2)
	SIDE = 4
	P2(SIDE) = CC(1,1)
	P1(SIDE) = CC(4,1)
	L2(SIDE) = CC(1,2)
	L1(SIDE) = CC(4,2)

C  For each side

	DO SIDE = 1, 4

C  Constants involved in great circle equation and corner area results

	  A = DTAN(L1(SIDE)*DTR)
	  B = DTAN(L2(SIDE)*DTR)
	  C = DSIN((P2(SIDE)-P1(SIDE))*DTR)
	  Q1 = -A*DCOS(P2(SIDE)*DTR) + B*DCOS(P1(SIDE)*DTR)
	  Q2 = -A*DSIN(P2(SIDE)*DTR) + B*DSIN(P1(SIDE)*DTR)
	  K0 = DSQRT(Q1*Q1+Q2*Q2)/C
	  K = K0
	  PS = DATAN(Q2/Q1)/DTR
	  L0 = DATAN(K*DSIN((P1(SIDE)-PS)*DTR))/DTR
	  IF ( DABS((L1(SIDE)/L0)-1).GT.1.0D0)K=-K0
	  L = DSQRT(1.0D0+K*K)
	  Q = K/L

C  Assign parameters as listed above

	  PARS(1,SIDE) = K
	  PARS(2,SIDE) = PS
	  PARS(3,SIDE) = Q
	  PARS(4,SIDE) = DTR
	  PARS(5,SIDE) = MIN(P2(SIDE),P1(SIDE))
	  PARS(6,SIDE) = MAX(P2(SIDE),P1(SIDE))

	ENDDO

C  Orientation

	IF ( CC(4,1).LT.CC(2,1) ) THEN
	  ORIENT = +1
	ELSE
	  ORIENT = -1
	ENDIF

C  Finish

	END



	SUBROUTINE TESSELATIONS_PATCH1
     I    ( MAXRUNS, MAX_YDIM, MAX_XDIM, MAX_ADIM,
     I      NRUNS, FOOT_COORDS, GLOBAL_ALBEDO_1, 
     I      ETOP05_CLIMDATA, OXGRID, OYGRID,
     O      FOOT_ALBEDOS, FOOT_HEIGHTS, FOOT_AREAS )

C  ARGUMENTS
C  =========

C  input numbers

	INTEGER		MAX_ADIM, MAX_XDIM, MAX_YDIM
	INTEGER		MAXRUNS, NRUNS

C  latitude and longitudes + albedo set

	REAL*8		OXGRID(MAX_XDIM), OYGRID(MAX_YDIM)
	REAL		GLOBAL_ALBEDO_1(MAX_YDIM,MAX_ADIM)
	INTEGER		ETOP05_CLIMDATA(MAX_YDIM,MAX_ADIM)

C  input coordinates

	DOUBLE PRECISION FOOT_COORDS ( 5, 2, MAXRUNS )

C  Output albedos

	DOUBLE PRECISION FOOT_ALBEDOS ( MAXRUNS )
	DOUBLE PRECISION FOOT_HEIGHTS ( MAXRUNS )
	DOUBLE PRECISION FOOT_AREAS   ( MAXRUNS )

C  LOCAL VARIABLES
C  ===============

	INTEGER		MAX_XBOX, MAX_YBOX
	PARAMETER	( MAX_XBOX = 50,   MAX_YBOX = 50 )

C  geolocation input

	REAL*8		GEO_CC(5,2)

C  corner coordinates and side equation parameters

	REAL*8		PARS(6,4), CC(4,2)

C  Limits output

	INTEGER		ISTART1, ISTART2, ISTART3, ISTART4
	INTEGER		JSTART, JFINIS, ORIENT

C  box-grid offsets and limits

	INTEGER		ISTART1_T, ISTART2_T, ISTART3_T, ISTART4_T
	INTEGER		JSTART_T, JFINIS_T
	INTEGER		NXUSED, NYUSED

C  Box grids, tesselated areas and albedos in box

	REAL*8		XGRID_T(MAX_XBOX), YGRID_T(MAX_YBOX)
	REAL*8		AREA(MAX_YBOX,MAX_XBOX)
	REAL*8		ALBUSED(MAX_YBOX,MAX_XBOX)
	REAL*8		TPGUSED(MAX_YBOX,MAX_XBOX)

C  Circularity flag

	LOGICAL		CIRCULARITY

C  Special case flags

	LOGICAL		SPECIAL_SIDE1
	LOGICAL		SPECIAL_SIDE3

C  major output results (area sum and weighted albedo)

	REAL*8		SUM, ALBEDO, HEIGHT
	INTEGER		YLIMIT_LOWER(MAX_XBOX)
	INTEGER		YLIMIT_UPPER(MAX_XBOX)

C  other local variables

	INTEGER		I, J, I1, J1, IG, JG, IZ, RANK(4), K, M,N
	REAL*8		LIMA, MINL, LIM1, LIM2, EP
	LOGICAL		TODO(4)
	INTEGER		OFFSET
	EXTERNAL	OFFSET

C  NRUNS loop

	DO N = 1, NRUNS

C  initialise output

	  FOOT_ALBEDOS(N) = 0.0D0
	  FOOT_AREAS(N)   = 0.0D0

C  set local variables

	  DO I = 1, 5
	    DO J = 1, 2
	      GEO_CC(I,J) = FOOT_COORDS(I,J,N)
	    ENDDO
	  ENDDO

C  test for circularity (corners not yet ranked so must do generally)

	  LIMA = 180.0
	  CIRCULARITY = .FALSE.
	  DO K = 1, 4
	    IF ( GEO_CC(K,2) .GT. LIMA ) THEN
	      DO I = 1, 4
	        IF ( I.NE.K ) THEN
	          IF ( DABS(GEO_CC(I,2)-GEO_CC(K,2)).GT.LIMA ) THEN
	            GEO_CC(I,2) = 360.0 + GEO_CC(I,2)
	            CIRCULARITY = .TRUE.
	          ENDIF
	        ENDIF
	      ENDDO
	    ENDIF
	  ENDDO

C  Rank corners by increasing Longitude, 1 = smallest.

	  DO K = 1, 4
	    TODO(K) = .TRUE.
	  ENDDO
	  DO K = 1, 4
	    MINL = +1000.0
	    DO I = 1, 4
	      IF ( TODO(I) ) THEN
	        MINL = MIN(MINL,GEO_CC(I,2))
	        IF ( MINL .EQ. GEO_CC(I,2) ) RANK(K) = I
	      ENDIF
	    ENDDO
	    TODO(RANK(K)) = .FALSE.
	  ENDDO

C assign ranked corner coordinates for use in Tessellation algorithm

	  CC(1,1) = GEO_CC(RANK(1),2)
	  CC(1,2) = GEO_CC(RANK(1),1)
	  CC(3,1) = GEO_CC(RANK(4),2)
	  CC(3,2) = GEO_CC(RANK(4),1)
	  IF ( GEO_CC(RANK(2),1) .GT. CC(1,2) ) THEN
	    CC(2,1) = GEO_CC(RANK(2),2)
	    CC(2,2) = GEO_CC(RANK(2),1)
	    CC(4,1) = GEO_CC(RANK(3),2)
	    CC(4,2) = GEO_CC(RANK(3),1)
	  ELSE
	    CC(2,1) = GEO_CC(RANK(3),2)
	    CC(2,2) = GEO_CC(RANK(3),1)
	    CC(4,1) = GEO_CC(RANK(2),2)
	    CC(4,2) = GEO_CC(RANK(2),1)
	  ENDIF

C  exact values are not allowed, make small displacements

	  EP = 1.0D-05
	  IF (CC(1,1)-DFLOAT(INT(CC(1,1))).EQ.0.0 )CC(1,1)=CC(1,1)+EP
	  IF (CC(1,2)-DFLOAT(INT(CC(1,2))).EQ.0.0 )CC(1,2)=CC(1,2)+EP
	  IF (CC(2,1)-DFLOAT(INT(CC(2,1))).EQ.0.0 )CC(2,1)=CC(2,1)-EP
	  IF (CC(2,2)-DFLOAT(INT(CC(2,2))).EQ.0.0 )CC(2,2)=CC(2,2)-EP
	  IF (CC(3,1)-DFLOAT(INT(CC(3,1))).EQ.0.0 )CC(3,1)=CC(3,1)-EP
	  IF (CC(3,2)-DFLOAT(INT(CC(3,2))).EQ.0.0 )CC(3,2)=CC(3,2)-EP
	  IF (CC(4,1)-DFLOAT(INT(CC(4,1))).EQ.0.0 )CC(4,1)=CC(4,1)+EP
	  IF (CC(4,2)-DFLOAT(INT(CC(4,2))).EQ.0.0 )CC(4,2)=CC(4,2)+EP

C  Tessellation SETUP
C  =============+====

C  Find parameters for the footprint sides

	  CALL PARSETUP ( CC, PARS, ORIENT )

C  Special cases (only flagged in this version, not computed)

	  SPECIAL_SIDE1 = .FALSE.
	  LIM1 = PARS(2,1) + 90.0
	  LIM2 = PARS(2,1) - 90.0
	  IF ( ( LIM1.GT.CC(1,1).AND.LIM1.LT.CC(2,1) )   .OR.
     &         ( LIM2.GT.CC(1,1).AND.LIM2.LT.CC(2,1) ) ) THEN
	    SPECIAL_SIDE1 = .TRUE.
	  ENDIF
	  SPECIAL_SIDE3 = .FALSE.
	  LIM1 = PARS(2,3) + 90.0
	  LIM2 = PARS(2,3) - 90.0
	  IF ( ( LIM1.GT.CC(4,1).AND.LIM1.LT.CC(3,1) )   .OR.
     &         ( LIM2.GT.CC(4,1).AND.LIM2.LT.CC(3,1) ) ) THEN
	    SPECIAL_SIDE3 = .TRUE.
	  ENDIF
	  IF ( SPECIAL_SIDE1 .OR. SPECIAL_SIDE3 ) THEN
	    write(*,'(I5,L2,A)')M,CIRCULARITY,' Special cases'
	    GOTO 999
	  ENDIF

C  Find Corner Offsets in the original X-Y grids

	  ISTART1 = OFFSET ( 1,       OXGRID, MAX_XDIM, 1, CC(1,1) )
	  ISTART3 = OFFSET ( ISTART1, OXGRID, MAX_XDIM, 1, CC(3,1) )
	  JSTART  = OFFSET ( 1,       OYGRID, MAX_YDIM, 1, CC(4,2) )
	  JFINIS  = OFFSET ( 1,       OYGRID, MAX_YDIM, 1, CC(2,2) )
	  ISTART2 = OFFSET ( ISTART1, OXGRID, MAX_XDIM, 1, CC(2,1) )
	  ISTART4 = OFFSET ( ISTART1, OXGRID, MAX_XDIM, 1, CC(4,1) )

C  prepare albedo data in box around footprint
C  (  take account of the circularity at meridian crossing)

	  IG = 0
	  IF ( CIRCULARITY ) THEN
	    DO I = ISTART1, MAX_ADIM
	      IG = IG + 1
	      JG = 0
	      DO J = JSTART, JFINIS
	        JG = JG + 1
	        ALBUSED(JG,IG) = GLOBAL_ALBEDO_1(J,I)
	        TPGUSED(JG,IG) = DFLOAT(ETOP05_CLIMDATA(J,I))/1000.0d0
	      ENDDO
	    ENDDO
	    DO I = MAX_ADIM+1, ISTART3
	      IG = IG + 1
	      JG = 0
	      DO J = JSTART, JFINIS
	        JG = JG + 1
	        IZ = I-MAX_ADIM
	        ALBUSED(JG,IG) = GLOBAL_ALBEDO_1(J,IZ)
	        TPGUSED(JG,IG) = DFLOAT(ETOP05_CLIMDATA(J,IZ))/1000.0d0
	      ENDDO
	    ENDDO
	  ELSE
	    DO I = ISTART1, ISTART3
	      IG = IG + 1
	      JG = 0
	      DO J = JSTART, JFINIS
	        JG = JG + 1
	        ALBUSED(JG,IG) = GLOBAL_ALBEDO_1(J,I)
	        TPGUSED(JG,IG) = DFLOAT(ETOP05_CLIMDATA(J,I))/1000.0d0
	      ENDDO
	    ENDDO
	  ENDIF

C  define offsets in narrow box covering only the pixel

	  NXUSED = ISTART3 - ISTART1 + 2
	  NYUSED = JFINIS  - JSTART  + 2
	  ISTART3_T = NXUSED - 1
	  ISTART1_T = 1
	  ISTART2_T = ISTART2 - ISTART1 + 1
	  ISTART4_T = ISTART4 - ISTART1 + 1
	  JFINIS_T  = NYUSED - 1
	  JSTART_T  = 1

C  define tesselation grid box round footprint

	  DO J = 1, NYUSED
	    J1 = J + JSTART - 1
	    YGRID_T(J) = OYGRID(J1)
	  ENDDO
	  DO I = 1, NXUSED
	    I1 = I + ISTART1 - 1
	    XGRID_T(I) = OXGRID(I1)
	  ENDDO

C  single call to tessellation master routine

	  CALL TESSELATE_AREAMASTER
     &    ( XGRID_T, YGRID_T, MAX_XBOX, MAX_YBOX,
     I      CC, PARS, ORIENT, 
     I      ISTART1_T, ISTART2_T, ISTART3_T, ISTART4_T, JFINIS_T,
     O      AREA, SUM, YLIMIT_LOWER, YLIMIT_UPPER )

C  Assign output

	  ALBEDO = 0.0D0
	  HEIGHT = 0.0D0
	  DO I = ISTART1_T, ISTART3_T
	    DO J = YLIMIT_LOWER(I), YLIMIT_UPPER(I)
	      ALBEDO = ALBEDO + AREA(J,I)*ALBUSED(J,I)
	      HEIGHT = HEIGHT + AREA(J,I)*TPGUSED(J,I)
	    ENDDO
	  ENDDO
	  FOOT_ALBEDOS(N) = ALBEDO/SUM
	  FOOT_HEIGHTS(N) = HEIGHT/SUM
	  FOOT_AREAS(N)   = SUM

C  control point for special cases

999	  CONTINUE

C  End loop

	ENDDO

C  Finish

	RETURN
	END
