NAME          supportcase33_ds
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  capConstr:c27
 L  delivOnlyOnce:c27(1)
 L  delivOrder:c27(0)
 L  delivOrder:c27(1)
 G  noOverlapCust:c27(0)
 G  releaseTime:c27(1)
 L  timeLag:c27(0)
 E  flowPreserv:c27(1),k0
 E  flowPreserv:c27(1),k1
 L  incompatible:c21,c39
 L  noPartialDeliveries:c0
 L  noPartialDeliveries:c1
 L  noPartialDeliveries:c10
 L  noPartialDeliveries:c11
 L  noPartialDeliveries:c12
 L  noPartialDeliveries:c13
 L  noPartialDeliveries:c14
 G  capConstr:c21
 G  capConstr:c34
 G  capConstr:c39
 G  deadline:c21(0)
 G  deadline:c21(1)
 G  deadline:c34(0)
 G  deadline:c34(1)
 G  deadline:c34(2)
 G  deadline:c39(0)
 G  deadline:c39(1)
 G  deadline:c39(2)
 E  flowPreserv:c0(0),k0
 E  flowPreserv:c0(0),k1
 E  flowPreserv:c1(0),k0
 E  flowPreserv:c1(0),k1
 E  flowPreserv:c10(0),k0
 E  flowPreserv:c10(0),k1
 E  flowPreserv:c11(0),k0
 E  flowPreserv:c11(0),k1
 E  flowPreserv:c12(0),k0
 E  flowPreserv:c12(0),k1
 E  flowPreserv:c13(0),k0
 E  flowPreserv:c13(0),k1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    yc39             OBJ                           -45 incompatible:c21,c39                1
    yc39             capConstr:c39                 -45 deadline:c39(0)            -1e+06
    yc39             deadline:c39(1)            -1e+06 deadline:c39(2)            -1e+06
    yc21             OBJ                           -30 incompatible:c21,c39                1
    yc21             capConstr:c21                 -30 deadline:c21(0)            -1e+06
    yc21             deadline:c21(1)            -1e+06
    yc34             OBJ                           -40 capConstr:c34                 -40
    yc34             deadline:c34(0)            -1e+06 deadline:c34(1)            -1e+06
    yc34             deadline:c34(2)            -1e+06
    x[c27(1),c0(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c0(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c0(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c0(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c0(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c0                1
    x[c27(1),c0(0),0] flowPreserv:c0(0),k0                1
    x[c27(1),c0(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c0(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c0(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c0(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c0(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c0                1
    x[c27(1),c0(0),1] flowPreserv:c0(0),k1                1
    x[c27(1),c1(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c1(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c1(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c1(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c1(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c1                1
    x[c27(1),c1(0),0] flowPreserv:c1(0),k0                1
    x[c27(1),c1(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c1(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c1(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c1(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c1(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c1                1
    x[c27(1),c1(0),1] flowPreserv:c1(0),k1                1
    x[c27(1),c10(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c10(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c10(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c10(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c10(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c10                1
    x[c27(1),c10(0),0] flowPreserv:c10(0),k0                1
    x[c27(1),c10(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c10(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c10(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c10(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c10(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c10                1
    x[c27(1),c10(0),1] flowPreserv:c10(0),k1                1
    x[c27(1),c11(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c11(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c11(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c11(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c11(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c11                1
    x[c27(1),c11(0),0] flowPreserv:c11(0),k0                1
    x[c27(1),c11(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c11(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c11(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c11(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c11(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c11                1
    x[c27(1),c11(0),1] flowPreserv:c11(0),k1                1
    x[c27(1),c12(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c12(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c12(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c12(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c12(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c12                1
    x[c27(1),c12(0),0] flowPreserv:c12(0),k0                1
    x[c27(1),c12(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c12(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c12(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c12(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c12(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c12                1
    x[c27(1),c12(0),1] flowPreserv:c12(0),k1                1
    x[c27(1),c13(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c13(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c13(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c13(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c13(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c13                1
    x[c27(1),c13(0),0] flowPreserv:c13(0),k0                1
    x[c27(1),c13(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c13(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c13(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c13(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c13(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c13                1
    x[c27(1),c13(0),1] flowPreserv:c13(0),k1                1
    x[c27(1),c14(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c14(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c14(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c14(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c14(0),0] flowPreserv:c27(1),k0               -1 noPartialDeliveries:c14                1
    x[c27(1),c14(0),1] OBJ                             0 capConstr:c27                  20
    x[c27(1),c14(0),1] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c14(0),1] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -20
    x[c27(1),c14(0),1] releaseTime:c27(1)              -20 timeLag:c27(0)                -20
    x[c27(1),c14(0),1] flowPreserv:c27(1),k1               -1 noPartialDeliveries:c14                1
    x[c27(1),c15(0),0] OBJ                             0 capConstr:c27                  15
    x[c27(1),c15(0),0] delivOnlyOnce:c27(1)                1 delivOrder:c27(0)                1
    x[c27(1),c15(0),0] delivOrder:c27(1)               -1 noOverlapCust:c27(0)              -15
    x[c27(1),c15(0),0] releaseTime:c27(1)              -15 timeLag:c27(0)                -15
    x[c27(1),c15(0),0] flowPreserv:c27(1),k0               -1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1              capConstr:c27                   0
    RHS1              delivOnlyOnce:c27(1)                0
    RHS1              delivOrder:c27(0)                0
    RHS1              delivOrder:c27(1)                0
    RHS1              noOverlapCust:c27(0)                0
    RHS1              releaseTime:c27(1)                0
    RHS1              timeLag:c27(0)                  0
    RHS1              flowPreserv:c27(1),k0                0
    RHS1              flowPreserv:c27(1),k1                0
    RHS1              incompatible:c21,c39                0
    RHS1              noPartialDeliveries:c0                0
    RHS1              noPartialDeliveries:c1                0
    RHS1              noPartialDeliveries:c10                0
    RHS1              noPartialDeliveries:c11                0
    RHS1              noPartialDeliveries:c12                0
    RHS1              noPartialDeliveries:c13                0
    RHS1              noPartialDeliveries:c14                0
    RHS1              capConstr:c21                   0
    RHS1              capConstr:c34                   0
    RHS1              capConstr:c39                   0
    RHS1              deadline:c21(0)                 0
    RHS1              deadline:c21(1)                 0
    RHS1              deadline:c34(0)                 0
    RHS1              deadline:c34(1)                 0
    RHS1              deadline:c34(2)                 0
    RHS1              deadline:c39(0)                 0
    RHS1              deadline:c39(1)                 0
    RHS1              deadline:c39(2)                 0
    RHS1              flowPreserv:c0(0),k0                0
    RHS1              flowPreserv:c0(0),k1                0
    RHS1              flowPreserv:c1(0),k0                0
    RHS1              flowPreserv:c1(0),k1                0
    RHS1              flowPreserv:c10(0),k0                0
    RHS1              flowPreserv:c10(0),k1                0
    RHS1              flowPreserv:c11(0),k0                0
    RHS1              flowPreserv:c11(0),k1                0
    RHS1              flowPreserv:c12(0),k0                0
    RHS1              flowPreserv:c12(0),k1                0
    RHS1              flowPreserv:c13(0),k0                0
    RHS1              flowPreserv:c13(0),k1                0
BOUNDS
 BV BND1              yc39
 BV BND1              yc21
 BV BND1              yc34
 BV BND1              x[c27(1),c0(0),0]
 BV BND1              x[c27(1),c0(0),1]
 BV BND1              x[c27(1),c1(0),0]
 BV BND1              x[c27(1),c1(0),1]
 BV BND1              x[c27(1),c10(0),0]
 BV BND1              x[c27(1),c10(0),1]
 BV BND1              x[c27(1),c11(0),0]
 BV BND1              x[c27(1),c11(0),1]
 BV BND1              x[c27(1),c12(0),0]
 BV BND1              x[c27(1),c12(0),1]
 BV BND1              x[c27(1),c13(0),0]
 BV BND1              x[c27(1),c13(0),1]
 BV BND1              x[c27(1),c14(0),0]
 BV BND1              x[c27(1),c14(0),1]
 BV BND1              x[c27(1),c15(0),0]
ENDATA
