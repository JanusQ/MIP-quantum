NAME          tbfp-network_ds
OBJSENSE
 MIN
ROWS
 N  OBJ
 E  Balance[0]
 E  Visit_Once_Network['AT&T_Park']
 E  Visit_Once_Network['Minute_Maid_Park']
 E  Visit_Once_Network['U.S._Cellular_Field']
 E  Balance[1000]
 E  Balance[1001]
 E  Balance[1002]
 E  Balance[1003]
 E  Balance[1004]
 E  Balance[1005]
 E  Balance[1006]
 E  Balance[1007]
 E  Balance[1008]
 E  Balance[1009]
 E  Balance[100]
 E  Balance[1010]
 E  Balance[1011]
 E  Balance[1012]
 E  Balance[1013]
 E  Balance[1014]
 E  Balance[1015]
 E  Balance[1016]
 E  Visit_Once_Network['Busch_Stadium']
 E  Visit_Once_Network['Citi_Field']
 E  Visit_Once_Network['Citizens_Bank_Park']
 E  Visit_Once_Network['Comerica_Park']
 E  Visit_Once_Network['Dodger_Stadium']
 E  Visit_Once_Network['Fenway_Park']
 E  Visit_Once_Network['Marlins_Park']
 E  Visit_Once_Network['Miller_Park']
 E  Visit_Once_Network['O.co_Coliseum']
 E  Visit_Once_Network['Oriole_Park_at_Camden_Yards']
 E  Visit_Once_Network['Progressive_Field']
 E  Visit_Once_Network['Safeco_Field']
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    UseArc[0,1000]   OBJ                             1 Balance[0]                      1
    UseArc[0,1000]   Visit_Once_Network['U.S._Cellular_Field']                1 Balance[1000]                  -1
    UseArc[0,1001]   OBJ                             1 Balance[0]                      1
    UseArc[0,1001]   Visit_Once_Network['Minute_Maid_Park']                1 Balance[1001]                  -1
    UseArc[0,1002]   OBJ                             1 Balance[0]                      1
    UseArc[0,1002]   Balance[1002]                  -1 Visit_Once_Network['Miller_Park']                1
    UseArc[0,1003]   OBJ                             1 Balance[0]                      1
    UseArc[0,1003]   Balance[1003]                  -1 Visit_Once_Network['Busch_Stadium']                1
    UseArc[0,1004]   OBJ                             1 Balance[0]                      1
    UseArc[0,1004]   Balance[1004]                  -1 Visit_Once_Network['O.co_Coliseum']                1
    UseArc[0,1005]   OBJ                             1 Balance[0]                      1
    UseArc[0,1005]   Balance[1005]                  -1 Visit_Once_Network['Safeco_Field']                1
    UseArc[0,1006]   OBJ                             1 Balance[0]                      1
    UseArc[0,1006]   Balance[1006]                  -1 Visit_Once_Network['Dodger_Stadium']                1
    UseArc[0,1007]   OBJ                             1 Balance[0]                      1
    UseArc[0,1007]   Visit_Once_Network['AT&T_Park']                1 Balance[1007]                  -1
    UseArc[0,1008]   OBJ                             1 Balance[0]                      1
    UseArc[0,1008]   Visit_Once_Network['U.S._Cellular_Field']                1 Balance[1008]                  -1
    UseArc[0,1009]   OBJ                             1 Balance[0]                      1
    UseArc[0,1009]   Balance[1009]                  -1 Visit_Once_Network['Citizens_Bank_Park']                1
    UseArc[0,100]    OBJ                             1 Balance[0]                      1
    UseArc[0,100]    Balance[100]                   -1 Visit_Once_Network['Progressive_Field']                1
    UseArc[0,1010]   OBJ                             1 Balance[0]                      1
    UseArc[0,1010]   Balance[1010]                  -1 Visit_Once_Network['Oriole_Park_at_Camden_Yards']                1
    UseArc[0,1011]   OBJ                             1 Balance[0]                      1
    UseArc[0,1011]   Balance[1011]                  -1 Visit_Once_Network['Fenway_Park']                1
    UseArc[0,1012]   OBJ                             1 Balance[0]                      1
    UseArc[0,1012]   Visit_Once_Network['AT&T_Park']                1 Balance[1012]                  -1
    UseArc[0,1013]   OBJ                             1 Balance[0]                      1
    UseArc[0,1013]   Balance[1013]                  -1 Visit_Once_Network['Comerica_Park']                1
    UseArc[0,1014]   OBJ                             1 Balance[0]                      1
    UseArc[0,1014]   Visit_Once_Network['Minute_Maid_Park']                1 Balance[1014]                  -1
    UseArc[0,1015]   OBJ                             1 Balance[0]                      1
    UseArc[0,1015]   Balance[1015]                  -1 Visit_Once_Network['Marlins_Park']                1
    UseArc[0,1016]   OBJ                             1 Balance[0]                      1
    UseArc[0,1016]   Balance[1016]                  -1 Visit_Once_Network['Citi_Field']                1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1              Balance[0]                      1
    RHS1              Visit_Once_Network['AT&T_Park']                1
    RHS1              Visit_Once_Network['Minute_Maid_Park']                1
    RHS1              Visit_Once_Network['U.S._Cellular_Field']                1
    RHS1              Balance[1000]                   0
    RHS1              Balance[1001]                   0
    RHS1              Balance[1002]                   0
    RHS1              Balance[1003]                   0
    RHS1              Balance[1004]                   0
    RHS1              Balance[1005]                   0
    RHS1              Balance[1006]                   0
    RHS1              Balance[1007]                   0
    RHS1              Balance[1008]                   0
    RHS1              Balance[1009]                   0
    RHS1              Balance[100]                    0
    RHS1              Balance[1010]                   0
    RHS1              Balance[1011]                   0
    RHS1              Balance[1012]                   0
    RHS1              Balance[1013]                   0
    RHS1              Balance[1014]                   0
    RHS1              Balance[1015]                   0
    RHS1              Balance[1016]                   0
    RHS1              Visit_Once_Network['Busch_Stadium']                1
    RHS1              Visit_Once_Network['Citi_Field']                1
    RHS1              Visit_Once_Network['Citizens_Bank_Park']                1
    RHS1              Visit_Once_Network['Comerica_Park']                1
    RHS1              Visit_Once_Network['Dodger_Stadium']                1
    RHS1              Visit_Once_Network['Fenway_Park']                1
    RHS1              Visit_Once_Network['Marlins_Park']                1
    RHS1              Visit_Once_Network['Miller_Park']                1
    RHS1              Visit_Once_Network['O.co_Coliseum']                1
    RHS1              Visit_Once_Network['Oriole_Park_at_Camden_Yards']                1
    RHS1              Visit_Once_Network['Progressive_Field']                1
    RHS1              Visit_Once_Network['Safeco_Field']                1
BOUNDS
 BV BND1              UseArc[0,1000]
 BV BND1              UseArc[0,1001]
 BV BND1              UseArc[0,1002]
 BV BND1              UseArc[0,1003]
 BV BND1              UseArc[0,1004]
 BV BND1              UseArc[0,1005]
 BV BND1              UseArc[0,1006]
 BV BND1              UseArc[0,1007]
 BV BND1              UseArc[0,1008]
 BV BND1              UseArc[0,1009]
 BV BND1              UseArc[0,100]
 BV BND1              UseArc[0,1010]
 BV BND1              UseArc[0,1011]
 BV BND1              UseArc[0,1012]
 BV BND1              UseArc[0,1013]
 BV BND1              UseArc[0,1014]
 BV BND1              UseArc[0,1015]
 BV BND1              UseArc[0,1016]
ENDATA
