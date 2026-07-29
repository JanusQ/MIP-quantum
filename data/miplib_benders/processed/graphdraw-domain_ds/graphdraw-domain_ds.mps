NAME          graphdraw-domain_ds
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  DistAxisLB1[r,CertificateTemplate,CourseSession]
 E  choose1[CertificateTemplate,CourseSession]
 L  triangle1[Client,CertificateTemplate,CourseSession,c]
 L  triangle2[CertificateTemplate,Course,CourseSession,c]
 L  triangle2[CertificateTemplate,CourseSession,Instructor,c]
 L  triangle2[CertificateTemplate,CourseSession,Location,c]
 L  triangle2[CertificateTemplate,CourseSession,Therapist,c]
 L  triangle2[Client,CertificateTemplate,CourseSession,c]
 L  triangle2[Client,CourseSession,Instructor,c]
 L  triangle2[Client,CourseSession,Location,c]
 L  triangle2[Client,CourseSession,Therapist,c]
 L  triangle2[Client,Location,Instructor,c]
 L  triangle2[Client,Location,Therapist,c]
 L  triangle2[Client,User,Instructor,c]
 L  triangle2[Client,User,Therapist,c]
 L  triangle2[Clinic,Location,Instructor,c]
 L  triangle2[Clinic,Location,Therapist,c]
 L  triangle2[Course,CourseSession,Instructor,c]
 L  triangle2[Course,CourseSession,Location,c]
 L  triangle2[Course,CourseSession,Therapist,c]
 L  triangle2[CourseSession,Location,Instructor,c]
 L  triangle2[CourseSession,Location,Therapist,c]
 L  triangle2[Role,User,Instructor,c]
 L  triangle2[Role,User,Therapist,c]
 G  DistAxisLB1[r,Client,CertificateTemplate]
 G  DistAxisLB1[r,Client,CourseSession]
 G  DistAxisLB1[r,Client,Instructor]
 G  DistAxisLB1[r,Client,Location]
 G  DistAxisLB1[r,Client,User]
 G  DistAxisLB1[r,Clinic,Location]
 G  DistAxisLB1[r,Clinic,Therapist]
 G  DistAxisLB1[r,Course,CourseSession]
 G  DistAxisLB1[r,CourseSession,Instructor]
 G  DistAxisLB1[r,Instructor,Location]
 G  DistAxisLB1[r,Location,CourseSession]
 G  DistAxisLB1[r,Location,Therapist]
 G  DistAxisLB1[r,Role,User]
 G  DistAxisLB1[r,Therapist,CourseSession]
 G  DistAxisLB1[r,User,Instructor]
 G  DistAxisLB1[r,User,Therapist]
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    zV[c,lb,CertificateTemplate,CourseSession] OBJ                             0 DistAxisLB1[r,CertificateTemplate,CourseSession]                8
    zV[c,lb,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession]                1 triangle2[CertificateTemplate,CourseSession,Instructor,c]                1
    zV[c,lb,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,CourseSession,Location,c]                1 triangle2[CertificateTemplate,CourseSession,Therapist,c]                1
    zV[c,lb,CertificateTemplate,CourseSession] triangle2[Client,CertificateTemplate,CourseSession,c]                1
    zV[c,lb,Client,CertificateTemplate] OBJ                             0 triangle2[Client,CertificateTemplate,CourseSession,c]                1
    zV[c,lb,Client,CertificateTemplate] DistAxisLB1[r,Client,CertificateTemplate]                8
    zV[c,lb,Client,CourseSession] OBJ                             0 triangle1[Client,CertificateTemplate,CourseSession,c]                1
    zV[c,lb,Client,CourseSession] triangle2[Client,CourseSession,Instructor,c]                1 triangle2[Client,CourseSession,Location,c]                1
    zV[c,lb,Client,CourseSession] triangle2[Client,CourseSession,Therapist,c]                1 DistAxisLB1[r,Client,CourseSession]                9
    zV[c,lb,Client,Instructor] OBJ                             0 DistAxisLB1[r,Client,Instructor]               19
    zV[c,lb,Client,Location] OBJ                             0 triangle2[Client,Location,Instructor,c]                1
    zV[c,lb,Client,Location] triangle2[Client,Location,Therapist,c]                1 DistAxisLB1[r,Client,Location]                9
    zV[c,lb,Client,User] OBJ                             0 triangle2[Client,User,Instructor,c]                1
    zV[c,lb,Client,User] triangle2[Client,User,Therapist,c]                1 DistAxisLB1[r,Client,User]                8
    zV[c,lb,Clinic,Location] OBJ                             0 triangle2[Clinic,Location,Instructor,c]                1
    zV[c,lb,Clinic,Location] triangle2[Clinic,Location,Therapist,c]                1 DistAxisLB1[r,Clinic,Location]               17
    zV[c,lb,Clinic,Therapist] OBJ                             0 DistAxisLB1[r,Clinic,Therapist]                9
    zV[c,lb,Course,CourseSession] OBJ                             0 triangle2[CertificateTemplate,Course,CourseSession,c]                1
    zV[c,lb,Course,CourseSession] triangle2[Course,CourseSession,Instructor,c]                1 triangle2[Course,CourseSession,Location,c]                1
    zV[c,lb,Course,CourseSession] triangle2[Course,CourseSession,Therapist,c]                1 DistAxisLB1[r,Course,CourseSession]               15
    zV[c,lb,CourseSession,Instructor] OBJ                             0 triangle2[CertificateTemplate,CourseSession,Instructor,c]                1
    zV[c,lb,CourseSession,Instructor] triangle2[Client,CourseSession,Instructor,c]                1 triangle2[Course,CourseSession,Instructor,c]                1
    zV[c,lb,CourseSession,Instructor] DistAxisLB1[r,CourseSession,Instructor]               19
    zV[c,lb,CourseSession,Location] OBJ                             0 triangle2[CertificateTemplate,CourseSession,Location,c]                1
    zV[c,lb,CourseSession,Location] triangle2[Client,CourseSession,Location,c]                1 triangle2[Course,CourseSession,Location,c]                1
    zV[c,lb,CourseSession,Location] triangle2[CourseSession,Location,Instructor,c]                1 triangle2[CourseSession,Location,Therapist,c]                1
    zV[c,lb,CourseSession,Location] DistAxisLB1[r,Location,CourseSession]                9
    zV[c,lb,CourseSession,Therapist] OBJ                             0 triangle2[CertificateTemplate,CourseSession,Therapist,c]                1
    zV[c,lb,CourseSession,Therapist] triangle2[Client,CourseSession,Therapist,c]                1 triangle2[Course,CourseSession,Therapist,c]                1
    zV[c,lb,CourseSession,Therapist] DistAxisLB1[r,Therapist,CourseSession]               19
    zV[c,lb,Location,Instructor] OBJ                             0 triangle2[Client,Location,Instructor,c]                1
    zV[c,lb,Location,Instructor] triangle2[Clinic,Location,Instructor,c]                1 triangle2[CourseSession,Location,Instructor,c]                1
    zV[c,lb,Location,Instructor] DistAxisLB1[r,Instructor,Location]               19
    zV[c,lb,Location,Therapist] OBJ                             0 triangle2[Client,Location,Therapist,c]                1
    zV[c,lb,Location,Therapist] triangle2[Clinic,Location,Therapist,c]                1 triangle2[CourseSession,Location,Therapist,c]                1
    zV[c,lb,Location,Therapist] DistAxisLB1[r,Location,Therapist]               19
    zV[c,lb,Role,User] OBJ                             0 triangle2[Role,User,Instructor,c]                1
    zV[c,lb,Role,User] triangle2[Role,User,Therapist,c]                1 DistAxisLB1[r,Role,User]               13
    zV[c,lb,User,Instructor] OBJ                             0 triangle2[Client,User,Instructor,c]                1
    zV[c,lb,User,Instructor] triangle2[Role,User,Instructor,c]                1 DistAxisLB1[r,User,Instructor]               17
    zV[c,lb,User,Therapist] OBJ                             0 triangle2[Client,User,Therapist,c]                1
    zV[c,lb,User,Therapist] triangle2[Role,User,Therapist,c]                1 DistAxisLB1[r,User,Therapist]               17
    zV[c,rt,CertificateTemplate,CourseSession] OBJ                             0 DistAxisLB1[r,CertificateTemplate,CourseSession]                8
    zV[c,rt,CertificateTemplate,CourseSession] choose1[CertificateTemplate,CourseSession]                1 triangle1[Client,CertificateTemplate,CourseSession,c]                1
    zV[c,rt,CertificateTemplate,CourseSession] triangle2[CertificateTemplate,Course,CourseSession,c]                1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1              DistAxisLB1[r,CertificateTemplate,CourseSession]                8
    RHS1              choose1[CertificateTemplate,CourseSession]                1
    RHS1              triangle1[Client,CertificateTemplate,CourseSession,c]                2
    RHS1              triangle2[CertificateTemplate,Course,CourseSession,c]                2
    RHS1              triangle2[CertificateTemplate,CourseSession,Instructor,c]                2
    RHS1              triangle2[CertificateTemplate,CourseSession,Location,c]                2
    RHS1              triangle2[CertificateTemplate,CourseSession,Therapist,c]                2
    RHS1              triangle2[Client,CertificateTemplate,CourseSession,c]                2
    RHS1              triangle2[Client,CourseSession,Instructor,c]                2
    RHS1              triangle2[Client,CourseSession,Location,c]                2
    RHS1              triangle2[Client,CourseSession,Therapist,c]                2
    RHS1              triangle2[Client,Location,Instructor,c]                2
    RHS1              triangle2[Client,Location,Therapist,c]                2
    RHS1              triangle2[Client,User,Instructor,c]                2
    RHS1              triangle2[Client,User,Therapist,c]                2
    RHS1              triangle2[Clinic,Location,Instructor,c]                2
    RHS1              triangle2[Clinic,Location,Therapist,c]                2
    RHS1              triangle2[Course,CourseSession,Instructor,c]                2
    RHS1              triangle2[Course,CourseSession,Location,c]                2
    RHS1              triangle2[Course,CourseSession,Therapist,c]                2
    RHS1              triangle2[CourseSession,Location,Instructor,c]                2
    RHS1              triangle2[CourseSession,Location,Therapist,c]                2
    RHS1              triangle2[Role,User,Instructor,c]                2
    RHS1              triangle2[Role,User,Therapist,c]                2
    RHS1              DistAxisLB1[r,Client,CertificateTemplate]                8
    RHS1              DistAxisLB1[r,Client,CourseSession]                9
    RHS1              DistAxisLB1[r,Client,Instructor]               19
    RHS1              DistAxisLB1[r,Client,Location]                9
    RHS1              DistAxisLB1[r,Client,User]                8
    RHS1              DistAxisLB1[r,Clinic,Location]               17
    RHS1              DistAxisLB1[r,Clinic,Therapist]                9
    RHS1              DistAxisLB1[r,Course,CourseSession]               15
    RHS1              DistAxisLB1[r,CourseSession,Instructor]               19
    RHS1              DistAxisLB1[r,Instructor,Location]               19
    RHS1              DistAxisLB1[r,Location,CourseSession]                9
    RHS1              DistAxisLB1[r,Location,Therapist]               19
    RHS1              DistAxisLB1[r,Role,User]               13
    RHS1              DistAxisLB1[r,Therapist,CourseSession]               19
    RHS1              DistAxisLB1[r,User,Instructor]               17
    RHS1              DistAxisLB1[r,User,Therapist]               17
BOUNDS
 BV BND1              zV[c,lb,CertificateTemplate,CourseSession]
 BV BND1              zV[c,lb,Client,CertificateTemplate]
 BV BND1              zV[c,lb,Client,CourseSession]
 BV BND1              zV[c,lb,Client,Instructor]
 BV BND1              zV[c,lb,Client,Location]
 BV BND1              zV[c,lb,Client,User]
 BV BND1              zV[c,lb,Clinic,Location]
 BV BND1              zV[c,lb,Clinic,Therapist]
 BV BND1              zV[c,lb,Course,CourseSession]
 BV BND1              zV[c,lb,CourseSession,Instructor]
 BV BND1              zV[c,lb,CourseSession,Location]
 BV BND1              zV[c,lb,CourseSession,Therapist]
 BV BND1              zV[c,lb,Location,Instructor]
 BV BND1              zV[c,lb,Location,Therapist]
 BV BND1              zV[c,lb,Role,User]
 BV BND1              zV[c,lb,User,Instructor]
 BV BND1              zV[c,lb,User,Therapist]
 BV BND1              zV[c,rt,CertificateTemplate,CourseSession]
ENDATA
