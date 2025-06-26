# Local Dataset Evaluation

This document provides analysis of local datasets for anomaly detection testing.

## 5000000 HRA Records.csv

- **File Type**: CSV
- **Rows Analyzed**: 1000
- **Columns (35)**: Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

### Sample Data (First 5 rows)

```
   Age Attrition     BusinessTravel  DailyRate              Department  DistanceFromHome  Education    EducationField  EmployeeCount  EmployeeNumber  EnvironmentSatisfaction  Gender  HourlyRate  JobInvolvement  JobLevel                JobRole  JobSatisfaction MaritalStatus  MonthlyIncome  MonthlyRate  NumCompaniesWorked Over18 OverTime  PercentSalaryHike  PerformanceRating  RelationshipSatisfaction  StandardHours  StockOptionLevel  TotalWorkingYears  TrainingTimesLastYear  WorkLifeBalance  YearsAtCompany  YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager
0   53       Yes         Non-Travel        862                 Support                46          5         Marketing              1               1                        3    Male          88               3         2        Sales Executive                2      Divorced          23450       351750                   1      Y      Yes                 37                  1                         3             80                 3                 29                      2                4              28                   4                        3                     8
1   59        No  Travel_Frequently       1269                   Sales                37          5   Human Resources              1               2                        3    Male         153               2         4                Manager                2       Married          50361       503610                   0      Y       No                 41                  2                         4             80                 3                 19                      2                1              15                  10                       15                    15
2   43        No         Non-Travel        383                Software                32          2  Technical Degree              1               3                        2    Male         166               1         2      Research Director                1      Divorced          17102       119714                   7      Y      Yes                 11                  2                         3             80                 3                 29                      3                4              22                  17                       16                     7
3   51        No      Travel_Rarely       1353  Research & Development                45          3         Marketing              1               4                        4  Female          72               1         5   Sales Representative                4       Married          29280       468480                   1      Y      Yes                 13                  2                         1             80                 2                  3                      4                4               2                   2                        2                     1
4   54       Yes         Non-Travel        216                Software                14          1         Marketing              1               5                        2  Female          57               4         2  Laboratory Technician                4        Single          39923       918229                   3      Y      Yes                 13                  2                         1             80                 2                 36                      1                3              30                  23                       21                    14
```

### Data Statistics

```
               Age    DailyRate  DistanceFromHome    Education  EmployeeCount  EmployeeNumber  EnvironmentSatisfaction   HourlyRate  JobInvolvement     JobLevel  JobSatisfaction  MonthlyIncome   MonthlyRate  NumCompaniesWorked  PercentSalaryHike  PerformanceRating  RelationshipSatisfaction  StandardHours  StockOptionLevel  TotalWorkingYears  TrainingTimesLastYear  WorkLifeBalance  YearsAtCompany  YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager
count  1000.000000  1000.000000       1000.000000  1000.000000         1000.0     1000.000000              1000.000000  1000.000000     1000.000000  1000.000000      1000.000000    1000.000000  1.000000e+03         1000.000000        1000.000000        1000.000000               1000.000000         1000.0       1000.000000         1000.00000            1000.000000      1000.000000     1000.000000         1000.000000              1000.000000           1000.000000
mean     39.579000   770.324000         25.744000     2.945000            1.0      500.500000                 2.500000   112.280000        2.541000     2.901000         2.460000   26876.595000  4.267059e+05            3.964000          24.805000           2.474000                  2.507000           80.0          2.487000           20.93100               3.472000         2.492000       11.259000            6.004000                 6.140000              6.087000
std      12.472101   400.624404         14.307801     1.435979            0.0      288.819436                 1.102368    49.634335        1.125871     1.425915         1.117877   14019.498242  3.362962e+05            2.672423          14.452884           1.126318                  1.147723            0.0          1.138036           11.49323               1.756787         1.133676        9.315938            6.259806                 6.152744              6.241184
min      18.000000   101.000000          1.000000     1.000000            1.0        1.000000                 1.000000    30.000000        1.000000     1.000000         1.000000    1083.000000  2.346000e+03            0.000000           0.000000           1.000000                  1.000000           80.0          1.000000            1.00000               1.000000         1.000000        1.000000            1.000000                 1.000000              1.000000
25%      29.000000   429.750000         13.000000     2.000000            1.0      250.750000                 2.000000    69.000000        2.000000     2.000000         1.000000   14889.500000  1.400430e+05            1.000000          12.000000           1.000000                  1.000000           80.0          1.000000           11.00000               2.000000         1.000000        3.000000            2.000000                 1.000000              2.000000
50%      40.000000   755.500000         25.000000     3.000000            1.0      500.500000                 3.000000   111.000000        3.000000     3.000000         2.000000   28156.000000  3.455150e+05            4.000000          25.000000           2.000000                  3.000000           80.0          2.000000           20.50000               3.000000         3.000000        9.000000            4.000000                 4.000000              3.000000
75%      51.000000  1112.000000         38.000000     4.000000            1.0      750.250000                 3.000000   155.000000        4.000000     4.000000         3.000000   38521.750000  6.579938e+05            6.000000          37.000000           3.000000                  4.000000           80.0          4.000000           31.00000               5.000000         4.000000       17.000000            8.000000                 9.000000              8.000000
max      60.000000  1500.000000         50.000000     5.000000            1.0     1000.000000                 4.000000   200.000000        4.000000     5.000000         4.000000   50921.000000  1.476720e+06            8.000000          49.000000           4.000000                  4.000000           80.0          4.000000           40.00000               6.000000         4.000000       39.000000           36.000000                33.000000             36.000000
```

### Missing Values

```
Age                         0
Attrition                   0
BusinessTravel              0
DailyRate                   0
Department                  0
DistanceFromHome            0
Education                   0
EducationField              0
EmployeeCount               0
EmployeeNumber              0
EnvironmentSatisfaction     0
Gender                      0
HourlyRate                  0
JobInvolvement              0
JobLevel                    0
JobRole                     0
JobSatisfaction             0
MaritalStatus               0
MonthlyIncome               0
MonthlyRate                 0
NumCompaniesWorked          0
Over18                      0
OverTime                    0
PercentSalaryHike           0
PerformanceRating           0
RelationshipSatisfaction    0
StandardHours               0
StockOptionLevel            0
TotalWorkingYears           0
TrainingTimesLastYear       0
WorkLifeBalance             0
YearsAtCompany              0
YearsInCurrentRole          0
YearsSinceLastPromotion     0
YearsWithCurrManager        0
```

### Suitability Analysis

- Contains numeric features: Yes
- Contains categorical features: Yes
- Contains timestamp features: Yes

**Assessment**: Suitable for anomaly detection. Time-series analysis possible. Contains categorical features that may need encoding.

---

## ip-creditcard-email.csv

- **File Type**: CSV
- **Rows Analyzed**: 100
- **Columns (5)**: id, name, email, ip_address, credit_card

### Sample Data (First 5 rows)

```
   id            name                              email     ip_address       credit_card
0   1    Bert Eshelby              beshelby0@t-online.de   13.84.33.231  3543951842051288
1   2      Wayne Jelk             wjelk1@miibeian.gov.cn   219.115.22.1  3574922401749937
2   3   Jeno Woodruff  jwoodruff2@scientificamerican.com  60.76.116.163  3588744702964559
3   4  Marisa Branton                  mbranton3@msn.com  169.9.150.151  5100136211210919
4   5  Freddi Poulson              fpoulson4@4shared.com    67.21.14.81  3546839366604342
```

### Data Statistics

```
               id   credit_card
count  100.000000  1.000000e+02
mean    50.500000  1.599719e+17
std     29.011492  8.862911e+17
min      1.000000  4.000059e+12
25%     25.750000  3.542909e+15
50%     50.500000  3.569462e+15
75%     75.250000  5.318487e+15
max    100.000000  6.011505e+18
```

### Missing Values

```
id             0
name           0
email          0
ip_address     0
credit_card    0
```

### Suitability Analysis

- Contains numeric features: Yes
- Contains categorical features: Yes
- Contains timestamp features: No

**Assessment**: Suitable for anomaly detection. Contains categorical features that may need encoding.

---

## telephone-owned-property.xlsx

- **File Type**: Excel (XLSX)
- **Rows Analyzed**: 20
- **Columns (6)**: ID, Name, Address, Telephone number, Vehicle, VIN number

### Sample Data (First 5 rows)

```
   ID               Name                 Address Telephone number                   Vehicle         VIN number
0   1      Perla V. Gray      3026 Heavner Court     515-962-6895  1995 Toyota Mega Cruiser  2B5WB35Y72K119615
1   2      Jeanie J. Yoo  4848 Meadow View Drive     860-548-4307        2009 Ford Ecosport  1FTRW12W07FA61725
2   3   Debbie R. Durant        1220 Wood Street     985-974-6159      2005 Hyundai Dynasty  JM1GJ1U64F1177346
3   4  Donald T. Morales        2890 Chatham Way     240-638-0928      2004 Daihatsu Terios  1FTYR10U64PB51553
4   5    Allan L. Wright         964 Late Avenue     580-765-2526      2003 Volkswagen Polo  3D4PG5FV3AT278828
```

### Data Statistics

```
             ID
count  20.00000
mean   10.50000
std     5.91608
min     1.00000
25%     5.75000
50%    10.50000
75%    15.25000
max    20.00000
```

### Missing Values

```
ID                  0
Name                0
Address             0
Telephone number    0
Vehicle             0
VIN number          0
```

### Suitability Analysis

- Contains numeric features: Yes
- Contains categorical features: Yes
- Contains timestamp features: No

**Assessment**: Suitable for anomaly detection. Contains categorical features that may need encoding.

---

