# Social Vulnerability Index (SVI) Methodology

## Overview

The Social Vulnerability Index (SVI) is a composite measure of social vulnerability calculated for each census tract in the top 100 highest-risk food desert predictions. The SVI score ranges from 0.0 to 1.0, where higher values indicate higher social vulnerability.

## Purpose

SVI helps identify tracts where social vulnerability compounds food access challenges. Tracts with high SVI scores are more vulnerable to becoming food deserts because they lack resources to adapt to food access disruptions. This enables more targeted interventions that address both food access and underlying social vulnerabilities.

## Calculation Framework

The SVI is based on the CDC/ATSDR Social Vulnerability Index methodology, adapted to use available data sources in this project. It consists of four component themes:

### Theme 1: Socioeconomic Status (30% weight)

Measures economic resources and stability.

**Components:**
- **Poverty Rate** (40% of theme weight): Percentage of population below poverty level
  - Calculation: `min(PovertyRate / 100, 1.0)`
  - Higher poverty = higher vulnerability
  
- **Income Level** (30% of theme weight): Median household income
  - Threshold-based scoring:
    - < $30,000: Score = 1.0 (very high vulnerability)
    - $30,000 - $50,000: Score = 0.7 (high vulnerability)
    - $50,000 - $75,000: Score = 0.4 (moderate vulnerability)
    - > $75,000: Score = 0.1 (low vulnerability)
  
- **Education Attainment** (20% of theme weight): Percentage with bachelor's degree or higher
  - Calculation: `1 - (EducationRate / 100)`
  - Lower education = higher vulnerability
  
- **Rent Burden** (10% of theme weight): Percentage of income spent on rent
  - Threshold-based scoring:
    - > 50%: Score = 1.0 (severely burdened)
    - 30-50%: Score = 0.7 (burdened)
    - < 30%: Score = 0.3 (not burdened)

**Theme Score**: Weighted average of available components, normalized by total weight

### Theme 2: Household Composition (25% weight)

Measures household characteristics that indicate vulnerability.

**Components:**
- **Low-Income Households** (50% of theme weight): Official low-income tract designation
  - Binary score: 1 = low-income tract, 0 = not
  
- **Vehicle Ownership** (50% of theme weight): Percentage of households without vehicle
  - Calculation: `1 - VehicleOwnershipRate` or `HouseholdsNoVehicle / TotalHouseholds`
  - Lower vehicle ownership = higher vulnerability (transportation dependency)

**Theme Score**: Weighted average of available components

### Theme 3: Minority Status & Language (25% weight)

Measures indicators of historically underserved communities.

**Note**: Limited race/ethnicity data available in current features. Uses socioeconomic proxies.

**Components:**
- **Low-Income Proxy** (60% of theme weight): Low-income tracts as proxy
  - Binary score: 1 = low-income tract, 0 = not
  - Rationale: Historical correlation between low-income and minority communities
  
- **Poverty Indicator** (40% of theme weight): Poverty rate as additional vulnerability indicator
  - Calculation: `min(PovertyRate / 100, 1.0)`

**Theme Score**: Weighted average (capped at 1.0)

**Limitation**: This is a simplified proxy. Ideally would use Census race/ethnicity (B03002 series) and language data (B16001 series) for more accurate calculation.

### Theme 4: Housing & Transportation (20% weight)

Measures housing conditions and transportation access.

**Components:**
- **Vehicle Access** (50% of theme weight): Percentage of households without vehicle
  - Same calculation as Theme 2
  - Critical for food access in areas with limited transit
  
- **Housing Crowding** (30% of theme weight): Average household size as proxy
  - Calculation:
    - If AvgHouseholdSize > 3.0: `min((AvgSize - 2.5) / 2.0, 1.0)`
    - If AvgHouseholdSize ≤ 3.0: 0.2 (low crowding)
  
- **Rent Burden** (20% of theme weight): Percentage of income spent on rent
  - Same calculation as Theme 1

**Theme Score**: Weighted average of available components

## Final SVI Score Calculation

```
SVI = (Socioeconomic_Score × 0.30) + 
      (Household_Score × 0.25) + 
      (Minority_Score × 0.25) + 
      (Housing_Score × 0.20)
```

**Normalization**:
- Each component theme is normalized to [0, 1] range
- If some components are missing, weights are normalized to sum to 1.0
- Final SVI score is clipped to [0, 1]

## Interpretation

| SVI Range | Vulnerability Level | Interpretation |
|-----------|-------------------|----------------|
| 0.0 - 0.3 | Low | Tracts with adequate resources and low vulnerability |
| 0.3 - 0.5 | Moderate | Tracts with some vulnerability indicators |
| 0.5 - 0.7 | High | Tracts with multiple vulnerability factors |
| 0.7 - 1.0 | Very High | Tracts with severe vulnerability across multiple dimensions |

## Data Sources

The SVI calculation uses the following data sources:

1. **Census ACS Data** (via API):
   - `Census_PovertyRate`: Poverty rate
   - `Census_MedianHouseholdIncome`: Median household income
   - `Census_EducationRate`: Education attainment (% bachelor's+)
   - `Census_RentBurden`: Rent burden (% income on rent)
   - `Census_VehicleOwnershipRate`: Vehicle ownership rate
   - `Census_HouseholdsNoVehicle`: Count of households without vehicle
   - `Census_TotalHouseholds`: Total households
   - `Census_AvgHouseholdSize`: Average household size

2. **USDA Food Access Atlas**:
   - `LowIncomeTracts`: Official low-income tract designation

3. **Zillow Housing Data** (if available):
   - Rent affordability metrics

## Limitations

1. **Race/Ethnicity Data**: Limited race/ethnicity data available, so Minority Status component uses socioeconomic proxies
2. **Age Data**: No direct age data available, so elderly vulnerability is proxied through other indicators
3. **Disability Data**: No disability data available in current features
4. **Language Barriers**: No language data available
5. **Group Quarters**: No group quarters data available

## Future Improvements

To improve SVI accuracy, the following Census ACS variables should be integrated:

1. **Race/Ethnicity** (B03002 series):
   - White alone, not Hispanic
   - Black or African American alone
   - American Indian and Alaska Native alone
   - Asian alone
   - Hispanic or Latino population

2. **Age Demographics** (B01001 series):
   - Population 65 and older
   - Population under 18

3. **Disability** (B18101 series):
   - Disability status by age

4. **Language** (B16001 series):
   - Limited English proficiency households

5. **Group Quarters** (B26001 series):
   - Institutionalized population

## Usage in Food Desert Prediction

SVI is included in the final output CSV (`top100_highest_risk_tracts.csv`) to provide context for food desert risk predictions. Tracts with both:
- **High food desert risk probability** (> 0.7)
- **High SVI score** (> 0.7)

Are particularly vulnerable and may require more comprehensive interventions addressing both food access and underlying social vulnerabilities.

## References

- CDC/ATSDR Social Vulnerability Index: https://www.atsdr.cdc.gov/placeandhealth/svi/index.html
- Flanagan, B. E., et al. (2011). "A Social Vulnerability Index for Disaster Management." Journal of Homeland Security and Emergency Management, 8(1).

---

**Last Updated**: 2025-01-27

