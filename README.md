# Renewable Energy Generation Analysis

## Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the **State Control Renewable Energy Generation** dataset. The objective is to analyze renewable energy production across different states and regions of India, identify trends over time, compare renewable energy sources, and generate insights that can support policy-making and sustainable energy planning.

The project uses **Python, Pandas, Matplotlib, and Seaborn** for data preprocessing, analysis, and visualization.

---

## Objectives

- Clean and preprocess renewable energy data.
- Handle missing values and duplicate records.
- Analyze renewable energy production by:
  - Region
  - State
  - Year
  - Month
- Compare contributions from:
  - Wind Energy
  - Solar Energy
  - Other Renewable Energy
- Identify top-performing states and regions.
- Study seasonal variations in renewable energy generation.
- Analyze regional disparities.
- Generate policy-related insights.

---

## Dataset

**Dataset Name:** `state-control-renewable-energy-generation.csv`

### Dataset Features

| Column Name | Description |
|------------|-------------|
| date | Date of observation |
| state_name | Name of the state |
| region | Region of India |
| wind_energy | Wind energy generation |
| solar_energy | Solar energy generation |
| other_renewable_energy | Other renewable energy generation |
| total_renewable_energy | Total renewable energy generation |

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Required Libraries

Install the required packages using:

```bash
pip install pandas matplotlib seaborn numpy
```

---

## Project Workflow

### 1. Data Loading
- Load CSV dataset using Pandas.

### 2. Data Preprocessing
- Check dataset information.
- Identify missing values.
- Detect duplicate records.
- Fill missing region values.
- Replace missing energy values with 0.
- Convert the date column into datetime format.
- Extract year and month features.

### 3. Exploratory Data Analysis

#### Region-wise Analysis
- Total renewable energy by region.

#### State-wise Analysis
- Top 10 states based on renewable energy production.

#### Year-wise Analysis
- Renewable energy trends from 2020–2024.

#### Month-wise Analysis
- Average monthly renewable energy generation.

#### Energy Source Analysis
- Wind vs Solar vs Other renewable sources.

#### Regional Disparity Analysis
- Distribution of renewable energy production across regions.

#### Correlation Analysis
- Relationship between different energy types.

---

## Visualizations Generated

The project automatically saves the following plots:

| File Name | Description |
|-----------|-------------|
| obj1_region_totals.png | Total renewable energy by region |
| obj1_state_avg.png | Average renewable energy by top states |
| obj2_yearly_trends.png | Yearly renewable energy trends |
| obj3_energy_mix_pie.png | Overall renewable energy mix |
| obj3_region_mix_stacked.png | Regional energy mix |
| obj4_top_states.png | Top 10 renewable energy producing states |
| obj5_seasonal_variations.png | Monthly renewable energy trends |
| obj6_regional_disparities.png | Regional disparity boxplot |
| obj7_top_states_growth.png | Growth of top 5 states |
| eda_correlation_heatmap.png | Correlation heatmap |

---

## Key Insights

- Renewable energy production varies significantly across different regions.
- A few states contribute a major share of renewable energy generation.
- Solar and wind energy are the dominant renewable sources.
- Renewable energy production shows clear yearly growth trends.
- Seasonal patterns affect energy generation.
- Strong positive correlations exist between individual renewable sources and total renewable energy production.

---

## Project Structure

```text
Renewable-Energy-Analysis/
│
├── state-control-renewable-energy-generation.csv
├── renewable_energy_analysis.py
├── README.md
│
├── obj1_region_totals.png
├── obj1_state_avg.png
├── obj2_yearly_trends.png
├── obj3_energy_mix_pie.png
├── obj3_region_mix_stacked.png
├── obj4_top_states.png
├── obj5_seasonal_variations.png
├── obj6_regional_disparities.png
├── obj7_top_states_growth.png
└── eda_correlation_heatmap.png
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/renewable-energy-analysis.git
```

2. Navigate to the project folder:

```bash
cd renewable-energy-analysis
```

3. Run the Python script:

```bash
python renewable_energy_analysis.py
```

4. The analysis results and visualizations will be generated automatically.

---

## Future Improvements

- Build a Machine Learning model to forecast renewable energy generation.
- Create an interactive dashboard using Plotly or Streamlit.
- Add state-wise geographical visualization using maps.
- Integrate real-time renewable energy datasets.

---

## Learning Outcomes

This project demonstrates the practical application of:

- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Correlation Analysis
- Feature Engineering
- Insight Generation

---

## Author

**Pragati Patel**  
B.Tech CSE | Data Science & Machine Learning Enthusiast

---

## License

This project is created for educational and research purposes. Feel free to use and modify it for learning purposes.
