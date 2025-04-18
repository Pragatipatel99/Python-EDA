import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/praga/OneDrive/Desktop/state-control-renewable-energy-generation.csv")

# Basic info
print("First 5 rows:\n", df.head())
print("\nInfo:\n"); df.info()
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Handle missing values
state_region_map = df[df['region'].notnull()].drop_duplicates('state_name').set_index('state_name')['region'].to_dict()
df['region'] = df.apply(lambda r: state_region_map.get(r['state_name'], 'Unknown') if pd.isnull(r['region']) else r['region'], axis=1)

df[['wind_energy', 'solar_energy', 'other_renewable_energy']] = df[['wind_energy', 'solar_energy', 'other_renewable_energy']].fillna(0)

# Convert date and extract time features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Final checks
print("\nRemaining Missing Values:\n", df.isnull().sum())
print("\nDate Breakdown:\n", df[['date', 'year', 'month']].head())
print("\nStatistical Summary:\n", df.describe())

# Analysis: Region-wise, Year-wise, State-wise, Month-wise
region_summary = df.groupby('region')[['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy']].sum()
print("\nRenewable Energy by Region:\n", region_summary)

yearly_trends = df.groupby('year')[['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy']].sum()
print("\nYearly Trends:\n", yearly_trends)

state_summary = df.groupby('state_name')['total_renewable_energy'].sum().sort_values(ascending=False)
print("\nTop 10 States by Renewable Energy:\n", state_summary.head(10))

monthly_avg = df.groupby('month')['total_renewable_energy'].mean()
print("\nAverage Monthly Renewable Energy:\n", monthly_avg)


# %% [markdown]
# VISUALIZATION

# %% [markdown]
# Objective 1: Summarize Renewable Energy Production by Region and State

# %%
# Drop rows where state_name is 'All India'
df = df.drop(df[df['state_name'] == 'All India'].index)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Bar charts for total production by region and state.

# Step 1.1: Total Renewable Energy by Region
region_totals = df.groupby('region')['total_renewable_energy'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=region_totals.index, y=region_totals.values, hue=region_totals.index, palette='viridis', legend=False)
plt.title('Total Renewable Energy by Region (2020-2024)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Energy (Units)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('obj1_region_totals.png')
plt.show()
print("Region totals bar chart saved as 'obj1_region_totals.png'")

# %%
# Step 1.2: Average Renewable Energy by State (Top 10)
state_avg = df.groupby('state_name')['total_renewable_energy'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=state_avg.index, y=state_avg.values, hue=state_avg.index, palette='magma', legend=False)
plt.title('Average Renewable Energy by State (Top 10, 2020-2024)', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Average Energy (Units)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('obj1_state_avg.png')
plt.show()
print("State averages bar chart saved as 'obj1_state_avg.png'")

# %% [markdown]
# Objective 2: Track Trends in Renewable Energy Over Time
# 

# %%
# Step 2: Yearly Trends
yearly_trends = df.groupby('year')[['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy']].sum()

plt.figure(figsize=(10, 6))
plt.plot(yearly_trends.index, yearly_trends['wind_energy'], label='Wind', marker='o', color='blue')
plt.plot(yearly_trends.index, yearly_trends['solar_energy'], label='Solar', marker='o', color='orange')
plt.plot(yearly_trends.index, yearly_trends['other_renewable_energy'], label='Other', marker='o', color='green')
plt.plot(yearly_trends.index, yearly_trends['total_renewable_energy'], label='Total', marker='o', color='black', linewidth=2)
plt.title('Renewable Energy Trends by Year (2020-2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Energy Production (Units)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('obj2_yearly_trends.png')
plt.show()
print("Yearly trends line plot saved as 'obj2_yearly_trends.png'")

# %% [markdown]
# Objective 3: Compare Contributions of Wind, Solar, and Other Renewable Sources

# %%
# Step 3.1: Overall Energy Mix (Pie Chart)
energy_mix = df[['wind_energy', 'solar_energy', 'other_renewable_energy']].sum()

plt.figure(figsize=(5, 5))
plt.pie(energy_mix, labels=energy_mix.index, autopct='%1.1f%%', colors=['skyblue', 'orange', 'green'], startangle=90)
plt.title('Overall Renewable Energy Mix (2020-2024)', fontsize=14)
plt.tight_layout()
plt.savefig('obj3_energy_mix_pie.png')
plt.show()
print("Pie chart saved as 'obj3_energy_mix_pie.png'")

# %%
# Step 3.2: Regional Energy Mix (Stacked Bar)
region_mix = df.groupby('region')[['wind_energy', 'solar_energy', 'other_renewable_energy']].sum()

plt.figure(figsize=(8, 5))
plt.bar(region_mix.index, region_mix['wind_energy'], label='Wind', color='skyblue')
plt.bar(region_mix.index, region_mix['solar_energy'], bottom=region_mix['wind_energy'], label='Solar', color='orange')
plt.bar(region_mix.index, region_mix['other_renewable_energy'], bottom=region_mix['wind_energy'] + region_mix['solar_energy'], label='Other', color='green')
plt.title('Renewable Energy Mix by Region (2020-2024)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Energy Production (Units)', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('obj3_region_mix_stacked.png')
plt.show()
print("Stacked bar chart saved as 'obj3_region_mix_stacked.png'")

# %% [markdown]
# Objective 4: Identify Top-Performing States and Regions

# %%
# Step 4.1: Top Regions (reused from Obj 1)
# See 'obj1_region_totals.png'

# Step 4.2: Top 10 States
state_totals = df.groupby('state_name')['total_renewable_energy'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=state_totals.index, y=state_totals.values, hue=state_totals.index, palette='coolwarm', legend=False)
plt.title('Top 10 States by Total Renewable Energy (2020-2024)', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Total Energy (Units)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('obj4_top_states.png')
plt.show()
print("Top states bar chart saved as 'obj4_top_states.png'")


monthly_avg = df.groupby('month')['total_renewable_energy'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, hue=monthly_avg.index, palette='Blues', legend=False)
plt.title('Average Renewable Energy by Month (2020-2024)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Energy (Units)', fontsize=12)
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('obj5_seasonal_variations.png')
plt.show()
print("Seasonal variations bar chart saved as 'obj5_seasonal_variations.png'")

# %% [markdown]
# Objective 6: Analyze Regional Disparities in Renewable Energy Adoption

# %%
# Step 6: Boxplot by Region
plt.figure(figsize=(12, 6))
sns.boxplot(x='region', y='total_renewable_energy', data=df, palette='Set2')
plt.title('Regional Disparities in Renewable Energy Production (2020-2024)', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Energy (Units)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('obj6_regional_disparities.png')
plt.show()
print("Boxplot saved as 'obj6_regional_disparities.png'")

# %% [markdown]
# Objective 7: Provide Insights for Policy Recommendations

# %%
# Step 7: Growth of Top 5 States Over Time
top_states = df.groupby('state_name')['total_renewable_energy'].sum().sort_values(ascending=False).head(5).index
top_states_trends = df[df['state_name'].isin(top_states)].groupby(['year', 'state_name'])['total_renewable_energy'].sum().unstack()

plt.figure(figsize=(12, 6))
for state in top_states_trends.columns:
    plt.plot(top_states_trends.index, top_states_trends[state], label=state, marker='o')
plt.title('Growth of Top 5 States (2020-2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Energy (Units)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('obj7_top_states_growth.png')
plt.show()
print("Growth plot saved as 'obj7_top_states_growth.png'")

# %%


# Correlation matrix
correlation_matrix = df[['wind_energy', 'solar_energy', 'other_renewable_energy', 'total_renewable_energy']].corr()

# Heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Between Energy Types (2020-2024)', fontsize=14)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.show()
print("Correlation heatmap saved as 'eda_correlation_heatmap.png'")





