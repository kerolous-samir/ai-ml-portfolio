# FoodHub — Order Data Analysis

> Exploratory data analysis of a New York food-aggregator platform to uncover demand patterns,
> fulfilment bottlenecks and revenue concentration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
![Type](https://img.shields.io/badge/Type-Exploratory%20Data%20Analysis-blue)

## Business Context

FoodHub is a food aggregator that gives customers access to multiple restaurants through a single
app. The company earns revenue by taking a fixed margin on each delivery order. With restaurant
numbers in New York growing daily, FoodHub needs to understand which restaurants and cuisines
drive demand, and where the fulfilment process loses time — so it can improve customer experience
and allocate couriers intelligently.

## Objective

Analyse historical order data to answer the business team's key questions about demand,
fulfilment time, ratings behaviour and revenue concentration, and turn the answers into
operational recommendations.

## Dataset

**1,898 orders × 9 columns**, one row per completed order.

| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier |
| `customer_id` | Customer who placed the order |
| `restaurant_name` | Restaurant fulfilling the order |
| `cuisine_type` | Cuisine category ordered |
| `cost_of_the_order` | Order value (USD) |
| `day_of_the_week` | Weekday or Weekend |
| `rating` | Customer rating out of 5 (not always given) |
| `food_preparation_time` | Minutes from order confirmation to courier pick-up |
| `delivery_time` | Minutes from pick-up to drop-off |

## Approach

1. **Data quality** — shape, dtypes, missing-value audit and treatment of unrated orders.
2. **Statistical summary** — distribution of order cost, preparation time and delivery time.
3. **Univariate analysis** — demand by cuisine type, restaurant, day of week and cost band.
4. **Bivariate analysis** — rating vs. cost, cuisine vs. weekday/weekend demand, cost vs. delivery time.
5. **Business rules** — revenue modelling against the platform's tiered commission structure.
6. **Conclusions** — translation of each finding into an operational recommendation.

## Key Findings

| Finding | Detail |
|---------|--------|
| Demand is weekend-heavy | Order volume peaks Saturday–Sunday, while courier capacity is flat |
| American cuisine leads | The single most-ordered cuisine category on the platform |
| Prep time dominates fulfilment | ≈ **27 min** preparation vs. ≈ **24 min** delivery — the kitchen, not the road, is the bottleneck |
| Rating data is sparse | ≈ **39%** of orders are never rated, weakening quality monitoring |
| Revenue is concentrated | Orders above **$20** account for ≈ **29%** of sales and the bulk of commission income |

## Recommendations

- **Encourage ratings** — small incentives to close the 39% feedback gap and improve quality signals.
- **Weekend staffing** — schedule additional couriers against the observed weekend demand peak.
- **Promote fast-track partners** — surface the four high-rated, high-volume restaurants to drive traffic.
- **Attack preparation time** — work with restaurants whose prep time exceeds 30 minutes to streamline kitchen workflow.
- **Tiered delivery pricing** — introduce a premium fee for sub-20-minute delivery to monetise time-sensitive customers.

## Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis with all charts and output |
| [`report.html`](report.html) | Standalone HTML report — open in any browser |
