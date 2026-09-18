# FoodHub — Order Data Analysis

> Exploratory data analysis of a New York food-aggregator platform to uncover demand patterns,
> fulfilment bottlenecks and how order value maps onto the platform's commission tiers.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5.3-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
![Type](https://img.shields.io/badge/Type-Exploratory%20Data%20Analysis-blue)
![Records](https://img.shields.io/badge/Orders-1%2C898-lightgrey)

## Business Context

FoodHub is a food aggregator that gives customers access to multiple restaurants through a single
app. The company earns revenue by taking a margin on each delivery order. With restaurant numbers
in New York growing daily, FoodHub needs to understand which restaurants and cuisines drive demand,
and where the fulfilment process loses time — so it can improve customer experience and plan
operations around real demand peaks.

## Objective

Analyse historical order data to answer the business team's questions about demand, fulfilment
time, ratings behaviour and the commission mix, and turn the answers into operational
recommendations.

## Dataset

**1,898 orders × 9 columns**, one row per completed order (cells 11 and 14).

| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier |
| `customer_id` | Customer who placed the order |
| `restaurant_name` | Restaurant fulfilling the order |
| `cuisine_type` | Cuisine category ordered |
| `cost_of_the_order` | Order value in USD — mean **$16.50**, median **$14.14**, range **$4.47–$35.41** |
| `day_of_the_week` | Two values only: `Weekday` or `Weekend` |
| `rating` | Customer rating, stored as text with `'Not given'` for unrated orders; observed values **3–5** |
| `food_preparation_time` | Minutes from order confirmation to courier pick-up — mean **27.37**, range **20–35** |
| `delivery_time` | Minutes from pick-up to drop-off — mean **24.16**, range **15–33** |

No courier, fleet, staffing, shift or location fields exist in the data.

## Approach

1. **Data quality** — shape, dtypes and missing-value audit; `'Not given'` ratings converted to
   `NaN` and left unimputed (cells 11–23).
2. **Statistical summary** — `describe()` across order cost, preparation time and delivery time
   (cell 20).
3. **Univariate analysis** — distributions of cost, prep time and delivery time; order counts by
   day type; top-10 cuisines; top-5 restaurants; share of orders above $20 (cells 28–36).
4. **Multivariate analysis** — correlation across order cost, preparation time and delivery time;
   cost vs. delivery time split by day type; mean delivery time by day type (cells 46 and 57).
5. **Business rules** — the platform's tiered commission model (25% above $20, 15% above $5, none
   at or below $5) and the end-to-end 60-minute breach rate (cells 51 and 54).
6. **Conclusions** — each finding mapped to an operational recommendation (cells 61–62).

## Key Findings

| Finding | Detail |
|---------|--------|
| Demand skews to weekends | The day-type countplot shows more orders on `Weekend` than `Weekday` (cell 28), yet weekend deliveries are the faster ones — ≈ **22.5 min** vs ≈ **28.3 min** on weekdays (cell 57) |
| American cuisine leads | Top of the top-10 cuisine chart (cell 28) and the most-ordered cuisine on weekends (cell 33) |
| Prep time dominates fulfilment | Mean preparation ≈ **27.4 min** vs mean delivery ≈ **24.2 min** (cells 20 and 39) — the kitchen, not the road, is the longer leg |
| One order in ten breaches the hour | **200 orders (10.54%)** take more than 60 minutes from placement to drop-off (cell 54) |
| Rating data is sparse | **736 of 1,898 orders (38.8%)** are unrated, so every rating-based result rests on 61.2% of the data (cells 17 and 23) |
| High-value orders sit in the top commission tier | **29.24%** of *orders* cost more than $20 (cell 36) and are billed at 25% rather than 15% (cell 51) |
| Commission base | Net revenue across all 1,898 orders: **$6,166.30** (cell 51) |

### Promotion-eligible restaurants

Restaurants with more than 50 ratings and a mean rating above 4 (cell 48):

| Restaurant | Ratings | Mean rating |
|------------|--------:|------------:|
| Shake Shack | 133 | 4.28 |
| The Meatball Shop | 84 | 4.51 |
| Blue Ribbon Sushi | 73 | 4.22 |
| Blue Ribbon Fried Chicken | 64 | 4.33 |

These four are also the platform's highest-volume restaurants by order count, alongside Parm
(cell 30).

## Recommendations

- **Encourage ratings** — small incentives to close the 38.8% feedback gap and improve quality signals.
- **Weekend staffing** — schedule against the observed weekend order peak.
- **Promote fast-track partners** — surface the four high-rated, high-volume restaurants above to drive traffic.
- **Attack preparation time** — prep time is the larger half of fulfilment; work with slow kitchens to streamline workflow.
- **Tiered delivery pricing** — test a premium fee for fast delivery to monetise time-sensitive customers.

The last two are directional proposals carried over from the notebook's conclusions, not results
the analysis quantifies — see below.

## Known Limitations

Written plainly, because a reviewer can open the notebook next to this file.

- **The notebook is not reproducible from this repo.** Cell 8 reads
  `/content/drive/MyDrive/foodhub_order.csv`, a Google Drive path, and no CSV ships here. Every
  figure above is read from stored cell outputs, not re-executed.
- **The notebook and the published report contain a "29% of sales" error that this README does
  not repeat.** Markdown cell 61 and `report.html` state that orders above $20 "account for ~29%
  of sales." Cell 36 computes `(cost > 20).mean()`, which is the share of order *count*. The
  notebook never computes a value or commission share for that segment, so only the count share is
  claimed here. The notebook and the rendered report have not been corrected.
- **Rating results rest on a self-selected subset.** The 736 unrated orders are dropped rather than
  imputed, so the promotion shortlist (cell 48) describes customers who chose to rate. Nothing
  tests whether rating propensity is independent of restaurant, cuisine or delivery time.
- **The promo filter does not implement the stated rule.** Cell 48 uses `mean >= 4` where the brief
  specifies a mean above 4. All four qualifying restaurants average above 4.2, so the output is
  unchanged, but the code and the criterion disagree.
- **Weekend demand is never normalised for exposure.** `day_of_the_week` holds only `Weekday` and
  `Weekend`, so there is no single-day resolution. Weekend spans 2 calendar days against 5 for
  weekday, and the notebook compares absolute counts only — never a per-day order rate.
- **The weekday/weekend delivery gap is untested.** The 22.5 vs 28.3 minute difference (cell 57) is
  a raw mean difference: no confidence interval, no significance test, and no control for obvious
  confounders such as traffic.
- **Correlation is shown but never quoted.** Cell 46 renders the heatmap as an image and does not
  print the coefficients, so no correlation value appears in this README.
- **Descriptive only.** There is no model, no train/test split and no out-of-sample validation.
  Nothing here is a prediction.
- **Two recommendations exceed the data.** No cell computes preparation time per restaurant, so
  "slow kitchens" names no restaurant; and courier capacity, staffing levels and price elasticity
  are absent from the 9-column schema entirely, so the delivery-pricing proposal is untested.
- **The environment is not a clean install.** Cell 3 pins numpy 1.25.2 / pandas 1.5.3 /
  matplotlib 3.7.1 / seaborn 0.13.1 over Colab's preinstalled versions; pip reports unresolved
  dependency conflicts and the notebook requires a kernel restart before a clean run.
- **`report.html` is a static Colab export.** Its DataFrame outputs carry Colab's interactive-table
  and quick-chart buttons, which are inert outside Colab.

## Tech Stack

`Python 3.10+` · `pandas 1.5.3` · `NumPy 1.25.2` · `Matplotlib 3.7.1` · `Seaborn 0.13.1` · `Google Colab`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed analysis with all charts and output |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/01-foodhub-data-analysis/report.html)** |
