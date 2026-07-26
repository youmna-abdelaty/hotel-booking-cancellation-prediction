# Hotel Booking Cancellation Prediction

## Overview
This project focuses on preparing a hotel booking dataset for a cancellation prediction task. It covers the full pipeline from raw data to a model-ready dataset: exploratory data analysis (EDA), data cleaning, feature engineering, categorical encoding, and train-test splitting with feature scaling.

## Project Structure
```
hotel-booking-cancellation-prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   └── hotel_bookings.csv
│
└── notebooks/
    └── hotel_booking_cancellation_prediction.ipynb
```

## Dataset
- **File:** `data/hotel_bookings.csv`
- **Original shape:** 119,390 rows × 32 columns
- **Target variable:** `is_canceled` (1 = booking canceled, 0 = not canceled)
- The dataset contains booking details such as lead time, arrival date, stay duration, guest counts, market segment, room type, deposit type, and reservation status.

## Steps Performed

### 1. Exploratory Data Analysis (EDA)
- Reviewed dataset structure, data types, and summary statistics (`info()`, `describe()`).
- Visualized missing data patterns using `missingno` matrix and heatmap.
- Identified a strong negative correlation (-0.6) between missing values in `company` and `agent`, indicating a Missing Not At Random (MNAR) pattern — bookings typically go through either an agent or a company, rarely both.
- Detected outliers in `adr` and `lead_time` using boxplots and the IQR method.

### 2. Data Cleaning
- **Missing values:**
  - `company` and `agent`: filled with `0` (representing "No Company" / "No Agent").
  - `country`: imputed with the mode (most frequent country).
  - `children`: imputed with the mode (0 children).
- **Duplicates:** checked and removed exact duplicate rows, then reset the index.
- **Outliers:** capped `adr` and `lead_time` using the IQR method (Q3 + 1.5 × IQR upper bound, lower bound fixed at 0).
- **Data type fixes:**
  - `reservation_status_date` converted from string to `datetime`.
  - `children` converted from `float64` to `int64`.

### 3. Feature Engineering
Created new features to better capture booking patterns:
- `total_guests` = adults + children + babies
- `total_nights` = weekend nights + week nights
- `is_family` = binary flag (1 if the booking includes children or babies, else 0)

### 4. Categorical Encoding
- Analyzed cardinality of all categorical columns.
- **Low-cardinality columns** (`arrival_date_month`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `assigned_room_type`, `deposit_type`, `customer_type`): encoded using **One-Hot Encoding** (`drop_first=True`).
- **High-cardinality column** (`country`, 113 unique values): rare categories (below 1% frequency) grouped into an `"Other"` category before applying One-Hot Encoding, to avoid excessive dimensionality.

### 5. Data Leakage Prevention & Train-Test Split
- Dropped `reservation_status` and `reservation_status_date`, since both directly reveal the booking outcome and would leak target information.
- Dropped `hotel` (zero-variance column, only 1 unique value).
- Split features (`X`) and target (`y`), then performed an 80/20 train-test split (`random_state=42`).
- Standardized numerical features using `StandardScaler` (fit on train, transform on test).

## Tools Used
- Python
- Pandas, NumPy
- Missingno
- Seaborn, Matplotlib
- Scikit-learn (`train_test_split`, `StandardScaler`)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/youmna-abdelaty/hotel-booking-cancellation-prediction.git
   cd hotel-booking-cancellation-prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. The dataset is already included in `data/hotel_bookings.csv`.
4. Open `notebooks/hotel_booking_cancellation_prediction.ipynb` in Jupyter or Google Colab and run all cells.

## Final Dataset
After cleaning, feature engineering, and encoding, the dataset was expanded to **81 columns** (from the original 32), ready to be used for training classification models to predict booking cancellations.
