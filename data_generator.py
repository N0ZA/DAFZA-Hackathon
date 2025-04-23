import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from hijri_converter import Hijri, Gregorian

print("Starting dataset generation...")

# Configuration
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"
PRODUCTS_CSV = "products.csv"
WEEKLY_OPERATIONAL_CSV = "weekly_operational_data.csv"
WEEKLY_DEMAND_CSV = "weekly_demand.csv"
N_ITEMS = 50

print("Generating item master data (products.csv)...")
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

categories = ['Poultry', 'Meat_Beef', 'Meat_Lamb', 'Meat_Pork', 'Seafood_Fish', 'Seafood_Shellfish',
              'Produce_Fruit_Tropical', 'Produce_Fruit_Berry', 'Produce_Veg_Leafy', 'Produce_Veg_Root',
              'Dairy_Milk', 'Dairy_Cheese', 'Dairy_Yogurt', 'Bakery_Bread', 'Bakery_Pastry',
              'Dry_Goods_Flour', 'Dry_Goods_Rice', 'Dry_Goods_Pasta', 'Dry_Goods_Oil', 'Dry_Goods_Spice',
              'Beverage_Soft_Cola', 'Beverage_Soft_Juice', 'Beverage_Alcoholic_Beer', 'Beverage_Alcoholic_Wine_Red',
              'Beverage_Alcoholic_Wine_White', 'Beverage_Alcoholic_Spirit', 'Beverage_Water_Still', 'Beverage_Water_Sparkling',
              'Dates_Sweets', 'Frozen_Goods']
units = ['KG', 'Litre', 'Each', 'Bottle_750ml', 'Bottle_330ml', 'Case_24', 'Pack_6', 'Dozen', 'Bag_5KG', 'Box_10KG']

items_master = []
for i in range(N_ITEMS):
    cat = random.choice(categories)
    unit = random.choice(units)
    sku = f'SKU{1000+i}'
    if unit in ['KG', 'Litre', 'Bag_5KG']:
        base_need = np.random.uniform(5, 150); base_cost = np.random.uniform(8, 120); moq = random.choice([1, 5, 10, 20])
    elif 'Bottle' in unit or unit == 'Each' or unit == 'Dozen' or unit == 'Pack_6':
        base_need = np.random.uniform(12, 500); base_cost = np.random.uniform(1.5, 40); moq = random.choice([6, 12, 24, 48])
    else: base_need = np.random.uniform(2, 60); base_cost = np.random.uniform(40, 250); moq = random.choice([1, 5, 10])
    shelf_life = 180
    if cat in ['Poultry', 'Meat_Beef', 'Meat_Lamb', 'Seafood_Fish', 'Seafood_Shellfish', 'Dairy_Milk', 'Dairy_Yogurt', 'Bakery_Pastry', 'Produce_Fruit_Berry', 'Produce_Veg_Leafy']: shelf_life = random.randint(3, 10)
    elif cat in ['Produce_Fruit_Tropical', 'Produce_Veg_Root', 'Dairy_Cheese', 'Bakery_Bread']: shelf_life = random.randint(7, 21)
    elif 'Wine' in cat or 'Spirit' in cat: shelf_life = 365 * 5
    elif 'Beer' in cat or 'Soft' in cat or 'Water' in cat: shelf_life = random.randint(120, 365)
    lead_time = random.randint(1, 5)
    if random.random() < 0.15 or 'Wine' in cat or 'Spirit' in cat: lead_time = random.randint(7, 21)
    items_master.append({
        'SKU': sku, 'Item_Name': f'{cat.split("_")[0]}_{fake.word().capitalize()}_{unit}', 'Item_Category': cat,
        'Unit_of_Measure': unit, 'Avg_Unit_Cost_Base': round(base_cost, 2), 'Supplier_Lead_Time_Days': lead_time,
        'Min_Order_Quantity': moq, 'Shelf_Life_Days': shelf_life, 'Base_Weekly_Need': base_need
    })
products_df = pd.DataFrame(items_master)
product_cols = ['SKU', 'Item_Name', 'Item_Category', 'Unit_of_Measure', 'Avg_Unit_Cost_Base',
                'Supplier_Lead_Time_Days', 'Min_Order_Quantity', 'Shelf_Life_Days']
products_df[product_cols].to_csv(PRODUCTS_CSV, index=False)
print(f"Generated {len(products_df)} products and saved to {PRODUCTS_CSV}")
print("-" * 30)

# --- Date Range & Holiday Helpers ---
print("Setting up date ranges and holiday helpers...")
weekly_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='W-SUN')
def get_uae_public_holidays(year):
    holidays = set(); holidays.add(datetime(year, 1, 1).date()); holidays.add(datetime(year, 12, 1).date()); holidays.add(datetime(year, 12, 2).date()); holidays.add(datetime(year, 12, 3).date())
    hijri_year_approx = year - (2024 - 1445)
    try: h_fitr_start = Hijri(hijri_year_approx, 10, 1); g_fitr_start_obj = h_fitr_start.to_gregorian(); eid_fitr_g = datetime(g_fitr_start_obj.year, g_fitr_start_obj.month, g_fitr_start_obj.day).date(); holidays.add(eid_fitr_g); holidays.add(eid_fitr_g + timedelta(days=1)); holidays.add(eid_fitr_g + timedelta(days=2))
    except (ValueError, OverflowError) as e: print(f"Warn: Eid Fitr calc fail {year}: {e}")
    try: h_arafat = Hijri(hijri_year_approx, 12, 9); g_arafat_obj = h_arafat.to_gregorian(); arafat_g = datetime(g_arafat_obj.year, g_arafat_obj.month, g_arafat_obj.day).date(); holidays.add(arafat_g); h_adha_start = Hijri(hijri_year_approx, 12, 10); g_adha_start_obj = h_adha_start.to_gregorian(); eid_adha_g = datetime(g_adha_start_obj.year, g_adha_start_obj.month, g_adha_start_obj.day).date(); holidays.add(eid_adha_g); holidays.add(eid_adha_g + timedelta(days=1)); holidays.add(eid_adha_g + timedelta(days=2))
    except (ValueError, OverflowError) as e: print(f"Warn: Eid Adha calc fail {year}: {e}")
    return holidays
uae_holidays = set()
for year in range(int(START_DATE[:4]), int(END_DATE[:4]) + 2): uae_holidays.update(get_uae_public_holidays(year))
def get_ramadan_dates(year):
    hijri_year_approx = year - (2024 - 1445)
    try: h_ramadan_start = Hijri(hijri_year_approx, 9, 1); g_ramadan_start_obj = h_ramadan_start.to_gregorian(); ramadan_start = datetime(g_ramadan_start_obj.year, g_ramadan_start_obj.month, g_ramadan_start_obj.day).date(); h_ramadan_end_29 = Hijri(hijri_year_approx, 9, 29); g_ramadan_end_29_obj = h_ramadan_end_29.to_gregorian(); ramadan_end_29 = datetime(g_ramadan_end_29_obj.year, g_ramadan_end_29_obj.month, g_ramadan_end_29_obj.day).date(); return ramadan_start, ramadan_end_29
    except (ValueError, OverflowError) as e: print(f"Warn: Ramadan calc fail {year}: {e}"); return None, None
ramadan_periods = {}
for year in range(int(START_DATE[:4]) -1, int(END_DATE[:4]) + 2): ramadan_periods[year] = get_ramadan_dates(year)
def is_in_ramadan(date_obj):
    current_date = date_obj.date() if isinstance(date_obj, datetime) else date_obj
    for year in [current_date.year, current_date.year - 1]:
        if year in ramadan_periods and ramadan_periods[year][0] is not None: start, end = ramadan_periods[year];
        if start <= current_date <= end: return True
    return False
def check_holiday_in_week(start_date, holiday_set):
    for i in range(7):
        if (start_date + timedelta(days=i)).date() in holiday_set: return 1
    return 0
def check_eid_in_week(start_date):
    year = start_date.year
    hijri_year_approx = year - (2024-1445)
    try:
        h_fitr_start = Hijri(hijri_year_approx, 10, 1); g_fitr_start_obj = h_fitr_start.to_gregorian()
        eid_fitr_start_date = datetime(g_fitr_start_obj.year, g_fitr_start_obj.month, g_fitr_start_obj.day).date()
        h_adha_start = Hijri(hijri_year_approx, 12, 10); g_adha_start_obj = h_adha_start.to_gregorian()
        eid_adha_start_date = datetime(g_adha_start_obj.year, g_adha_start_obj.month, g_adha_start_obj.day).date()
        week_dates_set = set([(start_date + timedelta(days=d)).date() for d in range(7)])
        if eid_fitr_start_date in week_dates_set or eid_adha_start_date in week_dates_set: return 1
    except (ValueError, OverflowError): pass
    return 0

def generate_percentages(n_categories, concentrations=None):
    if concentrations is None: concentrations = np.ones(n_categories)
    concentrations = np.array(concentrations).astype(float); concentrations[concentrations <= 0] = 0.01
    if len(concentrations) != n_categories: concentrations = np.ones(n_categories)
    return np.random.dirichlet(concentrations) * 100

# --- Generate Weekly Operational Data ---
print("Generating weekly operational data (weekly_operational_data.csv)...")
weekly_ops_data = []
total_weeks = len(weekly_dates)
for i, week_start_date in enumerate(weekly_dates):
    if (i + 1) % 52 == 0: print(f"  Processing week {i+1}/{total_weeks} for operational data...")
    month = week_start_date.month; year = week_start_date.year; week_of_year = week_start_date.isocalendar()[1]
    is_peak = 1 if month in [11, 12, 1, 2, 3] or (month == 4 and week_start_date.day < 15) or (month == 10 and week_start_date.day >=15) else 0
    is_low = 1 if month in [6, 7, 8] or (month == 9 and week_start_date.day < 15) else 0
    is_ramadan = 1 if is_in_ramadan(week_start_date) else 0
    is_public_holiday = check_holiday_in_week(week_start_date, uae_holidays)
    is_eid_week = check_eid_in_week(week_start_date) if is_public_holiday else 0
    occupancy_base = 65; occupancy_seasonality = 30 * np.sin(2 * np.pi * (week_of_year - 2) / 52); occupancy_low_season_dip = -20 if is_low else 0; occupancy_holiday_boost = 15 if is_public_holiday and not is_low else 0; occupancy_ramadan_dip = -10 if is_ramadan and not is_eid_week else 0
    occupancy = occupancy_base + occupancy_seasonality + occupancy_low_season_dip + occupancy_holiday_boost + occupancy_ramadan_dip + np.random.normal(0, 1.5)
    occupancy = np.clip(occupancy, 25, 99)
    event_guests_base = np.random.poisson(80); event_guests_season_boost = np.random.poisson(200) if is_peak and not is_public_holiday else 0; event_guests_holiday_boost = np.random.poisson(300) if is_public_holiday else 0
    event_guests = event_guests_base + event_guests_season_boost + event_guests_holiday_boost + np.random.poisson(5)
    event_guests = max(0, event_guests)
    region_con = [1.5, 1.5, 1.5, 0.5, 1, 0.5]; purpose_con = [1.5, 1, 0.5]
    if is_peak: region_con[1] = 5; region_con[3] = 0.7; purpose_con[0] = 4; purpose_con[2] = 0.8
    elif is_low: region_con[0] = 4; region_con[2] = 3; region_con[4] = 2
    else: purpose_con[1] = 2
    if is_ramadan or is_eid_week: region_con[0] = 6; region_con[4] = 2.5
    if is_public_holiday and not (is_ramadan or is_eid_week) : purpose_con[0] = 4
    regions_pct = generate_percentages(6, concentrations=region_con); purpose_pct = generate_percentages(3, concentrations=purpose_con)
    family_base = 15; family_boost = 25 if is_peak or is_eid_week or month in [7, 8] else 0

    family_pct = np.clip(family_base + family_boost + np.random.normal(0, 2.0), 5, 65)
    adult_only_pct = 100 - family_pct
    weekly_ops_data.append({
        'Week_Start_Date': week_start_date, 'Forecasted_Occupancy_Percent': round(occupancy, 2), 'Forecasted_Event_Guests': event_guests,
        'Week_of_Year': week_of_year, 'Month': month, 'Is_Peak_Season': is_peak, 'Is_Low_Season': is_low, 'Is_Ramadan_Week': is_ramadan,
        'Is_Eid_Week': is_eid_week, 'Is_Public_Holiday_Week': is_public_holiday, 'Percent_Guests_GCC': round(regions_pct[0], 2),
        'Percent_Guests_Europe': round(regions_pct[1], 2), 'Percent_Guests_Asia': round(regions_pct[2], 2), 'Percent_Guests_Americas': round(regions_pct[3], 2),
        'Percent_Guests_Africa_ME': round(regions_pct[4], 2), 'Percent_Guests_Other_Region': round(regions_pct[5], 2), 'Percent_Guests_Leisure': round(purpose_pct[0], 2),
        'Percent_Guests_Business': round(purpose_pct[1], 2), 'Percent_Guests_Group': round(purpose_pct[2], 2), 'Percent_Families': round(family_pct, 2),
        'Percent_Adult_Only_Bookings': round(adult_only_pct, 2)
    })
weekly_ops_df = pd.DataFrame(weekly_ops_data)
weekly_ops_df['Week_Start_Date'] = pd.to_datetime(weekly_ops_df['Week_Start_Date'])
weekly_ops_df.to_csv(WEEKLY_OPERATIONAL_CSV, index=False, date_format='%Y-%m-%d')
print(f"Generated {len(weekly_ops_df)} weeks of operational data and saved to {WEEKLY_OPERATIONAL_CSV}")
print("-" * 30)

# --- Generate Weekly Demand Data ---
print("Generating weekly demand data (weekly_demand.csv)...")
all_demand_data = []
weekly_ops_df_indexed = weekly_ops_df.set_index('Week_Start_Date')
for i, week_start_date in enumerate(weekly_dates):
    if (i + 1) % 26 == 0: print(f"  Processing week {i+1}/{total_weeks} for demand data...")
    try: ops_row = weekly_ops_df_indexed.loc[week_start_date]
    except KeyError: print(f"  Skipping week {week_start_date.date()} - op data not found."); continue
    occupancy = ops_row['Forecasted_Occupancy_Percent']; event_guests = ops_row['Forecasted_Event_Guests']; is_ramadan = ops_row['Is_Ramadan_Week']; is_eid_week = ops_row['Is_Eid_Week']; is_public_holiday = ops_row['Is_Public_Holiday_Week']; is_peak = ops_row['Is_Peak_Season']; is_low = ops_row['Is_Low_Season']; regions_pct = [ops_row['Percent_Guests_GCC'], ops_row['Percent_Guests_Europe'], ops_row['Percent_Guests_Asia'], ops_row['Percent_Guests_Americas'], ops_row['Percent_Guests_Africa_ME'], ops_row['Percent_Guests_Other_Region']]; family_pct = ops_row['Percent_Families']
    for _, item in products_df.iterrows():
        cost_fluctuation = np.random.normal(1, 0.08); seasonal_cost_factor = 1
        if item['Item_Category'] in ['Produce_Fruit_Tropical', 'Produce_Fruit_Berry', 'Produce_Veg_Leafy', 'Seafood_Fish', 'Seafood_Shellfish'] and ops_row['Month'] in [6,7,8,9]: seasonal_cost_factor = 1.20
        avg_unit_cost = item['Avg_Unit_Cost_Base'] * cost_fluctuation * seasonal_cost_factor
        avg_unit_cost = max(0.5, round(avg_unit_cost, 2))

        # --- Simulate Target Quantity Needed ---
        base = item['Base_Weekly_Need']; occ_effect = base * (occupancy / 100)**0.8; event_effect = 0
        if item['Item_Category'] in ['Poultry', 'Meat_Beef', 'Meat_Lamb', 'Seafood_Fish', 'Bakery_Bread', 'Beverage_Soft_Cola', 'Beverage_Water_Still', 'Beverage_Water_Sparkling'] or 'Alcoholic' in item['Item_Category']: event_effect = (event_guests / 300) * base * 0.6
        guest_mix_mod = 1.0
        if regions_pct[0] > 35:
            if item['Item_Category'] in ['Meat_Lamb', 'Dates_Sweets', 'Dry_Goods_Rice']: guest_mix_mod *= 1.3
            if item['Item_Category'] == 'Meat_Pork': guest_mix_mod *= 0.05
        if regions_pct[1] > 30:
            if item['Item_Category'] in ['Dairy_Cheese', 'Bakery_Pastry', 'Beverage_Alcoholic_Wine_Red', 'Beverage_Alcoholic_Wine_White']: guest_mix_mod *= 1.25
            if item['Item_Category'] == 'Dry_Goods_Spice': guest_mix_mod *= 0.8
        if regions_pct[2] > 25:
            if item['Item_Category'] in ['Poultry', 'Dry_Goods_Rice', 'Produce_Veg_Root', 'Dry_Goods_Spice']: guest_mix_mod *= 1.2
            if item['Item_Category'] == 'Meat_Beef': guest_mix_mod *= 0.7
        if family_pct > 35:
             if item['Item_Category'] in ['Beverage_Soft_Juice', 'Dairy_Milk', 'Bakery_Pastry', 'Produce_Fruit_Berry', 'Poultry']: guest_mix_mod *= 1.15
             if 'Alcoholic' in item['Item_Category']: guest_mix_mod *= 0.85
        temporal_mod = 1.0
        if is_ramadan:
            if item['Item_Category'] in ['Dates_Sweets', 'Meat_Lamb', 'Dry_Goods_Rice', 'Beverage_Soft_Juice']: temporal_mod *= 1.7
            elif 'Alcoholic' in item['Item_Category'] or item['Item_Category'] == 'Meat_Pork': temporal_mod *= 0.1
            elif item['Item_Category'] in ['Bakery_Pastry', 'Dairy_Cheese']: temporal_mod *= 0.5
            else: temporal_mod *= 0.9
        elif is_eid_week:
            if item['Item_Category'] in ['Meat_Lamb', 'Dates_Sweets', 'Poultry', 'Dry_Goods_Rice']: temporal_mod *= 1.8
            else: temporal_mod *= 1.2
        elif is_public_holiday: temporal_mod *= 1.15
        season_mod = 1.0
        if is_peak: season_mod *= 1.05
        if is_low: season_mod *= 0.9
        simulated_need = base + (occ_effect * guest_mix_mod * season_mod * temporal_mod) + (event_effect * guest_mix_mod * temporal_mod)

        noise_std_dev = max(base * 0.01, simulated_need * 0.02)

        noise = np.random.normal(0, noise_std_dev)
        simulated_need_with_noise = simulated_need + noise
        target_quantity = max(0, round(simulated_need_with_noise, 2))
        # --- End Simulation Logic Modification ---

        all_demand_data.append({ 'Week_Start_Date': week_start_date, 'SKU': item['SKU'],
                                 'Avg_Unit_Cost': avg_unit_cost, 'Target_Weekly_Quantity_Needed': target_quantity })

# --- Post-Process Demand Data (Lags) ---
print("Post-processing demand data (calculating lags)...")
demand_df = pd.DataFrame(all_demand_data)
demand_df['Week_Start_Date'] = pd.to_datetime(demand_df['Week_Start_Date'])
demand_df = demand_df.sort_values(by=['SKU', 'Week_Start_Date']).reset_index(drop=True)
demand_df['Needed_Qty_Lag_1W'] = demand_df.groupby('SKU')['Target_Weekly_Quantity_Needed'].shift(1)
demand_df['Needed_Qty_Lag_2W'] = demand_df.groupby('SKU')['Target_Weekly_Quantity_Needed'].shift(2)
demand_df['Needed_Qty_Lag_4W'] = demand_df.groupby('SKU')['Target_Weekly_Quantity_Needed'].shift(4)
demand_df['Avg_Needed_Qty_Last_4W'] = demand_df.groupby('SKU')['Needed_Qty_Lag_1W'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
initial_rows = len(demand_df)
demand_df.dropna(subset=['Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W', 'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W'], inplace=True)
rows_dropped = initial_rows - len(demand_df)
print(f"Dropped {rows_dropped} demand rows due to insufficient lag data.")
demand_cols = ['Week_Start_Date', 'SKU', 'Avg_Unit_Cost', 'Needed_Qty_Lag_1W', 'Needed_Qty_Lag_2W',
               'Needed_Qty_Lag_4W', 'Avg_Needed_Qty_Last_4W', 'Target_Weekly_Quantity_Needed']
demand_df = demand_df[demand_cols]

# --- Save Demand Data ---
demand_df.to_csv(WEEKLY_DEMAND_CSV, index=False, date_format='%Y-%m-%d')
print(f"Generated {len(demand_df)} weekly demand records and saved to {WEEKLY_DEMAND_CSV}")
print("-" * 50)
print("Generation complete. You should now have three CSV files:")
print(f"1. {PRODUCTS_CSV} (Item details)")
print(f"2. {WEEKLY_OPERATIONAL_CSV} (Weekly hotel context)")
print(f"3. {WEEKLY_DEMAND_CSV} (Weekly demand per item)")
print("-" * 50)