# Insurance Quote Generation & Comparison System

## Overview

This system provides automated insurance quote generation and comparison using machine learning models to assess health risk from CT scans and city wellness metrics. The system integrates with Azure SQL Database and Azure Blob Storage to deliver data-driven, risk-adjusted insurance premiums.

## Key Features

- **Automated Quote Generation**: Generate insurance quotes in seconds using ML risk assessment
- **Multi-Scenario Comparison**: Compare quotes across different cities, premiums, and risk scenarios
- **ML-Powered Risk Assessment**: 
  - CT Risk Model (ResNet18) analyzes CT scan images
  - Wellness Model (Neural Network) evaluates city wellness metrics
- **Azure Cloud Integration**: Seamless integration with Azure SQL Database and Azure Blob Storage

## System Components

### Notebooks

1. **`quotegeneration.ipynb`**: Main pipeline for generating single insurance quotes
   - Retrieves customer data and CT images
   - Computes ML risk scores
   - Calculates premiums
   - Creates contracts and records

2. **`quote_comparison.ipynb`**: Comparison tool for analyzing multiple quote scenarios
   - Executes multiple scenarios
   - Generates comparison visualizations
   - Identifies best/worst options
   - Exports results

## Data Sources

### Azure SQL Database
- **Purpose**: Transactional data (customers, contracts, health factors)
- **Connection**: `pymssql` library
- **Tables**: Customer, CustomerHealthFactor, Contract, ContractBenefit, ContractPremium

### Azure Blob Storage
- **Container**: `datalake`
- **CT Images**: `medical_images/` folder (PNG/JPG format)
- **City Wellness Data**: `city_wellness_curated.parquet` (Parquet format)
- **Access**: Storage Account Key authentication


## ML Models

### CT Risk Model
- **Architecture**: ResNet18 (pretrained) + Custom head
- **Input**: CT scan image (224×224 RGB)
- **Output**: CT risk score (0-1, higher = more risk)
- **Preprocessing**: Resize to 224×224, normalize to tensor

### Wellness Model
- **Architecture**: Neural Network (9→32→16→1)
- **Input**: 9 city wellness features
- **Output**: Wellness score (0-1, higher = better wellness)
- **Preprocessing**: StandardScaler normalization

## Premium Calculation

**Formula**: `Premium = Base × (1 + 0.4 × CT_Risk + 0.6 × (1 - Wellness))`

- **Base**: Base premium amount (e.g., $200)
- **CT Risk Weight**: 40% (α = 0.4)
- **Wellness Weight**: 60% (β = 0.6)
- Higher CT risk = higher premium
- Lower wellness = higher premium

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy torch torchvision pymssql azure-storage-blob scikit-learn pyarrow matplotlib seaborn
   ```

2. **Configure**:
   - Update database credentials in notebook
   - Update Azure Blob Storage credentials
   - Ensure model files exist in `models/` directory

3. **Run Quote Generation**:
   - Open `quotegeneration.ipynb`
   - Set `CUSTOMER_ID` and `BASE_PREMIUM`
   - Run all cells

4. **Run Quote Comparison**:
   - Open `quote_comparison.ipynb`
   - Define scenarios in `SCENARIOS` list
   - Run all cells


## Key Workflows

### Quote Generation Workflow
1. Retrieve customer data from SQL Database
2. Download CT image from Azure Blob Storage
3. Load city wellness data from Parquet
4. Preprocess data (image resize, feature scaling)
5. Run ML model inference (CT risk + Wellness scores)
6. Calculate premium using formula
7. Create contract records in database
8. Return quote summary

### Quote Comparison Workflow
1. Define multiple scenarios (cities, premiums, risk scores)
2. Execute quote generation for each scenario
3. Aggregate results
4. Generate comparison visualizations
5. Identify best/worst options
6. Export results (optional)


