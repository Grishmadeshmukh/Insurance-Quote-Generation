# End-to-End Reference Architecture Documentation
## Insurance Quote Generation & Comparison System


---

## Executive Summary

This document presents a comprehensive Reference Architecture (RA) for the Insurance Quote Generation and Comparison System, spanning across business, application, DIKW (Data-Information-Knowledge-Wisdom), and infrastructure domains. The system enables automated, ML-driven insurance quote generation and multi-scenario comparison capabilities, leveraging Azure cloud services, machine learning models, and enterprise-grade data governance practices.

**Key Capabilities**:
- Automated insurance quote generation using ML risk assessment
- Multi-scenario quote comparison with visual analytics
- Integration with Azure SQL Database and Azure Blob Storage
- Bias-aware ML models with fairness, accountability, and transparency
- Comprehensive data governance and lifecycle management

---

## Table of Contents

1. [Business Domain Architecture](#1-business-domain-architecture)
2. [Application Domain Architecture](#2-application-domain-architecture)
3. [DIKW Pyramid Architecture](#3-dikw-pyramid-architecture)
4. [Infrastructure Domain Architecture](#4-infrastructure-domain-architecture)
5. [Database Design & Query Optimization](#5-database-design--query-optimization)
6. [System Integration & Data Flow](#6-system-integration--data-flow)
7. [Security & Compliance](#7-security--compliance)
8. [Future Roadmap](#8-future-roadmap)

---

## 1. Business Domain Architecture

### 1.1 Business Objectives

#### 1.1.1 Primary Goals
1. **Automate Quote Generation**: Reduce manual underwriting time from days to minutes
2. **Risk-Based Pricing**: Implement data-driven, objective premium calculation
3. **Customer Experience**: Provide fast, transparent quote generation
4. **Decision Support**: Enable "what-if" analysis for customers and agents

#### 1.1.2 Business Value Propositions
- **Speed**: Quote generation in 1-2 seconds vs. days
- **Consistency**: Eliminate human bias and inconsistency
- **Accuracy**: ML models improve risk assessment precision
- **Transparency**: Clear breakdown of premium components
- **Flexibility**: Compare multiple scenarios side-by-side

### 1.2 Business Use Cases

#### 1.2.1 Use Case 1: Quote Generation (`quotegeneration.ipynb`)
**Actor**: Insurance Agent / System  
**Precondition**: Customer exists in database  
**Flow**:
1. Agent provides Customer ID and Base Premium
2. System retrieves customer data and CT image reference
3. System computes CT risk score from image
4. System computes wellness score from city data
5. System calculates risk-adjusted premium
6. System creates contract and records
7. System returns quote summary

**Postcondition**: Contract created, premium recorded, quote delivered

#### 1.2.2 Use Case 2: Quote Comparison (`quote_comparison.ipynb`)
**Actor**: Insurance Agent / Customer  
**Precondition**: Customer exists in database  
**Flow**:
1. Agent defines multiple scenarios (different cities, premiums, risk scores)
2. System executes each scenario
3. System computes quotes for all scenarios
4. System generates comparison visualizations
5. System provides summary with best/worst options
6. Agent exports results (optional)

**Postcondition**: Comparison report generated, insights provided

### 1.3 Business Rules

#### 1.3.1 Premium Calculation Formula
```
Premium = Base × (1 + α × CT_Risk + β × (1 - Wellness))

Where:
- Base = Base premium amount (e.g., $200)
- α = CT Risk weight (0.4 = 40%)
- β = Wellness weight (0.6 = 60%)
- CT_Risk = CT risk score (0-1, higher = more risk)
- Wellness = Wellness score (0-1, higher = better wellness)
```

#### 1.3.2 Risk Score Interpretation
- **CT Risk Score**: 0.0-1.0 scale
  - 0.0-0.3: Low risk
  - 0.3-0.7: Medium risk
  - 0.7-1.0: High risk
- **Wellness Score**: 0.0-1.0 scale
  - 0.0-0.4: Poor wellness
  - 0.4-0.7: Moderate wellness
  - 0.7-1.0: Excellent wellness

#### 1.3.3 Contract Creation Rules
- Both CT Risk and Wellness scores must be computed
- Contract requires valid Customer ID
- Contract Benefit requires Contract ID (foreign key)
- Contract Premium requires Contract Benefit ID (foreign key)

### 1.4 Business Process Flows

#### 1.4.1 Quote Generation Process
```
┌─────────────────────────────────────────────────────────────┐
│              QUOTE GENERATION BUSINESS PROCESS              │
└─────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────┐
│ Receive Request │ Customer ID + Base Premium
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate Input │ Check Customer ID exists
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Retrieve Customer Data          │ FROM SQL Database
│ - Customer Info                 │
│ - City                          │
│ - CT Image Blob Reference       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Load Unstructured Data           │ FROM Azure Blob Storage
│ - Download CT Image              │
│ - Load City Wellness Parquet     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Preprocess Data                  │
│ - Image: Resize, Normalize       │
│ - Features: Scale, Clean         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ ML Model Inference              │
│ - CT Risk Model (ResNet18)       │ → CT Risk Score
│ - Wellness Model (Neural Net)   │ → Wellness Score
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Calculate Premium                │
│ Premium = Base × (1 + α×CT + β×(1-Wellness)) │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Create Contract                  │ INSERT INTO Contract
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Create Contract Benefit          │ INSERT INTO ContractBenefit
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Record Premium                   │ INSERT INTO ContractPremium
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Generate Quote Summary           │ Display Results
└────────┬────────────────────────┘
         │
         ▼
END
```

#### 1.4.2 Quote Comparison Process
```
┌─────────────────────────────────────────────────────────────┐
│            QUOTE COMPARISON BUSINESS PROCESS                 │
└─────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────┐
│ Define Scenarios│ Multiple scenarios (cities, premiums, risks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute Each    │ Run quote generation for each scenario
│ Scenario        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate       │ Collect all quote results
│ Results         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │ Create comparison table and visualizations
│ Comparison      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze &       │ Identify best/worst options, savings
│ Summarize       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export Results   │ CSV export (optional)
└────────┬────────┘
         │
         ▼
END
```

---

## 2. Application Domain Architecture

### 4.1 Application Components

#### 4.1.1 Quote Generation System Components

##### Data Ingestion Layer
- **SQL Database Connector** (`pymssql`)
  - Functions: `get_connection()`, `get_customer()`, `get_health_factors()`
  - Purpose: Retrieve customer and contract data
  - Error Handling: Connection retry, query validation

- **Azure Blob Storage Connector** (`BlobServiceClient`)
  - Functions: `get_blob_service_client()`, `load_image_from_blob()`, `load_city_wellness_from_parquet()`
  - Purpose: Retrieve CT images and city wellness data
  - Error Handling: Blob existence checks, download retries

##### ML Inference Layer
- **CT Risk Model** (ResNet18)
  - Architecture: Pretrained ResNet18 + Custom head (512→1)
  - Input: CT scan image (224×224 RGB)
  - Output: CT risk score (0-1)
  - Preprocessing: Resize, ToTensor
  - Location: `models/ct_risk_model.pth`

- **Wellness Model** (Neural Network)
  - Architecture: 9→32→16→1 (ReLU activations)
  - Input: 9 city wellness features
  - Output: Wellness score (0-1)
  - Preprocessing: StandardScaler normalization
  - Location: `models/wellness_model.pth`, `models/wellness_scaler.pkl`

##### Business Logic Layer
- **Premium Calculator** (`compute_premium()`)
  - Formula: `Base × (1 + 0.4 × CT_Risk + 0.6 × (1 - Wellness))`
  - Input: Base premium, CT risk score, wellness score
  - Output: Final premium amount

- **Score Manager** (`ensure_scores()`)
  - Purpose: Get or compute health scores
  - Caching: Checks existing scores in database
  - Computation: Triggers ML inference if scores missing

- **Contract Manager**
  - Functions: `insert_contract()`, `insert_contract_benefit()`, `insert_contract_premium()`
  - Purpose: Create contract records
  - Transaction Management: Atomic operations with rollback

##### Model Retraining Module
- **Data Change Detection** (`trigger_model_retraining()`)
  - Purpose: Detect new data in Azure Blob Storage
  - Method: Compare blob modification times
  - Trigger: Automatic retraining when new data detected

- **Model Versioning** (`model_versions.json`)
  - Tracks: Model version, training timestamp, data version
  - Purpose: Audit trail and rollback capability

#### 4.1.2 Quote Comparison System Components

##### Scenario Definition Module
- **Scenario Structure**:
  ```python
  {
      'name': 'Scenario Name',
      'city': 'City Name' or None,
      'base_premium': 200,
      'ct_image_blob': 'blob_name' or None,
      'ct_risk_score': 0.5 or None  # Override option
  }
  ```

##### Multi-Scenario Execution Module
- **Function**: `compare_quote_scenarios()`
  - Purpose: Execute multiple scenarios in sequence
  - Error Handling: Individual scenario failures don't stop execution
  - Output: DataFrame with all scenario results

##### Comparison Analysis Module
- **Functions**: 
  - `display_comparison_table()`: Formatted table output
  - Premium difference calculations
  - Best/worst option identification
  - Savings potential calculation

##### Visualization Module
- **Function**: `visualize_premium_comparison()`
  - Charts:
    1. Premium Comparison Bar Chart
    2. Premium Breakdown (Stacked Bar)
    3. Risk Scores Comparison
    4. Premium Difference from Baseline
  - Libraries: `matplotlib`, `seaborn`

##### Export Module
- **Function**: `export_comparison_to_csv()`
  - Purpose: Export comparison results to CSV
  - Format: All scenario data with formatted values

### 4.2 Application Workflow

#### 4.2.1 Quote Generation Workflow
```
Input: Customer ID, Base Premium
  ↓
[1] Validate Input
  ↓
[2] Retrieve Customer Data (SQL)
  ↓
[3] Load CT Image (Blob Storage)
  ↓
[4] Load City Wellness Data (Parquet)
  ↓
[5] Preprocess Data
  │   ├─ Image: Resize, Tensor
  │   └─ Features: Scale, Normalize
  ↓
[6] ML Inference
  │   ├─ CT Risk Model → CT Risk Score
  │   └─ Wellness Model → Wellness Score
  ↓
[7] Calculate Premium
  ↓
[8] Create Contract (SQL)
  ↓
[9] Create Contract Benefit (SQL)
  ↓
[10] Record Premium (SQL)
  ↓
Output: Quote Summary
```

#### 4.2.2 Quote Comparison Workflow
```
Input: Customer ID, Scenarios List
  ↓
For each scenario:
  ↓
[1] Execute Quote Generation
  │   (Reuse quotegeneration logic)
  ↓
[2] Collect Results
  ↓
[3] Calculate Differences
  ↓
[4] Generate Visualizations
  ↓
[5] Create Summary
  ↓
Output: Comparison Report + Visualizations
```

### 4.3 Technology Stack

#### 4.3.1 Core Technologies
- **Language**: Python 3.13
- **ML Framework**: PyTorch 2.0+
- **Database**: Azure SQL Server (via pymssql)
- **Storage**: Azure Blob Storage
- **Data Processing**: pandas, numpy
- **Image Processing**: PIL, torchvision
- **Visualization**: matplotlib, seaborn

#### 4.3.2 Dependencies
```python
# Core
pandas, numpy, torch, torchvision
# Database
pymssql, pyodbc, sqlalchemy
# Azure
azure-storage-blob, azure-identity
# ML
scikit-learn
# Parquet
pyarrow or fastparquet
# Visualization
matplotlib, seaborn
```

---

## 3. DIKW Pyramid Architecture

### 5.1 Data Layer (Raw Data)

#### 5.1.1 Data Sources
- **CT Scan Images** (Azure Blob Storage)
  - Format: PNG/JPG
  - Location: `datalake/medical_images/`
  - Volume: Thousands of images
  - Characteristics: Unstructured, binary

- **City Wellness Data** (Azure Blob Storage - Parquet)
  - Format: Parquet (converted from CSV)
  - Location: `datalake/city_wellness_curated.parquet`
  - Volume: ~43 cities
  - Characteristics: Semi-structured, tabular

- **Customer Data** (Azure SQL Database)
  - Tables: Customer, CustomerHealthFactor
  - Volume: Hundreds of customers
  - Characteristics: Structured, relational

- **Contract Data** (Azure SQL Database)
  - Tables: Contract, ContractBenefit, ContractPremium
  - Volume: Growing with each quote
  - Characteristics: Structured, transactional

#### 5.1.2 Data Characteristics (4Vs)
- **Volume**: 
  - Images: ~GB scale
  - Parquet: ~MB scale
  - Database: ~GB scale
- **Velocity**: 
  - Real-time quote generation
  - Batch model retraining
- **Variety**: 
  - Images (unstructured)
  - Parquet (semi-structured)
  - SQL (structured)
- **Veracity**: 
  - Validated through preprocessing
  - Error handling at ingestion
  - Data quality checks

### 5.2 Information Layer (Processed Data)

#### 5.2.1 Information Extraction

##### From CT Images
- **Transformation**: Image → Tensor (224×224×3)
- **Process**: 
  1. Download from Blob Storage
  2. Convert to PIL Image
  3. Resize to 224×224
  4. Convert to Tensor
  5. Normalize (0-1 range)
- **Output**: Preprocessed image tensor ready for ML model

##### From City Wellness Parquet
- **Transformation**: Parquet → Feature Array (9 features)
- **Process**:
  1. Download parquet from Blob Storage
  2. Parse with pandas
  3. Clean data (remove %, currency symbols)
  4. Extract 9 numeric features:
     - Sunshine hours
     - Obesity levels
     - Life expectancy
     - Pollution index
     - Hours worked
     - Happiness levels
     - Outdoor activities
     - Water cost
     - Gym membership cost
  5. Handle missing values (fillna with mean)
  6. Convert to NumPy array
- **Output**: 9-dimensional feature vector

##### From Database
- **Transformation**: SQL Query → Structured Records
- **Process**:
  1. Execute SQL query
  2. Fetch results as dictionary
  3. Validate data types
  4. Handle null values
- **Output**: Python dictionaries/DataFrames

#### 5.2.2 Information Artifacts
- **Preprocessed CT Images**: Normalized tensors
- **City Feature Vectors**: 9-dimensional arrays
- **Customer Health Factors**: CT_RiskScore, WellnessScore
- **Contract Records**: Structured contract data

### 5.3 Knowledge Layer (ML Models & Business Rules)

#### 5.3.1 ML Models (Knowledge Artifacts)

##### CT Risk Model
- **Type**: Supervised Learning (Image Classification)
- **Architecture**: ResNet18 (pretrained) + Custom Head
- **Training Data**: Labeled CT scan images
- **Knowledge Encoded**: Patterns in CT scans indicating health risk
- **Output**: Risk probability (0-1)
- **Interpretation**: Higher score = higher health risk

##### Wellness Model
- **Type**: Supervised Learning (Regression)
- **Architecture**: Neural Network (9→32→16→1)
- **Training Data**: City features → Wellness scores
- **Knowledge Encoded**: Relationship between city characteristics and wellness
- **Output**: Wellness score (0-1)
- **Interpretation**: Higher score = better wellness

#### 5.3.2 Business Rules (Knowledge)

##### Premium Calculation Rule
- **Formula**: `Premium = Base × (1 + 0.4 × CT_Risk + 0.6 × (1 - Wellness))`
- **Rationale**: 
  - CT risk contributes 40% to premium adjustment
  - Wellness contributes 60% to premium adjustment
  - Higher risk = higher premium
  - Lower wellness = higher premium

##### Contract Creation Rules
- Both scores must be computed before contract creation
- Contract requires valid Customer ID
- Foreign key constraints ensure data integrity

#### 5.3.3 Preprocessing Knowledge
- **Image Preprocessing**: Standard resize and normalization
- **Feature Scaling**: StandardScaler (mean=0, std=1)
- **Data Cleaning**: Remove special characters, handle missing values

### 5.4 Wisdom Layer (Decision Making)

#### 5.4.1 Decision Making Process

##### Quote Generation Decision
1. **Input**: Customer ID, Base Premium
2. **Analysis**: Compute risk scores using ML models
3. **Calculation**: Apply premium formula
4. **Decision**: Generate quote with calculated premium
5. **Action**: Create contract and record premium

##### Quote Comparison Decision
1. **Input**: Multiple scenarios
2. **Analysis**: Compute quotes for all scenarios
3. **Comparison**: Identify best/worst options
4. **Decision**: Recommend optimal scenario
5. **Action**: Present comparison with visualizations

#### 5.4.2 Strategic Insights

##### Risk Pattern Recognition
- Identify correlations between CT scans and health risk
- Understand city wellness impact on premiums
- Recognize patterns in customer risk profiles

##### Model Performance Monitoring
- Track model accuracy over time
- Detect data drift
- Trigger retraining when needed

##### Business Intelligence
- Premium distribution analysis
- Risk score trends
- City wellness rankings
- Customer segmentation insights

### 5.5 DIKW Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        DIKW PYRAMID                          │
└─────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │   WISDOM     │  Strategic Decisions
                    │              │  - Premium Pricing
                    │              │  - Scenario Comparison
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  KNOWLEDGE  │  ML Models & Rules
                    │              │  - CT Risk Model
                    │              │  - Wellness Model
                    │              │  - Premium Formula
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ INFORMATION  │  Processed Data
                    │              │  - Preprocessed Images
                    │              │  - Feature Vectors
                    │              │  - Health Scores
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    DATA      │  Raw Data
                    │              │  - CT Images
                    │              │  - City CSV/Parquet
                    │              │  - Database Records
                    └──────────────┘

Data Flow:
Raw Data → Information (Preprocessing)
  → Knowledge (ML Models)
    → Wisdom (Business Decisions)
```

---

## 4. Infrastructure Domain Architecture

### 4.1 Infrastructure Components

#### 4.1.1 Compute Infrastructure
- **Platform**: Jupyter Notebook / JupyterLab
- **Language Runtime**: Python 3.13
- **ML Inference**: CPU-based (CUDA optional for GPU acceleration)
- **Resource Requirements**: ~2-4GB memory, multi-core CPU recommended

#### 4.1.2 Storage Infrastructure

##### Azure SQL Database
- **Service**: Azure SQL Server (PaaS)
- **Connection**: `pymssql` library
- **Tables**: Customer, CustomerHealthFactor, Contract, ContractBenefit, ContractPremium
- **Materialized Views**: Analytics aggregations
- **Indexes**: Primary keys, foreign keys (from Reports 1-3)
- **Backup**: Azure automated backups

##### Azure Blob Storage
- **Service**: Azure Blob Storage (PaaS)
- **Container**: `datalake`
- **Structure**:
  - `medical_images/`: CT scan images (PNG/JPG)
  - `city_wellness_curated.parquet`: City wellness data
- **Access**: Storage Account Key authentication
- **Encryption**: Azure Storage encryption at rest

##### Local Model Storage
- **Location**: `models/` directory
- **Files**: `ct_risk_model.pth`, `wellness_model.pth`, `wellness_scaler.pkl`, `model_versions.json`

### 4.2 Infrastructure Security

#### 4.2.1 Authentication & Authorization

##### SQL Database
- **Authentication**: SQL Authentication (Username/Password)
- **Authorization**: Role-based access control (RBAC)
- **Connection String**: Encrypted in-transit (TLS)
- **Credentials**: Stored in notebook (should use environment variables in production)

##### Blob Storage
- **Authentication**: Storage Account Key
- **Authorization**: Container-level access
- **Connection**: HTTPS (encrypted in-transit)
- **Credentials**: Stored in notebook (should use Managed Identity in production)

#### 4.2.2 Data Encryption

##### In-Transit Encryption
- **SQL Database**: TLS 1.2+ for all connections
- **Blob Storage**: HTTPS for all API calls
- **Local**: N/A (local file system)

##### At-Rest Encryption
- **SQL Database**: Azure Transparent Data Encryption (TDE)
- **Blob Storage**: Azure Storage Service Encryption (SSE)
- **Local Models**: Not encrypted (should encrypt in production)

### 4.3 Infrastructure Monitoring
- **Database Performance**: Query execution time, connection pool usage
- **Blob Storage**: Download latency, error rates
- **ML Inference**: Model inference time, memory usage
- **Application**: Error rates, success/failure counts
- **Logging**: Application logs, database logs, model version tracking

---

## 5. Database Design & Query Optimization

### 5.1 Database Schema Design

#### 10.1.1 Schema Overview

##### Core Tables in context
- **Customer**: Customer master data
  - Primary Key: `CustomerID`
  - Indexes: `CustomerID` (clustered), `City` (non-clustered)

- **CustomerHealthFactor**: Health scores and CT image references
  - Primary Key: `FactorID`
  - Foreign Key: `CustomerID` → `Customer.CustomerID`
  - Indexes: `CustomerID` (non-clustered), `FactorName` (non-clustered)

- **Contract**: Insurance contracts
  - Primary Key: `ContractID`
  - Foreign Key: `OwnerCustomerID` → `Customer.CustomerID`
  - Indexes: `ContractID` (clustered), `OwnerCustomerID` (non-clustered)

- **ContractBenefit**: Contract benefits
  - Primary Key: `ContractBenefitID`
  - Foreign Key: `ContractID` → `Contract.ContractID`
  - Indexes: `ContractBenefitID` (clustered), `ContractID` (non-clustered)

- **ContractPremium**: Premium records
  - Primary Key: `PremiumID` (or similar)
  - Foreign Key: `ContractBenefitID` → `ContractBenefit.ContractBenefitID`
  - Indexes: `ContractBenefitID` (non-clustered)

#### 10.1.2 Indexing Strategy

##### Primary Indexes
- All tables have clustered indexes on primary keys
- Ensures fast primary key lookups

##### Secondary Indexes
- **CustomerHealthFactor.CustomerID**: Fast lookup of health factors by customer
- **CustomerHealthFactor.FactorName**: Fast lookup by factor name
- **Contract.OwnerCustomerID**: Fast lookup of contracts by customer
- **ContractBenefit.ContractID**: Fast lookup of benefits by contract

##### Composite Indexes
- Consider composite index on `(CustomerID, FactorName)` for `CustomerHealthFactor`
- Consider composite index on `(OwnerCustomerID, Status)` for `Contract`

### 5.2 Query Optimization

#### 10.2.1 Query Patterns

##### Pattern 1: Customer Lookup
```sql
SELECT * FROM Customer WHERE CustomerID = %s
```
- **Optimization**: Clustered index on `CustomerID`
- **Performance**: O(log n) lookup

##### Pattern 2: Health Factors Lookup
```sql
SELECT FactorName, FactorValue
FROM CustomerHealthFactor
WHERE CustomerID = %s
```
- **Optimization**: Non-clustered index on `CustomerID`
- **Performance**: Index seek + key lookup

##### Pattern 3: CT Image Blob Lookup
```sql
SELECT TOP 1 ImageFileName 
FROM CustomerHealthFactor 
WHERE CustomerID = %s AND ImageFileName IS NOT NULL
ORDER BY FactorID DESC
```
- **Optimization**: 
  - Non-clustered index on `CustomerID`
  - Filter on `ImageFileName IS NOT NULL`
  - `TOP 1` limits result set
- **Performance**: Index seek + filter + sort (limited to 1 row)

#### 10.2.2 Query Optimization Techniques

##### Connection Management
- **Connection Pooling**: Reuse connections (pymssql supports connection pooling)
- **Connection Timeout**: Set appropriate timeout values
- **Connection Cleanup**: Always close connections in finally blocks

##### Query Optimization
- **Parameterized Queries**: Use parameterized queries to prevent SQL injection and enable plan caching
- **Selective Queries**: Only select required columns (avoid `SELECT *` in production)
- **Limit Results**: Use `TOP N` or `LIMIT` to limit result sets
- **Avoid N+1 Queries**: Batch queries when possible

##### Index Optimization
- **Covering Indexes**: Create covering indexes for frequent queries
- **Index Maintenance**: Regular index maintenance (rebuild, reorganize)
- **Index Statistics**: Keep statistics up-to-date

#### 10.2.3 Materialized Views

##### Purpose
- Pre-compute aggregations for analytics
- Improve query performance for reporting

##### Refresh Strategy
- **Manual Refresh**: `EXEC Refresh_Materialized_Views`
- **Scheduled Refresh**: Can be scheduled via Azure SQL Agent
- **Incremental Refresh**: Refresh only changed data (if supported)

### 5.3 Database Performance

#### 10.3.1 Performance Characteristics
- **Query Latency**: ~10-50ms per query (depends on query complexity)
- **Connection Time**: ~100-200ms (initial connection)
- **Transaction Time**: ~50-100ms per transaction

#### 10.3.2 Performance Optimization
- **Index Usage**: Ensure indexes are used (check execution plans)
- **Query Tuning**: Tune slow queries
- **Connection Pooling**: Reuse connections
- **Batch Operations**: Batch inserts/updates when possible

---

## 6. System Integration & Data Flow

### 6.1 Integration Points

#### 11.1.1 External Systems
1. **Azure SQL Database**: Customer and contract data
2. **Azure Blob Storage**: CT images and parquet files
3. **ML Training System**: `mlanalysis.ipynb` (separate, produces models)

#### 11.1.2 Internal Components
1. **Data Layer**: Database and blob storage connectors
2. **ML Layer**: Model loading and inference
3. **Business Logic**: Premium calculation, contract creation
4. **Persistence Layer**: Database operations

### 6.2 Data Flow Architecture

#### 11.2.1 Quote Generation Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│              QUOTE GENERATION DATA FLOW                     │
└─────────────────────────────────────────────────────────────┘

Input: Customer ID, Base Premium
  │
  ▼
┌─────────────────┐
│ SQL Database    │ → Customer Info, City, CT Image Blob Ref
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Azure Blob      │ → Download CT Image
│ Storage         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Azure Blob      │ → Download City Wellness Parquet
│ Storage         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │ → Image Tensor, Feature Vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ML Models       │ → CT Risk Score, Wellness Score
│ (Local)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Business Logic  │ → Premium Calculation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SQL Database    │ → Insert Contract, Benefit, Premium
└────────┬────────┘
         │
         ▼
Output: Quote Summary
```

#### 11.2.2 Quote Comparison Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│            QUOTE COMPARISON DATA FLOW                       │
└─────────────────────────────────────────────────────────────┘

Input: Customer ID, Scenarios List
  │
  ▼
For each scenario:
  │
  ▼
┌─────────────────┐
│ Quote Generation│ → Execute quote generation workflow
│ (Reuse Logic)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aggregate       │ → Collect all scenario results
│ Results         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Comparison      │ → Calculate differences, rankings
│ Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │ → Generate charts and tables
└────────┬────────┘
         │
         ▼
Output: Comparison Report + Visualizations
```

### 6.3 Data Transformation Pipeline

#### 11.3.1 Image Processing Pipeline
```
Raw Image (PNG/JPG)
  ↓
Download from Blob Storage
  ↓
PIL Image Object
  ↓
Resize to 224×224
  ↓
Convert to Tensor
  ↓
Normalize (0-1 range)
  ↓
Preprocessed Tensor (1×3×224×224)
  ↓
ML Model Input
```

#### 11.3.2 Feature Processing Pipeline
```
City Wellness Parquet
  ↓
Download from Blob Storage
  ↓
Parse with pandas
  ↓
Extract City Row
  ↓
Extract 9 Features
  ↓
Clean Data (remove %, currency)
  ↓
Convert to Numeric
  ↓
Handle Missing Values (fillna)
  ↓
Feature Vector (9 dimensions)
  ↓
StandardScaler Transform
  ↓
Normalized Feature Vector
  ↓
ML Model Input
```

---



---

## 8. Future Roadmap

- Automated model retraining pipeline
- Performance monitoring dashboard
- Advanced visualization capabilities
- Real-time streaming data ingestion
- Model A/B testing framework
- Advanced analytics and reporting



---

