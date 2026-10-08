[README.md](https://github.com/user-attachments/files/33208207/README.md)
# Accessify ♿

## AI-Powered Web Accessibility Analyzer

Accessify is a Python-based web accessibility analysis application designed to identify potential accessibility barriers in websites. It combines HTML analysis, accessibility rules, a violation dataset, and machine learning to provide an explainable accessibility assessment.

## 🎯 Problem Statement

Many websites contain accessibility problems such as:

- Images without alternative text
- Buttons without accessible names
- Links without meaningful names
- Incorrect heading structure
- Missing page language information
- Missing or incorrect landmarks
- Form fields without proper labels
- Duplicate HTML IDs
- Empty headings or table headers

These issues can make websites harder to use with screen readers and other assistive technologies.

## 💡 Proposed Solution

Accessify accepts a website URL and analyzes its webpage structure.

The application:

1. Accepts a website URL.
2. Fetches the webpage HTML.
3. Parses the HTML using BeautifulSoup.
4. Detects accessibility-related issues.
5. Matches detected issues with the project dataset.
6. Uses machine learning to estimate issue impact.
7. Calculates an overall accessibility risk score.
8. Displays explainable results through a Streamlit dashboard.

## 🧠 Key Features

### 🌐 Website URL Analysis
Users can enter a public website URL and analyze its HTML structure.

### ♿ Accessibility Detection
The application checks for issues including:

- Missing image alternative text
- Unnamed buttons
- Unnamed links
- Missing or incorrect H1 structure
- Heading-order problems
- Missing main landmark
- Duplicate landmarks
- Missing regions
- Empty headings
- Empty table headers
- Form-field labeling problems
- Duplicate IDs
- Missing HTML language attribute

### 🤖 Machine Learning
The application uses the supplied accessibility dataset to train a machine-learning model.

The model uses information such as:

- Violation name
- Violation category
- Affected HTML elements

These features are converted into TF-IDF representations and classified using a Random Forest model to predict violation impact.

### 📊 Accessibility Risk Score
The score combines factors including:

- Issue severity
- Number of detected violations
- Different violation types
- Dataset-based violation information
- Website structure and violation density

### 🔎 Explainable Results
Accessify displays:

- What the problem is
- Number of detected instances
- Severity
- Affected HTML elements
- Relevant WCAG reference
- ML-predicted impact and confidence

### 📈 Visual Dashboard
The Streamlit interface provides:

- Overall accessibility score
- Risk classification
- Violation breakdown
- Website structure statistics
- Interactive charts
- Detailed findings

## 🏗️ System Architecture

```text
                USER
                  │
                  ▼
          Enter Website URL
                  │
                  ▼
        Fetch Website HTML
                  │
                  ▼
        HTML Parsing / DOM
                  │
                  ▼
       Accessibility Scanner
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Rule-based Checks     Dataset Matching
        │                   │
        └─────────┬─────────┘
                  ▼
          Machine Learning
          TF-IDF + Random
             Forest
                  │
                  ▼
       Severity / Risk Analysis
                  │
                  ▼
        Accessibility Score
                  │
                  ▼
       Explainable Dashboard
```

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| BeautifulSoup | HTML parsing |
| Requests | Fetching webpages |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Random Forest | Impact classification |
| Plotly | Interactive visualizations |

## 📂 Project Structure

```text
Accessify/
│
├── accessify_app.py
├── Original_full_dataN.csv
├── requirements.txt
└── README.md
```

## 📋 Dataset

The application uses an accessibility violation dataset containing information about different web accessibility problems.

Important dataset attributes include:

- `domain_category`
- `violation_name`
- `violation_score`
- `violation_description`
- `violation_category`
- `violation_impact`
- `wcag_reference`
- `affected_html_elements`
- `violation_count`

The dataset acts as both a knowledge source and training data for the ML component.

## 🤖 Machine Learning Pipeline

### 1. Feature Construction

Relevant textual fields are combined:

```text
violation_name
+
violation_category
+
affected_html_elements
```

### 2. TF-IDF

TF-IDF converts the textual information into numerical features.

### 3. Random Forest

A Random Forest classifier learns relationships between the textual features and the recorded violation impact.

### 4. Prediction

For detected violations, the model predicts an impact category such as:

```text
Minor
Moderate
Serious
Critical
```

The prediction is displayed together with dataset information so that the result is explainable.

## ⚙️ Installation

### Step 1 — Install Python

Install Python 3.10 or later.

Verify:

```bash
python --version
```

### Step 2 — Install Dependencies

Open a terminal inside the project folder.

Run:

```bash
python -m pip install -r requirements.txt
```

If `python` is not recognized, try:

```bash
py -m pip install -r requirements.txt
```

## ▶️ Running the Application

Run:

```bash
python -m streamlit run accessify_app.py
```

Streamlit will start the application locally. Open the displayed local address in a browser.

## 📄 Example Workflow

```text
Enter:
https://example.com

        ↓

Accessify fetches the webpage

        ↓

HTML is parsed

        ↓

Accessibility checks are performed

        ↓

Detected issues are matched
with the dataset

        ↓

ML predicts issue impact

        ↓

Accessibility score is calculated

        ↓

Results are displayed
```

## 📊 Output

The dashboard provides an overall score together with detailed findings.

```text
Accessibility Risk Score
        ↓
LOW / MODERATE / HIGH / CRITICAL

Detected Violations
        ↓
Missing alt text
Unnamed buttons
Heading issues
Missing labels
etc.

ML Analysis
        ↓
Predicted impact
Confidence
```

## 🌍 Social Impact

Accessify is designed around **technology for an inclusive society**.

Better website accessibility can help people who rely on:

- Screen readers
- Keyboard navigation
- Alternative input methods
- Accessible page structures
- Clear semantic HTML

The project supports the broader goal of making digital information more accessible.

## 🌱 SDG Alignment

### SDG 4 — Quality Education

Accessible websites can make digital educational resources easier to access for students with disabilities.

### SDG 10 — Reduced Inequalities

Improving accessibility helps reduce barriers between users with different abilities.

### SDG 1 — No Poverty

Digital accessibility can support broader access to online services, information, education, and opportunities.

## 🔐 Limitations

Accessify is an analytical prototype and should not be treated as a complete WCAG certification system.

Possible limitations include:

- Some websites block automated requests.
- JavaScript-generated content may not be fully available through a simple HTML request.
- Some accessibility issues require human evaluation.
- The scanner checks a defined set of accessibility rules rather than every possible WCAG criterion.
- ML predictions depend on the quality and coverage of the supplied dataset.

The score should therefore be interpreted as an **accessibility risk indicator**, not an official compliance percentage.

## 🚀 Future Improvements

Possible future versions could include:

- Browser-based live accessibility inspection
- JavaScript-rendered page analysis using Selenium
- Automatic accessibility repair suggestions
- Before/after accessibility comparison
- More WCAG rules
- Screenshot-based issue highlighting
- Multi-page website crawling
- Accessibility report generation
- Larger training datasets
- Personalized explanations for different assistive-technology users

## 👥 Project Concept

**Project:** Accessify  
**Domain:** Accessibility / Inclusive Technology  
**Technology:** Python + Machine Learning + Web Analysis  
**Theme:** Technology for Social Good – Technology for an Inclusive Society

## ⚠️ Disclaimer

Accessify is an educational and research prototype demonstrating how Python, web analysis, datasets, and machine learning can be combined to identify potential accessibility problems.

It does not replace professional accessibility auditing or formal WCAG conformance testing.
