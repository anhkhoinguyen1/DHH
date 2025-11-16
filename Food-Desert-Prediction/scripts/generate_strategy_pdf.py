"""
Generate a PDF document with the strategy and mathematical equations
formatted clearly using LaTeX rendering.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path

# Use matplotlib's built-in math rendering (more reliable)
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "docs"

def create_strategy_pdf():
    """Create a PDF with strategy and mathematical equations."""
    
    output_file = OUTPUT_DIR / "strategy_equations.pdf"
    
    with PdfPages(output_file) as pdf:
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = 'Food Desert Prediction Strategy'
        subtitle_text = 'Mathematical Formulations and Methodology'
        
        ax.text(0.5, 0.9, title_text, ha='center', va='top', 
                fontsize=24, transform=ax.transAxes, weight='bold')
        ax.text(0.5, 0.85, subtitle_text, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        ax.text(0.1, 0.75, 'Research Question:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        
        overview = (
            'Can we predict which U.S. census tracts are at risk of becoming ' +
            'food deserts in the next decade, based on demographic shifts, ' +
            'retailer patterns, transit access, and socioeconomic change?'
        )
        
        ax.text(0.1, 0.68, overview, ha='left', va='top', 
                fontsize=12, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Feature Normalization
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title = 'Feature Normalization'
        ax.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        y_pos = 0.85
        equations = [
            ('Min-Max Normalization:',
             r'$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$'),
            ('Z-Score Normalization:',
             r'$z = \frac{x - \mu}{\sigma}$'),
            ('Population Normalized Features:',
             r'$normalized\_lapop = \frac{lapop1}{Pop2010}$'),
            ('Low-Income Normalized:',
             r'$normalized\_lalowi = \frac{lalowi1}{TractLOWI}$'),
            ('Income Score:',
             r'$income\_score = \frac{MedianIncome - min_{income}}{max_{income} - min_{income}}$'),
            ('Poverty Score:',
             r'$poverty\_score = \frac{PovertyRate}{100}$'),
            ('Education Score:',
             r'$education\_score = \frac{BachelorDegreeCount}{TotalPopulation}$'),
            ('Rent Burden:',
             r'$rent\_burden = \frac{MedianGrossRent}{MedianFamilyIncome}$'),
        ]
        
        for label, eq in equations:
            ax.text(0.1, y_pos, label, ha='left', va='top', 
                    fontsize=12, transform=ax.transAxes, weight='bold')
            ax.text(0.5, y_pos - 0.03, eq, ha='left', va='top', 
                    fontsize=14, transform=ax.transAxes)
            y_pos -= 0.12
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Category Scores and Risk Calculation
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title = 'Category Scores and Risk Calculation'
        ax.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        y_pos = 0.85
        
        # Category Score Formula
        ax.text(0.1, y_pos, 'Category Score:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        ax.text(0.5, y_pos - 0.03, 
                r'$Category\_Score = \sum_{i} (w_i \times f_i)$',
                ha='left', va='top', fontsize=14, transform=ax.transAxes)
        y_pos -= 0.15
        
        # Feature Weights
        ax.text(0.1, y_pos, 'Feature Category Weights:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        y_pos -= 0.08
        
        weights = [
            r'$w_A = 0.25$ (Food Access Status)',
            r'$w_B = 0.30$ (Socioeconomic Indicators)',
            r'$w_C = 0.25$ (Retail Environment Changes)',
            r'$w_D = 0.15$ (Transportation Access)',
            r'$w_E = 0.05$ (Demographic and Health Indicators)'
        ]
        for w in weights:
            ax.text(0.15, y_pos, w, ha='left', va='top', 
                    fontsize=11, transform=ax.transAxes)
            y_pos -= 0.06
        
        y_pos -= 0.05
        
        # Overall Risk Score
        ax.text(0.1, y_pos, 'Overall Risk Score:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        risk_eq = (
            r'$Risk\_Score = w_A \times A + w_B \times B + w_C \times C + w_D \times D + w_E \times E$'
        )
        ax.text(0.5, y_pos - 0.03, risk_eq, ha='left', va='top', 
                fontsize=13, transform=ax.transAxes)
        y_pos -= 0.1
        
        expanded_eq = (
            r'$Risk\_Score = 0.25 \times A + 0.30 \times B + 0.25 \times C + 0.15 \times D + 0.05 \times E$'
        )
        ax.text(0.5, y_pos - 0.03, expanded_eq, ha='left', va='top', 
                fontsize=12, transform=ax.transAxes)
        y_pos -= 0.15
        
        # Probability Conversion
        ax.text(0.1, y_pos, 'Probability Conversion (Logistic Function):', 
                ha='left', va='top', fontsize=12, transform=ax.transAxes, weight='bold')
        prob_eq = r'$P = \frac{1}{1 + e^{-(Risk\_Score - \theta)}}$'
        ax.text(0.5, y_pos - 0.03, prob_eq, ha='left', va='top', 
                fontsize=14, transform=ax.transAxes)
        y_pos -= 0.1
        
        ax.text(0.1, y_pos, r'where $\theta$ is the decision threshold', 
                ha='left', va='top', fontsize=11, transform=ax.transAxes, style='italic')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: SVI Calculation
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title = 'Social Vulnerability Index (SVI) Calculation'
        ax.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        y_pos = 0.85
        
        # SVI Formula
        ax.text(0.1, y_pos, 'SVI Score Formula:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        svi_eq = (
            r'$SVI = 0.30 \times S_{socio} + 0.25 \times S_{household} + ' +
            r'0.25 \times S_{minority} + 0.20 \times S_{housing}$'
        )
        ax.text(0.5, y_pos - 0.03, svi_eq, ha='left', va='top', 
                fontsize=12, transform=ax.transAxes)
        y_pos -= 0.15
        
        # Component Calculations
        components = [
            ('Socioeconomic Status (30%):',
             r'$S_{socio} = 0.40 \times Poverty + 0.30 \times Income + 0.20 \times Education + 0.10 \times Rent$'),
            ('Household Composition (25%):',
             r'$S_{household} = 0.50 \times LowIncome + 0.50 \times (1 - VehicleOwnership)$'),
            ('Minority Status (25%):',
             r'$S_{minority} = 0.60 \times LowIncomeProxy + 0.40 \times Poverty$'),
            ('Housing & Transportation (20%):',
             r'$S_{housing} = 0.50 \times (1 - VehicleOwnership) + 0.30 \times Crowding + 0.20 \times Rent$'),
        ]
        
        for label, eq in components:
            ax.text(0.1, y_pos, label, ha='left', va='top', 
                    fontsize=11, transform=ax.transAxes, weight='bold')
            ax.text(0.1, y_pos - 0.04, eq, ha='left', va='top', 
                    fontsize=10, transform=ax.transAxes)
            y_pos -= 0.12
        
        # Interpretation
        y_pos -= 0.05
        ax.text(0.1, y_pos, 'Interpretation:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        y_pos -= 0.08
        
        interpretation = [
            r'$0.0 - 0.3$: Low vulnerability',
            r'$0.3 - 0.5$: Moderate vulnerability',
            r'$0.5 - 0.7$: High vulnerability',
            r'$0.7 - 1.0$: Very high vulnerability'
        ]
        for interp in interpretation:
            ax.text(0.1, y_pos, interp, ha='left', va='top', 
                    fontsize=11, transform=ax.transAxes)
            y_pos -= 0.06
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Additional Calculations
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title = 'Additional Feature Calculations'
        ax.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        y_pos = 0.85
        
        calculations = [
            ('Grocery Store Density Change:',
             r'$\Delta_{density} = \frac{stores_t - stores_{t-5}}{stores_{t-5}}$'),
            ('Density Risk Score:',
             r'$risk\_score = -\Delta_{density}$'),
            ('Vehicle Ownership Rate:',
             r'$vehicle\_ownership = 1 - \frac{HouseholdsNoVehicle}{TotalHouseholds}$'),
            ('Transit Access Score:',
             r'$transit\_access = TransitFrequency$ if accessible, else $0$'),
            ('Income-Based Vulnerability:',
             r'$income\_vuln = 1.0$ if income < \$30K, $0.7$ if \$30K-50K, $0.4$ if \$50K-75K, $0.1$ if $\geq$ \$75K'),
            ('Rent Burden Score:',
             r'$rent\_burden = 1.0$ if rent > 50% income, $0.7$ if 30-50%, $0.3$ if $\leq$ 30%'),
        ]
        
        for label, eq in calculations:
            ax.text(0.1, y_pos, label, ha='left', va='top', 
                    fontsize=11, transform=ax.transAxes, weight='bold')
            ax.text(0.1, y_pos - 0.05, eq, ha='left', va='top', 
                    fontsize=10, transform=ax.transAxes)
            y_pos -= 0.15
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Model Performance Metrics
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title = 'Model Performance Metrics'
        ax.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=18, transform=ax.transAxes, weight='bold')
        
        y_pos = 0.85
        
        metrics = [
            ('ROC-AUC:',
             r'$AUC = \int_0^1 TPR(FPR^{-1}(x)) dx$'),
            ('Precision:',
             r'$Precision = \frac{TP}{TP + FP}$'),
            ('Recall:',
             r'$Recall = \frac{TP}{TP + FN}$'),
            ('F1-Score:',
             r'$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$'),
            ('Accuracy:',
             r'$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$'),
        ]
        
        for label, eq in metrics:
            ax.text(0.1, y_pos, label, ha='left', va='top', 
                    fontsize=12, transform=ax.transAxes, weight='bold')
            ax.text(0.5, y_pos - 0.03, eq, ha='left', va='top', 
                    fontsize=14, transform=ax.transAxes)
            y_pos -= 0.15
        
        # Risk Classification
        y_pos -= 0.1
        ax.text(0.1, y_pos, 'Risk Level Classification:', ha='left', va='top', 
                fontsize=12, transform=ax.transAxes, weight='bold')
        y_pos -= 0.08
        
        risk_levels = [
            r'High Risk: $P \in [0.70, 1.00]$',
            r'Moderate Risk: $P \in [0.40, 0.69]$',
            r'Low Risk: $P \in [0.20, 0.39]$',
            r'Very Low Risk: $P \in [0.00, 0.19]$'
        ]
        for level in risk_levels:
            ax.text(0.1, y_pos, level, ha='left', va='top', 
                    fontsize=11, transform=ax.transAxes)
            y_pos -= 0.06
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ PDF generated: {output_file}")
    return output_file

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    create_strategy_pdf()

