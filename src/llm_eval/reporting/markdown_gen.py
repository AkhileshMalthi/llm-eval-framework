import pandas as pd
from tabulate import tabulate

def generate_markdown_report(df: pd.DataFrame, eval_name: str) -> str:
    summary = df.describe().loc[['mean', 'min', 'max']].to_markdown()
    detailed_table = df.to_markdown(index=False)
    
    report = f"""# Evaluation Report: {eval_name}

## Summary Statistics
{summary}

## Detailed Results
{detailed_table}
"""
    return report