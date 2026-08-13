import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids GUI issues
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

from utils import config


class ReportGenerator:
    """
    Generates a final PDF violation report (with statistics and pie chart)
    and a summary CSV, based on data collected by ViolationManager.
    """

    def __init__(self, student_name):
        self.student_name = student_name

    def _generate_pie_chart(self, violation_counts, output_path):
        """
        Creates a pie chart of violation type distribution.
        Skips chart generation if there are no violations at all.
        """
        filtered = {k: v for k, v in violation_counts.items() if v > 0}

        if not filtered:
            return None

        labels = list(filtered.keys())
        sizes = list(filtered.values())

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Violation Distribution")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def generate_pdf(self, summary):
        """
        Generates the final PDF report.

        Args:
            summary: dict returned by ViolationManager.get_summary()

        Returns:
            path to the generated PDF file
        """
        session_start = summary["session_start_time"]
        session_end = summary["session_end_time"]
        duration = session_end - session_start
        duration_str = str(duration).split(".")[0]  # strip microseconds

        total_score = summary["total_score"]
        risk_level = summary["risk_level"]
        violation_counts = summary["violation_counts"]
        total_violations = sum(violation_counts.values())

        timestamp_str = session_start.strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(config.REPORTS_DIR, f"chart_{timestamp_str}.png")
        pdf_path = os.path.join(config.REPORTS_DIR, f"report_{self.student_name}_{timestamp_str}.pdf")

        chart_file = self._generate_pie_chart(violation_counts, chart_path)

        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, "AI Smart Exam Proctor - Violation Report", ln=True, align="C")
        pdf.ln(5)

        # Student & session info
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Student Name: {self.student_name}", ln=True)
        pdf.cell(0, 8, f"Exam Date: {session_start.strftime('%Y-%m-%d')}", ln=True)
        pdf.cell(0, 8, f"Start Time: {session_start.strftime('%H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"End Time: {session_end.strftime('%H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"Duration: {duration_str}", ln=True)
        pdf.ln(5)

        # Violation summary table
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Violation Summary", ln=True)
        pdf.set_font("Arial", "", 11)

        pdf.set_fill_color(230, 230, 230)
        pdf.cell(120, 8, "Violation Type", border=1, fill=True)
        pdf.cell(60, 8, "Count", border=1, fill=True, ln=True)

        for v_type, count in violation_counts.items():
            pdf.cell(120, 8, v_type, border=1)
            pdf.cell(60, 8, str(count), border=1, ln=True)

        pdf.ln(5)

        # Score & risk
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Final Assessment", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Total Violations: {total_violations}", ln=True)
        pdf.cell(0, 8, f"Final Risk Score: {total_score}", ln=True)

        # Color-coded risk level
        risk_colors = {
            "Safe": (0, 150, 0),
            "Warning": (200, 150, 0),
            "Suspicious": (230, 100, 0),
            "High Risk": (200, 0, 0)
        }
        r, g, b = risk_colors.get(risk_level, (0, 0, 0))
        pdf.set_text_color(r, g, b)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Risk Level: {risk_level}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Pie chart
        if chart_file:
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, "Violation Distribution", ln=True)
            pdf.image(chart_file, x=45, w=120)
        else:
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, "No violations recorded — clean exam session.", ln=True)

        pdf.output(pdf_path)
        print(f"✅ PDF report generated: {pdf_path}")

        return pdf_path

    def generate_summary_csv(self, summary):
        """
        Generates a simple summary CSV (separate from the detailed
        violations.csv log) — one row per session.
        """
        session_start = summary["session_start_time"]
        summary_path = os.path.join(config.REPORTS_DIR, "session_summaries.csv")

        row = {
            "Student Name": self.student_name,
            "Exam Date": session_start.strftime("%Y-%m-%d"),
            "Start Time": session_start.strftime("%H:%M:%S"),
            "End Time": summary["session_end_time"].strftime("%H:%M:%S"),
            "Total Score": summary["total_score"],
            "Risk Level": summary["risk_level"],
            "Total Violations": sum(summary["violation_counts"].values())
        }

        file_exists = os.path.exists(summary_path)
        df = pd.DataFrame([row])
        df.to_csv(summary_path, mode="a", header=not file_exists, index=False)

        print(f"✅ Summary CSV updated: {summary_path}")
        return summary_path