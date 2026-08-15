from datetime import datetime, timedelta
from modules.report_generator import ReportGenerator

# Fake summary data to test report generation
fake_summary = {
    "total_score": 27,
    "risk_level": "Suspicious",
    "violation_counts": {
        "Unknown Face": 0,
        "No Face": 1,
        "Multiple Faces": 0,
        "Phone Detected": 2,
        "Looking Away": 5,
        "Head Turned": 3,
        "Student Left Camera": 0,
    },
    "session_start_time": datetime.now() - timedelta(hours=1, minutes=30),
    "session_end_time": datetime.now()
}

generator = ReportGenerator(student_name="Test Student")
pdf_path = generator.generate_pdf(fake_summary)
csv_path = generator.generate_summary_csv(fake_summary)

print(f"\n📄 Check these files:")
print(f"   PDF: {pdf_path}")
print(f"   CSV: {csv_path}")