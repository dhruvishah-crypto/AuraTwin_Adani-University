from pathlib import Path

train_model_path = Path("src/train_model.py")
content = train_model_path.read_text(encoding="utf-8")

# Fix line 1506-1507
content = content.replace(
    "f\"`{row['alerts_per_patient_day']:.4f}` |\n\"",
    "f\"`{row['alerts_per_patient_day']:.4f}` |\\n\""
)

# Fix line 1537-1538
content = content.replace(
    'next_section = post.find("\n## ")',
    'next_section = post.find("\\n## ")'
)

# Fix line 1546-1548
content = content.replace(
    'path.write_text(text + "\n\n" + section_content, encoding="utf-8")',
    'path.write_text(text + "\\n\\n" + section_content, encoding="utf-8")'
)

# Fix line 1699-1700
content = content.replace(
    "f\"`{row['alerts_per_patient_day']:.4f}` |\n\"",
    "f\"`{row['alerts_per_patient_day']:.4f}` |\\n\""
)

train_model_path.write_text(content, encoding="utf-8")
print("Fixed raw newlines in f-strings.")
