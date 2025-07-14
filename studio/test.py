# test.py
from report_pdf import generate_pdf_report 
from test_report_pdf import generate_pdf_with_code_and_plots


# Read the entire contents of debug_output.txt
with open("debug_output.txt", "r", encoding="utf-8") as f:
    debug_text = f.read()

with open('extracted_code.py', 'r', encoding= 'utf-8') as f:
    debug_output = f.read()


# Call your report function with the text
generate_pdf_report(debug_text, 'test_output.pdf')
generate_pdf_with_code_and_plots(debug_output, 'test_output_code.pdf')
