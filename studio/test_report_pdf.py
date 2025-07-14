import matplotlib.pyplot as plt
import tempfile
import os
from fpdf import FPDF
from io import StringIO
import contextlib

def run_code_and_capture_plots(code: str, image_dir: str):
    # Clear any existing figures
    plt.close('all')
    
    # Execute the code in a safe global/local scope
    local_vars = {}
    with contextlib.redirect_stdout(StringIO()):  # Suppress print output
        exec(code, {}, local_vars)
    
    image_paths = []
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        img_path = os.path.join(image_dir, f"plot_{i}.png")
        fig.savefig(img_path, bbox_inches='tight')
        image_paths.append(img_path)
    
    return image_paths

def generate_pdf_with_code_and_plots(code: str, output_pdf: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    
    # Add code as formatted text
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    for line in code.split('\n'):
        pdf.multi_cell(0, 5, line)
    
    # Run code and capture plots
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = run_code_and_capture_plots(code, tmpdir)
        
        for img_path in image_paths:
            pdf.add_page()
            pdf.image(img_path, x=10, y=20, w=180)
    
    pdf.output(output_pdf)
