import os
import opendataloader_pdf

# Set JAVA_HOME to ensure Java 11 is used
os.environ['JAVA_HOME'] = r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot"

opendataloader_pdf.convert(
input_path="DATYS.pdf",
output_dir="output/parsed_pdf",
format="markdown,json"
)