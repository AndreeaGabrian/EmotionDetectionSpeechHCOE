import pandas as pd


def json2excel(json_input_filename):
    # Load JSON data into a pandas DataFrame
    json_data = pd.read_json(f"{json_input_filename}.json")

    # Write DataFrame to Excel file
    excel_file = f"{json_input_filename}.xlsx"
    json_data.to_excel(excel_file, index=False)

    print("Conversion complete. Excel file saved as:", excel_file)
