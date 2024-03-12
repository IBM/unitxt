import os
import difflib
import sys

"""
This file contains a script that compare the content of text files in 2 directories. The script parameters  (argv) are:
  1. First dir to compare (path)
  2. Second dir to compare (path)
  3. Path to the results dir (the script will create this dir, and will create the reports inside).

  After running the script, the result file will include the main report main_report.html. This file will contain a 
  table with all the unique files names which exist in the two compared directories. For each file, the table will show 
  if it exist on A (first directory), B (second directory) or both. In case that the same file name appears in both 
  directories, the file name will be a link to a report that compares the content of the two files (the diffs will be 
  highlighted by color). 
"""


def generate_diff_html(dir1, dir2, file1, file2, output_html):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
    d = difflib.HtmlDiff(wrapcolumn=120)
    html_diff = d.make_file(file1_lines, file2_lines, dir1, dir2)
    custom_css = """
        <style>
        table.diff {width: 100% !important; table-layout: fixed; word-wrap: break-word;}
        td {word-wrap: break-word; max-width: 0; overflow-wrap: break-word; white-space: pre-wrap !important;}
        </style>
        """
    html_diff = html_diff.replace('</head>', f'{custom_css}</head>')
    with open(output_html, 'w') as html_file:
        html_file.write(html_diff)
    print(f"HTML report generated: {output_html}")


def get_dirs_files_lists(dir_a, dir_b):
    files_a = set(os.listdir(dir_a))
    files_b = set(os.listdir(dir_b))
    only_in_a = files_a - files_b
    only_in_b = files_b - files_a
    common_files = files_a & files_b
    diff_files = []
    for file_name in common_files:
        with open(os.path.join(dir_a, file_name), 'r') as f_a, open(os.path.join(dir_b, file_name), 'r') as f_b:
            content_a = f_a.readlines()
            content_b = f_b.readlines()
            if content_a != content_b:
                diff_files.append(file_name)
    return only_in_a, only_in_b, common_files, diff_files


def get_diff_report_name(file_name):
    return f"diff_{file_name}.html"


def create_main_report(report_dir, only_in_a, only_in_b, common_files, diff_files):
    os.makedirs(report_dir, exist_ok=True)
    main_report_path = os.path.join(report_dir, 'main_report.html')
    with open(main_report_path, 'w') as report_file:
        report_file.write("""
            <html><head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
            </head><body>
            <h2>Directory Comparison Report</h2>
            """)
        report_file.write(
            f"<h3>Directories contained {len(common_files) - len(diff_files)} shared files with no diff </h2>")
        report_file.write("""
            <table>
                <tr><th>#</th><th>File</th><th>Directory</th></tr>
            """)
        for i, file_name in enumerate(sorted(only_in_a), 1):
            report_file.write(f'<tr><td>{i}</td><td>{file_name}</td><td>A</td></tr>')
        for i, file_name in enumerate(sorted(only_in_b), len(only_in_a) + 1):
            report_file.write(f'<tr><td>{i}</td><td>{file_name}</td><td>B</td></tr>')
        for i, file_name in enumerate(sorted(diff_files), len(only_in_a) + len(only_in_b) + 1):
            link = f'<a href="{get_diff_report_name(file_name)}">{file_name}</a>'
            report_file.write(f'<tr><td>{i}</td><td>{link}</td><td>Both (Different Content)</td></tr>')
        report_file.write('</table></body></html>')
    print(f"Main report generated: {main_report_path}")


def compare_dirs(dir_a, dir_b, report_dir):
    only_in_a, only_in_b, common_files, diff_files = get_dirs_files_lists(dir_a, dir_b)
    create_main_report(report_dir=report_dir, only_in_a=only_in_a, only_in_b=only_in_b,
                       common_files=common_files, diff_files=diff_files)
    for i, file in enumerate(sorted(diff_files)):
        print(f'{i}/{len(diff_files)}: creating report for {file}')
        generate_diff_html(dir_a, dir_b, os.path.join(dir_a, file), os.path.join(dir_b, file),
                           os.path.join(report_dir, get_diff_report_name(file)))


if __name__ == "__main__":
    dir_a = sys.argv[1]
    dir_b = sys.argv[2]
    report_dir = sys.argv[3]
    compare_dirs(dir_a, dir_b, report_dir)