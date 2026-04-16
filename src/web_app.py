import os
import dash
from dash import html

# --- DIAGNOSTIC SECTION ---
# This will print to your Render Logs
current_file_path = os.path.abspath(__file__)
current_working_dir = os.getcwd()

print("\n" + "="*40)
print("RENDER SYSTEM DIAGNOSTICS")
print(f"1. Absolute path of this file: {current_file_path}")
print(f"2. Current Working Directory: {current_working_dir}")

# List everything in the directory ABOVE 'src'
try:
    parent_dir = os.path.dirname(current_working_dir)
    print(f"3. Contents of Parent Dir ({parent_dir}): {os.listdir(parent_dir)}")
    
    # List everything in the actual Root
    # We'll try to find the 'project' folder by going up until we can't
    print(f"4. Contents of Project Root: {os.listdir(os.path.dirname(parent_dir))}")
except Exception as e:
    print(f"Error scanning directories: {e}")
print("="*40 + "\n")

# --- MINIMAL APP TO KEEP RENDER HAPPY ---
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Diagnostic Mode Active"),
    html.P("Check your Render Logs to see the full directory tree.")
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
