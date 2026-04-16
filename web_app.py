import dash
from dash import html
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Expose the flask server for gunicorn
server = app.server

# Simple layout with no external assets
app.layout = html.Div([
    html.H1("MVP Launch Successful"),
    html.P("If you see this, the server, port, and environment are working correctly.")
])

if __name__ == '__main__':
    # Required port logic for Render
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
