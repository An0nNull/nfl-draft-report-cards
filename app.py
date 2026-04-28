from flask import Flask, abort, send_from_directory

app = Flask(__name__)
PUBLIC_DATA_FILES = {"team_scores.json", "predictions.csv"}


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.route("/")
def dashboard():
    return send_from_directory(".", "dashboard.html")


@app.route("/data/<path:filename>")
def data_file(filename):
    if filename not in PUBLIC_DATA_FILES:
        abort(404)
    return send_from_directory("data", filename)


@app.route("/api/teams")
def api_teams():
    return send_from_directory("data", "team_scores.json")


@app.route("/api/predictions")
def api_predictions():
    return send_from_directory("data", "predictions.csv")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1")
