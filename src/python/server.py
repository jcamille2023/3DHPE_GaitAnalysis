from flask import Flask, request, jsonify
from gait_measures import gait_measures
app = Flask(__name__)


@app.route('/gait_data', methods=['POST'])
def handle_gait_data():
    if request.method == 'POST':
        # Add new gait data from JSON payload
        new_data = request.get_json()
        if not new_data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        # Assign an ID to the new data
        return jsonify(gait_measures(new_data.pose_landmarks,"undistorted.xlsx")), 201

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
