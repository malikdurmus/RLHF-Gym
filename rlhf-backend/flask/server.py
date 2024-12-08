from flask import Flask, send_from_directory
from flask_cors import CORS

VIDEO_DIRECTORY = ""
VIDEO_FILE = "video1.mp4"
app = Flask(__name__)
CORS(app)
@app.route("/members")



@app.route('/video', methods=['GET'])
def get_video():
    return send_from_directory(VIDEO_DIRECTORY, VIDEO_FILE)

if __name__ == "__main__":
    a = {"members": ["Member1", "Member2", "Member3"]}
    print(type(a))

    app.run(debug=True)
