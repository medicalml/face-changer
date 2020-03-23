import dash
import dash_core_components as dcc
import dash_html_components as html

from flask import Flask, Response
import cv2


class FaceDetector:
    def __init__(self):
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.last_faces = []

    def detect(self, img, cached=False):
        if not cached:
            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            self.last_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in self.last_faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        return img

    def __call__(self, img, cached=False):
        return self.detect(img, cached)


class VideoCamera:
    def __init__(self, post_processor=None):
        self.video = cv2.VideoCapture(0)
        self.post_processor = post_processor
        self.frame_counter = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        self.frame_counter += 1
        # print(self.frame_counter)
        success, image = self.video.read()
        if success and self.post_processor is not None:
            image = self.post_processor(image, self.frame_counter % 10 != 0)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


server = Flask(__name__)
app = dash.Dash(__name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(FaceDetector())),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([
    html.H1("medical.ml", style={"font-family": "Ubuntu"}),
    html.Img(src="/video_feed")
])


if __name__ == '__main__':
    app.run_server(debug=True)
