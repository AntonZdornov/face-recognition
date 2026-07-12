"""
Минимальный проект для детекции лиц через OpenCV Haar Cascade.
Работает с веб-камерой, видеофайлом или одной картинкой.
Специально сделан лёгким, чтобы без проблем запускался на Raspberry Pi.
"""

import argparse
import sys

import cv2


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, len(faces)


def run_on_image(path, face_cascade):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Не удалось загрузить изображение: {path}")
        sys.exit(1)

    frame, count = detect_faces(frame, face_cascade)
    print(f"Найдено лиц: {count}")
    cv2.imshow("Face Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_stream(source, face_cascade):
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Не удалось открыть источник видео/камеру")
        sys.exit(1)

    print("Нажмите 'q' в окне видео, чтобы выйти")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, count = detect_faces(frame, face_cascade)
        cv2.putText(
            frame,
            f"Faces: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Минимальный демо-проект по детекции лиц")
    parser.add_argument(
        "--source",
        default="0",
        help="Индекс камеры (0, 1, ...), путь к видеофайлу или к картинке (по умолчанию: 0)",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Если указан, --source трактуется как путь к одной картинке",
    )
    args = parser.parse_args()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Не удалось загрузить каскад Хаара. Проверь установку OpenCV.")
        sys.exit(1)

    if args.image:
        run_on_image(args.source, face_cascade)
    else:
        run_on_stream(args.source, face_cascade)


if __name__ == "__main__":
    main()
