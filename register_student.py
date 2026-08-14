import cv2
import face_recognition
import pickle
import os

def register_student(student_name, num_samples=5):
    """
    Captures face images from webcam, generates embeddings,
    and saves them for later verification during the exam.
    """
    save_dir = os.path.join("dataset", "student_faces", student_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print(f"📸 Registering student: {student_name}")
    print("Press SPACE to capture a photo, ESC to cancel.")

    captured = 0
    while captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Captured: {captured}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture | ESC = cancel",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Student Registration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("❌ Registration cancelled")
            break
        elif key == 32:  # SPACE
            img_path = os.path.join(save_dir, f"{student_name}_{captured}.jpg")
            cv2.imwrite(img_path, frame)
            captured += 1
            print(f"✅ Captured image {captured}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()

    if captured == 0:
        print("⚠️ No images captured. Registration aborted.")
        return

    generate_embeddings(student_name, save_dir)


def generate_embeddings(student_name, image_dir):
    """
    Generates face embeddings from captured images and
    appends them to the shared embeddings database.
    """
    embeddings_path = os.path.join("models", "face_embeddings.pkl")

    # Load existing database or create new
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            database = pickle.load(f)
    else:
        database = {}

    encodings = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) != 1:
            print(f"⚠️ Skipping {filename} — expected 1 face, found {len(face_locations)}")
            continue

        encoding = face_recognition.face_encodings(image, face_locations)[0]
        encodings.append(encoding)

    if not encodings:
        print("❌ No valid face encodings generated. Try recapturing with better lighting.")
        return

    database[student_name] = encodings

    with open(embeddings_path, "wb") as f:
        pickle.dump(database, f)

    print(f"✅ Embeddings saved for '{student_name}' ({len(encodings)} samples) → {embeddings_path}")


if __name__ == "__main__":
    name = input("Enter student name: ").strip()
    if name:
        register_student(name)
    else:
        print("❌ Name cannot be empty")