from collections import Counter
from flask import Flask, json, jsonify, request, send_file
from keras.models import load_model # type: ignore
from PIL import Image, ImageOps # type: ignore
import numpy as np # type: ignore
import os

model = load_model("assets/keras_model.h5", compile=False)
class_names = open("assets/labels.txt", "r").readlines()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST", "PUT"])
def index():
    if request.method == "GET":
        size = (224, 224)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        label = request.args.get('label')
        
        if not label:
            return jsonify({"error": "Road label not provided"})
        
        road_path = os.path.join("images", label)
        
        if not os.path.exists(road_path):
            return jsonify({"error": "Road not found"})
        
        best_image_path = None
        best_avg_score = -1
        
        for camera_id in os.listdir(road_path):
            camera_path = os.path.join(road_path, camera_id)
            if os.path.isdir(camera_path):
                predictions = []
                confidence_scores = []
                for filename in os.listdir(camera_path):
                    image_path = os.path.join(camera_path, filename)
                    image = Image.open(image_path).convert("RGB")
                    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                    image_array = np.asarray(image)
                    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                    data[0] = normalized_image_array
                    
                    prediction = model.predict(data)
                    index = np.argmax(prediction)
                    
                    class_name = class_names[index]
                    confidence_score = prediction[0][index]
                    
                    predictions.append(class_name)
                    confidence_scores.append(confidence_score)
                
                avg_confidence_score = np.mean(confidence_scores)
                
                if avg_confidence_score > best_avg_score:
                    best_avg_score = avg_confidence_score
                    best_image_path = image_path
                    best_predictions = predictions
                
        if best_image_path is None:
            return jsonify({"error": "No images found in the road folder"})
        
        most_common_class = Counter(best_predictions).most_common(1)[0][0]
        
        return jsonify({
            "road_name": label,
            "image_path": best_image_path,
            "class_name": most_common_class,
            "average_confidence_score": float(best_avg_score)
        })
    
    elif request.method == "PUT":
        return " ............ ---------nothing here to do--------- ............"
    
    elif request.method == "POST":
        # Parse multipart/form-data
        if 'metadata' in request.form:
            metadata = json.loads(request.form['metadata'])
            road_name = metadata.get('road_name')
            camera_id = metadata.get('camera_id')
        else:
            return jsonify({"error": "Missing metadata"}), 400

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_data = request.files['image']

        if not road_name or not camera_id:
            return jsonify({"error": "Missing road_name or camera_id"}), 400

    try:
        # Create directories if they don't exist
        road_path = os.path.join("images", road_name)
        camera_path = os.path.join(road_path, camera_id)

        if not os.path.exists(camera_path):
            os.makedirs(camera_path)

        # Ensure only 10 images are stored per camera
        existing_images = sorted(os.listdir(camera_path), key=lambda x: os.path.getctime(os.path.join(camera_path, x)))

        if len(existing_images) >= 10:
            # Remove the oldest image (first in the sorted list)
            os.remove(os.path.join(camera_path, existing_images[0]))
            existing_images.pop(0)

        # Determine the next image number
        if existing_images:
            last_image = existing_images[-1]
            last_image_number = int(last_image.split('_')[-1].split('.')[0])
            new_image_number = last_image_number + 1
        else:
            new_image_number = 1

        # Save image to file
        image_filename = f"{road_name}_{camera_id}_{new_image_number}.jpg"  # Naming new image uniquely
        image_path = os.path.join(camera_path, image_filename)

        image_data.save(image_path)

        return jsonify({"message": "Image saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/images/<road_name>/<camera_id>/<filename>')
def get_image(road_name, camera_id, filename):
    return send_file(os.path.join("images", road_name, camera_id, filename))

@app.route("/roads/number-of-cam", methods=["GET"])
def get_number_of_cameras():
    road_name = request.args.get('road_name')
    if not road_name:
        return jsonify({"error": "No road_name provided"}), 400

    image_dir = "images"
    road_path = os.path.join(image_dir, road_name)
    
    if not os.path.exists(road_path):
        return jsonify({"error": "Road not found"})
    
    if not os.path.isdir(road_path):
        return jsonify({"error": "Invalid road directory"})
    
    camera_count = len([name for name in os.listdir(road_path) if os.path.isdir(os.path.join(road_path, name))])
    
    return jsonify({road_name: camera_count})

@app.route("/images", methods=["GET"])
def list_images():
    image_dir = "images"
    
    if not os.path.exists(image_dir):
        return jsonify({"error": "No images directory found"})
    
    folder_structure = {}
    
    for road_name in os.listdir(image_dir):
        road_path = os.path.join(image_dir, road_name)
        if os.path.isdir(road_path):
            folder_structure[road_name] = {}
            for camera_id in os.listdir(road_path):
                camera_path = os.path.join(road_path, camera_id)
                if os.path.isdir(camera_path):
                    folder_structure[road_name][camera_id] = os.listdir(camera_path)
    
    return jsonify(folder_structure)


from flask import send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
