from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import os
from io import BytesIO
import numpy as np

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']

            # 修正点: BytesIOを使用して画像データを処理
            img_data = BytesIO(img_file.read())

            # 画像の前処理
            img = load_img(img_data, target_size=(224, 224))  # VGG16の入力サイズ
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # バッチ次元を追加
            img_array = preprocess_input(img_array)  # 正規化処理

            # モデルのロードと予測
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            predictions = model.predict(img_array)

            # 上位5つのカテゴリと確率を取得
            decoded_predictions = decode_predictions(predictions, top=5)
            top_results = [(result[1], f"{result[2]*100:.2f}%") for result in decoded_predictions[0]]

            # 結果をテンプレートに渡す
            return render(request, 'home.html', {'form': form, 'top_results': top_results})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
