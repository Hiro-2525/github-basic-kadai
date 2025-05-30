from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import save_model

# VGG16モデルを取得
model = VGG16(weights='imagenet')

# モデルを保存
save_model(model, './prediction/models/vgg16.h5')  # modelsフォルダに保存
