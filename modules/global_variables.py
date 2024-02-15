import os

images_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models') 
pred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictions')
val_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'validation')  

if not os.path.exists(models_path):
    os.mkdir(models_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(pred_path):
    os.mkdir(pred_path)
if not os.path.exists(val_path):
    os.mkdir(val_path)

