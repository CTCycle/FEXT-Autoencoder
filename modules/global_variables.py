import os

images_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models') 

if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)

