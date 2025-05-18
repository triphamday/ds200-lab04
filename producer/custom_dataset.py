import os

def scan_images(root):
    # root = path tới dataset/data_split/train (hoặc val, test)
    class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for cls in class_dirs:
        img_dir = os.path.join(root, cls)
        for img in os.listdir(img_dir):
            if img.lower().endswith(('jpg', 'jpeg', 'png')):
                yield {
                    'path': os.path.abspath(os.path.join(img_dir, img)),
                    'label': cls
                }
