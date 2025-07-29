import os
import shutil
import tempfile
import pytest
from PIL import Image
import numpy as np

import subset_data

@pytest.fixture
def temp_src_dir():
    # Create a temporary source directory with subfolders and dummy images
    temp_dir = tempfile.mkdtemp()
    class_names = ['classA', 'classB']
    for cls in class_names:
        cls_dir = os.path.join(temp_dir, cls)
        os.makedirs(cls_dir)
        # Create 5 dummy images per class
        for i in range(5):
            img = Image.fromarray(np.uint8(np.random.rand(32,32,3)*255))
            img.save(os.path.join(cls_dir, f"{cls}_{i}.jpg"))
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_copy_subset_per_class_basic(temp_src_dir):
    with tempfile.TemporaryDirectory() as dst_dir:
        subset_data.copy_subset_per_class(temp_src_dir, dst_dir, max_per_class=3, exts=('.jpg',))
        # Check that destination has correct structure
        for cls in ['classA', 'classB']:
            cls_dst = os.path.join(dst_dir, cls)
            assert os.path.isdir(cls_dst)
            images = os.listdir(cls_dst)
            # Should be at most 3 images per class
            assert len(images) == 3

def test_copy_subset_per_class_less_than_max(temp_src_dir):
    # Remove images so one class has less than max_per_class
    classA_dir = os.path.join(temp_src_dir, 'classA')
    for img_file in os.listdir(classA_dir)[3:]:
        os.remove(os.path.join(classA_dir, img_file))
    with tempfile.TemporaryDirectory() as dst_dir:
        subset_data.copy_subset_per_class(temp_src_dir, dst_dir, max_per_class=5, exts=('.jpg',))
        # classA should have only 3 images, classB should have 5
        assert len(os.listdir(os.path.join(dst_dir, 'classA'))) == 3
        assert len(os.listdir(os.path.join(dst_dir, 'classB'))) == 5

def test_copy_subset_per_class_ext_filter(temp_src_dir):
    # Add a PNG file that should not be copied
    classA_dir = os.path.join(temp_src_dir, 'classA')
    img = Image.fromarray(np.uint8(np.random.rand(32,32,3)*255))
    img.save(os.path.join(classA_dir, "extra.png"))
    with tempfile.TemporaryDirectory() as dst_dir:
        subset_data.copy_subset_per_class(temp_src_dir, dst_dir, max_per_class=10, exts=('.jpg',))
        images = os.listdir(os.path.join(dst_dir, 'classA'))
        # PNG should not be copied
        assert all(img.endswith('.jpg') for img in images)
