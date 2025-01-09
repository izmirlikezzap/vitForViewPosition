import cv2
from pathlib import Path
from tqdm import tqdm

# CLAHE settings
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Paths
main = "/media/envisage/backup8tb/Padchest"
dataset_folder = Path(main) / "projectionClassificationImagesTwoClass"
train_folder = dataset_folder / "train"
test_folder = dataset_folder / "test"
val_folder = dataset_folder / "validation"

clahe_path = Path(main) / "clahe_images"
clahe_train_folder = clahe_path / "train"
clahe_test_folder = clahe_path / "test"
clahe_val_folder = clahe_path / "validation"

def apply_clahe_to_images(input_folder, output_folder):
    """
    Applies CLAHE to all images in the input folder and saves them to the output folder.

    Parameters:
        input_folder (Path): Path to the folder containing input images.
        output_folder (Path): Path to the folder to save CLAHE-enhanced images.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for subfolder in input_folder.iterdir():
        if subfolder.is_dir():
            sub_output_folder = output_folder / subfolder.name
            sub_output_folder.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(list(subfolder.rglob("*.png")), desc=f"Processing {subfolder}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Apply CLAHE
                enhanced_img = clahe.apply(img)

                # Save the enhanced image
                relative_path = img_path.relative_to(input_folder)
                output_path = output_folder / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), enhanced_img)

# Apply CLAHE to train, test, and validation folders
apply_clahe_to_images(train_folder, clahe_train_folder)
apply_clahe_to_images(test_folder, clahe_test_folder)
apply_clahe_to_images(val_folder, clahe_val_folder)
