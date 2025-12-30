from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 1 ======================  Load dataset from disk for Visualization  ======================= 
train_dataset = datasets.ImageFolder(
    root="C:\\Users\\ALSHELSS\OneDrive\\Desktop\\fire_detection\\fire_dataset\Train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="C:\\Users\\ALSHELSS\\OneDrive\\Desktop\\fire_detection\\fire_dataset\Vali",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="C:\\Users\\ALSHELSS\\OneDrive\Desktop\\fire_detection\\fire_dataset\Test",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2 ========================  Load and transform data for train model  ========================= 


train_data_path = "C:\\Users\\ALSHELSS\OneDrive\\Desktop\\fire_detection\\fire_dataset\Train"
test_data_path = "C:\\Users\\ALSHELSS\\OneDrive\Desktop\\fire_detection\\fire_dataset\Test"


train_data_simple = datasets.ImageFolder(root=train_data_path, transform=transform)
test_data_simple = datasets.ImageFolder(root=test_data_path, transform=transform)



BATCH_SIZE = 32  # Setup batch size and number of workers
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and workers.")

# Create DataLoader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)

print(f"---> {train_dataloader_simple}"), print(test_dataloader_simple)




# 3.A =============================   Visualization   =====================================
img = Image.open("C:\\Users\\ALSHELSS\\OneDrive\Desktop\\fire_detection\\fire_dataset\\Train\\Fire\\dayroad265.jpg")

print(f"image height is :{img.height} and width :{img.width}")
img

print(train_dataset.classes)

class_name = train_dataset.classes
class_to_idx = train_dataset.class_to_idx



# 3.B  ================================ Function Visualization  =================================== 
def visualize_batch(
    dataloader,
    class_name,
    class_to_idx,
    num_images=10
):
    """
    Visualize images from a training batch filtered by class name.
    
    Args:
        dataloader: PyTorch DataLoader (train_loader)
        class_name: 'fire' or 'no_fire'
        class_to_idx: dataset.class_to_idx
        num_images: number of images to display
    """

    class_idx = class_to_idx[class_name]
    images_shown = 0

    for images, labels in dataloader:
        mask = labels == class_idx
        filtered_images = images[mask]

        for img in filtered_images:
            if images_shown >= num_images:
                plt.show()
                return

            img = img.permute(1, 2, 0)  # CHW → HWC
            img = img.clamp(0, 1)

            plt.figure(figsize=(3,3))
            plt.imshow(img)
            plt.title(class_name)
            plt.axis("off")

            images_shown += 1

    plt.show()

# 3.C  ================================ Assuming ImageFolder dataset  =================================== 
class_to_idx = train_dataset.class_to_idx
print(class_to_idx)
num_classes = len(train_loader.dataset.classes)
print(num_classes)
# {'fire': 0, 'non_fire': 1}


# 3.D  ================================ Show 10 fire images =================================== 
visualize_batch(
    dataloader=train_loader,
    class_name="Fire",
    class_to_idx=class_to_idx,
    num_images=1
)


# 3.E  ================================ Show 10 non_fire images  =================================== 
visualize_batch(
    dataloader=train_loader,
    class_name="Non-Fire",
    class_to_idx=class_to_idx,
    num_images=1
)


# ===============================  Done ============================