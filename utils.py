import matplotlib.pyplot as plt
import torchvision

# Helper function to show images
def show_images(images):
    images = (images + 1) / 2  # Scale images to [0, 1]
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.show()