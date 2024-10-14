from torchvision import  transforms
import torch
def get_normalize():
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return normalize


def save_weights(best_model_path):
    # Load the best checkpoint to retrieve the model state dict
    checkpoint = torch.load(best_model_path)
    # Extract the model's state dictionary (weights)
    model_state_dict = checkpoint['state_dict']
    # Save the model weights as a .pth file
    torch.save(model_state_dict, "best_model_weights.pth")
    print(f"Best model weights saved as 'best_model_weights.pth' from checkpoint {best_model_path}")


def get_transformation():

    # Data transformations
    transform_train = transforms.Compose(
        [

            transforms.ToTensor(),
            #transforms.RandomRotation(10),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5),
            #transforms.RandomHorizontalFlip(),
            #transforms.Resize(128),
            #get_normalize()

        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(128),

        ]
    )
    return transform_train,transform_test