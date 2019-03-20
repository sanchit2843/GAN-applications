from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from params import *
#create custom data class with transform
class dog_dataset(Dataset):
  def __init__(self,image_dir,transform = None):

    self.img_dir = image_dir
    self.transform = transform
    self.id = os.listdir(self.img_dir)
  def __len__(self):
    return len(os.listdir(self.img_dir))
  def __getitem__(self,idx):
    img_name = os.path.join(self.img_dir, self.id[idx])
    image = cv2.imread(img_name)

    if self.transform:
      image = self.transform(image)
    return image
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((im_size,im_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
#normalize the image with mean (0.5,0.5,0.5) and standard deviation = (0.5,0.5,0.5)
dog_data = dog_dataset(des ,transform)
dataloader = DataLoader(dog_data, batch_size=128)
