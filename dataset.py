from config import*
# ------------------Augment 1---------------------------
# from torch.utils.data import Dataset  # Thêm dòng này
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=0, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2()
])

valid_transform = A.Compose([
	A.Resize(height=256, width=256),
	A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
	ToTensorV2()
])
# Không dùng augmentation
# transforms = transforms.Compose([transforms.ToPILImage(),
#  	transforms.Resize((INPUT_IMAGE_HEIGHT,
# 		INPUT_IMAGE_WIDTH)),
# 	transforms.ToTensor()])

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# Dùng Augmentation
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		if self.transforms:
			image = cv2.resize(image, (256, 256))
			mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
		    	augmented = self.transforms(image=image, mask=mask)
			image = augmented["image"]
			# mask = augmented["mask"]
			# print("shape_mask1: ", mask.shape)
	        	mask = (mask > 127).astype("float32")        # chuyển về float32: giá trị 0.0 hoặc 1.0
			mask = torch.from_numpy(mask)  
			mask = mask.unsqueeze(0)                     # shape (1, H, W)

			# print("shape_image: ", image.shape)
			# print("shape_mask: ", mask.shape)
		# return (image, mask)
		return image, mask, imagePath
		# Không có augmentation
		# grab the image path from the current index
		# imagePath = self.imagePaths[idx]
		# # load the image from disk, swap its channels from BGR to RGB,
		# # and read the associated mask from disk in grayscale mode
		# image = cv2.imread(imagePath)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# mask = cv2.imread(self.maskPaths[idx], 0)
		# # check to see if we are applying any transformations
		# if self.transforms is not None:
		# 	# apply the transformations to both image and its mask
		# 	image = self.transforms(image)
		# 	# mask = self.transforms(mask)
		# 	mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
		# 	mask = (mask > 127).astype("float32")
		# 	mask = torch.from_numpy(mask)
		# 	mask = mask.unsqueeze(0)    
		# return image, mask, imagePath

# load the image and mask filepaths in a sorted manner
trainImagesPaths = sorted(list(paths.list_images(IMAGE_TRAIN_PATH)))
trainMasksPaths = sorted(list(paths.list_images(MASK_TRAIN_PATH)))

validImagesPaths = sorted(list(paths.list_images(IMAGE_VALID_PATH)))
validMasksPaths = sorted(list(paths.list_images(MASK_VALID_PATH)))

testImagesPaths = sorted(list(paths.list_images(IMAGE_TEST_PATH)))
testMasksPaths = sorted(list(paths.list_images(MASK_TEST_PATH)))

# Không dùng Augmentation 
# # create the train and test datasets
# trainDS = SegmentationDataset(imagePaths=trainImagesPaths, maskPaths=trainMasksPaths,
# 	transforms=transforms)
# validDS = SegmentationDataset(imagePaths=validImagesPaths, maskPaths=validMasksPaths,
#     transforms=transforms)
# testDS = SegmentationDataset(imagePaths=testImagesPaths, maskPaths=testMasksPaths,
#     transforms=transforms)
# Dùng Augmentation

trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms = train_transform)
validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms = valid_transform)
testDS = SegmentationDataset(imagePaths=testImagesPaths, maskPaths=testMasksPaths,
    transforms=valid_transform)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(validDS)} examples in the valid set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=bach_size, pin_memory=PIN_MEMORY,
	num_workers=4)
validLoader = DataLoader(validDS, shuffle=False,
	batch_size=bach_size, pin_memory=PIN_MEMORY,
	num_workers=4)
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=bach_size, pin_memory=PIN_MEMORY,
	num_workers=4)


