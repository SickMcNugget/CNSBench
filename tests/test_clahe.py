import albumentations as A
import cv2

transform = A.Compose([A.CLAHE()])

image = cv2.imread("MoNuSeg/yolo_snv/segment/test/TCGA-2Z-A9J9-01A-01-TS1.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transformed = transform(image=image)
transformed_image = transformed["image"]

cv2.imshow("transformed", transformed_image)
cv2.imshow("original", transformed_image)
cv2.waitKey(0)