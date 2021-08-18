# ceiq-python
This is a Python implementation of CEIQ
## Requirements
```
pip install -r requirements.txt
```
## Import and go
The model accepts two kinds of input type:
- Option 0: Predicting score with inputs are paths to images.
- Option 1: Predicting score with inputs are RGB matrix reprentation of images.

```
model = CEIQ()
results0 = model.predict(['test_imgs/1.png', 'test_imgs/2.png'], 0) # 'option' is set to 0 to indicate prediction from paths

img1 = cv2.imread('test_imgs/1.png')
img2 = cv2.imread('test_imgs/2.png')
results1 = model.predict([img1, img2], 1) # 'option' is set to 1 to indicate prediction from BGR matrix representations of images

print(results0, results1) # the two outputs are supposed to be the same
```
