Align with Config and Model checkpoint name

```python
#pip install gdown
import gdown

# Replace FILE_ID with the actual file ID
file_id = "1-3oRGdeXR5c6G58Ok5KbosZXfmhZNAu_"
url = f"https://drive.google.com/uc?id={file_id}"

# Replace FILE_NAME with the name you want to save the file as
output = "/home/SuperResolutionDentalXray/weights/model.pth"
gdown.download(url, output, quiet=False)
```