# Recaption Images with Generated Neural Image Caption
 ----


### Example Usage:

Commandline: Recaption all images from folder `/home/feng/Downloads/images` to folder `/home/feng/Downloads/recaptioned_images`.

```bash
python3 ./recaption_images.py -i /home/feng/Downloads/images -o /home/feng/Downloads/recaptioned_images
```

Python:

```python
from recaption_images import recaption_images
recaption_images( '/home/feng/Downloads/images', '/home/feng/Downloads/recaptioned_images' ) # the first argument is for the input image folder, the second argument is for the output image folder
```


The images names in the input folder

```
├── p1540655619.jpg
├── p1540655903.jpg
├── p1540656197.jpg
├── p1957821761.jpg
├── p2184401089.jpg
├── p2184401118.jpg
├── p2212204439.jpg
├── p2264739838.jpg
├── p2321825526.jpg
├── p2556763206.jpg
├── p794593280.jpg
├── p960078680.jpg
├── p960078915.jpg
├── p960119762.jpg
├── p97138677.jpg
├── p979605183.jpg
└── p979608270.jpg
```

become

```
├── a woman holding a cell phone in her hand_p960078680.jpg
├── a woman holding a cup of coffee_p2264739838.jpg
├── a woman holding a teddy bear in front of a building_p960119762.jpg
├── a woman in a bikini holding a purse_p1957821761.jpg
├── a woman in a bikini sitting on a chair_p2212204439.jpg
├── a woman in a white dress and a white dress_p2556763206.jpg
├── a woman in a white dress holding a white dog_p960078915.jpg
├── a woman in a white dress holding a white flower_p1540655619.jpg
├── a woman in a white dress holding a white flower_p1540656197.jpg
├── a woman in a white dress is holding a flower_p1540655903.jpg
├── a woman in a white dress sitting on a bed_p2321825526.jpg
├── a woman in a white shirt holding a remote_p2184401118.jpg
├── a woman is holding a glass of wine_p2184401089.jpg
├── a woman is holding a spoon in her hand_p794593280.jpg
├── a young girl eating a piece of cake_p97138677.jpg
├── a young girl holding a box of donuts_p979608270.jpg
└── a young girl wearing a scarf and a tie_p979605183.jpg
```



### Reference

- Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. PMLR, 2015.

### Acknowledgements

- [a PyTorch tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)


