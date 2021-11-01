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

### Reference

- Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. PMLR, 2015.

### Acknowledgements

- [a PyTorch tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)


