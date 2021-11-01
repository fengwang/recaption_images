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
recaption_images( '/home/feng/Downloads/images', '/home/feng/Downloads/recaptioned_images' )
```


