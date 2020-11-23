Chạy chương trình 
``` 
        pip install torch and torchvision
        pip install -r requirements.txt
        cd Pytorch_Retinaface
        mkdir weights
        ## download weights from this link and move to Pytorch_Retinaface/weights
        https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view?usp=sharing
        python main.py
```
        
config_file 
change this line to the testing image folder
```
    [moire]
    in=./images 
```
results are stored in ./save_image/..csv
```
    0 live image
    1, 2 fake detected
```

### SWITCH BETWEEN GPU AND CPU
```
    change in config file config.cfg
    [dl_model]
    device_id=-1 ## for cpu version
    [dl_model]
    device_id=-1 ## for gpu version
    
```
