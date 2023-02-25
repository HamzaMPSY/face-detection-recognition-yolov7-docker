# Face Detection and Recognition using YOLOv7 and Docker

In this project is my first attempt on  the world of microsevices, the idea behind it os to separate the detection from the recognition and even separate the ENV's, and thats is done using docker

## Face detection
The face detection service is implemeted using a pretrained YOLOv7 model, i dockerize it based on the [pytorch](https://hub.docker.com/r/pytorch/pytorch/) image that use the gpu as an accelerator and have acces to the display so we can see the plots, and i managed to make it happen after declaring this env variables and volumes, and i reserve the nvidia gpu for the container:

```markdown

environment:
    - DISPLAY=unix$DISPLAY
    - QT_X11_NO_MITSHM=1
volumes:
    - /dev/video0:/dev/video0
    - /tmp/.X11-unix:/tmp/.X11-unix
deploy:
    resources:
    reservations:
        devices:
        - driver: nvidia
            count: 1
            capabilities: [ gpu ]
```

## face rocognizer 
The face recognizer service is imlemented using the face recognition library , i use only the function that compute the face embeedings and a simple function to get  the closest person from a data base of faces, this service also had access to the gpu the same way declared in the previous code