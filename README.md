# NoSoLuckyImaging
Live quality check for Lucky Image Astrophotography with libcamera interface

# Build for the IMX290C Sensor

# Find Target with libcamera device

VNC desktop sessions are to slow to get a good live video.
Use the tcp streaming option from libcamera:
```
libcamera-vid -t 0 --width 1920 --height 1080 --nopreview --codec h264 --inline --listen -o tcp://0.0.0.0:8888
```
And on your host use VLC to display with less network buffer (200 ms) for a real time view:
```
vlc tcp/h264://camera:8888/ --network-caching=200
```