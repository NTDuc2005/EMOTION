import serial

ser = serial.Serial("COM3", 9600, timeout=1)

print("Dang lang nghe VC02...")

while True:
    if ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        print("VC02:", line)