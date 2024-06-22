import LEDPmodelforzip
import sys
import math
import numpy as np

def read_file(open_file):
    with open(open_file, 'r', encoding='utf-8') as file:
        return file.read()

def write(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(content)

def zip2(data):
    Data_x = [ord(c) for c in data[:-1]]
    Data_x2 = list(range(len(data) - 1))
    Data_y = [ord(c) for c in data[1:]]
    model = LEDPmodelforzip.LEDPModel()  # Instantiate the LEDPModel class
    model.reset_training()
    model.build_model(Data_x, Data_x2, Data_y, len(data) - 2, data, 1, 1)
    model.training_model()
    outr = list(model.return_data())
    outr.extend([len(data), ord(data[0])])  # Store the length of data and the ASCII value of the first character
    return outr


def un_zip(data):
    # Remove trailing slash if it exists
    data = data.rstrip('/')
    out_list = list(map(float, data.split("/")))
    
    # Extract length and start character ASCII value
    length = int(out_list[-2])  # Length is the second last element
    start_char = int(out_list[-1])  # Start character ASCII value is the last element
    
    reout = ""
    for x2 in range(length):
        x = start_char
        value = (
            out_list[1] * x + out_list[2] * x**2 + out_list[3] * x**3 +
            out_list[4] * math.cos(x) + out_list[5] * math.sin(x) +
            out_list[6] * math.factorial(int(x)) + out_list[7] * x2 +
            out_list[8] * x2**2 + out_list[9] * x2**3 +
            out_list[10] * math.cos(x2) + out_list[11] * math.sin(x2) +
            out_list[12] * math.factorial(int(x2)) + out_list[0]
        )
        reout += chr(int(value) % 256)  # Ensure the value is within valid ASCII range
    return reout


def main():
    while True:
        inpuz = input("command: ")
        if inpuz.startswith("/zip "):
            file_name = inpuz[len("/zip "):]
            content = read_file(file_name)
            zipped_content = zip2(content)
            zipped_string = "/".join(map(str, zipped_content)) + "/"
            write(file_name, zipped_string)
            print(f"ZIP file {file_name} is complete")
        elif inpuz.startswith("/unzip "):
            file_name = inpuz[len("/unzip "):]
            content = read_file(file_name)
            unzipped_content = un_zip(content)
            write(file_name, unzipped_content)
            print(f"UNZIP file {file_name} is complete")

main()
