import tkinter as tk                                # TK is used to display and update the GUI itself.
from tkinter import ttk
from PIL import ImageTk, Image                      # PIL is used to display the images in the GUI.
import numpy as np                                  # NumPy is necessary for many TS operations.
import fft                                          # The package provided to us for Fast Fourier Transforms.
from ifxdaq.sensor.radar_ifx import RadarIfxAvian   # This package is used to handle everything relating to the radar.
from tensorflow import keras                        # Keras is the library we used to build and train the ML model.

reconstructed_model = keras.models.load_model("radar_model_fit")  # Here we're importing the model from our folder.
overallcount = 0                                                  # Used to count the passed frames
config_file = "radar_configs/RadarIfxBGT60.json"                  # The provided radar json config file

root = tk.Tk()

lastframe = 0                       # Prediction of the previous frame, used for ensuring output stability.

variable = tk.StringVar()
var2 = tk.StringVar()

root.geometry("900x500")            # Defining the default screen size for the main window

your_label = tk.Label(root, textvariable=variable, font=("Tahoma", 30))
your_label.pack()

l0 = tk.Label(root, text="0", anchor="w")  # This part of the code is used for defining the probability progress bars.
l0.pack()
pb0 = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
pb0.pack()

l1 = tk.Label(root, text="1", anchor="w")
l1.pack()
pb1 = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
pb1.pack()

l2 = tk.Label(root, text="2", anchor="w")
l2.pack()
pb2 = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
pb2.pack()

l3 = ttk.Label(root, text="3", anchor="w")
l3.pack()
pb3 = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
pb3.pack()

lab2 = tk.Label(root, textvariable=var2, font=("Tahoma", 10))
lab2.pack()

canvas = tk.Canvas(root, width=300, height=300)             # Used to draw the Doppler Range matrix as a GUI image.
canvas.pack()
image_container = canvas.create_image(0, 0, image=None)
canvas.place(anchor='e', relx=0.4, rely=0.5)


def runAfterMainloop():
    with RadarIfxAvian(config_file) as device:    # Initialize the radar with configurations

        for i_frame, frame in enumerate(device):  # Loop through the frames coming from the radar

            global overallcount
            overallcount += 1
            if overallcount % 4 == 0:
                data = np.squeeze(frame['radar'].data[0] / (4095.0))            # Here we're normalising by dividing.

                range_data = np.abs(fft.range_doppler_fft(data))[16:48, 0:32]   # From raw data to FFT Doppler range
                # plt.imshow(range_data)
                # plt.show
                templist = np.array([range_data])

                data = Image.fromarray(range_data * 100)  # Generate an image from the Doppler range to display in GUI

                maxind = 0
                max = 0
                count = 0
                secondmax = 0

                predicted_data = reconstructed_model.predict(templist, verbose=0)  # This line uses the saved model
                # to predict the outputs (people count) of the current frame data.

                global lastframe
                predicted_data[0][lastframe] += 0.05 * lastframe
                if lastframe == 3:
                    predicted_data[0][lastframe] += 0.05

                for i in predicted_data[0]:  # Iterate through the predicted outputs, making sure to keep the index
                    # of the most probable one in order to output that as a prediction
                    if max < i:
                        max = i
                        maxind = count
                    count += 1

                for i in predicted_data[0]:  # Iterate through the predicted outputs, making sure to keep the index
                    # of the most probable one in order to output that as a prediction
                    if i > secondmax and i < max:
                        secondmax = i

                if max > 1:
                    max = 1
                lastframe = maxind
                variable.set(f"Predicted {maxind} with a probability of {int((max) * 100)}% !")
                pb0['value'] = int(predicted_data[0][0] * 100)
                pb1['value'] = int(predicted_data[0][1] * 100)
                pb2['value'] = int(predicted_data[0][2] * 100)
                pb3['value'] = int(predicted_data[0][3] * 100)
                img2 = ImageTk.PhotoImage(data.resize([600, 600], Image.NEAREST))  # We're resizing the data to fit a
                # 600 by 600 image, without antialiasing to keep as much neatly visible data as possible.
                canvas.itemconfig(image_container, image=img2)
                canvas.image = img2
                var2.set(f"Relative confidence: {int((max-secondmax) * 100)}%")
                root.update()  # Updating the GUI with the new image and text data.


root.after(0, runAfterMainloop)
root.mainloop()  # Launching the main window, and starting the function
