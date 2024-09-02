import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks import vision
import tkinter as tk
from tkinter import filedialog

class MainWindow:
    def __init__():
        win = tk.Tk()
        win.title("Pose estimation model")
        labels = []
        
        labels.append(tk.Label(win,text="Enter the path to the video dataset..."))
        labels[0].pack()
        
        inputs = []
        inputs.append(tk.Entry(win))
        inputs[0].pack()

        buttons = []
        buttons.append(tk.Button(text="Search",command=lambda: folder_search(inputs[0])))
        buttons[0].pack()
        
        buttons.append(tk.Button(win,text="Train model"))
        return win



def main():
    win = MainWindow()
    win.mainloop()

    return 0

def folder_search(entry):
    entry.delete(0,'end')
    entry.insert(0,filedialog.askdirectory())
main()