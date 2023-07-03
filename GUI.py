import tkinter
import tkinter.font as font
from tkinter import *
#from task_NN import *

window_main = tkinter.Tk(className='Tkinter - TutorialKart', )
window_main.geometry("500x500")


#################################################################################################
labele=tkinter.Label(window_main, text=" Enter number of hidden layers ")
labele.pack()

check_hidden=IntVar()

hidden=tkinter.Entry(window_main,width=20,textvariable=check_hidden)
hidden.pack()

#################################################################################################

labele=tkinter.Label(window_main, text=" Enter number of neurons in each hidden layer ")
labele.pack()

check_neurons=IntVar()

neurons=tkinter.Entry(window_main,width=20,textvariable=check_neurons)
neurons.pack()

#################################################################################################

labele=tkinter.Label(window_main, text=" Enter learning rate (eta) ")
labele.pack()

Check_eta=DoubleVar()

eta=tkinter.Entry(window_main,width=20,textvariable=Check_eta)
eta.pack()
#################################################################################################

labele=tkinter.Label(window_main, text=" Enter number of epochs (m) ")
labele.pack()

check_epochs=IntVar()

epochs=tkinter.Entry(window_main,width=20,textvariable=check_epochs)
epochs.pack()
#################################################################################################

Check_bias = IntVar()

bias=tkinter.Checkbutton(window_main, text="Add bias",variable = Check_bias ,onvalue = 1,  offvalue = 0)
bias.pack()
#################################################################################################

OPTIONS = ["sigmoid","tanh"]
activation = StringVar(window_main)
activation.set("   activation function   ")
w = OptionMenu(window_main, activation, *OPTIONS)
w.pack()

#################################################################################################

def close_window ():
    window_main.destroy()

button = tkinter.Button(window_main, text="Excute Training",command = close_window )
button.pack()

window_main.mainloop()