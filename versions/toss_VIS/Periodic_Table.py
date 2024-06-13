#this is a GUI for the plots of the distance distribution
#it can be packaged by the pyinstaller to generate a exe file
#NEEDED assign the plot paths

import tkinter as tk
import tkinter.font as tf
from PIL import Image
import sys
import os

element_list = ['None','H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
                'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga','Ge', 'As', 
                'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 
                'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 
                'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 
                'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
                'Fl', 'Mc', 'Lv', 'Ts', 'Og']

master = tk.Tk()
master.title("Periodic table")
master.geometry("1100x800")
def open_plot(ele_pair): 
    global element_list
    path = os.getcwd()

    if element_list.index(ele_pair[0]) > element_list.index(ele_pair[1]):
        temp = ele_pair[0]
        ele_pair[0] = ele_pair[1]
        ele_pair[1] = temp
        #print("Position Changed!")
        img = Image.open(path + "\\redefined_length_png\\Threshold of %s and %s.png"%(ele_pair[0],ele_pair[1]))
        img.show()
    else:
        img = Image.open(path + "\\redefined_length_png\\Threshold of %s and %s.png"%(ele_pair[0],ele_pair[1]))
        img.show()
    #print("Plot has been openned!")


var = tk.StringVar()
l = tk.Label(master, textvariable=var, font=("Helvetica", 16), width = 20, height = 10)
l.place(x = 0, y = 700, width = 1100, height = 50)
    

ele_pair = []
def get_ele(i):
    global ele_pair
    ele_pair.append(element_list[i])
    if len(ele_pair) == 1:
        var.set("Click once again!")
    if len(ele_pair) == 2:
        try:
            open_plot(ele_pair)
        except:
            var.set("Sorry! We do not have the data you want.")
    if len(ele_pair) == 3:
        ele_pair.remove(ele_pair[0])
        ele_pair.remove(ele_pair[0])
        var.set("Click once again!")

font = tf.Font(family="Helvetica", size=16)

for i in range(1,119):
    if i == 1:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100,y=100, width=50, height=50)"%i)
    if i == 2:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=950,y=100, width=50, height=50)"%i)
    if 2 < i <= 4:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-3),y=150, width=50, height=50)"%(i,i))
    if 4 < i <= 10:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(12+%d-5),y=150, width=50, height=50)"%(i,i))
    if 10 < i <= 12:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-11),y=200, width=50, height=50)"%(i,i))
    if 12 < i <= 18:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(12+%d-13),y=200, width=50, height=50)"%(i,i))
    if 18 < i <= 36:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-19),y=250, width=50, height=50)"%(i,i))
    if 36 < i <= 54:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-37),y=300, width=50, height=50)"%(i,i))
    if 54 < i <= 56:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-55),y=350, width=50, height=50)"%(i,i))
    if 56 < i <= 71:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-55),y=470, width=50, height=50)"%(i,i))
    if 71 < i <= 86:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-69),y=350, width=50, height=50)"%(i,i))
    if 86 < i <= 88:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-87),y=400, width=50, height=50)"%(i,i))
    if 88 < i <= 103:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-87),y=520, width=50, height=50)"%(i,i))
    if 103 < i <= 119:
        exec("button_%s = tk.Button(master, text=element_list[i], font=font, command=lambda: get_ele(%d), activebackground='red', activeforeground='black',relief='raised')"%(i,i))
        exec("button_%s.place(x=100+50*(%d-101),y=400, width=50, height=50)"%(i,i))
        
button_quit = tk.Button(master, text="Quit", command = master.quit)
button_quit.place(x = 500, y = 600, width = 100, height = 40)

tk.mainloop()

"""END HERE"""