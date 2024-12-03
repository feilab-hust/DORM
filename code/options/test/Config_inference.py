import os.path

# import matplotlib

# matplotlib.use('TkAgg')
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
import tkinter as tk
from tkinter import ttk
import numpy as np
import os.path as osp


def new_file(testdir):
    # get the latest generated folder
    list = os.listdir(testdir)
    list.sort(key=lambda fn: os.path.getmtime(testdir + '\\' + fn))
    # filepath = os.path.join(testdir, list[-1])
    return list[-1]


def Inference_cfg():
    win = tk.Tk()
    win.title("Config Inference")  # Add title
    win.geometry('480x160')
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    model_root_path = os.path.join(root_path, 'experiments')
    lastet_model = new_file(model_root_path)

    def clickMe():
        nonlocal label_tag, v_path, net_type
        label_tag = label_tag_entered.get()
        v_path = v_path_Choose.get()
        net_type = net_type_Chosen.get()
        win.destroy()

    style = ttk.Style()
    style.configure("test.TButton", background="white", foreground="blue")
    action = ttk.Button(win, text="Start running", command=clickMe, style="test.TButton")
    action.grid(column=2, row=10, columnspan=5)

    tk.Label(win, text="Label tag :").grid(column=0, row=0, sticky=tk.W)
    label_tag = tk.StringVar()
    label_tag.set(lastet_model)
    label_tag_entered = ttk.Entry(win, width=40, textvariable=label_tag)
    label_tag_entered.grid(column=1, row=0, columnspan=14, sticky=tk.W)
    label_tag_entered.focus()

    g_row = 1

    ttk.Label(win, text="Net type:").grid(column=0, row=g_row + 3, sticky=tk.W)
    net_type = tk.StringVar()
    net_type_Chosen = ttk.Combobox(win, width=13, textvariable=net_type, state='readonly')
    net_type_Chosen['values'] = ('3D_net', 'ISO_net')
    net_type_Chosen.grid(column=1, row=g_row + 3, columnspan=8, sticky=tk.W)
    net_type_Chosen.current(0)

    def select_LR_Path():
        path_lr = askdirectory(title="Please choose the Validation path")
        v_path.set(path_lr)

    v_path = tk.StringVar()
    row_2 = 5 + g_row
    Label(win, text="Validation path:").grid(column=0, row=row_2, sticky=tk.N)
    v_path_Choose = ttk.Entry(win, width=40, textvariable=v_path)
    v_path_Choose.grid(column=1, row=row_2, columnspan=7, sticky=tk.W)
    lr_path_Choose_button = ttk.Button(win, text="Choose", command=select_LR_Path, style="test.TButton")
    lr_path_Choose_button.grid(column=12, row=row_2, sticky=tk.E)

    win.mainloop()
    return label_tag, v_path, net_type


def inference_pa():
    label_tag, v_path, net_type = Inference_cfg()
    label_tag = str(label_tag)
    v_path = str(v_path)
    net_dim = 5 if net_type == '3D_net' else 4
    return label_tag, v_path, net_dim


if __name__ == '__main__':
    label_tag, v_path,net_dim = inference_pa()

# kk = out
