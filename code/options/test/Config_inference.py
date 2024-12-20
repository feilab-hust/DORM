import os.path
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
    return list[-1]


def Inference_cfg():
    win = tk.Tk()
    win.title("Config Inference")  # Add title
    win.geometry('480x160')
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    model_root_path = os.path.join(root_path, 'experiments')
    lastet_model = new_file(model_root_path)

    def clickMe():
        nonlocal label_tag, v_path
        label_tag = label_tag_entered.get()
        v_path = v_path_Choose.get()
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
    return label_tag, v_path


def inference_pa():
    label_tag, v_path = Inference_cfg()
    label_tag = str(label_tag)
    v_path = str(v_path)
    return label_tag, v_path


if __name__ == '__main__':
    label_tag, v_path = inference_pa()

