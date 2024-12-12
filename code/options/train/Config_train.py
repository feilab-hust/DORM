from data.chunking import generate_training_data

# import matplotlib

# matplotlib.use('TkAgg')

from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
import tkinter as tk
from tkinter import ttk
import numpy as np


def tkinter_input():
    win = tk.Tk()
    win.title("Config train")  # Add title
    win.geometry('480x240')

    lm = 7

    def clickMe():
        nonlocal label_tag, hr_path, lr_path,  patch_size_d, patch_size_h, patch_size_w, \
            thre, poisson_noise, gauss_noise, Gan_model
        label_tag = label_tag_entered.get()
        hr_path = hr_path_Choose.get()
        lr_path = lr_path_Choose.get()
        patch_size_d = patch_size_d_entered.get()
        patch_size_h = patch_size_h_entered.get()
        patch_size_w = patch_size_w_entered.get()
        thre = thre_entered.get()
        poisson_noise = poisson_noise.get()
        Gan_model = Gan_model.get()
        gauss_noise = gauss_noise_entered.get()
        win.destroy()

    style = ttk.Style()
    style.configure("test.TButton", background="white", foreground="blue")
    action = ttk.Button(win, text="Start running", command=clickMe, style="test.TButton")
    action.grid(column=3, row=15, columnspan=2)


    tk.Label(win, text="Label tag :").grid(column=0, row=1)
    label_tag = tk.StringVar()
    label_tag.set("Digital_DOFe")
    label_tag_entered = ttk.Entry(win, width=40, textvariable=label_tag)
    label_tag_entered.grid(column=1, row=1, columnspan=14, sticky=tk.W)
    label_tag_entered.focus()

    g_column = 0
    g_row = 1

    def select_HR_Path():
        path_hr = askdirectory(title="Please choose the GT path")
        hr_path.set(path_hr)

    hr_path = tk.StringVar()
    Label(win, text="GT path:").grid(column=0, row=g_row + 1, sticky=tk.N)
    hr_path_Choose = ttk.Entry(win, width=40, textvariable=hr_path)
    hr_path_Choose.grid(column=g_column + 1, row=g_row + 1, columnspan=lm, sticky=tk.W)
    hr_path_Choose_button = ttk.Button(win, text="Choose", command=select_HR_Path, style="test.TButton")
    hr_path_Choose_button.grid(column=g_column + 6, row=g_row + 1, sticky=tk.W)

    size_row = 2
    tk.Label(win, text="Patch size:").grid(column=g_column, row=g_row + size_row, sticky=tk.N)
    patch_size_d = tk.StringVar()
    patch_size_d.set('32')
    tk.Label(win, text="Depth:").grid(column=1, row=g_row + size_row, sticky=tk.W)
    patch_size_d_entered = ttk.Entry(win, width=5, textvariable=patch_size_d)
    patch_size_d_entered.grid(column=2, row=g_row + size_row, sticky=tk.W)

    tk.Label(win, text="Height:").grid(column=g_column + 3, row=g_row + size_row, sticky=tk.W)
    patch_size_h = tk.StringVar()
    patch_size_h.set('64')
    patch_size_h_entered = ttk.Entry(win, width=5, textvariable=patch_size_h)
    patch_size_h_entered.grid(column=g_column + 4, row=g_row + size_row, sticky=tk.W)

    tk.Label(win, text="Width:").grid(column=g_column + 5, row=g_row + size_row, sticky=tk.W)
    patch_size_w = tk.StringVar()
    patch_size_w.set('64')
    patch_size_w_entered = ttk.Entry(win, width=5, textvariable=patch_size_w)
    patch_size_w_entered.grid(column=g_column + 6, row=g_row + size_row, sticky=tk.W)

    tk.Label(win, text="Threshold:").grid(column=0, row=g_row + 3, sticky=tk.N)
    thre = tk.StringVar()
    thre.set(0.9)
    thre_entered = ttk.Entry(win, width=5, textvariable=thre)
    thre_entered.grid(column=g_column + 1, row=g_row + 3, sticky=tk.W)

    tk.Label(win, text="Gaussian noise:").grid(column=g_column + 2, row=g_row + 3, columnspan=2, sticky=tk.W)
    gauss_noise = tk.StringVar()
    gauss_noise.set(0)
    gauss_noise_entered = ttk.Entry(win, width=5, textvariable=gauss_noise)
    gauss_noise_entered.grid(column=g_column + 4, row=g_row + 3, sticky=tk.W)

    poisson_noise = tk.IntVar()
    check2 = tk.Checkbutton(win, text="Possion noise", variable=poisson_noise)
    # check2.select()
    check2.grid(column=g_column + 5, row=3 + g_row, columnspan=g_row + 2, sticky=tk.W)

    def select_LR_Path():
        path_lr = askdirectory(title="Please choose the Raw data path")
        lr_path.set(path_lr)


    lr_path = tk.StringVar()
    row_2 = 5 + g_row
    Label(win, text="Raw data path:").grid(column=0, row=row_2, sticky=tk.N)
    lr_path_Choose = ttk.Entry(win, width=40, textvariable=lr_path)
    lr_path_Choose.grid(column=1, row=row_2, columnspan=lm, sticky=tk.W)
    lr_path_Choose_button = ttk.Button(win, text="Choose", command=select_LR_Path, style="test.TButton")
    lr_path_Choose_button.grid(column=6, row=row_2, sticky=tk.E)

    Gan_model = tk.IntVar()
    check3 = tk.Checkbutton(win, text="Gan model", variable=Gan_model)
    # check3.select()
    check3.grid(column=1, row=row_2 + 1, columnspan=g_row + 2, sticky=tk.W)


    win.mainloop()
    return label_tag, hr_path, lr_path,  patch_size_d, patch_size_h, patch_size_w, \
        thre, poisson_noise, gauss_noise, Gan_model


def chunking_data():
    label_tag, hr_path, lr_path, patch_size_d, patch_size_h, patch_size_w, \
        thre, poisson_noise, gauss_noise, Gan_model = tkinter_input()

    d = int(patch_size_d)
    h = int(patch_size_h)
    w = int(patch_size_w)
    hr_path = hr_path
    lr_path = lr_path
    train_img_size_lr = [h, w] if d == 1 else [d, h, w]  # [d,h,w] or [h,w]
    label_tag = str(label_tag)
    Gan_model = str(Gan_model)
    threshold = float(thre)
    poisson_noise = poisson_noise
    gauss_sigma = float(gauss_noise)

    hr, lr = generate_training_data(
        hr_path=hr_path,
        lr_path=lr_path,
        patch_size=train_img_size_lr,
        factor=1,
        psf=None,
        z_sub_sample=1,
        threshold=threshold,
        poisson_noise=poisson_noise,
        gauss_sigma=gauss_sigma,
    )
    return label_tag, Gan_model


