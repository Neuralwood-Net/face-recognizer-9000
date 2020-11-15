import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
f.suptitle("WoodNet and SqueezeNet training on images cropped to faces")

df = pd.read_csv(
    "/Users/larsankile/GitLocal/face-recognizer-9000/logs/WoodNet-1605365270.1111202_cropped.csv",
    names=["time", "epoch", "phase", "img", "loss", "acc"],
)

df["acc"] = df["acc"] * 100

trn = df[df["phase"] == "train"]
val = df[df["phase"] == "val"]
val["img"] = trn["img"]

epochs = 3
num_row = min(trn.shape[0], val.shape[0], 103 * epochs)

max_img = trn["img"].max()

trn = trn.iloc[:num_row, :]
val = val.iloc[:num_row, :]

trn['acc_sma'] = trn.loc[:,"acc"].rolling(window=60).mean()
val['acc_sma'] = val.loc[:,"acc"].rolling(window=60).mean()

trn['loss_sma'] = trn.loc[:,"loss"].rolling(window=60).mean()
val['loss_sma'] = val.loc[:,"loss"].rolling(window=60).mean()

ax1.plot(trn["img"], list(trn["acc_sma"]), label="Training accuracy")
ax1.plot(trn["img"], list(val["acc_sma"]), label="Validation accuracy")

for x in range(109000, 109000*epochs + 1, 109000):
    ax1.axvline(x=x, linewidth=1, color="black")

ax1.set_xlabel("Number of images")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim((80, 100))
ax1.set_title("WoodNet Accuracy")
ax1.legend()

ax2.plot(trn["img"], list(trn["loss_sma"]), label="Training loss")
ax2.plot(trn["img"], list(val["loss_sma"]), label="Validation loss")

for x in range(109000, 109000*epochs + 1, 109000):
    ax2.axvline(x=x, linewidth=1, color="black")

ax2.set_xlabel("Number of images")
ax2.set_ylabel("Loss measured by cross-entropy")
ax2.set_title("WoodNet Loss")
ax2.legend()

df = pd.read_csv(
    "/Users/larsankile/GitLocal/face-recognizer-9000/logs/SqueezeNet-1605361529.9021263_cropped.csv",
    names=["time", "epoch", "phase", "img", "loss", "acc"],
)

df["acc"] = df["acc"] * 100

trn = df[df["phase"] == "train"]
val = df[df["phase"] == "val"]
val["img"] = trn["img"]

epochs = 3
num_row = min(trn.shape[0], val.shape[0], 103 * epochs)

max_img = trn["img"].max()

trn = trn.iloc[:num_row, :]
val = val.iloc[:num_row, :]

trn['acc_sma'] = trn.loc[:,"acc"].rolling(window=60).mean()
val['acc_sma'] = val.loc[:,"acc"].rolling(window=60).mean()

trn['loss_sma'] = trn.loc[:,"loss"].rolling(window=60).mean()
val['loss_sma'] = val.loc[:,"loss"].rolling(window=60).mean()

ax3.plot(trn["img"], list(trn["acc_sma"]), label="Training accuracy")
ax3.plot(trn["img"], list(val["acc_sma"]), label="Validation accuracy")

for x in range(109000, 109000*epochs + 1, 109000):
    ax3.axvline(x=x, linewidth=1, color="black")

ax3.set_xlabel("Number of images")
ax3.set_ylabel("Accuracy (%)")
ax3.set_ylim((80, 100))
ax3.set_title("SqueezeNet Accuracy")
ax3.legend()

ax4.plot(trn["img"], list(trn["loss_sma"]), label="Training loss")
ax4.plot(trn["img"], list(val["loss_sma"]), label="Validation loss")

for x in range(109000, 109000*epochs + 1, 109000):
    ax4.axvline(x=x, linewidth=1, color="black")

ax4.set_xlabel("Number of images")
ax4.set_ylabel("Loss measured by cross-entropy")
ax4.set_title("SqueezeNet Loss")
ax4.legend()

plt.show()

