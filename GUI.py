from tkinter import *
from math import cos, sin, sqrt, radians
from PIL import Image, ImageTk

"""
class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("HEX board")
        #mw.geometry("500x500")
        self.label = Label(master, text="HEX")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

    def play_piece(self):
        pass

root = Tk()
root.geometry("900x600")
my_gui = MyFirstGUI(root)
root.mainloop()
"""

    #         (x,y) (rows, cols)       R (3)
    # -4-            (0,0)                         1
    # -3-         (1,0) (0,1)                     6  2
    # -2-      (2,0) (1,1) (0,2)                 11  7  3
    # -1-   (3,0) (2,1) (1,2) (0,3)             15  12  8  4       
    #    (4,0) (3,1) (2,2) (1,3) (0,4)         15 16    13  9  5       
    #       (4,1) (3,2) (2,3) (1,4)              18  17  14 10
    #          (4,2) (3,3) (2,4)                   20 19 18
    #             (4,3) (3,4)                        21 22
    #                (4,4)                             23


#[[math.cos(rot), -math.sin(rot)],
#              [math.sin(rot), math.cos(rot)]
#              ]
GRID_SIZE = 5
IMG_SIZE = 35
XPAD = 20
YPAD = 40
WIN_HEIGHT = 2 * YPAD + GRID_SIZE * IMG_SIZE + 100
WIN_WIDTH = 2 * XPAD + (3 * GRID_SIZE - 1) * IMG_SIZE

class gameGrid():
    #white = Image.open(fp="images/hex_50.png")
    def __init__(self, frame):
        self.frame = frame
        self.white = PhotoImage(file="images/white35.gif")
        #self.white = self.white.zoom(30)
        self.red = PhotoImage(file="images/hex_white.png")
        self.blue = PhotoImage(file="images/blue35.gif")
        self.drawGridHEX()
        #self.playInfo = playInfo()

    def drawGrid(self):
        for yi in range(0, GRID_SIZE):
            xi = XPAD + yi * IMG_SIZE
            for i in range(0, GRID_SIZE):
                l = Label(self.frame, image=self.white)
                l.pack()
                l.image = self.white
                l.place(anchor=NW, x=xi, y=YPAD + yi * IMG_SIZE)
                l.bind('<Button-1>', lambda e: self.on_click(e))
                xi += 2 * IMG_SIZE

    def drawGridHEX(self):
        hex_row = 5 # start at row_0, increment for every 1, 2, ... , dim by 1
        # x - horisonta, y - vertical
        count = 0
        rotation = -0.785398 # radians aprox 45 degree
        for xi in range(0, GRID_SIZE): # 
            #yi = XPAD + xi * IMG_SIZE
            for yi in range(0, GRID_SIZE):
                xpad = cos(rotation)*xi - sin(rotation)*yi
                ypad = sin(rotation)*yi + cos(rotation)*xi
                print(xi,yi, "->",xpad, ypad)
                xpad = GRID_SIZE - xi
                l = Label(self.frame, image=self.white)
                l.pack()
                l.image = self.white
                l.place(anchor=NW, x=xpad*IMG_SIZE, y=YPAD*yi + IMG_SIZE)
                l.bind('<Button-1>', lambda e: self.on_click(e))
                #yi += 2 * IMG_SIZE
                count += 1 
                if(count%5 == 0):
                    print("hex_row",hex_row,xi,yi)
                    hex_row -= 1
                    if(hex_row == 0):
                        hex_row = 5

    def getCoordinates(self, widget):
        row = (widget.winfo_y() - YPAD) / IMG_SIZE
        col = (widget.winfo_x() - XPAD - row * IMG_SIZE) / (2 * IMG_SIZE)
        return row + 1, col + 1

    def toggleColor(self, widget):
        widget.config(image=self.blue)
        widget.image = self.blue
        #if self.playInfo.mode == 1:
            #widget.config(image=self.red)
            #widget.image = self.red
        #else:
            #widget.config(image=self.blue)
            #widget.image = self.blue

    def display_winner(self, winner):
        winner_window = Tk()
        winner_window.wm_title("Winner")
        frame = Frame(winner_window, width=40, height=40)
        frame.pack()
        label = Label(frame,text = "Winner is Player : " + winner )
        label.pack()
        label.place(anchor=NW, x = 20, y = 20)
    
    def on_click(self,event):
        print(event)

    """
    def on_click(self, event):
        if event.widget.image != self.white:
            return
        self.toggleColor(event.widget)
        a, b = self.getCoordinates(event.widget)
        self.playInfo.board[a][b] = self.playInfo.mode
        #self.playInfo.printBoard()
        if self.playInfo.isWinning(a, b):
            winner = ""
            if self.playInfo.mode == 0:
                winner = " 1 ( Blue ) "
            else:
                winner += " 2 ( Blue ) "
            self.display_winner(winner)
            self.quit()
        self.playInfo.mode = (self.playInfo.mode + 1) % 2
    """
# [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]
class gameWindow:
    def __init__(self, window):
        self.frame = Frame(window, width=WIN_WIDTH, height=WIN_HEIGHT)
        self.frame.pack()
        self.gameGrid = gameGrid(self.frame)


if(__name__ == "__main__"):
    window = Tk()
    window.wm_title("Hex Game")
    gameWindow(window)
    window.mainloop()