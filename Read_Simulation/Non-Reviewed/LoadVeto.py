import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))
import warnings as warn

global_dtype = \
    np.array([np.uint32, np.uint8, np.uint8, np.uint8, np.uint8,
              np.uint8, np.uint8, np.uint8, np.uint8,
              np.uint8, np.uint8, np.uint8, np.uint8,
              np.uint8, np.uint8, np.uint8, np.uint8,
              np.uint8, np.uint8, np.uint8, np.uint8,
              np.uint8, np.uint8, np.uint8, np.uint8])

column_event = "event"
column_front0 = "front0"
column_front1 = "front1"
column_front2 = "front2"
column_front3 = "front3"
column_back0 = "back0"
column_back1 = "back1"
column_back2 = "back2"
column_back3 = "back3"
column_left0 = "left0"
column_left1 = "left1"
column_left2 = "left2"
column_left3 = "left3"
column_right0 = "right0"
column_right1 = "right1"
column_right2 = "right2"
column_right3 = "right3"
column_top0 = "top0"
column_top1 = "top1"
column_top2 = "top2"
column_top3 = "top3"
column_bottom0 = "bottom0"
column_bottom1 = "bottom1"
column_bottom2 = "bottom2"
column_bottom3 = "bottom3"

global_columns = \
    np.array([column_event, column_front0, column_front1, column_front2, column_front3,
              column_back0, column_back1, column_back2, column_back3,
              column_left0, column_left1, column_left2, column_left3,
              column_right0, column_right1, column_right2, column_right3,
              column_top0, column_top1, column_top2, column_top3,
              column_bottom0, column_bottom1, column_bottom2, column_bottom3])

def load_column(path, col, dtype, mt=False):

    def get_arr(path):
        p = os.path.join(path, "Event")
        ending = "_" + col + ".binary"

        if mt:
            for i in range(4):
                if i == 0:
                    arr = np.fromfile(
                        os.path.join(p, str(int(i)) + ending), dtype=dtype)
                else:
                    arr = np.append(
                        arr, np.fromfile(
                            os.path.join(p, str(int(i)) + ending), dtype=dtype))
        else:
            try :
                arr = np.fromfile(
                    os.path.join(p, str(int(-2)) + ending), dtype=dtype)
            except FileNotFoundError:
                arr = np.fromfile(
                    os.path.join(p, str(int(-1)) + ending), dtype=dtype)
        return arr

    if type(path) == str:
        return get_arr(path)
    else:
        temp = None
        for p in path:
            if temp is None:
                temp = get_arr(p)
            else:
                if col == prims_column_event:
                    temp = np.append(temp, get_arr(p) + temp.shape[0])
                else:
                    temp = np.append(temp, get_arr(p))
        return temp
        
def load_events(path, mt=False, **kwargs):
    #print("---------------------------------------")
    #print("----------- Loading events ------------")
    #print("---------------------------------------")
    arrays = {}
    data = {}

    for i in range(global_columns.shape[0]):
        arrays[i] = load_column(path, global_columns[i], global_dtype[i], mt=mt)
        data[global_columns[i]] = arrays[i]

    frame = pd.DataFrame(data=data)
    arrays.clear()
    data.clear()
    #print("---------------------------------------")
    #print("------ Completed loading events -------")
    #print("---------------------------------------")
    #print(" ")
    return frame
