import tkinter

import matplotlib
import numpy as np
import random
from tkinter import *
from tkinter.ttk import Combobox
from tkinter.ttk import Checkbutton
import imgkit
import PIL
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import PolynomialFeatures
from bokeh.plotting import figure, show, output_file
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression
matplotlib.style.use('ggplot')

window = Tk()

wb2020_2021 = load_workbook('./1628665048603762.xlsx')
wb2021_2022 = load_workbook('./1628611575213049.xlsx')

sheet_1 = wb2020_2021.worksheets[0]
sheet_2 = wb2021_2022.worksheets[0]


year1 = 2020
year2 = 2021


def GetDayInMonth(year, month):
    day = 0
    coef = 0
    if year % 4 == 0:
        coef = 1
    else:
        coef = 0
    if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        day = 31
    elif month == 2:
        day = 28 + coef
    else:
        day = 30
    return day


def CreateEmptyList():
    list = [[]]
    for month in range(1, 13):
        List1 = [[]]
        days = GetDayInMonth(2020, month)
        for day in range(1, days+1):
            List2 = []
            for hour in range(1, 25):
                List3 = [hour]
                List2.extend(List3)
            List1.append(List2)
        list.append(List1)
    return list


def CreateTestList(List):
    row = 7
    percent = 0
    for month in range(1, 13):
        days = GetDayInMonth(2020, month)
        for day in range(1, days+1):
            for hour in range(0, 24):
                if sheet_1[row][6].value == hour:
                    List[month][day][hour] = sheet_1[row][8].value
                    row += 1
                else:
                    List[month][day][hour] = random.randint(0, 20)
        percent += 4
        print(percent, '%')
    print('end read exel')
    return List


def CreatePredictList(List):
    row = 7
    percent = 52
    for month in range(1, 13):
        days = GetDayInMonth(2021, month)
        for day in range(1, days + 1):
            for hour in range(0, 24):
                if sheet_2[row][5].value == hour:  # Так как ексель с пропущенными данными, то мы их добавляем рандомайзером от 1 до 10
                    List[month][day][hour] = sheet_2[row][7].value
                    row += 1
                else:
                    List[month][day][hour] = random.randint(0, 20)
        percent += 4
        print(percent, '%')
    print('end read second exel')
    return List


checkList = CreateEmptyList()
ListLavina = CreateTestList(checkList)
checkList = CreateEmptyList()
ListPredictLavina = CreatePredictList(checkList)

enterWeekWithYellowBlock = [0, 0, 0, 0]
enterWeekWithRedBlock = [0, 0, 0, 0]


def PrintGraphics(x, y, y_pred):
    p = figure(title="Actual vs Predict", width=1350, height=900)
    p.title.align = 'center'
    p.circle(x, y_pred)
    p.line(x, y, legend_label='Actual', line_width=3, line_alpha=0.4)
    p.circle(x, y_pred, color="red")
    p.line(x, y_pred, color="red", legend_label='Predict', line_width=3, line_alpha=0.4)
    p.xaxis.axis_label = 'day in month by hour (day * hour)'
    p.yaxis.axis_label = 'Visitors'
    show(p)


def PolynomialRegress(month):
    pf = PolynomialFeatures(degree=5, include_bias=False)
    day = GetDayInMonth(year1, month)
    day2 = GetDayInMonth(year2, month-1)
    coef = 1
    lowcoef = 0.5
    ListDay = []
    ListHour = []
    ListDataActual = []
    ListDataPredict = []
    ListSecondDataTrainingTemp = []
    ListDataTemp = []
    startBlock = 0
    endBlock = 0
    for chBlock in range(0, 4):
        if enterWeekWithYellowBlock[chBlock] == 1:
            if startBlock == 0:
                startBlock = 1 + 7 * chBlock
            endBlock = 7 + 7 * chBlock
            coef = 0.9467  # coef
        if enterWeekWithRedBlock[chBlock] == 1:
            if startBlock == 0:
                startBlock = 1 + 7 * chBlock
            endBlock = 7 + 7 * chBlock
            coef = 0.8682  # coef
    for days in range(1, day * 24 + 1):
        ListDay.append(days)
    for hour in range(0, 24):
        ListHour.append(hour)
        ListHour.append(hour)
    x = np.array(ListHour).reshape((-1, 1))
    pf.fit(x)
    for days in range(1, day + 1):
        ListDataTemp.clear()
        ListSecondDataTrainingTemp.clear()
        index = 0
        for hour in range(0, 24):
            if days <= day2:
                ListSecondDataTrainingTemp.append(ListPredictLavina[month-1][days][hour])
            ListDataActual.append(ListPredictLavina[month][days][hour])
            if days != day:
                ListDataTemp.append(ListLavina[month][days+1][hour])
                if days <= day2:
                    ListDataTemp.append(ListSecondDataTrainingTemp[index])
                elif days > day2:
                    ListDataTemp.append(ListLavina[month][days+1][hour])
            elif days == day:
                ListDataTemp.append(ListLavina[month+1][1][hour])
                if days <= day2:
                    ListDataTemp.append(ListSecondDataTrainingTemp[index])
                elif days > day2:
                    ListDataTemp.append(ListLavina[month+1][1][hour])
            index += 1
        x_ = pf.transform(x)
        model = LinearRegression().fit(x_, ListDataTemp)
        y_pred = model.predict(x_)
        if startBlock < days <= endBlock:
            for i in range(0, 48):
                y_pred[i] = int(y_pred[i] * coef)
        for iters in range(0, 48):
            if y_pred[iters] < 0:
                y_pred[iters] = int(y_pred[iters] * -1)
            else:
                y_pred[iters] = int(y_pred[iters])
            if 0 <= iters < 8:
                y_pred[iters] = int(y_pred[iters] * lowcoef)
        ListDataPredict.extend(y_pred)
    ListDataPredict = mathAverage(ListDataPredict)
    writeArray(ListDataPredict, ListDataActual, month)
    PrintGraphics(ListDay, ListDataActual, ListDataPredict)


def mathAverage(array):
    arraytemp = []
    for lenar in range(0, int(array.__len__() / 2)):
        average = (array[(lenar * 2)] + array[(lenar * 2 + 1)]) / 2
        arraytemp.append(average)
    return arraytemp


def writeArray(array, arraylavina, month):
    day = GetDayInMonth(year1, month)
    array2 = []
    array3 = []
    for days in range(1, day+1):
        for hour in range(0, 24):
            array2.append(hour)
            array3.append(days)
    df = pd.DataFrame({'Day': array3, 'Hour': array2, 'Visitors': array, 'Lavina': arraylavina})
    df.to_excel('./predict.xlsx')


def writeYellowOrRed(lvl, count):
    if lvl == 1:
        enterWeekWithYellowBlock[count-1] = 1
    if lvl == 2:
        enterWeekWithRedBlock[count-1] = 1


def clicked():
    lvl = 0
    enterMonth = int(combo.get())
    text = int(selected.get())
    if text == 1:
        lvl = 1
    elif text == 2:
        lvl = 2
    if chk1_state.get():
        writeYellowOrRed(lvl, 1)
    if chk2_state.get():
        writeYellowOrRed(lvl, 2)
    if chk3_state.get():
        writeYellowOrRed(lvl, 3)
    if chk4_state.get():
        writeYellowOrRed(lvl, 4)
    PolynomialRegress(enterMonth)


window.title("Interface of program")
window.geometry('550x250')
lbl = Label(window, text="Enter a month number")
lbl.grid(column=0, row=0)
combo = Combobox(window)
combo['values'] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
combo.current(0)
combo.grid(column=1, row=0)
lbl2 = Label(window, text="Add restrictions")
lbl2.grid(column=0, row=1)
chk1_state = BooleanVar()
chk2_state = BooleanVar()
chk3_state = BooleanVar()
chk4_state = BooleanVar()
chk1_state.set(False)
chk2_state.set(False)
chk3_state.set(False)
chk4_state.set(False)
chk1 = Checkbutton(window, text='1', var=chk1_state, width=3)
chk2 = Checkbutton(window, text='2', var=chk2_state, width=3)
chk3 = Checkbutton(window, text='3', var=chk3_state, width=3)
chk4 = Checkbutton(window, text='4', var=chk4_state, width=3)
chk1.grid(column=1, row=1)
chk2.grid(column=2, row=1)
chk3.grid(column=3, row=1)
chk4.grid(column=4, row=1)
selected = IntVar()
rad1 = Radiobutton(window, text='Yellow', value=1, variable=selected)
rad2 = Radiobutton(window, text='Red', value=2, variable=selected)
rad1.grid(column=5, row=1)
rad2.grid(column=6, row=1)
btn = Button(window, text="Enter", command=clicked)
btn.grid(column=0, row=2)

window.mainloop()