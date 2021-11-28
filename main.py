import cv2
import pytesseract
import math
import numpy as np


def colored_mask(img, color):
    # Размытие для удаления мелких шумов.
    denoised = cv2.GaussianBlur(img, (3, 3), 3)
    cv2.imwrite('denoised.bmp', denoised)

    # Сохранение в ЧБ для получения маски.
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.bmp', gray)

    # Получение цветной части изображения.
    colored = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    if color == "RED":
        mask0 = cv2.inRange(colored, (30, 70, 70), (0, 225, 255))
        mask1 = cv2.inRange(colored, (140, 70, 70), (180, 225, 255))
        mask = mask0 + mask1
    else:
        mask = cv2.inRange(colored, (90, 70, 70), (130, 255, 255))

    # Создание маски цветной части изображения.
    dst = cv2.bitwise_and(gray, gray, mask=mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('colors_mask.bmp', closed)
    cv2.imwrite('test_res.bmp', closed)
    return closed


def contour_mask(img, color):
    binary = colored_mask(img, color)
    contours, hireratchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,)
    return contours


def find_color(path, color):
    counter = 1
    image = cv2.imread(path)
    image = cv2.resize(image, (600, 775))
    if color == "RED":
        contour = np.array(contour_mask(image, color))
    if color == "BLUE":
        contour = np.array(contour_mask(image, color))
    if (contour.ndim != 4):
        counter = contour.size

    return counter


# TABLE DETECT
def empty():
    pass


def find_table_pice(current_point, points):
    ru = (0, 0)
    ld = (0, 0)
    rd = (0, 0)

    for i in range(len(points)):
        if (points[i][0] >= current_point[0] - 7) and (points[i][0] <= current_point[0] + 7) and (points[i][1] > current_point[1]):
            ru = points[i]
            break
    for i in range(len(points)):
        if (points[i][1] >= current_point[1] - 7) and (points[i][1] <= current_point[1] + 7) and (points[i][0] > current_point[0]):
            ld = points[i]
            break
    for i in range(len(points)):
        if (points[i][0] >= ld[0] - 7) and (points[i][0] <= ld[0] + 7) and (points[i][1] >= ru[1] - 7) and (points[i][1] <= ru[1] + 7):
            rd = points[i]
            break

    return ((ru != (0, 0)) and (ld != (0, 0)) and (rd != (0, 0)))


def find_number_of_pices(points):
    table_counter = 0
    for point in points:
        if (find_table_pice(point, points)):
            table_counter += 1
    return table_counter


def find_samge_pixel_around(img, y, x):
    for row in range(y-7, y+7):
        for column in range(x-7, x+7):
            if not (row == y and column == x):
                try:
                    img[row][column] = [255, 255, 255]
                except:
                    pass
    return img


def findpoints(img):
    points = []
    for y in range(len(img)):
        for x in range(len(img[0])):
            if (img[y][x][0] == 0) and (img[y][x][1] == 255) and (img[y][x][2] == 0):
                points.append((y, x))
                img = find_samge_pixel_around(img, y, x)
    return img, points


def delete_colored(img, colored):
    for y in range(len(img)):
        for x in range(len(img[0])):
            if(colored[y][x] > 0):
                img[y][x] = [255, 255, 255]


def table_cells_count(path):

    image = cv2.imread(path)
    image = cv2.resize(image, (600, 775))

    delete_colored(image, colored_mask(image, "RED"))
    delete_colored(image, colored_mask(image, "BLUE"))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
                        gray, 255,
                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                        25, 5)
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)

    thresh_contours, hirarity = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in thresh_contours:
        cv2.drawContours(thresh, [c], -1, (255), thickness=1)

    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / 80)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)


    rows = vertical.shape[0]
    vertical_size = math.ceil(rows / 80)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    verticalContours, hirarity = cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    horizontalContours, hirarity = cv2.findContours(horizontal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    for c in verticalContours:
        cv2.drawContours(vertical, [c], -1, (255), thickness=3)
    for c in horizontalContours:
        cv2.drawContours(horizontal, [c], -1, (255), thickness=3)

    table = (horizontal / 255) * (vertical / 255) * 255

    table = np.array(table)

    for y in range(len(table)):
        for x in range(len(table[0])):
            if(table[y][x] > 0):
                 image[y][x] = [0, 255, 0]

    image, points = findpoints(image)

    return find_number_of_pices(points)


# tesseract
def colors_killer(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # маска для удаления всех цветов, кроме черного и белого с изображения
    lower_color = np.array([0, 30, 30])  # 0 30 30
    upper_color = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # mask = cv2.resize(mask, (0, 0), fx=0.4, fy=0.4)
    # cv2.imshow('Result', hsv)
    # cv2.waitKey(0)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] > 0:
                img[i][j] = [255, 255, 255]

    return img


def find_horizontal(img):
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255 - img_bin
    kernel_length = np.array(img).shape[1] // 40
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    horizontal_lines_list = []
    for y in range(0, len(horizontal_lines_img)):
        for x in range(len(horizontal_lines_img[0])):
            if horizontal_lines_img[y][x] > 0:
                if y not in horizontal_lines_list:
                    horizontal_lines_list.append(y)

    return horizontal_lines_list


# создание словаря из строк, где ключ - это кортеж формата (абзац, строка)
def make_lines_and_kill_table_words(img, data, horizontal_lines_list):
    for i, el in enumerate(data.splitlines()):
        if i == 0:
            continue
        el = el.split()
        par_num, line_num, x, y, w, h = int(el[3]), int(el[4]), int(el[6]), int(el[7]), int(el[8]), int(el[9])

        if x != 0 and y != 0 and el[10] != '-1':
            for y_table in horizontal_lines_list:
                if (y_table - 10 <= y + h <= y_table + 10 or y_table - 10 <= y <= y_table + 10) and y + h not in horizontal_lines_list:
                    if [(x, y), (x + w, y + h)] not in table_words_coord:
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                        table_words_coord.append([(x, y), (x + w, y + h)])

        if x != 0 and y != 0 and el[10] != '-1':
            text = el[11]
            if (par_num, line_num) not in lines:
                lines[par_num, line_num] = [x, x + w, text, y, y + h]
            else:
                # если след элем в строке очень далеко от предыдущего, то не записываем (а если строка в длинну меньше чем 2 своих высоты, то записываем)
                if (lines[par_num, line_num][1] - lines[par_num, line_num][0]) / 1.5 > x - lines[par_num, line_num][1] or (lines[par_num, line_num][4] - lines[par_num, line_num][3]) / (lines[par_num, line_num][1] - lines[par_num, line_num][0]) > 0.5:
                    lines[par_num, line_num][1] = x + w
                    lines[par_num, line_num][2] += ' ' + text
        # print(el)


def get_title(lines, width, horizontal_lines_list):
    title = ''
    lines_keys = list(lines.keys())
    found_title = None
    for i in range(len(lines_keys)):
        value = lines[lines_keys[i]]
        flag = True  # если найдена строка не из таблицы и не содержащая спец разметки
        #  x1 - левая часть строки, x2 - крайняя правая часть строки
        x1, x2, text, y1, y2 = value[0], value[1], value[2], value[3], value[4]
        # true, если расстояния до правого и левого конца страницы различаются не больше, чем на 1/6 (подобрал примерно)
        if math.fabs(x1 - (width - x2)) <= width / 6 and found_title is None:
            title = text
            found_title = lines_keys[i]

        for y_table in horizontal_lines_list:
            # print(y_table - 15, el[4], y_table + 15)
            if y_table - 15 <= y2 <= y_table + 15 or y_table - 15 <= y1 <= y_table + 15 or math.fabs(
                    x1 - (width - x2)) > width / 6 or found_title == lines_keys[i]:
                flag = False
                break
        if flag:
            clear_lines[lines_keys[i]] = value

    for replace_el in replace_mass:
        title = title.replace(replace_el, ' ')

    if title == title.upper():
        return title.upper()[0] + title.lower()[1:]
    return title


def get_desc(clear_lines, width):
    desc = ''
    clear_lines_keys = list(clear_lines.keys())
    start_desc_index = len(clear_lines_keys)
    for i in range(len(clear_lines_keys) - 1):
        value = clear_lines[clear_lines_keys[i]]
        value_next = clear_lines[clear_lines_keys[i + 1]]
        x1, x2, text, y1, y2 = value[0], value[1], value[2], value[3], value[4]
        x1_next, x2_next, text_next, y1_next, y2_next = value_next[0], value_next[1], value_next[2], value_next[3], value_next[4]
        if math.fabs(x1 - (width - x2)) <= width / 6 and math.fabs(x1_next - (width - x2_next)) <= width / 6 and y2_next - y1 < (y2 - y1) * 3.5:
            start_desc_index = i
            break

    counter = 0
    for i in range(start_desc_index, len(clear_lines_keys)):
        text = clear_lines[clear_lines_keys[i]][2]
        for replace_el in replace_mass:
            text = text.replace(replace_el, ' ')
        text = text.split()

        for j in text:
            if counter < 10:
                # desc += j.strip(' ,.;"\':!&?-  (){}[]') + ' '
                desc += j.strip() + ' '
                counter += 1
            else:
                break

    return desc[:len(desc)-1]


config = r'-l rus --oem 3 --psm 6'
# pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
lines = {}
cur_par_num, cur_line_num = 0, 0
page_list, par_list, line_list = [], [], []
table_words_coord = []
clear_lines = {}  # строки, максимально очищенные от строк из таблиц и спец разметки
replace_mass = [' ', ',', '.', ';', '"', "'", ':', '!', '&', '?', '- ', ' -', ' - ', '  ', '(', ')', '{', '}', '[', ']', '/', '\\', '‘', '’', '“', '”', '`']


def clear_data():
    global lines, cur_par_num, cur_line_num, page_list, par_list, line_list, table_words_coord, clear_lines
    lines = {}
    cur_par_num, cur_line_num = 0, 0
    page_list, par_list, line_list = [], [], []
    table_words_coord = []
    clear_lines = {}  # строки, максимально очищенные от строк из таблиц и спец разметки


# def tesseract_ocr(path):
#     img = cv2.imread(path)
#     img_gray = cv2.imread(path, 0)
#     img = colors_killer(img)
#     height, width, _ = img.shape
#     data = pytesseract.image_to_data(img, config=config)
#     horizontal_lines_list = find_horizontal(img_gray)
#     make_lines_and_kill_table_words(img, data, horizontal_lines_list)
#     # print(lines)
#     print(get_title(lines, width, horizontal_lines_list))
#     # print(clear_lines)
#     print(get_desc(clear_lines, width))
#     # img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
#     # cv2.imshow('Result', img)
#     # cv2.waitKey(0)


# MAIN

def extract_doc_features(filepath):
    img = cv2.imread(filepath)
    img_gray = cv2.imread(filepath, 0)
    img = colors_killer(img)
    height, width, _ = img.shape
    data = pytesseract.image_to_data(img, config=config)
    horizontal_lines_list = find_horizontal(img_gray)
    make_lines_and_kill_table_words(img, data, horizontal_lines_list)
    result = {
        'red_areas_count': find_color(filepath, "RED"), # количество красных участков
        'blue_areas_count': find_color(filepath, "BLUE"), # количество синих областей
        'text_main_title': get_title(lines, width, horizontal_lines_list), # текст главного заголовка страницы
        'text_block': get_desc(clear_lines, width), # текстовый блок параграфа страницы
        'table_cells_count': table_cells_count(filepath), # количество ячеек
    }
    clear_data()

    return result